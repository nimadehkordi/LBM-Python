"""
MPI-parallel utilities for the Lattice Boltzmann Method.

Provides ghost-cell communication for domain decomposition and
parallel file I/O for writing distributed arrays in NumPy format.

Requires mpi4py. Run with:
    mpirun -np <N> python lid_driven_cavity_parallel.py [args...]
"""

import numpy as np
from mpi4py import MPI


def communicate_ghosts(f, communicator):
    """Exchange ghost-cell layers between neighboring MPI processes.

    Uses a Cartesian communicator to send/receive boundary slices
    in all four directions (left, right, top, bottom).

    Parameters
    ----------
    f : ndarray, shape (local_Nx, local_Ny, Q)
        Local distribution function (modified in-place).
    communicator : MPI.Cartcomm
        2D Cartesian communicator.

    Returns
    -------
    ndarray
        Distribution function with updated ghost cells.
    """
    left_src, left_dst = communicator.Shift(1, -1)
    right_src, right_dst = communicator.Shift(1, 1)
    bottom_src, bottom_dst = communicator.Shift(0, 1)
    top_src, top_dst = communicator.Shift(0, -1)

    # Left <-> Right
    recvbuf = f[:, -1, :].copy()
    communicator.Sendrecv(f[:, 1, :].copy(), left_dst,
                          recvbuf=recvbuf, source=left_src)
    f[:, -1, :] = recvbuf

    recvbuf = f[:, 0, :].copy()
    communicator.Sendrecv(f[:, -2, :].copy(), right_dst,
                          recvbuf=recvbuf, source=right_src)
    f[:, 0, :] = recvbuf

    # Top <-> Bottom
    recvbuf = f[-1, :, :].copy()
    communicator.Sendrecv(f[1, :, :].copy(), top_dst,
                          recvbuf=recvbuf, source=top_src)
    f[-1, :, :] = recvbuf

    recvbuf = f[0, :, :].copy()
    communicator.Sendrecv(f[-2, :, :].copy(), bottom_dst,
                          recvbuf=recvbuf, source=bottom_src)
    f[0, :, :] = recvbuf

    return f


def save_mpiio(comm, filename, g_kl):
    """Write a distributed 2D array to a single .npy file using MPI-IO.

    The resulting file can be read with ``numpy.load()``.

    Parameters
    ----------
    comm : MPI.Cartcomm
        2D Cartesian communicator.
    filename : str
        Output file name (e.g. ``'velocity_x.npy'``).
    g_kl : ndarray, shape (local_Nx, local_Ny)
        Local portion of the global 2D array.
    """
    from numpy.lib.format import dtype_to_descr, magic

    magic_str = magic(1, 0)
    local_nx, local_ny = g_kl.shape

    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({
        "descr": dtype_to_descr(g_kl.dtype),
        "fortran_order": False,
        "shape": (int(nx), int(ny)),
    })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += " "
    arr_dict_str += "\n"
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny * local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    fh = MPI.File.Open(comm, filename, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        fh.Write(magic_str)
        fh.Write(np.int16(len(arr_dict_str)))
        fh.Write(arr_dict_str.encode("latin-1"))

    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    fh.Set_view(header_len + (offsety + offsetx) * mpitype.Get_size(),
                filetype=filetype)
    fh.Write_all(g_kl.copy())
    filetype.Free()
    fh.Close()
