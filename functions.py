#-------------imports-------------
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})
from mpi4py import MPI
import constants

# -------------functions-------------
# calculate density at x, t point on pdfGrid d
def f_rho(f):
    # should calculate whole grid
    return np.sum(f, axis=2)


def f_vel(f):
    # first calculate rho for the given f
    rho = f_rho(f)

    # calculate and return velocity
    """
    dimensions: 

    f -> length*width*directions
    c -> directions*2
    f.c -> length*width*2
    (f.c).T -> 2*width*length

    """
    return (np.dot(f, constants.c).T / rho.T).T

def streaming(f, width= None, length = None, boundry=None, lid_vel=None):
    f_old = np.copy(f)

    # stream
    for i in range(constants.q):
        f[:, :, i] = np.roll(np.roll(f[:, :, i], constants.c[i, 0], axis=0), constants.c[i, 1], axis=1)
    
    # bounce_back at existing not moving walls
    if boundry:
        for i in range(constants.q):
            b = set_boundry(boundry, width, length)
            # bounce back where the wall exists
            f[:, :, i] = np.where(b, f_old[:, :, constants.opposite_direction[i]], f[:, :, i])

    # if upper lid velocity exists
    if lid_vel:
        # moving lid in upper boundry
        # update pdf according to lid velocity
        rho_wall = 2 * (f_old[:, -1, 6] + f_old[:, -1, 2] + f_old[:, -1, 5]) + \
                   f_old[:, -1, 3] + f_old[:, -1, 0] + f_old[:, -1, 1]

        f[:, -1, 4] = f_old[:, -1, constants.opposite_direction[4]] - 6 * constants.w_i[4] * rho_wall * np.dot(constants.c[4], [lid_vel, 0])
        f[:, -1, 8] = f_old[:, -1, constants.opposite_direction[8]] - 6 * constants.w_i[8] * rho_wall * np.dot(constants.c[8], [lid_vel, 0])
        f[:, -1, 7] = f_old[:, -1, constants.opposite_direction[7]] - 6 * constants.w_i[7] * rho_wall * np.dot(constants.c[7], [lid_vel, 0])
    
    return f

# Equilibrium distribution function.
def equilibrium(rho, u):

    cu = np.dot(u,constants.c.T)
    cu2 = cu ** 2
    u2 = u[:,:,0] ** 2 + u[:,:,1] ** 2

    return ((1 + 3*(cu.T) + 9/2*(cu2.T) - 3/2*(u2.T)) * rho.T ).T * constants.w_i

def collision(f, omega):
    rho = f_rho(f)
    u = f_vel(f)
    # to compute the new f, first we need to compute feq
    feq = equilibrium(rho, u)
    # relaxation
    f += omega * (feq - f)
    return f, rho, u


def set_boundry(boundry, width, length):
    Mask = np.zeros((width, length))

    Mask[:,  0] = boundry[0]  # left
    Mask[:, -1] = boundry[1]  # right
    Mask[0,  :] = boundry[2]  # top
    Mask[-1, :] = boundry[3]  # bottom

    # convert the boundries to True/False Mask
    return Mask == 1

def plot_velocity(u, steps, milestone, re, lid_vel, omega, figsize = (15,15), width = 300, length = 300):
    fig = plt.figure(figsize=figsize)
    if lid_vel:
        plt.title("lattice dimensions: (%d * %d), omega: %.1f, lid velocity = %.1f, steps: %d, Re: %0.1f " %(width, length, omega, lid_vel, steps, re))
    else:
        plt.title("lattice dimensions: (%d * %d), omega: %.1f, steps: %d, elapsed_time: %0.1f seconds" %(width, length, omega,steps, elapsed_time))
    plt.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T)
    plt.xlabel("lenght")
    plt.ylabel("width")
    plt.xticks(np.arange(0, length+1, 25))
    plt.yticks(np.arange(0, width+1, 25))
    plt.savefig("milestone_%d_%d_%d_%.1f_%d.png" %(milestone, width, length, omega,steps))

def calculate_re(omega, length, lid_vel):
    nu = 1/3 * (1/omega - 1/2)
    return (lid_vel*length) / (nu)
 
# initialize MPI communicator
def communicate_f(f, communicator):
    
    # get MPI ranks of neighbor cells
    left_src, left_dst = communicator.Shift(1, -1)
    right_src, right_dst = communicator.Shift(1, 1)
    bottom_src, bottom_dst = communicator.Shift(0, 1)
    top_src, top_dst = communicator.Shift(0, -1)

    # Send to left
    recvbuf = f[:, -1, :].copy()
    communicator.Sendrecv(f[:, 1, :].copy(), left_dst,
                  recvbuf=recvbuf, source=left_src)
    f[:, -1, :] = recvbuf
    
    # Send to right
    recvbuf = f[:, 0, :].copy()
    communicator.Sendrecv(f[:, -2, :].copy(), right_dst,
                  recvbuf=recvbuf, source=right_src)
    f[:, 0, :] = recvbuf
    
    # Send to top
    recvbuf = f[-1, :, :].copy()
    communicator.Sendrecv(f[1, :, :].copy(), top_dst,
                  recvbuf=recvbuf, source=top_src)
    f[-1, :, :] = recvbuf
    
    # Send to bottom
    recvbuf = f[0, :, :].copy()
    communicator.Sendrecv(f[-2, :, :].copy(), bottom_dst,
                  recvbuf=recvbuf, source=bottom_src)
    f[0, :, :] = recvbuf
    return f

def save_mpiio(comm, fn, g_kl):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm
        MPI communicator.
    fn : str
        File name.
    g_kl : array_like
        Portion of the array on this MPI processes. This needs to be a
        two-dimensional array.
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

    arr_dict_str = str({ 'descr': dtype_to_descr(g_kl.dtype),
                         'fortran_order': False,
                         'shape': (np.asscalar(nx), np.asscalar(ny)) })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()
