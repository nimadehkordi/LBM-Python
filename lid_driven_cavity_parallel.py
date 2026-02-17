"""
Lid-Driven Cavity Flow — MPI-parallel Lattice Boltzmann simulation.

Distributes the cavity domain across multiple MPI processes using
a 2D Cartesian decomposition. Each process handles a subdomain and
communicates ghost cells with its neighbors at every time step.

Boundary conditions are identical to the serial version:
    - Top wall: moving lid with horizontal velocity U_lid
    - Left, right, bottom walls: no-slip (bounce-back)

Usage:
    mpirun -np 4 python lid_driven_cavity_parallel.py \\
        --nx 300 --ny 300 --px 2 --py 2 \\
        --steps 10000 --omega 1.7 --lid-velocity 0.1
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import lbm
import parallel

plt.rcParams.update({"font.size": 14})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lid-Driven Cavity — MPI parallel (LBM)"
    )
    parser.add_argument("--nx", type=int, default=300, help="Global grid width")
    parser.add_argument("--ny", type=int, default=300, help="Global grid height")
    parser.add_argument("--px", type=int, default=2, help="Processes in x-direction")
    parser.add_argument("--py", type=int, default=2, help="Processes in y-direction")
    parser.add_argument("--steps", type=int, default=10000, help="Number of time steps")
    parser.add_argument("--omega", type=float, default=1.7, help="Relaxation frequency")
    parser.add_argument("--lid-velocity", type=float, default=0.1, help="Lid velocity")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    # Create 2D Cartesian communicator (non-periodic)
    communicator = MPI.COMM_WORLD.Create_cart(
        (args.py, args.px), periods=(False, False)
    )
    rank = communicator.Get_rank()

    # Local subdomain dimensions
    local_nx = args.nx // args.px
    local_ny = args.ny // args.py

    if rank == 0:
        re = lbm.reynolds_number(args.omega, args.ny, args.lid_velocity)
        print(f"Lid-Driven Cavity (parallel): {args.nx}x{args.ny} grid, "
              f"{args.px}x{args.py} processes")
        print(f"  omega={args.omega}, lid_vel={args.lid_velocity}, Re={re:.1f}")
        print(f"  Local subdomain: {local_nx}x{local_ny}")

    # All walls solid except top (moving lid)
    boundary = (True, True, False, True)

    rho = np.ones((local_nx, local_ny), dtype=np.float64)
    u = np.zeros((local_nx, local_ny, 2), dtype=np.float64)
    f = lbm.equilibrium(rho, u)

    for t in range(args.steps):
        f = parallel.communicate_ghosts(f, communicator)
        f = lbm.streaming(f, boundary, local_nx, local_ny, args.lid_velocity)
        f, rho, u = lbm.collision(f, args.omega)

        if rank == 0 and (t + 1) % 1000 == 0:
            print(f"  Step {t + 1}/{args.steps}")

    # Save velocity components using parallel I/O
    parallel.save_mpiio(communicator, "output/ux.npy", u[:, :, 1])
    parallel.save_mpiio(communicator, "output/uy.npy", u[:, :, 0])

    # Rank 0 generates the streamline plot
    if rank == 0:
        ux = np.load("output/ux.npy")
        uy = np.load("output/uy.npy")
        nx_global, ny_global = ux.shape

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.streamplot(
            np.arange(nx_global), np.arange(ny_global), ux.T, uy.T
        )
        ax.set_aspect("equal")
        ax.set_title(
            f"Lid-Driven Cavity (parallel) — {args.nx}x{args.ny}, "
            f"$\\omega$={args.omega}, t={args.steps}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig("output/lid_driven_cavity_parallel.png", dpi=150)
        plt.close(fig)
        print("Done. Plot saved to output/lid_driven_cavity_parallel.png")


if __name__ == "__main__":
    main()
