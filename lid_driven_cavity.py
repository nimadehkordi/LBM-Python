"""
Lid-Driven Cavity Flow — Lattice Boltzmann simulation.

Simulates recirculating flow in a square cavity driven by a moving
top lid. This is a classic benchmark problem in computational fluid
dynamics, producing a primary vortex and (at higher Reynolds numbers)
secondary corner vortices.

Boundary conditions:
    - Top wall: moving lid with horizontal velocity U_lid
    - Left, right, bottom walls: no-slip (bounce-back)

Usage:
    python lid_driven_cavity.py --nx 300 --ny 300 --steps 10000 --omega 1.7 --lid-velocity 0.1
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import lbm

plt.rcParams.update({"font.size": 14})


def parse_args():
    parser = argparse.ArgumentParser(description="Lid-Driven Cavity (LBM)")
    parser.add_argument("--nx", type=int, default=300, help="Grid width")
    parser.add_argument("--ny", type=int, default=300, help="Grid height")
    parser.add_argument("--steps", type=int, default=10000, help="Number of time steps")
    parser.add_argument("--omega", type=float, default=1.7, help="Relaxation frequency")
    parser.add_argument("--lid-velocity", type=float, default=0.1, help="Lid velocity")
    return parser.parse_args()


def main():
    args = parse_args()
    nx, ny, steps, omega = args.nx, args.ny, args.steps, args.omega
    lid_vel = args.lid_velocity
    os.makedirs("output", exist_ok=True)

    re = lbm.reynolds_number(omega, ny, lid_vel)
    print(f"Lid-Driven Cavity: {nx}x{ny} grid, omega={omega}, "
          f"lid_vel={lid_vel}, Re={re:.1f}")

    # All walls solid except top (which has the moving lid)
    boundary = (True, True, False, True)

    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny, 2))
    f = lbm.equilibrium(rho, u)

    t_start = time.time()

    for t in range(steps):
        f = lbm.streaming(f, boundary, nx, ny, lid_vel)
        f, rho, u = lbm.collision(f, omega)

        if (t + 1) % 1000 == 0:
            print(f"  Step {t + 1}/{steps}")

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.1f}s")

    # Final streamline plot
    lbm.plot_streamlines(
        u,
        title=(f"Lid-Driven Cavity — {nx}x{ny}, $\\omega$={omega}, "
               f"$U_{{lid}}$={lid_vel}, Re={re:.1f}, t={steps}"),
        filename="output/lid_driven_cavity.png",
    )
    print("Done. Plot saved to output/lid_driven_cavity.png")


if __name__ == "__main__":
    main()
