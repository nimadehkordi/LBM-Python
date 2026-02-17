"""
Couette Flow — Lattice Boltzmann simulation.

Simulates shear-driven flow between a stationary bottom wall and a
moving top lid. The steady-state velocity profile is linear:
    u_x(y) = U_lid * y / H

Boundary conditions:
    - Top wall: moving lid with horizontal velocity U_lid
    - Bottom wall: no-slip (bounce-back)
    - Left and right: periodic

Usage:
    python couette_flow.py --nx 50 --ny 50 --steps 3000 --omega 1.7 --lid-velocity 0.1
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import lbm

plt.rcParams.update({"font.size": 14})


def parse_args():
    parser = argparse.ArgumentParser(description="Couette Flow (LBM)")
    parser.add_argument("--nx", type=int, default=50, help="Grid width")
    parser.add_argument("--ny", type=int, default=50, help="Grid height")
    parser.add_argument("--steps", type=int, default=3000, help="Number of time steps")
    parser.add_argument("--omega", type=float, default=1.7, help="Relaxation frequency")
    parser.add_argument("--lid-velocity", type=float, default=0.1, help="Lid velocity")
    return parser.parse_args()


def main():
    args = parse_args()
    nx, ny, steps, omega = args.nx, args.ny, args.steps, args.omega
    lid_vel = args.lid_velocity
    os.makedirs("output", exist_ok=True)

    re = lbm.reynolds_number(omega, ny, lid_vel)
    print(f"Couette Flow: {nx}x{ny} grid, omega={omega}, "
          f"lid_vel={lid_vel}, Re={re:.1f}")

    # Top wall is solid (bounce-back + moving lid); bottom is stationary solid
    boundary = (False, False, True, False)

    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny, 2))
    f = lbm.equilibrium(rho, u)

    # Prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Couette Flow — {nx}x{ny}, $\\omega$={omega}, "
        f"$U_{{lid}}$={lid_vel}, Re={re:.1f}",
        fontsize=16,
    )
    axes = axes.flatten()
    plot_idx = 0

    t_start = time.time()

    for t in range(steps):
        f = lbm.streaming(f, boundary, nx, ny, lid_vel)
        f, rho, u = lbm.collision(f, omega)

        if t % 200 == 199 and plot_idx < 6:
            ax = axes[plot_idx]
            ax.plot(u[:, 0, 1], np.arange(nx))
            ax.set_xlabel(r"$u_y$")
            ax.set_ylabel("x (cross-channel)")
            ax.set_title(f"t = {t + 1}")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            print(f"  Step {t + 1}/{steps}")

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.1f}s")

    fig.tight_layout()
    fig.savefig("output/couette_flow.png", dpi=150)
    plt.close(fig)
    print("Done. Plot saved to output/couette_flow.png")


if __name__ == "__main__":
    main()
