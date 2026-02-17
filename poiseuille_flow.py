"""
Poiseuille Flow — Lattice Boltzmann simulation.

Simulates pressure-driven flow between two parallel plates (top and bottom
walls). The velocity profile should converge to the analytical parabolic
solution:
    u_y(x) = (rho * g * L^2) / (8 * mu) * (1 - (2x/L)^2)

Boundary conditions:
    - Top and bottom walls: no-slip (bounce-back)
    - Left and right: periodic

Usage:
    python poiseuille_flow.py --nx 50 --ny 50 --steps 1200 --omega 0.8
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import lbm

plt.rcParams.update({"font.size": 14})


def parse_args():
    parser = argparse.ArgumentParser(description="Poiseuille Flow (LBM)")
    parser.add_argument("--nx", type=int, default=50, help="Grid width")
    parser.add_argument("--ny", type=int, default=50, help="Grid height")
    parser.add_argument("--steps", type=int, default=1200, help="Number of time steps")
    parser.add_argument("--omega", type=float, default=0.8, help="Relaxation frequency")
    return parser.parse_args()


def main():
    args = parse_args()
    nx, ny, steps, omega = args.nx, args.ny, args.steps, args.omega
    os.makedirs("output", exist_ok=True)

    print(f"Poiseuille Flow: {nx}x{ny} grid, omega={omega}, {steps} steps")

    # Solid walls on top and bottom; periodic left/right
    boundary = (False, False, True, True)

    # Initial conditions: uniform density, constant y-velocity
    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny, 2))
    u[:, :, 1] = 0.1  # initial driving velocity

    f = lbm.equilibrium(rho, u)

    # Prepare figure for velocity profile snapshots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Poiseuille Flow — {nx}x{ny}, $\\omega$={omega}",
        fontsize=16,
    )
    axes = axes.flatten()
    plot_idx = 0

    t_start = time.time()

    for t in range(steps):
        f = lbm.streaming(f, boundary, nx, ny)
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
    fig.savefig("output/poiseuille_flow.png", dpi=150)
    plt.close(fig)
    print("Done. Plot saved to output/poiseuille_flow.png")


if __name__ == "__main__":
    main()
