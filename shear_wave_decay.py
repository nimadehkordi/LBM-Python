"""
Shear Wave Decay — Lattice Boltzmann simulation.

Simulates the decay of a sinusoidal density perturbation in a periodic
domain, verifying that the LBM correctly reproduces viscous damping.

Two experiments are included:
  Part 1 — Density perturbation: rho(x) = rho_0 + epsilon * sin(2*pi*x/L)
  Part 2 — Velocity perturbation: u_y(x) = u_0 + epsilon * sin(2*pi*x/L)

Usage:
    python shear_wave_decay.py --nx 50 --ny 50 --steps 240 --omega 0.4
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import lbm

plt.rcParams.update({"font.size": 14})


def parse_args():
    parser = argparse.ArgumentParser(description="Shear Wave Decay (LBM)")
    parser.add_argument("--nx", type=int, default=50, help="Grid width")
    parser.add_argument("--ny", type=int, default=50, help="Grid height")
    parser.add_argument("--steps", type=int, default=240, help="Number of time steps")
    parser.add_argument("--omega", type=float, default=0.4, help="Relaxation frequency")
    return parser.parse_args()


# ── Part 1: Density perturbation ─────────────────────────────────────────────

def run_density_perturbation(nx, ny, steps, omega):
    """Simulate the decay of a sinusoidal density wave."""
    epsilon = 0.01
    rho_0 = 0.2

    # Initial conditions: sinusoidal density, zero velocity
    rho = rho_0 + np.fromfunction(
        lambda i, j: epsilon * np.sin(2 * np.pi * j / ny),
        (nx, ny), dtype=float,
    )
    u = np.zeros((nx, ny, 2))

    # Plot initial density profile
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(ny), rho[0, :])
    ax.set_title(r"Initial density $\rho(\mathbf{r}, t=0)$")
    ax.set_xlabel("y")
    ax.set_ylabel(r"$\rho$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/shear_wave_initial_density.png", dpi=150)
    plt.close(fig)

    # Run simulation
    f = lbm.equilibrium(rho, u)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        rf"Shear Wave Decay — $\rho_0$={rho_0}, $\varepsilon$={epsilon}, $\omega$={omega}",
        fontsize=16,
    )
    axes = axes.flatten()
    plot_idx = 0
    velocity_history = []

    for t in range(steps):
        f = lbm.streaming(f)
        f, rho, u = lbm.collision(f, omega)
        velocity_history.append(u[0, 0].copy())

        if t % 40 == 0 and plot_idx < 6:
            ax = axes[plot_idx]
            x, y = np.meshgrid(np.arange(ny), np.arange(nx))
            im = ax.pcolormesh(
                x, y, rho,
                vmin=rho_0 - 2 * rho_0 * epsilon,
                vmax=rho_0 + 2 * rho_0 * epsilon,
                shading="auto",
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"t = {t}")
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            plot_idx += 1

    fig.tight_layout()
    fig.savefig("output/shear_wave_density_evolution.png", dpi=150)
    plt.close(fig)

    # Plot velocity evolution over time
    velocity_history = np.array(velocity_history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(steps), velocity_history[:, 0], label=r"$u_x$")
    ax.plot(np.arange(steps), velocity_history[:, 1], label=r"$u_y$")
    ax.set_title("Velocity evolution at (0, 0)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/shear_wave_velocity_evolution.png", dpi=150)
    plt.close(fig)

    print(f"  Density perturbation: {steps} steps completed.")


# ── Part 2: Velocity perturbation ────────────────────────────────────────────

def run_velocity_perturbation(nx, ny, steps, omega):
    """Simulate the decay of a sinusoidal velocity wave."""
    epsilon = 0.02
    u0 = 0.1

    # Initial conditions: uniform density, sinusoidal y-velocity
    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny, 2))
    u[:, :, 1] = u0 + np.fromfunction(
        lambda i, j: epsilon * np.sin(2 * np.pi * i / nx),
        (nx, ny), dtype=float,
    )

    # Plot initial velocity profile
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u[:, 0, 1], np.arange(nx))
    ax.set_title(r"Initial velocity $u_y(x, t=0)$")
    ax.set_xlabel(r"$u_y$")
    ax.set_ylabel("x")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/shear_wave_initial_velocity.png", dpi=150)
    plt.close(fig)

    # Run simulation
    f = lbm.equilibrium(rho, u)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        rf"Velocity Decay — $u_0$={u0}, $\varepsilon$={epsilon}, $\omega$={omega}",
        fontsize=16,
    )
    axes = axes.flatten()
    plot_idx = 0

    for t in range(steps):
        f = lbm.streaming(f)
        f, rho, u = lbm.collision(f, omega)

        if t % 50 == 0 and plot_idx < 6:
            ax = axes[plot_idx]
            ax.plot(u[:, 0, 1], np.arange(nx))
            ax.set_xlabel(r"$u_y$")
            ax.set_ylabel("x")
            ax.set_xlim(0.08, 0.12)
            ax.set_title(f"t = {t}")
            plot_idx += 1

    fig.tight_layout()
    fig.savefig("output/shear_wave_velocity_decay.png", dpi=150)
    plt.close(fig)

    print(f"  Velocity perturbation: {steps} steps completed.")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    print(f"Shear Wave Decay: {args.nx}x{args.ny} grid, omega={args.omega}")

    import os
    os.makedirs("output", exist_ok=True)

    run_density_perturbation(args.nx, args.ny, args.steps, args.omega)
    run_velocity_perturbation(args.nx, args.ny, args.steps, args.omega)
    print("Done. Plots saved to output/.")
