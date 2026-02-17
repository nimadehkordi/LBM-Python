"""
Core Lattice Boltzmann Method (LBM) routines for 2D fluid simulations.

Implements the BGK (Bhatnagar-Gross-Krook) collision operator with
the D2Q9 lattice model. Provides functions for the two fundamental
LBM steps — streaming and collision — as well as equilibrium
distribution, boundary conditions, and post-processing utilities.
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice

plt.rcParams.update({"font.size": 14})


def density(f):
    """Compute the macroscopic density field from the distribution function.

    Parameters
    ----------
    f : ndarray, shape (Nx, Ny, Q)
        Particle distribution function.

    Returns
    -------
    ndarray, shape (Nx, Ny)
        Density field rho(x, y).
    """
    return np.sum(f, axis=2)


def velocity(f):
    """Compute the macroscopic velocity field from the distribution function.

    Parameters
    ----------
    f : ndarray, shape (Nx, Ny, Q)
        Particle distribution function.

    Returns
    -------
    ndarray, shape (Nx, Ny, 2)
        Velocity field u(x, y) with components [u_x, u_y].
    """
    rho = density(f)
    return (np.dot(f, lattice.C).T / rho.T).T


def equilibrium(rho, u):
    """Compute the equilibrium distribution function (Maxwell-Boltzmann).

    Uses the second-order expansion:
        f_eq_i = w_i * rho * (1 + 3*(c_i . u) + 9/2*(c_i . u)^2 - 3/2*(u . u))

    Parameters
    ----------
    rho : ndarray, shape (Nx, Ny)
        Density field.
    u : ndarray, shape (Nx, Ny, 2)
        Velocity field.

    Returns
    -------
    ndarray, shape (Nx, Ny, Q)
        Equilibrium distribution function.
    """
    cu = np.dot(u, lattice.C.T)
    cu2 = cu ** 2
    u2 = u[:, :, 0] ** 2 + u[:, :, 1] ** 2
    return ((1 + 3 * cu.T + 4.5 * cu2.T - 1.5 * u2.T) * rho.T).T * lattice.W


def collision(f, omega):
    """Perform the BGK collision step (relaxation toward equilibrium).

    f_new = f + omega * (f_eq - f)

    Parameters
    ----------
    f : ndarray, shape (Nx, Ny, Q)
        Pre-collision distribution function (modified in-place).
    omega : float
        Relaxation frequency (1/tau). Related to kinematic viscosity by:
        nu = (1/3) * (1/omega - 1/2)

    Returns
    -------
    f : ndarray, shape (Nx, Ny, Q)
        Post-collision distribution function.
    rho : ndarray, shape (Nx, Ny)
        Updated density field.
    u : ndarray, shape (Nx, Ny, 2)
        Updated velocity field.
    """
    rho = density(f)
    u = velocity(f)
    f_eq = equilibrium(rho, u)
    f += omega * (f_eq - f)
    return f, rho, u


def streaming(f, boundary=None, nx=None, ny=None, lid_velocity=None):
    """Perform the streaming step: propagate distributions along lattice links.

    Optionally applies bounce-back boundary conditions on solid walls
    and a moving-lid boundary condition (Zou-He style) at the top wall.

    Parameters
    ----------
    f : ndarray, shape (Nx, Ny, Q)
        Distribution function (modified in-place).
    boundary : tuple of 4 bools, optional
        Which walls are solid: (left, right, top, bottom).
    nx : int, optional
        Grid width (required if boundary is set).
    ny : int, optional
        Grid height (required if boundary is set).
    lid_velocity : float, optional
        Horizontal velocity of the top lid (for Couette / cavity flows).

    Returns
    -------
    ndarray, shape (Nx, Ny, Q)
        Post-streaming distribution function.
    """
    f_old = np.copy(f)

    # Propagate each population along its velocity vector
    for i in range(lattice.Q):
        f[:, :, i] = np.roll(
            np.roll(f[:, :, i], lattice.C[i, 0], axis=0),
            lattice.C[i, 1], axis=1,
        )

    # Bounce-back at stationary solid walls
    if boundary:
        mask = _build_wall_mask(boundary, nx, ny)
        for i in range(lattice.Q):
            f[:, :, i] = np.where(
                mask, f_old[:, :, lattice.OPPOSITE[i]], f[:, :, i]
            )

    # Moving lid (top wall) — Zou-He boundary condition
    if lid_velocity:
        rho_wall = (
            2 * (f_old[:, -1, 6] + f_old[:, -1, 2] + f_old[:, -1, 5])
            + f_old[:, -1, 3] + f_old[:, -1, 0] + f_old[:, -1, 1]
        )
        u_lid = np.array([lid_velocity, 0])
        for i in [4, 7, 8]:
            f[:, -1, i] = (
                f_old[:, -1, lattice.OPPOSITE[i]]
                - 6 * lattice.W[i] * rho_wall * np.dot(lattice.C[i], u_lid)
            )

    return f


def _build_wall_mask(boundary, nx, ny):
    """Create a boolean mask indicating solid wall cells.

    Parameters
    ----------
    boundary : tuple of 4 bools
        (left, right, top, bottom) — True means solid wall.
    nx, ny : int
        Grid dimensions.

    Returns
    -------
    ndarray, shape (Nx, Ny), dtype bool
    """
    mask = np.zeros((nx, ny), dtype=bool)
    if boundary[0]:
        mask[:, 0] = True     # left wall
    if boundary[1]:
        mask[:, -1] = True    # right wall
    if boundary[2]:
        mask[0, :] = True     # top wall
    if boundary[3]:
        mask[-1, :] = True    # bottom wall
    return mask


def reynolds_number(omega, length, velocity_scale):
    """Compute the Reynolds number for the simulation.

    Re = (U * L) / nu,  where  nu = (1/3) * (1/omega - 1/2)

    Parameters
    ----------
    omega : float
        Relaxation frequency.
    length : int
        Characteristic length (in lattice units).
    velocity_scale : float
        Characteristic velocity (in lattice units).

    Returns
    -------
    float
        Reynolds number.
    """
    nu = (1.0 / 3.0) * (1.0 / omega - 0.5)
    return velocity_scale * length / nu


def plot_streamlines(u, title, filename, figsize=(12, 10)):
    """Save a streamline plot of the velocity field.

    Parameters
    ----------
    u : ndarray, shape (Nx, Ny, 2)
        Velocity field.
    title : str
        Plot title.
    filename : str
        Output file path.
    figsize : tuple
        Figure dimensions.
    """
    nx, ny = u.shape[0], u.shape[1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.streamplot(np.arange(nx), np.arange(ny), u[:, :, 0].T, u[:, :, 1].T)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
