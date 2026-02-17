# Lattice Boltzmann Method in Python

A clean, from-scratch implementation of the **Lattice Boltzmann Method (LBM)** for 2D fluid dynamics simulations using the D2Q9 lattice model. Includes four classic benchmark flows — from simple wave decay to parallel lid-driven cavity simulation with MPI.

Developed as part of the *High-Performance Computing with Python* course at the [IMTEK Simulation Lab](https://www.imtek.uni-freiburg.de/professuren/simulation/simulation), University of Freiburg.

---

## What is the Lattice Boltzmann Method?

The LBM is a mesoscopic approach to computational fluid dynamics. Instead of solving the Navier-Stokes equations directly, it models the collective behavior of fictitious particle populations on a discrete lattice. At each time step, two operations are performed:

1. **Streaming** — particles propagate to neighboring lattice sites along discrete velocity directions
2. **Collision** — particle distributions relax toward a local equilibrium (BGK operator)

The macroscopic quantities (density, velocity) emerge naturally from moments of the distribution function. The method is particularly well-suited to parallelization and complex boundary geometries.

### D2Q9 Lattice

This implementation uses the **D2Q9** model (2 dimensions, 9 velocity directions):

```
    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8
```

Each direction *i* has:
- A velocity vector **c**_i
- A weight *w*_i (4/9 for rest, 1/9 for axis-aligned, 1/36 for diagonals)

The equilibrium distribution is:

```
f_eq_i = w_i · ρ · (1 + 3(c_i · u) + 9/2(c_i · u)² - 3/2(u · u))
```

The kinematic viscosity is related to the relaxation parameter ω by:

```
ν = (1/3)(1/ω - 1/2)
```

---

## Simulations

### 1. Shear Wave Decay
**`shear_wave_decay.py`** — Verifies viscous damping of sinusoidal perturbations in a periodic domain. Two modes are tested: density waves and velocity waves. The observed decay rate validates the LBM viscosity model.

```bash
python shear_wave_decay.py --nx 50 --ny 50 --steps 240 --omega 0.4
```

### 2. Poiseuille Flow
**`poiseuille_flow.py`** — Pressure-driven flow between two parallel no-slip walls. The velocity profile converges to the analytical parabolic solution, providing quantitative validation of the method.

```bash
python poiseuille_flow.py --nx 50 --ny 50 --steps 1200 --omega 0.8
```

### 3. Couette Flow
**`couette_flow.py`** — Shear-driven flow with a moving top lid and stationary bottom wall. The steady-state solution is a linear velocity profile.

```bash
python couette_flow.py --nx 50 --ny 50 --steps 3000 --omega 1.7 --lid-velocity 0.1
```

### 4. Lid-Driven Cavity
**`lid_driven_cavity.py`** — A classic CFD benchmark: flow in a closed square cavity driven by a moving top lid. Produces characteristic primary and secondary vortex structures depending on the Reynolds number.

```bash
python lid_driven_cavity.py --nx 300 --ny 300 --steps 10000 --omega 1.7 --lid-velocity 0.1
```

### 5. Lid-Driven Cavity (MPI Parallel)
**`lid_driven_cavity_parallel.py`** — The same cavity flow distributed across multiple MPI processes using 2D Cartesian domain decomposition with ghost-cell communication.

```bash
mpirun -np 4 python lid_driven_cavity_parallel.py \
    --nx 300 --ny 300 --px 2 --py 2 \
    --steps 10000 --omega 1.7 --lid-velocity 0.1
```

---

## Project Structure

```
├── lattice.py                      # D2Q9 lattice constants (weights, velocities)
├── lbm.py                          # Core LBM routines (streaming, collision, equilibrium)
├── parallel.py                     # MPI ghost-cell communication & parallel I/O
├── shear_wave_decay.py             # Simulation: shear wave decay
├── poiseuille_flow.py              # Simulation: Poiseuille flow
├── couette_flow.py                 # Simulation: Couette flow
├── lid_driven_cavity.py            # Simulation: lid-driven cavity
├── lid_driven_cavity_parallel.py   # Simulation: lid-driven cavity (MPI)
├── environment.yml                 # Conda environment specification
└── README.md
```

### Core Library

| Module | Description |
|--------|-------------|
| `lattice.py` | D2Q9 lattice definition — velocity vectors, weights, and opposite-direction mapping for bounce-back |
| `lbm.py` | Streaming step, BGK collision, equilibrium distribution, boundary handling (bounce-back & Zou-He moving wall), Reynolds number calculation, and visualization utilities |
| `parallel.py` | MPI domain decomposition utilities — ghost-cell exchange via `Sendrecv` and parallel NumPy array I/O via MPI-IO |

---

## Setup

### Requirements
- Python 3.8+
- NumPy
- Matplotlib
- mpi4py (only for the parallel simulation)

### Install with Conda

```bash
conda env create -f environment.yml
conda activate lbm-python
```

### Install with pip

```bash
pip install numpy matplotlib mpi4py
```

---

## References

- Krüger, T. et al. *The Lattice Boltzmann Method: Principles and Practice*. Springer, 2017.
- Bhatnagar, P.L., Gross, E.P., Krook, M. "A model for collision processes in gases." *Physical Review*, 94(3):511, 1954.
- Zou, Q. and He, X. "On pressure and velocity boundary conditions for the lattice Boltzmann BGK model." *Physics of Fluids*, 9(6):1591–1598, 1997.
