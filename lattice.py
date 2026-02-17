"""
D2Q9 Lattice constants for the Lattice Boltzmann Method.

The D2Q9 model uses a 2-dimensional lattice with 9 discrete velocity directions:

    6   2   5
      \\ | /
    3 - 0 - 1
      / | \\
    7   4   8

Each direction has an associated weight (w) and velocity vector (c).
"""

import numpy as np

# Number of discrete velocity directions
Q = 9

# Lattice weights for D2Q9
#   - Center (i=0): 4/9
#   - Axis-aligned (i=1..4): 1/9
#   - Diagonal (i=5..8): 1/36
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Discrete velocity vectors (c_i) for D2Q9
#   Index:  0       1       2       3       4       5       6       7       8
#   Dir:  rest   east   north   west   south    NE      NW      SW      SE
C = np.array([
    [0,  0],   # 0: rest
    [1,  0],   # 1: east
    [0,  1],   # 2: north
    [-1, 0],   # 3: west
    [0, -1],   # 4: south
    [1,  1],   # 5: north-east
    [-1, 1],   # 6: north-west
    [-1, -1],  # 7: south-west
    [1, -1],   # 8: south-east
])

# Mapping from each direction to its opposite (for bounce-back boundaries)
#   0 <-> 0,  1 <-> 3,  2 <-> 4,  5 <-> 7,  6 <-> 8
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8)
