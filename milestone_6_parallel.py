#-------------imports-------------
import numpy as np
import sys
from mpi4py import MPI
import functions
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

#-------------Input Arguments-------------
# "args": ["300", "300", "150", "150" ,"10000", "1.7", "0.1"]
# grid size
width = int(sys.argv[1])
length = int(sys.argv[2])

# grid  descomposition
decomposition_y = int(sys.argv[3])
decomposition_x = int(sys.argv[4])

#number of iterations
max_iter = int(sys.argv[5])

#collision frequency
omega = float(sys.argv[6])

#lid velocity
lid_vel = float(sys.argv[7])

#-------------Milestone 6 - Parallel-------------
# create a cartesian communicator
communicator = MPI.COMM_WORLD.Create_cart((decomposition_y, decomposition_x), periods=(False, False))

# calculate number of columns and rows for each MPI process
columns = length // decomposition_x 
rows = width // decomposition_y

# indicate solid non-moving walls
# left, right, top, bottom 
boundry = (True, True, False, True) 

#initialize rho and u
rho = np.ones((rows, columns), dtype= np.float64)
u = np.zeros((rows, columns, 2), dtype= np.float64)

# initialize density function w.r.t. rho and u 
f = functions.equilibrium(rho, u)

for t in range(max_iter):
    f = functions.communicate_f(f, communicator)
    
    # streaming step
    f = functions.streaming(f, rows, columns, boundry, lid_vel) 
    
    # collision step
    f, rho, u = functions.collision(f, omega)

functions.save_mpiio(communicator, 'ux_{}.npy'.format(max_iter), u[:, :, 1])
functions.save_mpiio(communicator, 'uy_{}.npy'.format(max_iter), u[:, :, 0])

ux_kl = np.load('ux_10000.npy')
uy_kl = np.load('uy_10000.npy')

nx, ny = ux_kl.shape

plt.figure()
x_k = np.arange(nx)
y_l = np.arange(ny)
plt.streamplot(x_k, y_l, ux_kl.T, uy_kl.T)
plt.savefig("fig.png")

    
    