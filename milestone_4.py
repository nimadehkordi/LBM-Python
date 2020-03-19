#-------------imports-------------
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import sys
import time
from matplotlib import cm
import functions

#-------------Input Arguments-------------
# grid size
width = int(sys.argv[1])
length = int(sys.argv[2])

#number of iterations
max_iter = int(sys.argv[3])

#collision frequency
omega = float(sys.argv[4])

#-------------Milestone 6-------------
# "args": ["50", "50", "1200", "0.8"]
# indicate solid non-moving walls
# left, right, top, bottom 
#boundry = (False, False, True, True) 
boundry = (False, False, True, True)
#initialize rho and u
rho = np.ones((width, length))
u = np.zeros((width, length, 2))

#initial velocity at the inlet 
u[:,:,1] = 0.1

# initialize density function w.r.t. rho and u 
f = functions.equilibrium(rho, u)

fig = plt.figure(figsize=(25,25))
fig.suptitle("Poiseuille flow, lattice dimensions: ({} * {}), omega: {}, steps: {},".format(width, length, omega, max_iter), fontsize="x-large")
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

initial_time = time.time()
i = 1

for t in range(max_iter):
    # streaming step
    f = functions.streaming(f, width, length, boundry) 
    # collision step
    f, rho, u = functions.collision(f, omega)
    
    if t%200 == 199:
        sys.stdout.write('iteration {}/{}\n'.format(t+1, max_iter))
        ax = plt.subplot(2,3,i)            
        ax.set_ylabel("width")
        ax.set_xlabel("velocity in x direction")
        ax.set_yticks(np.arange(0, width+1, 5))
        im = ax.plot(u[:,0,1], np.arange(width))
        ax.set_title('Velocity in x direction after {} iteration'.format(t+1))
        i+=1  
elapsed_time = time.time() - initial_time

print("elapsed time: ", elapsed_time)
plt.savefig("M4_velocity_evolution_over_time.png")
