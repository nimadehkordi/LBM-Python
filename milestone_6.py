#-------------imports-------------
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import sys
import time
from matplotlib import cm
import functions
import streaming

#-------------Input Arguments-------------
# grid size
width = int(sys.argv[1])
length = int(sys.argv[2])

#number of iterations
max_iter = int(sys.argv[3])

#collision frequency
omega = float(sys.argv[4])

#lid velocity
lid_vel = float(sys.argv[5])

#-------------Milestone 6-------------
# "args": ["300", "300", "10000", "1.7", "0.1"]
# "args": ["300", "300", "10000", "1.3", "0.3"]
# indicate solid non-moving walls
# left, right, top, bottom 
boundry = (True, True, False, True) 

#initialize rho and u
rho = np.ones((width, length))
u = np.zeros((width, length, 2))

# initialize density function w.r.t. rho and u 
f = functions.equilibrium(rho, u)

# calculate corresponding reynolds number
re = functions.calculate_re(omega=omega, length=length, lid_vel=lid_vel)

fig = plt.figure(figsize=(25,25))
fig.suptitle("Lid Driven Cavity, lattice dimensions: ({} * {}), omega: {}, lid velocity = {}, steps: {}, Re: {}".format(width, length, omega, lid_vel, max_iter, re), fontsize="x-large")
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

initial_time = time.time()
i = 1

for t in range(max_iter):
    # streaming step
    f = functions.streaming(f, width, length, boundry, lid_vel) 
    
    # collision step
    f, rho, u = functions.collision(f, omega)
    
    
    if t%500 == 499:
        sys.stdout.write('iteration {}/{}\n'.format(t+1, max_iter))
        ax = plt.subplot(5,5,i)
        ax.set_xticks(np.arange(0, length+1, 50))
        ax.set_yticks(np.arange(0, width+1, 50))
        ax.grid(True)
        ax.axis("equal")
        ax.streamplot(np.arange(width), np.arange(length), u[:,:, 0].T, u[:,:, 1].T)
        ax.set_title(t+1)
        i+=1   
elapsed_time = time.time() - initial_time

print("elapsed time: ", elapsed_time)

plt.savefig("milestone_%d_%d_%d_%.1f_%d_final.png" %(6, width, length, omega, max_iter))
functions.plot_velocity(u, steps = max_iter, milestone=6, re = re, lid_vel=lid_vel, omega = omega)