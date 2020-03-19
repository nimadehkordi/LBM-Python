#-------------imports-------------
import functions

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import sys

#-------------Input Arguments-------------
# grid size
width = int(sys.argv[1])
length = int(sys.argv[2])

#number of iterations
max_iter = int(sys.argv[3])

#collision frequency
omega = float(sys.argv[4])

#-------------Milestone 3 - Part 1-------------

# "args": ["50", "50", "240", "0.4", "0.1"]
epsilon = 0.01

# initialize rho and u
rho_0 = 0.2
rho = rho_0 + np.fromfunction(lambda i, j: epsilon*(np.sin(np.pi * 2 * j / length)), (width, length), dtype=float)
u = np.zeros((width, length, 2))

# plot initial density distribution 
plt.figure(figsize=(10,10))
plt.plot(np.arange(length), rho[0,:])
plt.title("Initial density distribution at $\\rho$(r, t=0)")
plt.grid(True)
plt.xlabel("length")
plt.ylabel("density")
plt.savefig("m3_initial density distribution")



# initialize probability density function
f = functions.equilibrium(rho, u)

fig = plt.figure(figsize=(21,14))
fig.suptitle("Shear Wave Decay, rho_0 = {}, epsilon = {}, omega = {}".format(rho_0, epsilon, omega), fontsize="x-large")
plt.title("Milestone 3")
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
i = 1
# to plot evolution of velocity in the end
vel_evolution = []

# main loop
for t in range(max_iter):
    # streaming step
    f = functions.streaming(f)

    # collision step
    f, rho, u = functions.collision(f, omega)
    vel_evolution.append(u[0,0])

    # plot the evloution of density which shows the decay in the end 
    if t%40 == 0:
        ax = plt.subplot(2,3,i)            
        x = np.arange(0, length)
        y = np.arange(0, width)
        X, Y = np.meshgrid(x, y)
        ax.set_xlabel("length")
        ax.set_ylabel("width")
        ax.set_xticks(np.arange(0, length+1, 5))
        ax.set_yticks(np.arange(0, width+1, 5))
        #cs = ax0.contourf(X, Y, rho, shading='gouraud')
        #ax.axis("equal")
        im = ax.pcolormesh(X, Y, rho, vmin=rho_0 - rho_0*2*epsilon, vmax=rho_0 + rho_0*2*epsilon)
        fig.colorbar(im, ax=ax)
        ax.set_title('Density Distribution after {} iteration'.format(t))
        i+=1

plt.savefig("M3.png")

# plot evolution of velocity 
plt.figure(figsize=(10,10))
plt.plot(np.arange(max_iter), vel_evolution)
plt.title("Velocity evolution")
plt.grid(True)
plt.xlabel("iteration")
plt.ylabel("velocity")
plt.savefig("m3_velocity_evolution")
"""
#-------------Milestone 3 - Part 2-------------
"""
# "args": ["50", "50", "300", "0.2", "0.1"]
epsilon = 0.02

# initialize rho and u
ux_0 = 0.1
u = np.zeros((width, length, 2))
u[:,:,1] = ux_0 + np.fromfunction(lambda i, j: epsilon*(np.sin(np.pi * 2 * i / width)), (width, length), dtype=float)
rho = np.ones((width, length))

# plot initial density distribution 
plt.figure(figsize=(10,10))
plt.plot(u[:,0,1], np.arange(width))
plt.title("Initial velocity in x direction")
plt.grid(True)
plt.xlabel("velocity")
plt.ylabel("width")
plt.savefig("m3_initial_velocity_in_x_direction")



# initialize probability density function
f = functions.equilibrium(rho, u)

fig = plt.figure(figsize=(21,14))
fig.suptitle("Velocity evolution, ux_0 = {}, epsilon = {}, omega = {}".format(ux_0, epsilon, omega), fontsize="x-large")
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
i = 1

# main loop
for t in range(max_iter):
    # streaming step
    f = functions.streaming(f)

    # collision step
    f, rho, u = functions.collision(f, omega)

    # plot the evloution of density which shows the decay in the end 
    if t%50 == 0:
        sys.stdout.write('iteration {}/{}\n'.format(t+1, max_iter))
        ax = plt.subplot(2,3,i)            
        ax.set_ylabel("width")
        ax.set_xlabel("velocity in x direction")
        ax.set_yticks(np.arange(0, width+1, 5))
        ax.set_xlim(0.08, 0.12)
        im = ax.plot(u[:,0,1], np.arange(width))
        ax.set_title('Velocity in x direction after {} iteration'.format(t))
        i+=1
plt.savefig("M3_velocity_evolution_over_time.png")

"""
omega = np.arange(0.1, 2, 0.01)
s = (1 / omega - 1 / 2) / 3

fig, ax = plt.subplots()
ax.plot(omega, s, label="Theoretical Viscosity")
ax.legend()

ax.set(xlabel=r'$\omega$', ylabel=r'$\nu$' )
ax.grid()

plt.savefig("omgea_vs_nu.png")
"""