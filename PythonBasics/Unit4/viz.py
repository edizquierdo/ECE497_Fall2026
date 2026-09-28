##################################################################################
# Visualize the behavior saved by sim.py (sim.npz).
##################################################################################
import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.load('sim.npz')
a_x = data['a_x']
a_y = data['a_y']
d = data['d']

# Visualize the agent's trajectories, one per trial
plt.plot(a_x.T,a_y.T)
plt.plot(a_x[:,0],a_y[:,0],"k^")    # Starting positions of the agent with black triangles
plt.plot(0.0,0.0,"yo",markersize=12,markeredgecolor="k")   # Light source at the origin with a yellow circle
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectories")
plt.axis("equal")
plt.show()

# Distance to the light over time for each trial
plt.plot(d.T)
plt.xlabel("Time")
plt.ylabel("Distance to light")
plt.title("Distance to light over time")
plt.show()
