##################################################################################
# Visualize the robot saved by sim.py (sim.npz).
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load data from file
data = np.load('sim.npz')
body = data['body']
pos = data['pos']
muscles = data['muscles']

# The evolved body, with the same colors EvoGym uses
colors = ListedColormap(["white", "black", "gray", "orange", "deepskyblue"])
plt.imshow(body, cmap=colors, vmin=-0.5, vmax=4.5)
plt.colorbar(ticks=range(5), format=plt.FuncFormatter(lambda v, _: ["empty","rigid","soft","horizontal","vertical"][int(v)]))
plt.title("Evolved body")
plt.xticks([])
plt.yticks([])
plt.show()

# How far the robot walked over time
plt.plot(pos[:,0] - pos[0,0])
plt.xlabel("Time")
plt.ylabel("Distance walked")
plt.title("Center of mass over time")
plt.show()

# What every muscle was told to do over time (one line per muscle)
plt.plot(muscles)
plt.xlabel("Time")
plt.ylabel("Muscle command (0.6 squeeze, 1.6 stretch)")
plt.title("Muscle commands over time")
plt.show()
