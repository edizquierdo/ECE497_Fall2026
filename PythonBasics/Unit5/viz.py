##################################################################################
# Visualize the robot saved by sim.py (sim.npz): the same plots as Project 5's
# `sim.py --showbody --viztraces`.
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load data from file
data = np.load('sim.npz')
body = data['body']
pos = data['pos']
muscles = data['muscles']

# The body, with the same colors EvoGym uses in its animations
colors = ListedColormap(["#ffffff", "#262626", "#bfbfbf", "#fd8e3e", "#6dafd6"])
plt.imshow(body, cmap=colors, vmin=-0.5, vmax=4.5)
plt.colorbar(ticks=range(5), format=plt.FuncFormatter(lambda v, _: ["empty","rigid","soft","horizontal","vertical"][int(v)]))
plt.title("Body")
plt.xticks([])
plt.yticks([])
plt.show()

# How far the robot walked over time, and how high its center of mass was (hops and falls show up here)
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axes[0].plot(pos[:,0] - pos[0,0], color="green")
axes[0].set_ylabel("Distance walked")
axes[0].set_title("Center of mass over time")
axes[1].plot(pos[:,1], color="purple")
axes[1].set_ylabel("Height")

# What every muscle was told to do over time: one row per muscle (blue = squeeze, red = stretch)
axes[2].imshow(muscles.T, aspect="auto", cmap="coolwarm", vmin=0.6, vmax=1.6, interpolation="nearest")
axes[2].set_ylabel("Muscle")
axes[2].set_xlabel("Time")
axes[2].set_title("Muscle commands (blue = squeeze, red = stretch)")
plt.tight_layout()
plt.show()
