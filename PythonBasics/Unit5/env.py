##################################################################################
# Soft robot body and world for the embodied task, using Evolution Gym (EvoGym).
# Same bodies and brains as in Project 5, written in plain Python/NumPy:
#   - The body is a small grid of voxels. Each voxel is one of five materials:
#       0 empty, 1 rigid, 2 soft, 3 horizontal muscle, 4 vertical muscle
#     Row 0 is the top of the robot; the last row rests on the ground.
#   - Every step, each muscle gets a command in [0.6, 1.6]: its target length as
#     a fraction of its rest length (0.6 = squeeze, 1.6 = stretch).
#   - The robot's "brain" is a neural network (fnn.FNN), in one of four versions:
#
#                 open-loop (no sensing)          closed-loop (+ sensing)
#       local     clock + muscle's (x, y)         + strain of its voxel and 4 neighbors
#                 -> 1 command (one copy of       -> 1 command
#                    the network per muscle)
#       global    clock                           + strain of every cell in the grid
#                 -> 1 command per grid cell      -> 1 command per grid cell
#
#     "Strain" is how stretched a voxel is: its (width, height) relative to its
#     rest size, minus 1 (0 at rest, positive when stretched, negative when squeezed).
#     All four fit ANY body on the grid, so all four can be evolved together with a body.
##################################################################################
import warnings
warnings.filterwarnings("ignore")   # EvoGym prints deprecation warnings that aren't about this code

import numpy as np
import gymnasium as gym
import evogym.envs                  # registers EvoGym's tasks with gymnasium
from evogym import is_connected, has_actuator

materials = 5

# Hand-designed bodies, the same as in Project 5. Row 0 is the top of the robot.
presets = {
    "biped":  [[3,3,3,3,3],     # two legs and a torso, every voxel a horizontal muscle
               [3,3,3,3,3],
               [3,3,0,3,3],
               [3,3,0,3,3],
               [3,3,0,3,3]],
    "worm":   [[0,0,0,0,0],     # a flat strip: vertical muscles on top of horizontal ones
               [0,0,0,0,0],
               [0,0,0,0,0],
               [4,4,4,4,4],
               [3,3,3,3,3]],
    "block":  [[1,1,1,1,1],     # a soft block with a rigid top and muscles along the bottom
               [2,2,2,2,2],
               [2,2,2,2,2],
               [4,2,4,2,4],
               [3,3,3,3,3]],
    "tripod": [[1,1,1,1,1],     # three legs hanging from a rigid back
               [3,3,3,3,3],
               [4,0,4,0,4],
               [4,0,4,0,4],
               [3,0,3,0,3]],
}

def loadBody(name):
    # A preset name, or a text file with one row of material codes per line (row 0 at the top)
    if name in presets:
        return np.array(presets[name])
    return np.loadtxt(name, dtype=int, ndmin=2)

def makeBody(genes, grid):
    # Each voxel gets 5 genes, one per material, and becomes whichever material's
    # gene is largest (just like fnn.py picks each layer's activation function)
    return np.argmax(genes.reshape(grid, grid, materials), axis=2)

def isValid(body):
    # A body can only be simulated if it is one connected piece with at least one muscle
    return is_connected(body) and has_actuator(body)

def brainLayers(controller, loop, shape, hidden):
    # Layer sizes [inputs, hidden, outputs] of the network for a body grid of the given shape
    cells = shape[0]*shape[1]
    if controller == "local":
        inputs = 4 + (10 if loop == "closed" else 0)        # clock (2) + position (2) [+ strains (10)]
        return [inputs, hidden, 1]                          # one command, for its own muscle
    inputs = 2 + (2*cells if loop == "closed" else 0)       # clock (2) [+ every cell's strain]
    return [inputs, hidden, cells]                          # one command per grid cell

class SoftRobot:

    def __init__(self, body, neuralnetwork, controller, loop, envname, duration, period, render=False):
        self.body = body
        self.brain = neuralnetwork
        self.controller = controller
        self.loop = loop
        self.period = period
        self.time = 0
        self.world = gym.make(envname, body=body, max_episode_steps=duration,
                              render_mode="human" if render else None)
        self.world.reset(seed=0)

        # The muscles, listed row by row: the same order EvoGym expects the commands in
        self.rows, self.cols = np.nonzero((body == 3) | (body == 4))
        self.cells = self.rows*body.shape[1] + self.cols     # each muscle's cell number in the grid
        # Where each muscle sits in the body, scaled to [0, 1]: x from left to right, y from bottom to top
        self.musclepos = np.column_stack([self.cols/max(body.shape[1]-1, 1), 1 - self.rows/max(body.shape[0]-1, 1)])
        self.muscles = np.ones(len(self.rows))
        if loop == "closed":
            self.findCorners()

    def findCorners(self):
        # EvoGym tells us where the robot's point masses are, but not which voxel they belong to.
        # Right after reset the robot is undeformed and every point mass sits exactly on a corner
        # of the voxel grid, so we can work it out from the starting positions, once.
        pts = self.points()
        lattice = np.round((pts - pts.min(axis=1, keepdims=True))/self.world.unwrapped.VOXEL_SIZE).astype(int)
        index = {(x, y): i for i, (x, y) in enumerate(lattice.T)}
        rows, cols = np.nonzero(self.body)
        bottom, left = rows.max(), cols.min()
        self.corners = {}   # (row, col) -> point masses at its bottom-left, bottom-right, top-left, top-right
        for r, c in zip(rows, cols):
            x, y = c - left, bottom - r     # grid coordinates of the voxel's bottom-left corner (y points up)
            self.corners[(r, c)] = [index[(x, y)], index[(x+1, y)], index[(x, y+1)], index[(x+1, y+1)]]

    def points(self):
        # Positions of all the robot's point masses, shape (2, number of points)
        sim = self.world.unwrapped
        return sim.object_pos_at_time(sim.get_time(), "robot")

    def strains(self):
        # How stretched every voxel is, as (width, height) relative to rest, minus 1
        # Returned as a grid with one extra cell of zeros all around, so neighbors are easy to look up
        pts = self.points()/self.world.unwrapped.VOXEL_SIZE
        s = np.zeros((self.body.shape[0]+2, self.body.shape[1]+2, 2))
        for (r, c), (bl, br, tl, tr) in self.corners.items():
            width = (np.linalg.norm(pts[:,br] - pts[:,bl]) + np.linalg.norm(pts[:,tr] - pts[:,tl]))/2
            height = (np.linalg.norm(pts[:,tl] - pts[:,bl]) + np.linalg.norm(pts[:,tr] - pts[:,br]))/2
            s[r+1, c+1] = [width - 1, height - 1]
        return s

    def think(self):
        # Every brain sees the same clock
        phase = 2*np.pi*self.time/self.period
        clock = np.array([np.sin(phase), np.cos(phase)])
        if self.controller == "local":
            # One row of inputs per muscle, and one forward pass for all of them at once:
            # every muscle runs the same network, on its own inputs
            inputs = np.hstack([np.tile(clock, (len(self.muscles), 1)), self.musclepos])
            if self.loop == "closed":
                s = self.strains()
                r, c = self.rows+1, self.cols+1     # +1 for the border of zeros
                inputs = np.hstack([inputs, s[r,c], s[r-1,c], s[r+1,c], s[r,c-1], s[r,c+1]])
            outputs = self.brain.forward(inputs)[:,0]
        else:
            # One network for the whole body: one output per grid cell, and each muscle
            # takes the output of the cell it sits in
            inputs = clock
            if self.loop == "closed":
                inputs = np.concatenate([clock, self.strains()[1:-1,1:-1].reshape(-1)])
            outputs = self.brain.forward(inputs)[0][self.cells]
        # Outputs are capped to [0, 1], like the motors in Unit 4, then shifted into EvoGym's [0.6, 1.6]
        self.muscles = 0.6 + np.clip(outputs, 0, 1)

    def move(self):
        # Step the physics; the reward is how far the robot moved to the right this step
        obs, reward, terminated, truncated, info = self.world.step(self.muscles)
        self.time += 1
        return reward, terminated or truncated

    def position(self):
        # Center of mass of the robot (average of all its point masses)
        return np.mean(self.points(), axis=1)

    def close(self):
        self.world.close()
