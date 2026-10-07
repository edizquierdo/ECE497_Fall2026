##################################################################################
# Soft robot body and world for the embodied task, using Evolution Gym (EvoGym).
#   - The body is a small grid of voxels. Each voxel is one of five materials:
#       0 empty, 1 rigid, 2 soft, 3 horizontal muscle, 4 vertical muscle
#     Row 0 is the top of the robot; the last row rests on the ground.
#   - Every step, each muscle gets a command in [0.6, 1.6]: its target length as
#     a fraction of its rest length (0.6 = squeeze, 1.6 = stretch).
#   - The robot's "brain" is ONE neural network (fnn.FNN) shared by all muscles.
#     Each muscle feeds it the same clock plus its own (x, y) position in the body,
#     so different muscles can do different things with the same weights.
##################################################################################
import warnings
warnings.filterwarnings("ignore")   # EvoGym prints deprecation warnings that aren't about this code

import numpy as np
import gymnasium as gym
import evogym.envs                  # registers EvoGym's tasks with gymnasium
from evogym import is_connected, has_actuator

materials = 5

def makeBody(genes, grid):
    # Each voxel gets 5 genes, one per material, and becomes whichever material's
    # gene is largest (just like fnn.py picks each layer's activation function)
    return np.argmax(genes.reshape(grid, grid, materials), axis=2)

def isValid(body):
    # A body can only be simulated if it is one connected piece with at least one muscle
    return is_connected(body) and has_actuator(body)

class SoftRobot:

    def __init__(self, body, neuralnetwork, envname, duration, period, render=False):
        self.body = body
        self.controller = neuralnetwork
        self.period = period
        self.time = 0
        self.world = gym.make(envname, body=body, max_episode_steps=duration,
                              render_mode="human" if render else None)
        self.world.reset()

        # Where each muscle sits in the body, scaled to [0, 1]: x from left to right,
        # y from bottom to top. Listed row by row, the same order EvoGym expects the commands in.
        rows, cols = np.nonzero((body == 3) | (body == 4))
        self.musclepos = np.column_stack([cols/(body.shape[1]-1), 1 - rows/(body.shape[0]-1)])
        self.muscles = np.ones(len(rows))

    def think(self):
        # Every muscle sees the same clock, plus its own position
        phase = 2*np.pi*self.time/self.period
        clock = np.tile([np.sin(phase), np.cos(phase)], (len(self.muscles), 1))
        inputs = np.hstack([clock, self.musclepos])
        # One forward pass for all muscles at once (one row of inputs per muscle)
        # Outputs are capped to [0, 1], like the motors in Unit 4, then shifted into EvoGym's [0.6, 1.6]
        self.muscles = 0.6 + np.clip(self.controller.forward(inputs)[:,0], 0, 1)

    def move(self):
        # Step the physics; the reward is how far the robot moved to the right this step
        obs, reward, terminated, truncated, info = self.world.step(self.muscles)
        self.time += 1
        return reward, terminated or truncated

    def position(self):
        # Center of mass of the robot (average of all its point masses)
        sim = self.world.unwrapped
        return np.mean(sim.object_pos_at_time(sim.get_time(), "robot"), axis=1)

    def close(self):
        self.world.close()
