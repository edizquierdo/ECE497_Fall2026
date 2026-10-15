"""
A continuous-time recurrent neural network (CTRNN), written as plainly as
possible.

Each neuron i has a state y_i that changes continuously in time:

    tau_i * dy_i/dt = -y_i + sum_j w_ji * o_j + I_i
    o_i = sigmoid(y_i + theta_i)

  y_i      state (think: membrane potential)
  o_i      output (think: firing rate), always between 0 and 1
  tau_i    time constant: how fast neuron i responds (bigger = slower)
  theta_i  bias: shifts where neuron i's sigmoid turns on
  w_ji     weight of the connection FROM neuron j TO neuron i
           (stored as Weights[j][i], so Weights.T @ Outputs is the input)
  I_i      external input

step(dt) advances the network by one small time step dt using Euler
integration: y <- y + dt * dy/dt. dt must be small compared to the
smallest time constant, or the simulation becomes inaccurate (or blows up).
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CTRNN:

    def __init__(self, size):
        self.Size = size
        self.States = np.zeros(size)
        self.TimeConstants = np.ones(size)
        self.invTimeConstants = 1.0 / self.TimeConstants
        self.Biases = np.zeros(size)
        self.Weights = np.zeros((size, size))
        self.Outputs = np.zeros(size)
        self.Inputs = np.zeros(size)

    def step(self, dt):
        netinput = self.Inputs + np.dot(self.Weights.T, self.Outputs)
        self.States += dt * (self.invTimeConstants * (-self.States + netinput))
        self.Outputs = sigmoid(self.States + self.Biases)

    def setWeights(self, weights):
        self.Weights = weights

    def setBiases(self, biases):
        self.Biases = biases

    def setTimeConstants(self, timeconstants):
        self.TimeConstants = timeconstants
        self.invTimeConstants = 1.0 / self.TimeConstants

    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10, 10, size=(self.Size, self.Size))
        self.Biases = np.random.uniform(-10, 10, size=(self.Size))
        self.TimeConstants = np.random.uniform(0.1, 5.0, size=(self.Size))
        self.invTimeConstants = 1.0 / self.TimeConstants

    def setCenterCrossing(self):
        # Center-crossing condition (Mathayomchan & Beer, 2002): choose each
        # neuron's bias so that its sigmoid is centered on the middle of its
        # range of possible inputs. Neuron i's net input lies between the sum
        # of its negative incoming weights (all inputs fully on, only the
        # negative ones counted) and the sum of its positive ones. Its center
        # is half the sum of all incoming weights, so theta_i = -sum_j w_ji / 2.
        # Each neuron then sits where its sigmoid is steepest, instead of being
        # stuck fully on or fully off: the region where interesting dynamics
        # (like oscillations) tend to live.
        self.Biases = -np.sum(self.Weights, axis=0) / 2

    def initializeState(self, s):
        self.Inputs = np.zeros(self.Size)
        self.States = s
        self.Outputs = sigmoid(self.States + self.Biases)

    def save(self, filename):
        np.savez(filename, size=self.Size, weights=self.Weights,
                  biases=self.Biases, timeconstants=self.TimeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.Size = params['size']
        self.Weights = params['weights']
        self.Biases = params['biases']
        self.TimeConstants = params['timeconstants']
        self.invTimeConstants = 1.0 / self.TimeConstants
