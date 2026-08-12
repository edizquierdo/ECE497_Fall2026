"""
Neural Network Controller for Embodied NeuroEvolution.

Defines a feedforward neural network controller that maps sensor inputs
to motor outputs for the Braitenberg vehicle.

Architecture:
    Input layer (2 neurons) -> Hidden layer (hidden_size neurons, Tanh) -> Output layer (2 neurons)

The genome is a flat vector containing all weights and biases.
"""

import torch
import torch.nn as nn


class NeuralController(nn.Module):
    """
    Neural controller for phototaxis: [2 sensors] -> hidden -> [2 motors].

    Args:
        hidden_size: Number of neurons in the hidden layer.

    The genome is a flat vector containing all weights and biases in order:
        - W1: input->hidden weights (2 * hidden_size values)
        - b1: hidden bias (hidden_size values)
        - W2: hidden->output weights (hidden_size * 2 values)
        - b2: output bias (2 values)

    The output layer uses Tanh activation to bound motor commands to [-1, 1].
    """

    def __init__(self, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),   # 2 sensors -> hidden layer
            nn.Tanh(),                   # nonlinearity (activation)
            nn.Linear(hidden_size, 2),   # hidden -> 2 motors
            nn.Tanh(),                   # bound outputs to [-1, 1]
        )

    def forward(self, sensors):
        return self.net(sensors)
