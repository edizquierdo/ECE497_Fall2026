"""
Neural Network Controller for CartPole NeuroEvolution.

Defines a feedforward neural network controller that maps cartpole observations
to action logits.

Architecture:
    Input layer (4 neurons) -> Hidden layer (hidden_size neurons, Tanh) -> Output layer (2 neurons)

The genome is a flat vector containing all weights and biases.
"""

import torch
import torch.nn as nn


class NeuralController(nn.Module):
    """
    Neural controller for CartPole: [4 observations] -> hidden -> [2 action logits].

    Args:
        hidden_size: Number of neurons in the hidden layer.

    The genome is a flat vector containing all weights and biases in order:
        - W1: input->hidden weights (4 * hidden_size values)
        - b1: hidden bias (hidden_size values)
        - W2: hidden->output weights (hidden_size * 2 values)
        - b2: output bias (2 values)

    The output layer has no activation (logits for discrete action selection).
    """

    def __init__(self, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),   # 4 observations -> hidden layer
            nn.Tanh(),                   # nonlinearity (activation)
            nn.Linear(hidden_size, 2),   # hidden -> 2 actions (logits)
        )

    def forward(self, obs):
        return self.net(obs)

    @torch.no_grad()
    def act(self, observation):
        """Select an action for the given observation."""
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        logits = self(obs_tensor)
        return int(torch.argmax(logits).item())
