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
    Neural controller for CartPole: [4 observations] -> hidden(s) -> [2 action logits].

    Args:
        hidden_size: Number of neurons in the (single) hidden layer. Ignored
            if hidden_sizes is given.
        hidden_sizes: Optional list of hidden layer sizes, e.g. [16, 16] for
            two 16-neuron hidden layers. Defaults to [hidden_size], which
            reproduces the original single-hidden-layer architecture.
        activation: Name of the activation function applied after each
            hidden layer -- one of 'tanh', 'relu', 'sigmoid' (default:
            'tanh').

    The genome is a flat vector containing all weights and biases in
    PyTorch's parameter order. The output layer has no activation (logits
    for discrete action selection).
    """

    ACTIVATIONS = {
        "tanh":    nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu":    nn.ReLU,
    }

    def __init__(self, hidden_size=8, hidden_sizes=None, activation="tanh"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(self.ACTIVATIONS)}")

        sizes = list(hidden_sizes) if hidden_sizes is not None else [hidden_size]

        layers = []
        in_size = 4
        for size in sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(self.ACTIVATIONS[activation]())
            in_size = size
        layers.append(nn.Linear(in_size, 2))  # hidden -> 2 action logits

        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    @torch.no_grad()
    def act(self, observation):
        """Select an action for the given observation."""
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        logits = self(obs_tensor)
        return int(torch.argmax(logits).item())


def _genome_layout(hidden_sizes):
    """Sizes (in order) of each parameter block within a flat NeuralController genome.

    Matches the order PyTorch's parameters_to_vector uses for
    nn.Sequential(Linear, Activation, ..., Linear): W1, b1, ..., W_n, b_n,
    walking the dimension chain [4] + hidden_sizes + [2].
    """
    dims = [4] + list(hidden_sizes) + [2]
    layout = {}
    for i in range(len(dims) - 1):
        in_dim, out_dim = dims[i], dims[i + 1]
        layout[f"w{i + 1}"] = out_dim * in_dim
        layout[f"b{i + 1}"] = out_dim
    return layout


def genome_size(hidden_size=8, hidden_sizes=None):
    """Total genome length (weights + biases) for a given NeuralController architecture.

    Computed analytically from `_genome_layout` rather than instantiating a
    throwaway `NeuralController` just to count `p.numel()` over its
    parameters.

    Args:
        hidden_size: Number of neurons in the single hidden layer. Used only
            when `hidden_sizes` is not given.
        hidden_sizes: Optional list of hidden-layer sizes, overriding `hidden_size`.

    Returns:
        Total number of weights + biases (int).
    """
    sizes = list(hidden_sizes) if hidden_sizes is not None else [hidden_size]
    return sum(_genome_layout(sizes).values())
