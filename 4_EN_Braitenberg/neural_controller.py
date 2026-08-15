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
    Neural controller for phototaxis: [2 sensors] -> hidden(s) -> [2 motors].

    Args:
        hidden_size: Number of neurons in the (single) hidden layer. Ignored
            if hidden_sizes is given.
        hidden_sizes: Optional list of hidden layer sizes, e.g. [16, 16] for
            two 16-neuron hidden layers. Defaults to [hidden_size], which
            reproduces the original single-hidden-layer architecture.
        activation: Name of the activation function applied after each
            hidden layer -- one of 'tanh', 'relu', 'sigmoid' (default:
            'tanh'). The output layer always uses Tanh, regardless of this
            setting, so motor commands stay bounded to [-1, 1].

    The genome is a flat vector containing all weights and biases of the
    resulting nn.Sequential, in PyTorch parameter order.
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
        in_size = 2
        for size in sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(self.ACTIVATIONS[activation]())
            in_size = size
        layers.append(nn.Linear(in_size, 2))  # hidden -> 2 motors
        layers.append(nn.Tanh())              # bound outputs to [-1, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, sensors):
        return self.net(sensors)


def _genome_layout(hidden_sizes):
    """Sizes (in order) of each parameter block within a flat NeuralController genome.

    Matches the order PyTorch's parameters_to_vector uses for
    nn.Sequential(Linear, Activation, ..., Linear, Tanh): W1, b1, ..., W_n,
    b_n, walking the dimension chain [2] + hidden_sizes + [2].
    """
    dims = [2] + list(hidden_sizes) + [2]
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
