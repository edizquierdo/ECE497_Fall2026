"""
Neural Network Controller for Embodied NeuroEvolution.

Defines a feedforward neural network controller that maps sensor inputs
to motor outputs for the Braitenberg vehicle.

Architecture:
    Input layer (2 sensors) -> Hidden layer(s) (Tanh by default) -> Output layer (2 motors, Tanh)

This is the same network you evolved for XOR in Project 3 (XORNet), with
two changes: it has 2 outputs instead of 1 (one per motor), and the output
layer is passed through Tanh so motor commands are bounded to [-1, 1].

The genome is a flat vector containing all weights and biases.
"""

import torch
import torch.nn as nn


class NeuralController(nn.Module):
    """
    Neural controller for phototaxis: [2 sensors] -> hidden(s) -> [2 motors].

    Args:
        hidden: Number of neurons in the (single) hidden layer. Ignored
            if hidden_sizes is given.
        hidden_sizes: Optional list of hidden layer sizes, e.g. [16, 16] for
            two 16-neuron hidden layers. Defaults to [hidden].
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

    def __init__(self, hidden=8, hidden_sizes=None, activation="tanh"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(self.ACTIVATIONS)}")

        sizes = list(hidden_sizes) if hidden_sizes is not None else [hidden]

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


def genome_size(hidden=8, hidden_sizes=None):
    """Total genome length (weights + biases) for a given NeuralController architecture.

    Computed analytically from `_genome_layout` rather than instantiating a
    throwaway `NeuralController` just to count `p.numel()` over its
    parameters.

    Args:
        hidden: Number of neurons in the single hidden layer. Used only
            when `hidden_sizes` is not given.
        hidden_sizes: Optional list of hidden-layer sizes, overriding `hidden`.

    Returns:
        Total number of weights + biases (int).
    """
    sizes = list(hidden_sizes) if hidden_sizes is not None else [hidden]
    return sum(_genome_layout(sizes).values())


# Elementwise activation functions used by batched_forward(). Must mirror
# NeuralController.ACTIVATIONS so a genome behaves identically whether it is
# loaded into a NeuralController (as sim.py does) or evaluated in a batch (as
# evolve.py does).
_ACTIVATION_FNS = {
    "tanh":    torch.tanh,
    "sigmoid": torch.sigmoid,
    "relu":    torch.relu,
}


def batched_forward(genomes, sensors, hidden=8, hidden_sizes=None, activation="tanh"):
    """Run a whole population of networks at once, one network per genome.

    This is the same trick Project 3's make_fitness_fn used: instead of
    loading each genome into its own nn.Module, slice every individual's
    W and b matrices directly out of the (pop, n_genes) genome batch and
    apply them with one batched einsum per layer.

    Args:
        genomes: Tensor of shape (pop, n_genes).
        sensors: Tensor of shape (pop, E, 2) -- the left/right sensor readings
            for each of E vehicles controlled by each individual.
        hidden, hidden_sizes, activation: The architecture (as in NeuralController).

    Returns:
        Tensor of shape (pop, E, 2): the (left, right) motor commands in [-1, 1].
    """
    sizes = list(hidden_sizes) if hidden_sizes is not None else [hidden]
    layout = _genome_layout(sizes)
    dims = [2] + sizes + [2]
    n_layers = len(dims) - 1
    act_fn = _ACTIVATION_FNS[activation]
    pop = genomes.shape[0]

    activ = sensors
    offset = 0
    for i in range(1, n_layers + 1):
        in_dim, out_dim = dims[i - 1], dims[i]
        w = genomes[:, offset:offset + layout[f"w{i}"]].view(pop, out_dim, in_dim)
        offset += layout[f"w{i}"]
        b = genomes[:, offset:offset + layout[f"b{i}"]].view(pop, out_dim)
        offset += layout[f"b{i}"]

        # 'pek,phk->peh': per-individual (out_dim=h, in_dim=k) weights applied
        # to each of that individual's E vehicles.
        pre = torch.einsum("pek,phk->peh", activ, w) + b.unsqueeze(1)
        activ = act_fn(pre) if i < n_layers else torch.tanh(pre)  # output layer is always Tanh

    return activ
