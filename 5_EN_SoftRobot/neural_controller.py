"""
Controllers for EvoGym soft robots.

Three very different ways to drive the same actuators:

NeuralController (closed-loop)
    [observation] -> hidden layer(s) -> [one command per actuator]
    A feedforward network, exactly like Project 4's. Its input size and
    output size both depend on the body (EvoGym's observation includes the
    position of every point mass in the robot; its action has one entry per
    actuator voxel), so a network only fits the one body it was built for.

OscillatorController (open-loop)
    command_i(t) = 1.1 + 0.5 * sin(2*pi*t / period + phase_i)
    Every actuator follows the same sine wave, shifted by its own phase. It
    never looks at the observation -- it is a pure rhythm generator, the
    simplest possible central pattern generator (CPG).
    Its genome is just one phase per actuator, which makes it the natural
    partner for evolving bodies: add a voxel, add a phase.

LocalController (closed-loop, distributed)
    The SAME small network, copied into every actuator voxel. Each copy sees
    only local information -- how stretched its own voxel and its four
    neighbors are, whether it's a horizontal or vertical actuator, plus a
    shared clock -- and sends one command to its own voxel. Because the
    network's size doesn't depend on the body, it can be co-designed with
    a body, just like the oscillator.

All three controllers share the same interface -- reset() once per episode, then
act(obs) once per step, returning a NumPy array of actuator commands in
[0.6, 1.6] -- so soft_robot.run_episode() doesn't care which one it gets.
"""

import numpy as np
import torch
import torch.nn as nn

from soft_robot import ACTION_MID, ACTION_AMP

DEFAULT_PERIOD = 25  # simulation steps per oscillation cycle


class NeuralController(nn.Module):
    """
    Feedforward controller: [obs_size] -> hidden(s) -> [n_actuators].

    Args:
        obs_size: Length of the environment's observation vector.
        n_actuators: Number of actuator voxels in the body.
        hidden_sizes: List of hidden layer sizes (default [16]).
        activation: 'tanh', 'relu' or 'sigmoid' on the hidden layer(s).

    The output layer is always Tanh, rescaled from [-1, 1] onto EvoGym's
    actuator range [0.6, 1.6], so every possible genome produces legal commands.
    """

    ACTIVATIONS = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}

    def __init__(self, obs_size, n_actuators, hidden_sizes=(16,), activation="tanh"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(self.ACTIVATIONS)}")
        layers = []
        in_size = obs_size
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(self.ACTIVATIONS[activation]())
            in_size = size
        layers.append(nn.Linear(in_size, n_actuators))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    @staticmethod
    def genome_size(obs_size, n_actuators, hidden_sizes=(16,)):
        """Total weights + biases, counted layer by layer."""
        dims = [obs_size] + list(hidden_sizes) + [n_actuators]
        return sum(dims[i + 1] * dims[i] + dims[i + 1] for i in range(len(dims) - 1))

    def load_genome(self, genome):
        genome = torch.as_tensor(genome, dtype=torch.float32)
        expected = sum(p.numel() for p in self.parameters())
        if genome.numel() != expected:
            raise ValueError(
                f"Genome has {genome.numel()} values but this network needs {expected}. "
                f"It was evolved with a different body, task, or --hidden setting."
            )
        nn.utils.vector_to_parameters(genome, self.parameters())
        return self

    def reset(self):
        pass  # feedforward: no internal state to reset

    @torch.no_grad()
    def act(self, obs):
        out = self.net(torch.as_tensor(obs, dtype=torch.float32))
        return (ACTION_MID + ACTION_AMP * out).numpy()


class OscillatorController:
    """
    Open-loop sine-wave controller: one phase per actuator, shared period.

    Args:
        phases: Array of phase offsets in radians, one per actuator.
        period: Oscillation period in simulation steps.
    """

    def __init__(self, phases, period=DEFAULT_PERIOD):
        self.phases = np.asarray(phases, dtype=float)
        self.period = period
        self.t = 0

    @staticmethod
    def genes_to_phases(genes):
        """Genes live in roughly [-1, 1]; map them onto phases in [-pi, pi]."""
        return np.pi * np.asarray(genes, dtype=float)

    def reset(self):
        self.t = 0

    def act(self, obs=None):
        command = ACTION_MID + ACTION_AMP * np.sin(2 * np.pi * self.t / self.period + self.phases)
        self.t += 1
        return command



class LocalController:
    """
    Distributed controller: one shared network, applied to every actuator voxel.

    Inputs to each copy (14 numbers):
        own voxel's (width, height) strain                       2
        the same for its neighbors above, below, left, right     8  (0 if no voxel there)
        its own type, one-hot (horizontal, vertical)             2
        a shared clock, sin and cos of 2*pi*t / period           2
    Output: one command for that voxel, in [0.6, 1.6].

    The clock is there because without it, every copy of the network sees
    nearly the same thing when the robot is at rest, so they all do nearly
    the same thing and no rhythm ever starts.

    The network itself is an ordinary NeuralController with 14 inputs and
    1 output; all actuators are run through it as one batch.

    Args:
        genome: The shared network's weights and biases.
        body: The body it's driving.
        env: The EvoGym environment (needed to read voxel shapes each step).
        hidden_sizes, activation: As for NeuralController.
        period: Clock period in simulation steps.
    """

    N_INPUTS = 14

    def __init__(self, genome, body, env, hidden_sizes=(8,), activation="tanh", period=DEFAULT_PERIOD):
        self.net = NeuralController(self.N_INPUTS, 1, hidden_sizes, activation).load_genome(genome)
        self.body = np.asarray(body)
        self.env = env
        self.period = period
        rows, cols = np.nonzero(np.isin(self.body, (3, 4)))  # row-major = action order
        # +1 because strains are looked up in a grid padded with a border of zeros
        self.rows, self.cols = rows + 1, cols + 1
        self.types = np.stack([self.body[rows, cols] == 3, self.body[rows, cols] == 4], axis=1).astype(float)

    @staticmethod
    def genome_size(hidden_sizes=(8,)):
        return NeuralController.genome_size(LocalController.N_INPUTS, 1, hidden_sizes)

    def reset(self):
        from soft_robot import voxel_corners
        self.corners = voxel_corners(self.env, self.body)  # robot is undeformed right after reset
        self.t = 0

    def local_inputs(self):
        from soft_robot import voxel_strains
        s = np.pad(voxel_strains(self.env, self.corners), ((1, 1), (1, 1), (0, 0)))
        r, c = self.rows, self.cols
        phase = 2 * np.pi * self.t / self.period
        clock = np.tile([np.sin(phase), np.cos(phase)], (len(r), 1))
        return np.concatenate([s[r, c], s[r - 1, c], s[r + 1, c], s[r, c - 1], s[r, c + 1],
                               self.types, clock], axis=1)

    def act(self, obs=None):
        command = self.net.act(self.local_inputs())[:, 0]
        self.t += 1
        return command

def traveling_wave_phases(body):
    """Fixed phases for morphology-only evolution: a wave that travels from
    the robot's front (right, the direction of travel) to its back (left).

    Each actuator's phase depends only on which column it sits in, so the
    same rule works for any body evolution produces.
    """
    from soft_robot import actuator_indices
    body = np.asarray(body)
    cols = actuator_indices(body) % body.shape[1]
    return 2 * np.pi * cols / body.shape[1]
