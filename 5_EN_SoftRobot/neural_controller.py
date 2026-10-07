"""
Controllers ("brains") for EvoGym soft robots.

Every brain here is a small feedforward network, and every one of them fits
ANY body on its grid -- which is what lets the same brain be evolved for a
fixed, hand-designed body or co-evolved with a body that keeps changing.
They differ along two independent choices:

                     open-loop (no sensing)         closed-loop (+ sensing)
    LocalController  clock + own (x, y) position    + own and neighbors' strain
                     -> 1 command, own muscle       -> 1 command, own muscle
    GlobalController clock                          + every cell's strain
                     -> 1 command per grid cell     -> 1 command per grid cell

Local: ONE small network, copied into every muscle (same weights in every
copy). Each copy only knows where it sits in the body and, if closed-loop,
how stretched its own voxel and its neighbors are.

Global: ONE network for the whole body. It has one output per cell of the
grid, and each muscle takes the output of the cell it sits in; outputs for
cells that aren't muscles are ignored. If closed-loop, it sees how
stretched every cell of the grid is (0 for empty cells).

Both get a shared clock, sin and cos of 2*pi*t / period, as a source of
rhythm. "Strain" is a voxel's (width, height) relative to its rest size,
minus 1: 0 at rest, positive when stretched, negative when squeezed.

All controllers share the same interface -- reset() once per episode, then
act(obs) once per step, returning a NumPy array of actuator commands in
[0.6, 1.6] -- so soft_robot.run_episode() doesn't care which one it gets.
"""

import numpy as np
import torch
import torch.nn as nn

from soft_robot import ACTION_MID, ACTION_AMP, actuator_indices, voxel_corners, voxel_strains

DEFAULT_PERIOD = 25  # simulation steps per clock cycle


class Network(nn.Module):
    """
    Feedforward network, as in Project 4: [n_inputs] -> hidden(s) -> [n_outputs].

    The output layer is always Tanh, rescaled from [-1, 1] onto EvoGym's
    actuator range [0.6, 1.6], so every possible genome produces legal commands.
    """

    ACTIVATIONS = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}

    def __init__(self, n_inputs, n_outputs, hidden_sizes=(8,), activation="tanh"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(self.ACTIVATIONS)}")
        layers = []
        in_size = n_inputs
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(self.ACTIVATIONS[activation]())
            in_size = size
        layers.append(nn.Linear(in_size, n_outputs))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    @staticmethod
    def genome_size(n_inputs, n_outputs, hidden_sizes=(8,)):
        """Total weights + biases, counted layer by layer."""
        dims = [n_inputs] + list(hidden_sizes) + [n_outputs]
        return sum(dims[i + 1] * dims[i] + dims[i + 1] for i in range(len(dims) - 1))

    def load_genome(self, genome):
        genome = torch.as_tensor(np.array(genome), dtype=torch.float32)
        expected = sum(p.numel() for p in self.parameters())
        if genome.numel() != expected:
            raise ValueError(f"Genome has {genome.numel()} values but this network needs {expected}.")
        nn.utils.vector_to_parameters(genome, self.parameters())
        return self

    @torch.no_grad()
    def act(self, inputs):
        out = self.net(torch.as_tensor(inputs, dtype=torch.float32))
        return (ACTION_MID + ACTION_AMP * out).numpy()


class _ClockedController:
    """Shared plumbing: the clock, and reading strains for closed-loop brains."""

    def __init__(self, body, env, closed_loop, period):
        self.body = np.asarray(body)
        self.env = env
        self.closed_loop = closed_loop
        self.period = period

    def reset(self):
        self.t = 0
        if self.closed_loop:
            # Right after env.reset() the robot is undeformed, which is when
            # voxel_corners() can work out which point masses belong to which voxel.
            self.corners = voxel_corners(self.env, self.body)

    def clock(self):
        phase = 2 * np.pi * self.t / self.period
        return np.array([np.sin(phase), np.cos(phase)])

    def strains(self):
        return voxel_strains(self.env, self.corners)  # (rows, cols, 2)

    def act(self, obs=None):
        command = self.think()
        self.t += 1
        return command


class LocalController(_ClockedController):
    """
    One small network, copied into every muscle.

    Inputs to each copy:
        clock: sin, cos of 2*pi*t / period                          2
        its own (x, y) position in the body, each in [0, 1]         2
          (x from left to right, y from bottom to top)
        [closed-loop only] its own voxel's (width, height) strain,  10
          then the same for the voxels above, below, left, right
          (0 where there's no voxel)
    Output: one command, for its own muscle.
    """

    @staticmethod
    def n_inputs(closed_loop):
        return 4 + (10 if closed_loop else 0)

    @staticmethod
    def genome_size(body_shape, closed_loop, hidden_sizes=(8,)):
        # body_shape is unused: a local network is the same size whatever the body.
        return Network.genome_size(LocalController.n_inputs(closed_loop), 1, hidden_sizes)

    def __init__(self, genome, body, env, closed_loop=True, hidden_sizes=(8,), activation="tanh",
                 period=DEFAULT_PERIOD):
        super().__init__(body, env, closed_loop, period)
        self.net = Network(self.n_inputs(closed_loop), 1, hidden_sizes, activation).load_genome(genome)
        n_rows, n_cols = self.body.shape
        # Muscles in row-major order, which is the order EvoGym expects commands in.
        self.rows, self.cols = np.nonzero(np.isin(self.body, (3, 4)))
        self.positions = np.column_stack([self.cols / max(n_cols - 1, 1),
                                          1 - self.rows / max(n_rows - 1, 1)])

    def think(self):
        n = len(self.rows)
        inputs = [np.tile(self.clock(), (n, 1)), self.positions]
        if self.closed_loop:
            s = np.pad(self.strains(), ((1, 1), (1, 1), (0, 0)))  # border of zeros
            r, c = self.rows + 1, self.cols + 1
            inputs += [s[r, c], s[r - 1, c], s[r + 1, c], s[r, c - 1], s[r, c + 1]]
        # One forward pass for all muscles at once (one row of inputs per muscle).
        return self.net.act(np.concatenate(inputs, axis=1))[:, 0]


class GlobalController(_ClockedController):
    """
    One network for the whole body, with one output per grid cell.

    Inputs:
        clock: sin, cos of 2*pi*t / period                          2
        [closed-loop only] every cell's (width, height) strain,     2 per cell
          row by row (0 for empty cells)
    Outputs: one command per grid cell. Each muscle takes the output of
    the cell it sits in; the rest are ignored.
    """

    @staticmethod
    def n_inputs(body_shape, closed_loop):
        return 2 + (2 * int(np.prod(body_shape)) if closed_loop else 0)

    @staticmethod
    def genome_size(body_shape, closed_loop, hidden_sizes=(16,)):
        n_cells = int(np.prod(body_shape))
        return Network.genome_size(GlobalController.n_inputs(body_shape, closed_loop), n_cells, hidden_sizes)

    def __init__(self, genome, body, env, closed_loop=True, hidden_sizes=(16,), activation="tanh",
                 period=DEFAULT_PERIOD):
        super().__init__(body, env, closed_loop, period)
        shape = self.body.shape
        self.net = Network(self.n_inputs(shape, closed_loop), int(np.prod(shape)),
                           hidden_sizes, activation).load_genome(genome)
        self.muscle_cells = actuator_indices(self.body)  # flat, row-major cell index of each muscle

    def think(self):
        inputs = [self.clock()]
        if self.closed_loop:
            inputs.append(self.strains().reshape(-1))
        return self.net.act(np.concatenate(inputs))[self.muscle_cells]


CONTROLLERS = {"local": LocalController, "global": GlobalController}
DEFAULT_HIDDEN = {"local": [8], "global": [16]}


class TravelingWaveController:
    """
    A FIXED brain, used only by --mode morphology (where only the body evolves).

    Every muscle follows the same sine wave, with a phase set by its column:
        command(t) = 1.1 + 0.5 * sin(2*pi*t / period + 2*pi*column / n_columns)
    so a wave of contraction sweeps across the body, whatever body it's in.
    Nothing about it is evolved.
    """

    def __init__(self, body, period=DEFAULT_PERIOD):
        body = np.asarray(body)
        cols = actuator_indices(body) % body.shape[1]
        self.phases = 2 * np.pi * cols / body.shape[1]
        self.period = period

    def reset(self):
        self.t = 0

    def act(self, obs=None):
        command = ACTION_MID + ACTION_AMP * np.sin(2 * np.pi * self.t / self.period + self.phases)
        self.t += 1
        return command
