"""
Controllers for the six-legged walker (WalkerEnv): two architectures,
sharing the same `act(...)` / `genome_size(...)` interface so `evolve.py`
and `sim.py` can drive either one through the same code paths.

  - `WalkerController`      -- feedforward net. Input -> Hidden layer(s)
    (Tanh) -> Output layer (Sigmoid). A pure function of the current
    observation: no memory of anything before this tick.
  - `WalkerCTRNNController`  -- continuous-time recurrent neural network.
    Has internal state that persists across an episode (reset once per
    episode via `reset_state()`), and can run with its sensory input on
    or off (see `act(..., sensors_on=...)`).

Both size their input/output to the walker's 6 observations (one leg-angle
sensor per leg) and 18 actions (forward, backward, and foot commands per
leg; see walker.py for exactly how each of the 18 values is used).
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from walker import N_LEGS, OBS_SIZE, ACTION_SIZE, TS

HiddenSizes = Union[int, Sequence[int]]


# Feedforward controller

def _normalize_hidden_sizes(hidden_sizes: HiddenSizes) -> tuple:
    if isinstance(hidden_sizes, int):
        return (hidden_sizes,)
    hidden_sizes = tuple(int(h) for h in hidden_sizes)
    if len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must contain at least one layer size")
    return hidden_sizes


class WalkerController(nn.Module):
    """[6 leg angles] -> hidden layer(s) -> [18 actuator commands in [0,1]].

    Sigmoid (not Tanh) on the output layer regardless of `activation`,
    because actions are interpreted in [0, 1], not [-1, 1] (see walker.py
    for exactly how each of the 18 values is used).

    Args:
        hidden_sizes: single int or sequence of hidden layer sizes.
        obs_size: input dimensionality (one leg-angle sensor per leg).
        activation: name of the activation applied after each hidden layer
            -- one of 'tanh', 'relu', 'sigmoid' (default: 'tanh').
    """

    ACTIVATIONS = {
        "tanh":    nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu":    nn.ReLU,
    }

    def __init__(self, hidden_sizes: HiddenSizes = 64, obs_size: int = OBS_SIZE, activation: str = "tanh"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(self.ACTIVATIONS)}")

        self.hidden_sizes = _normalize_hidden_sizes(hidden_sizes)
        self.obs_size = obs_size

        layers = []
        in_size = obs_size
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self.ACTIVATIONS[activation]())
            in_size = h
        layers.append(nn.Linear(in_size, ACTION_SIZE))
        layers.append(nn.Sigmoid())  # actions are in [0, 1], not [-1, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    @torch.no_grad()
    def act(self, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        action = self(obs_tensor)
        return action.cpu().numpy()

    @staticmethod
    def genome_size(hidden_sizes: HiddenSizes = 64, obs_size: int = OBS_SIZE) -> int:
        # activation doesn't affect parameter count, so it's not a parameter here.
        model = WalkerController(hidden_sizes=hidden_sizes, obs_size=obs_size)
        return sum(p.numel() for p in model.parameters())


# CTRNN controller

SM_DEFAULT = 3          # default modular module size: [FWD, BWD, FOOT] (no free interneurons)
MIN_MODULE_SIZE = 3     # a module needs at least FWD, BWD, FOOT -- no free interneurons below this
N_COMMAND = 3 * N_LEGS  # fully-connected topology's fixed command neurons: 3 roles x 6 legs = 18
RW = 16.0               # genotype-to-weight decoding range: raw gene in [0,1] -> weight in [-RW, RW]
RB = 16.0               # genotype-to-bias decoding range: raw gene in [0,1] -> bias in [-RB, RB]

# Local role indices within one modular leg's module (module_size local
# indices 0, 1, 2 -- anything >= 3 is a free interneuron).
FWD, BWD, FOOT = 0, 1, 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class WalkerCTRNNController:
    """[6 leg angles] -> CTRNN -> [18 actuator commands in [0,1]].

    Unlike a feedforward net, a CTRNN has state that evolves over time,
    so using one takes two extra steps a feedforward controller doesn't
    need: `reset_state()` once per episode (randomizes the starting
    neuron states, mirroring the original init_circuit()), and `act()`
    advances that state by one tick rather than being a pure function of
    the current observation.

    Two circuit topologies, selected by `topology=`:
      - "modular" (default): one small `module_size`-neuron circuit is
        duplicated across all six legs, sharing weights, plus a handful of
        fixed inter-leg connections (see the body-plan diagram below). This
        is the biologically-inspired, weight-sharing design -- and it's why
        the genome stays small (24 numbers at the default module_size=3)
        even though the expanded network has 18 neurons.
      - "fully_connected": one big network of `18 + n_interneurons` neurons,
        every one connected to every other with its own independently
        evolved weight -- no weight-sharing at all. The 18 "command"
        neurons (3 roles x 6 legs) are pre-specified; any extras are free
        interneurons.

    Body plan (looking down at the insect, head at the bottom) -- this is
    what the modular topology's CBW/ISW inter-leg wiring follows:

            y
            |
            ---> x

          Left
        2    1    0
      ------------+
      ------------+  "Head"
        3    4    5
          Right

    Genome layout:
      modular:         [ W (sm*sm) | WS (sm) | CBW (sm) | ISW (sm) | T (sm) | B (sm) ]
      fully_connected:  [ W (n*n) | WS (18) | T (n) | B (n) ],  n = 18 + n_interneurons
    """

    def __init__(self, topology: str = "modular", module_size: int = SM_DEFAULT, n_interneurons: int = 0):
        if topology not in ("modular", "fully_connected"):
            raise ValueError(f"topology must be 'modular' or 'fully_connected', got {topology!r}")
        self.topology = topology
        self.module_size = module_size
        self.n_interneurons = n_interneurons
        self.n_neurons = (
            N_LEGS * module_size if topology == "modular" else N_COMMAND + n_interneurons
        )

        # Per-neuron connectivity/sensing, filled in by set_genome().
        self.Weights = np.zeros((self.n_neurons, self.n_neurons))
        self.Biases = np.zeros(self.n_neurons)
        self.TimeConstants = np.ones(self.n_neurons)
        self.WS_full = np.zeros(self.n_neurons)          # per-neuron sensory input weight
        self.sensor_leg_of = -np.ones(self.n_neurons, dtype=np.int64)  # which leg's sensor, or -1
        self.command_idx = self._command_indices()        # (6,3): each leg's [FWD,BWD,FOOT] neuron index

        # Per-episode state, filled in by reset_state().
        self.States = np.zeros(self.n_neurons)
        self.Outputs = np.zeros(self.n_neurons)

    def _command_indices(self) -> np.ndarray:
        if self.topology == "modular":
            sm = self.module_size
            return np.array(
                [[leg * sm + FWD, leg * sm + BWD, leg * sm + FOOT] for leg in range(N_LEGS)],
                dtype=np.int64,
            )
        return np.array([[leg * 3, leg * 3 + 1, leg * 3 + 2] for leg in range(N_LEGS)], dtype=np.int64)

    @staticmethod
    def genome_size(topology: str = "modular", module_size: int = SM_DEFAULT, n_interneurons: int = 0) -> int:
        """Length of the flat genome vector for the given topology/size --
        what EvoTorch's `Problem(solution_length=...)` should be set to."""
        if topology == "modular":
            if module_size < MIN_MODULE_SIZE:
                raise ValueError(
                    f"module_size must be >= {MIN_MODULE_SIZE} (a module needs at least "
                    f"FWD, BWD, and FOOT command neurons); got {module_size}"
                )
            sm = module_size
            return sm * sm + 5 * sm  # W + WS + CBW + ISW + T + B
        elif topology == "fully_connected":
            if n_interneurons < 0:
                raise ValueError(f"n_interneurons must be >= 0, got {n_interneurons}")
            n = N_COMMAND + n_interneurons
            return n * n + 2 * n + N_COMMAND  # W + (T and B) + WS
        raise ValueError(f"topology must be 'modular' or 'fully_connected', got {topology!r}")

    def set_genome(self, genome) -> None:
        """Decode a flat genome vector (numpy array or torch tensor, each
        entry in [0,1]) into this controller's weights/biases/time
        constants. Call once per genome, before any `act()` calls."""
        genome = np.asarray(genome, dtype=float).reshape(-1)
        expected = self.genome_size(self.topology, self.module_size, self.n_interneurons)
        if genome.size != expected:
            raise ValueError(f"Expected genome length {expected}, got {genome.size}")
        if self.topology == "modular":
            self._set_genome_modular(genome)
        else:
            self._set_genome_fully_connected(genome)

    def _set_genome_modular(self, rG: np.ndarray) -> None:
        sm = self.module_size
        k = 0
        W = (rG[k : k + sm * sm].reshape(sm, sm) * 2.0 * RW) - RW
        k += sm * sm
        WS = (rG[k : k + sm] * 2.0 * RW) - RW
        k += sm
        CBW = (rG[k : k + sm] * 2.0 * RW) - RW
        k += sm
        ISW = (rG[k : k + sm] * 2.0 * RW) - RW
        k += sm
        T = 1.0 + rG[k : k + sm] * 19.0
        k += sm
        B = (rG[k : k + sm] * 2.0 * RB) - RB

        # W (the intra-leg sm x sm block) is copied onto the diagonal for
        # each of the 6 legs -- every leg's module is wired the same way.
        weights = np.zeros((self.n_neurons, self.n_neurons))
        for leg in range(N_LEGS):
            start = leg * sm
            weights[start : start + sm, start : start + sm] = W

        # CBW/ISW are added as sparse off-diagonal connections between
        # specific pairs of legs, following the body plan above.
        # dst_leg -> [(src_leg, weight_vector), ...]
        links = {
            0: [(5, CBW), (1, ISW)],
            1: [(4, CBW), (2, ISW), (0, ISW)],
            2: [(3, CBW), (1, ISW)],
            3: [(2, CBW), (4, ISW)],
            4: [(1, CBW), (5, ISW), (3, ISW)],
            5: [(0, CBW), (4, ISW)],
        }
        for dst_leg, sources in links.items():
            for src_leg, gains in sources:
                for j in range(sm):
                    weights[src_leg * sm + j, dst_leg * sm + j] += gains[j]

        self.Weights = weights
        self.Biases = np.tile(B, N_LEGS)
        self.TimeConstants = np.tile(T, N_LEGS)
        self.WS_full = np.tile(WS, N_LEGS)
        self.sensor_leg_of = np.repeat(np.arange(N_LEGS, dtype=np.int64), sm)

    def _set_genome_fully_connected(self, rG: np.ndarray) -> None:
        n = self.n_neurons
        k = 0
        W = (rG[k : k + n * n].reshape(n, n) * 2.0 * RW) - RW
        k += n * n
        WS = (rG[k : k + N_COMMAND] * 2.0 * RW) - RW
        k += N_COMMAND
        T = 1.0 + rG[k : k + n] * 19.0
        k += n
        B = (rG[k : k + n] * 2.0 * RB) - RB

        self.Weights = W
        self.Biases = B
        self.TimeConstants = T

        WS_full = np.zeros(n)
        WS_full[:N_COMMAND] = WS
        self.WS_full = WS_full

        # Only the 18 command neurons get direct sensory input, each from
        # its own leg; every interneuron gets sensor_leg_of == -1 (no
        # direct sensory input -- it can only learn about the body
        # through its recurrent connections to/from command neurons).
        sensor_leg_of = -np.ones(n, dtype=np.int64)
        for leg in range(N_LEGS):
            for role in range(3):
                sensor_leg_of[leg * 3 + role] = leg
        self.sensor_leg_of = sensor_leg_of

    def reset_state(self, rng: np.random.Generator | None = None) -> None:
        """Randomize the CTRNN's starting neuron states for a fresh
        episode. Call once at the start of every episode, after
        `set_genome()`."""
        rng = np.random.default_rng() if rng is None else rng
        offsets = rng.random(self.n_neurons) * 2.0 - 1.0
        self.States = -self.Biases + offsets
        self.Outputs = sigmoid(self.States + self.Biases)

    def act(self, observation, sensors_on: bool = True, dt: float = TS) -> np.ndarray:
        """Advance the CTRNN by one Euler tick and return the 18
        command-neuron outputs as a WalkerEnv-ready action.

        sensors_on=True feeds `observation` (each leg's live angle) into
        the neurons wired to hear it (RPG). sensors_on=False feeds zero
        sensory input regardless of `observation` (CPG) -- the circuit
        has to generate its rhythm from internal dynamics alone.

        `dt` is the CTRNN's own Euler step size -- how much its internal
        clock believes has elapsed on this call. It defaults to `TS`
        (walker.py's own tick length), which is what every call site in
        this repo except hexapod_bridge.py relies on implicitly (the
        CTRNN's internal clock and the body's physical clock are meant to
        advance in lockstep, one act() per one walker.py tick). Passing a
        different `dt` decouples them -- see hexapod_bridge.py's
        `--timescale`, for driving a real MuJoCo body (whose own per-step
        physical time isn't TS) with a genome evolved under walker.py's
        TS-paced assumption.
        """
        angle_sensor = np.asarray(observation, dtype=float) if sensors_on else np.zeros(N_LEGS)

        inputs = np.zeros(self.n_neurons)
        has_sensor = self.sensor_leg_of >= 0
        inputs[has_sensor] = self.WS_full[has_sensor] * angle_sensor[self.sensor_leg_of[has_sensor]]

        netinput = inputs + self.Weights.T @ self.Outputs
        self.States += dt * ((1.0 / self.TimeConstants) * (-self.States + netinput))
        self.Outputs = sigmoid(self.States + self.Biases)

        return self.Outputs[self.command_idx].reshape(-1)


# Center-crossing initialization (population-seeding helper)

def center_crossing_bias_genes(W: np.ndarray, WS: np.ndarray, CBW: np.ndarray, ISW: np.ndarray) -> np.ndarray:
    """Per-neuron center-crossing biases: set each neuron's bias so its
    steady-state input sits at 0, which keeps its output near the
    midpoint of the sigmoid (0.5) rather than saturated near 0 or 1 --
    a common way to seed a CTRNN population in a dynamically richer
    starting regime.

    For each local neuron `l` in the module:
        B[l] = -(sum_m W[m][l] + WS[l] + CBW[l] + ISW[l]) / 2

    i.e. the bias is half the negative sum of *every* weight feeding into
    that neuron -- the intra-module recurrent weights (column `l` of `W`,
    matching this file's `Weights[j, i]` = weight from `j` into `i`
    convention), the sensory weight, and the two inter-leg weights.
    """
    return -0.5 * (W.sum(axis=0) + WS + CBW + ISW)


def sample_center_crossing_genome(
    topology: str, module_size: int, n_interneurons: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample one random genome the same way Problem's default
    initial_bounds=(0,1) sampling would, except the bias (B) segment is
    replaced with the center-crossing bias for the sampled weights,
    re-encoded back into [0,1] gene space. Used only to *seed* the
    initial population -- evolution is still free to move biases away
    from the center-crossing point via ordinary mutation afterward, same
    as any other gene.

    `topology="fully_connected"` uses a direct generalization of the same
    idea: each neuron's bias is set from its own incoming recurrent
    weights plus its sensory weight, `WS_full` (0 for interneurons --
    there's no CBW/ISW-equivalent concept in this topology, since nothing
    is duplicated per leg).
    """
    size = WalkerCTRNNController.genome_size(topology, module_size, n_interneurons)
    genome = rng.random(size)

    if topology == "modular":
        sm = module_size
        k = 0
        W = (genome[k : k + sm * sm].reshape(sm, sm) * 2.0 * RW) - RW
        k += sm * sm
        WS = (genome[k : k + sm] * 2.0 * RW) - RW
        k += sm
        CBW = (genome[k : k + sm] * 2.0 * RW) - RW
        k += sm
        ISW = (genome[k : k + sm] * 2.0 * RW) - RW
        k += sm
        # k now points past T -- B is the last sm-length segment.
        b_start = k + sm  # skip T (sm genes), landing on B
        B = center_crossing_bias_genes(W, WS, CBW, ISW)
        genome[b_start : b_start + sm] = np.clip((B + RB) / (2.0 * RB), 0.0, 1.0)
    else:
        n = N_COMMAND + n_interneurons
        W = (genome[: n * n].reshape(n, n) * 2.0 * RW) - RW
        WS_full = np.zeros(n)
        WS_full[:N_COMMAND] = (genome[n * n : n * n + N_COMMAND] * 2.0 * RW) - RW
        B = -0.5 * (W.sum(axis=0) + WS_full)
        b_start = n * n + N_COMMAND + n  # after W, WS, T
        genome[b_start : b_start + n] = np.clip((B + RB) / (2.0 * RB), 0.0, 1.0)

    return genome
