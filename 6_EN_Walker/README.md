# Project 6: Embodied NeuroEvolution III — Six-Legged Walker

## Overview

In this project, you will combine the concepts from previous projects —
evolutionary algorithms and neural networks — to evolve a controller for a
six-legged (hexapod) insect body, in two phases:

- **Phase A: Feedforward networks.** A pure stimulus-response mapping —
  the network's output depends only on the current tick's sensor reading.
- **Phase B: CTRNNs (continuous-time recurrent neural networks).** A
  dynamical system with its own internal state, which can keep generating
  a rhythmic gait even without sensory input at all.

Both phases evolve the weights of a neural network controller with the
same evolutionary algorithm, on the exact same physical body and the exact
same fitness signal (average forward speed) — so the comparison between
them is architecture vs. architecture, nothing else. The network receives
each leg's current angle as input and produces actuator commands (forward
force, backward force, and foot up/down) for every leg as output.

---

## Learning Objectives

By completing this project, you will learn how to:

- Combine neural networks with evolutionary algorithms (neuroevolution)
- Encode a neural network's weights as a genome for evolutionary search
- Apply neuroevolution to a multi-legged embodied robotics task
- Investigate how network architecture (layer count and size, or CTRNN
  topology and circuit size) affects what evolution can learn
- Investigate how genome size affects evolutionary dynamics
- Contrast a stateless, purely reactive controller with a stateful,
  dynamical one — and see what that difference in architecture makes
  possible (or impossible) behaviorally

---

## Background

### Neuroevolution

**Neuroevolution** is the application of evolutionary algorithms to optimize
neural network weights. Instead of using gradient descent and backpropagation,
the entire weight vector (the "genome") is treated as a solution to be evolved:

1. **Encode**: Flatten all network weights into a single real-valued vector
2. **Evaluate**: Load the genome into the network, run it in the environment,
   compute a fitness score
3. **Evolve**: Apply selection, crossover, and mutation to produce the next
   generation

This approach works even when the objective has no usable gradient.

### Six-Legged Walker Environment

The **Walker** task is an insect-inspired locomotion problem:

- A six-legged body, legs arranged in two rows of three (left/right), each
  leg attached at a fixed point and swinging within angle limits
- The goal is to walk forward as far as possible while staying balanced on a
  stable tripod of supporting legs
- The system is dynamic — without coordinated stance/swing timing across
  legs, the body doesn't move forward at all (an unsupported body has its
  velocity zeroed out every tick)

#### Environment Configuration

The `WalkerEnv` class (in `walker.py`) provides a `reset()`/`step()`
interface to the walker's body physics, shared unchanged by both phases.

**Observations:** 6 values — each leg's current angle, in
`[BACKWARD_ANGLE_LIMIT, FORWARD_ANGLE_LIMIT]` radians.

**Actions:** 18 continuous values in [0, 1]: `(forward, backward, foot)` for
each of the 6 legs. `forward`/`backward` drive that leg's push force;
`foot > 0.5` means the foot is planted (stance), `foot ≤ 0.5` means it's
swinging.

**Reward:** Forward body speed (`vx`) each tick. A rep's overall fitness is
`cx / duration` — average forward speed over the whole episode.

**Balance constraint:** Every tick, the body only moves if the legs currently
in stance form a stable tripod under the body's center — otherwise forward
velocity is zeroed out for that tick, whatever the controller commanded. This
is not a termination condition by default — it's a built-in penalty for
uncoordinated leg timing, baked into the physics itself.

**Episode length:** Every episode runs for the full `--duration` simulated
seconds by default. Phase A can optionally shorten stalled-out episodes
early (see `--patience` below); there's no other fall/failure state.

### Two Controller Architectures

Both controllers see the same 6-value observation and produce the same
18-value action, through the same `act(...)` call — but they get there very
differently.

**Feedforward (`WalkerController`)** — Input → Hidden layer(s) (Tanh) →
Output layer (Sigmoid). Every tick's action depends *only* on that tick's
leg angles. It has no memory: give it the same observation twice, in any
order, and it produces the same action twice. Any rhythm in its behavior
has to be reconstructed indirectly, tick by tick, from the leg-angle
feedback alone.

**CTRNN (`WalkerCTRNNController`)** — a continuous-time recurrent neural
network: each neuron has its own internal state that decays and integrates
input over time (governed by a per-neuron time constant), and neurons are
recurrently connected to each other, not just arranged in feedforward
layers. That internal state is what makes it a *dynamical system* rather
than a lookup function — it can sustain an oscillating rhythm on its own,
even with zero sensory input, the way a central pattern generator (CPG) in
a real insect's nervous system does. Two topology choices:
- `modular` (default): a small circuit is duplicated once per leg,
  sharing weights across legs, plus a handful of fixed inter-leg
  connections — biologically-inspired, and keeps the genome small.
- `fully_connected`: one big network, no weight-sharing, with optional
  free interneurons beyond the 18 command neurons.

It can also be run in three sensory modes: `rpg` (live leg-angle feedback
every tick — a "reflex pattern generator"), `cpg` (no sensory input at
all — the rhythm must come entirely from internal dynamics), or `mpg`
(mixed pattern generator: fitness averaged across both, so evolution finds
a circuit that works either way).

**How `mpg` actually alternates cpg/rpg is per-*episode*, not within a
single episode.** During evolution (`evolve.py`), each genome's fitness
comes from `--episodes_per_eval` episodes; under `mpg`, odd-indexed
episodes run `rpg` and even-indexed ones run `cpg` (see
`_sensors_on_for_episode()` in `evolve.py`), and the reported fitness is
the average across all of them — so a genome only scores well under `mpg`
if it walks reasonably under *both* conditions, not just one. During
playback (`sim.py`), the same alternation happens across `--reps`: the
first rep runs `cpg`, the second `rpg`, and so on, and `--scores` reports
the average across that mix. If you want to see one specific genome's
gait under one specific condition rather than an alternating mix, run it
under `--mode cpg` or `--mode rpg` explicitly instead of `--mode mpg`.

### Phase C: Sim-to-Real Transfer Test (Optional / Advanced)

`walker.py`'s physics is idealized: angle-clamped legs, a hand-rolled
tripod-support check, no real contact forces, friction, or inertia. Phase
C asks a harder question than "does evolution converge": **does a genome
evolved on that idealized physics still do anything useful on a real
rigid-body simulation it never saw during evolution?**

`hexapod.xml` is a proper MuJoCo model of a six-legged body with real
contact dynamics, friction, and inertia. Its leg design is deliberately
**not** the hip+knee kinematic chain you'd see in Gymnasium's own `Ant-v5`
(a torso + 2-segment legs). Instead, each leg is a single rigid segment
pivoting at one mount point, with 2 co-located joints there — "sweep"
(forward/backward, matching `walker.py`'s single per-leg angle sensor
directly) and "lift" (up/down, driven by the `foot` signal). A hip+knee
chain is closer to how a real hexapod robot's legs are actually built,
but it would introduce a second, unrelated kind of mismatch on top of the
one this test is actually about: not just idealized-vs-real *physics*,
but also a structural 1-DOF-vs-2-DOF leg mismatch that has nothing to do
with physics fidelity. The single-pivot design keeps that variable fixed
so physics is the only thing that changed.

`hexapod_env.py` wraps `hexapod.xml` in a standard Gymnasium `MujocoEnv`.
Even with matched leg kinematics, this real environment still doesn't
exactly share `walker.py`'s interface:

| | `walker.py` (idealized) | `hexapod_env.py` (real MuJoCo) |
|---|---|---|
| Observation | 6 leg angles | 35 values (torso pose/velocity + 12 joint angles/velocities) |
| Action | 18 values: `(fwd, bwd, foot)` × 6 legs, each in [0,1] | 12 torque commands in [-1,1]: `(sweep, lift)` × 6 legs |

`hexapod_bridge.py` translates between them. The sweep half of this
mapping is a direct correspondence (`sweep torque = fwd - bwd`,
`sweep angle` extracted straight from the hexapod's observation as
`walker.py`'s per-leg angle). The lift half needs more care: `walker.py`'s
`foot` is a stance/swing signal its own physics interprets by zeroing
body velocity when unsupported, never as an actual commanded torque, so
turning `foot` into a lift-joint torque command every tick takes a real
design decision. `hexapod_bridge.py` uses a **closed-loop PD controller**
for this: it reads the lift joint's actual current angle and velocity
from the hexapod's own sensors and drives it toward a target angle (down
while planted, up while swinging), tapering the torque off as the joint
arrives instead of pushing at a constant rate regardless of where it
already is — the same idea as a servo holding a position.

Even so, transfer is not guaranteed. A genome evolved on `walker.py`'s
idealized physics may still drag a leg, slip, or fail to support its own
weight on the real body, because real contact dynamics, friction, and
inertia are all things evolution never saw. Whether — and how well — a
given genome transfers is itself worth investigating: see Part 4 below.

**Watching it run:** `hexapod_bridge.py --render` opens MuJoCo's live
interactive 3D viewer — but only works on a machine with a display (your
own laptop, not a headless server). For a headless GPU server (no
display), use `hexapod_render.py` instead, which renders offscreen and
saves an MP4:

```bash
python hexapod_render.py --controller feedforward --hidden 16 --genome ff_best.npy --duration 15 --out hexapod_demo.mp4
```

See the `hexapod_render.py` options table further below for the full
flag list (torque scale, controller type, resolution, fps, etc. — the
same set as `hexapod_bridge.py`, plus output/rendering options).

---

## Project Structure

| File | Purpose |
|------|---------|
| `neural_controller.py` | Defines both controller classes: `WalkerController` (feedforward) and `WalkerCTRNNController` (CTRNN) |
| `walker.py` | Body-physics environment for the six-legged walker, shared by both controllers |
| `evolve.py` | Runs neuroevolution using EvoTorch, for either controller (`--controller feedforward\|ctrnn`) |
| `sim.py` | Runs simulations with a controller — scoring, trajectory plots, and trace files for `visualize_walker.py` (`--controller feedforward\|ctrnn`) |
| `visualize_walker.py` | Turns a trace file from `sim.py` into a gait diagram, per-actuator/per-neuron output plots, and a top-down animation — works unmodified for either controller's trace |
| `hexapod.xml` | [Phase C] MuJoCo model (MJCF) of a real six-legged body with contact dynamics, friction, and inertia |
| `hexapod_env.py` | [Phase C] Gymnasium `MujocoEnv` wrapper around `hexapod.xml` |
| `hexapod_bridge.py` | [Phase C] Translates a `walker.py`-evolved genome's action space onto the real hexapod's actuators, and runs it there |
| `hexapod_lift_sweep.py` | [Phase C] Sweeps `--lift_gain` and `--timescale` together over a grid of values (with repetitions per cell) for a given genome, and plots average displacement (direction-agnostic distance from start) as a heatmap over the two parameters |
| `requirements.txt` | Pinned dependency versions for this project's virtual environment (Phases A/B plus optional Phase C extras) |
| `README.md` | This documentation |

---

## Installation

The project requires:

- Python 3.9 or newer
- NumPy
- Matplotlib
- PyTorch
- EvoTorch
- Pillow (for `visualize_walker.py`'s animation output)

All five are listed with tested version ranges in `requirements.txt`, so
you install them together into a project-specific virtual environment
rather than into your system Python. Doing this in a fresh venv (not
`pip install`-ing directly) matters more here than in a typical script:
EvoTorch's API has changed across versions, and Ray/PyTorch version
mismatches can cause `evolve.py` to behave differently machine to machine
even when the code hasn't changed — a venv pinned to `requirements.txt`
is what makes sure everyone in the class is running the same thing.

**Create and activate a virtual environment**, from inside this project's
folder:

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Windows (cmd.exe):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

You'll know it worked if your terminal prompt now starts with `(venv)`.
Do this every time you come back to work on the project, before running
any of the commands below — you'll need to `activate` again in each new
terminal session (no need to `venv` again; that step is one-time).

**Install the pinned dependencies:**

```bash
pip install -r requirements.txt
```

**To leave the environment** when you're done:

```bash
deactivate
```

If you'd rather use `conda`, that's fine too — just create an environment
with a matching Python version and run the same
`pip install -r requirements.txt` inside it.

`requirements.txt` also includes `gymnasium[mujoco]` and `mujoco`, needed
only for the optional Phase C sim-to-real transfer test — Phases A and B
don't touch them at all.

---

## Running Neuroevolution

`--controller` (default `feedforward`) selects which architecture
`evolve.py` evolves. Everything else about the EA — selection, crossover,
mutation, generation loop — is identical either way; only genome
construction and the per-episode simulation loop change underneath.

### Phase A: Feedforward

```bash
python evolve.py --controller feedforward
```

(`--controller feedforward` is the default, so plain `python evolve.py`
works too.)

| Option | Description | Default |
|--------|-------------|---------|
| `--hidden` | Hidden layer size(s). Give one value for a single hidden layer, or several for a deeper network, e.g. `--hidden 64 64 64` | `64` |
| `--activation` | Activation function on the hidden layer(s): `tanh`, `relu`, or `sigmoid`. The output layer always stays Sigmoid regardless, so actions stay in `[0, 1]` | `tanh` |
| `--patience` | Early-terminate an episode once the walker hasn't advanced `--min_progress` over the trailing `--patience` seconds. `0` disables this and always runs the full `--duration` | `0.0` |
| `--min_progress` | Minimum forward-position advance required over the trailing `--patience` window to keep an episode going (only used if `--patience > 0`) | `0.5` |

### Phase B: CTRNN

```bash
python evolve.py --controller ctrnn
```

| Option | Description | Default |
|--------|-------------|---------|
| `--topology` | `modular` (one small circuit duplicated per leg, weight-sharing) or `fully_connected` (one big network, no weight-sharing) | `modular` |
| `--module_size` | Neurons per leg module, `modular` topology only (minimum 3) | `3` |
| `--interneurons` | Extra free neurons beyond the 18 command neurons, `fully_connected` topology only | `0` |
| `--mode` | `cpg` (no sensory input), `rpg` (live leg-angle feedback every tick), or `mpg` (fitness averaged over both) | `rpg` |

`--patience`/`--min_progress` are ignored for `--controller ctrnn` — a
CTRNN's rhythm can take a while to spin up, so early-terminating on early
stillness would unfairly penalize slow-starting gaits that would have
walked well once established.

### Shared options (both phases)

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithm` | `ga` (GeneticAlgorithm: SBX crossover + Gaussian mutation) or `es` (CMA-ES, evotorch's `CMAES` — no separate crossover/mutation operators; adapts its own step size and covariance each generation) | `ga` |
| `--popsize` | Population size | `50` |
| `--gens` | Number of generations | `100` |
| `--episodes_per_eval` | Episodes to average per fitness evaluation | `3` |
| `--duration` | Simulated seconds per episode | `220.0` |
| `--mut_stdev` | `--algorithm ga`: Gaussian mutation standard deviation. `--algorithm es`: reused as CMA-ES's initial step size (`stdev_init`) | `0.05` |
| `--tournament_size` | [`--algorithm ga` only] Tournament size for SBX crossover | `3` |
| `--eta` | [`--algorithm ga` only] Distribution index for SBX crossover | `20` |
| `--no-elitism` | [`--algorithm ga` only] Disable elitism (best individual always survives by default) | off |
| `--seed_genome FILE` | Seed the initial population around a previously saved genome (`--output` from an earlier run) instead of starting from scratch — one exact copy plus the rest perturbed by `--seed_noise`. Must match the current `--controller`/architecture settings | `None` |
| `--seed_noise` | Stdev of the Gaussian perturbation applied to `--seed_genome` copies | `0.05` |
| `--center_crossing` | [`--controller ctrnn` only] Seed the initial population's biases at the center-crossing point (steady-state input near 0, output near 0.5) instead of uniformly at random — meant to start evolution in a region of richer circuit dynamics. `--topology fully_connected` uses a direct generalization of the same idea — see `neural_controller.py`. Ignored if `--seed_genome` is also given | off |
| `--workers` | Parallel workers for evaluating the population: `max` (all CPU cores), an integer, or `none` for a single process | `max` |
| `--vizperf` | Visualize fitness over generations | off |
| `--verbose` | Print per-generation statistics to console | off |
| `--seed` | Random seed for reproducibility | `None` |
| `--output FILE` | File path to save the best genome (e.g. `best_genome.npy`) | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves across configurations without re-running evolution | `None` |

For example:

```bash
python evolve.py --controller feedforward --hidden 32 --popsize 100 --gens 150 --vizperf --verbose
python evolve.py --controller ctrnn --topology modular --popsize 100 --gens 150 --vizperf --verbose
```

**Before running a big configuration like that, start small to make sure
everything works and to get a feel for how long things take:**

```bash
python evolve.py --controller feedforward --hidden 16 --popsize 20 --gens 40 --vizperf --verbose
python evolve.py --controller ctrnn --popsize 20 --gens 40 --vizperf --verbose
```

---

## Running Simulations

After evolution, `sim.py` loads a saved genome and runs it — scoring it,
plotting its trajectory, or writing a trace file for `visualize_walker.py`.
Pass the **same** `--controller` (and matching `--hidden` or
`--topology`/`--module_size`/`--interneurons`) that you evolved with —
`sim.py` will raise a clear error if the genome's size doesn't match what
you asked for.

| Option | Description | Default |
|--------|-------------|---------|
| `--controller` | `feedforward` or `ctrnn` — must match how `--genome` was evolved | `feedforward` |
| `--genome FILE` | Path to a saved genome (`.npy`). Omit to use a randomly initialized controller | `None` |
| `--hidden` | [feedforward only] must match what you evolved with | `64` |
| `--activation` | [feedforward only] must match what you evolved with | `tanh` |
| `--topology`, `--module_size`, `--interneurons` | [ctrnn only] must match what you evolved with | `modular`, `3`, `0` |
| `--mode` | [ctrnn only] sensor condition to run under (`cpg`/`rpg`/`mpg`) | `rpg` |
| `--duration` | Simulated seconds per episode | `220.0` |
| `--reps` | Number of independent repetitions/episodes to run | `5` |
| `--viztraces` | Plot the body's x-position over time | off |
| `--scores` | Print average forward speed over all repetitions | off |
| `--trace FILE` | Write a single episode's per-tick data to this path, for `visualize_walker.py` | `None` |
| `--seed` | Random seed for reproducibility | `None` |

Score an evolved controller:

```bash
python sim.py --controller feedforward --genome ff_best.npy --scores
python sim.py --controller ctrnn --genome ctrnn_best.npy --scores
```

Plot its trajectory over several episodes:

```bash
python sim.py --controller feedforward --genome ff_best.npy --viztraces --reps 10
```

Write a trace file and turn it into a gait diagram, actuator/neuron-output
plots, and an animation:

```bash
python sim.py --controller feedforward --genome ff_best.npy --trace ff_walk.dat --duration 1000
python visualize_walker.py ff_walk.dat --out_prefix ff_walk

python sim.py --controller ctrnn --genome ctrnn_best.npy --trace ctrnn_walk.dat --duration 1000
python visualize_walker.py ctrnn_walk.dat --out_prefix ctrnn_walk --topology modular --module_size 3
```

This produces `<prefix>_gait.png` (a stance/swing diagram, one row per
leg), `<prefix>_neurons.png` (actuator/neuron outputs plotted over time),
and `<prefix>_animation.gif` (a top-down animation of the body and legs
moving). Note the `--topology`/`--module_size` flags on
`visualize_walker.py` itself for the CTRNN trace — these only affect how
`_neurons.png` groups neurons into rows, and must match what you evolved
with (`sim.py`'s feedforward traces don't need them; the default
`fully_connected` view is what a feedforward trace's 18 flat outputs
expect).

| Option | Description | Default |
|--------|-------------|---------|
| `trace_path` | Positional. Trace file written by `sim.py --trace` | required |
| `--out_prefix` | Output filename prefix — produces `<prefix>_gait.png`, `<prefix>_neurons.png`, `<prefix>_animation.gif` | trace filename stem |
| `--topology` | `fully_connected` or `modular` — must match what the trace was generated with (only affects `_neurons.png`'s layout) | `fully_connected` |
| `--module_size` | Must match the module size the trace was generated with. Only used with `--topology modular` | `0` |
| `--max_time` | Max simulated seconds included in **all three** outputs — the gait diagram and neuron plots, not just the animation. `sim.py --trace` writes at least 1000 simulated seconds by design, so this is the parameter to shrink if those figures' time axes are showing more than you want to look at. `0` means the whole trace | `30.0` |
| `--stride` | Animation frame subsampling (only affects the `.gif`, not the gait/neuron plots — e.g. `stride=10` with `TS=0.1` gives one animation frame per 1.0s of simulated time) | `10` |
| `--fps` | Animation frames per second (only affects the `.gif`) | `20` |

### Running the Sim-to-Real Transfer Test (Phase C)

`hexapod_bridge.py` loads a genome evolved by `evolve.py` and runs it
against the real MuJoCo hexapod instead of `walker.py`:

```bash
python hexapod_bridge.py --controller feedforward --hidden 16 --genome ff_best.npy --duration 50
python hexapod_bridge.py --controller ctrnn --genome ctrnn_best.npy --duration 50
```

| Option | Description | Default |
|--------|-------------|---------|
| `--controller`, `--genome`, `--hidden`, `--activation`, `--topology`, `--module_size`, `--interneurons`, `--mode` | Same meaning as in `sim.py` — must match how the genome was evolved | see `sim.py` |
| `--torque_scale` | Scale on the mapped **sweep** torque. `walker.py`'s action magnitudes were never tuned against real actuator gains, so this is worth sweeping rather than assuming 1.0 is right | `1.0` |
| `--timescale` | [`--controller ctrnn` only] Scales the CTRNN's own internal clock relative to the real hexapod's physical clock (`dt` passed to `act()` = `TS * timescale`). The genome evolved with its internal clock and `walker.py`'s body advancing in lockstep at `TS=0.1s`/tick, but `hexapod_env.py`'s real physical step is only `0.05s`, so the two clocks don't automatically line up — worth exploring together with `--lift_gain` (see `hexapod_lift_sweep.py`) rather than assuming the default is right | `4.0` |
| `--lift_gain` | Multiplier on the closed-loop PD lift controller's gains (see `hexapod_bridge.py`'s module docstring), independent of `--torque_scale` (which scales sweep only) — worth exploring together with `--timescale` | `2.0` |
| `--duration` | Simulated seconds to run — same convention as `walker.py`/`sim.py`/`evolve.py`'s `--duration`. Not a raw step count: `hexapod_env.py`'s per-step `dt` is 0.05s (MuJoCo `timestep=0.01` × `frame_skip=5`), vs. `walker.py`'s `TS` of 0.1s, so steps actually run = `duration / 0.05` | `50.0` |
| `--render` | Open an interactive MuJoCo viewer window (only works with a display, not headless servers — see `hexapod_render.py` below for headless machines) | off |

Output reports whether the hexapod stayed upright and its total displacement
(straight-line distance from its start position, direction-agnostic — see
`hexapod_bridge.py`'s `run_transfer()` docstring for why raw x-axis progress
isn't a fair metric on the real MuJoCo body, which can drift or yaw off-heading
in ways `walker.py`'s idealized physics never allows). **Don't be surprised —
or discouraged — if this comes back near zero even for a genome that walked
well in `sim.py`.** That gap is the assignment.

To actually *see* it rather than just reading a displacement number, use
`hexapod_render.py` (works on headless machines too — see the note in the
previous section):

```bash
python hexapod_render.py --controller feedforward --hidden 16 --genome ff_best.npy --duration 15 --out hexapod_demo.mp4
```

| Option | Description | Default |
|--------|-------------|---------|
| `--controller`, `--genome`, `--hidden`, `--activation`, `--topology`, `--module_size`, `--interneurons`, `--mode`, `--torque_scale`, `--timescale`, `--lift_gain`, `--seed` | Same meaning as `hexapod_bridge.py` | see above |
| `--duration` | Simulated seconds to render — same convention as `hexapod_bridge.py`'s `--duration` (steps rendered = `duration / 0.05`) | `15.0` |
| `--out` | Output video path (`.mp4` recommended; `.gif` also works) | `hexapod.mp4` |
| `--width`, `--height` | Frame resolution in pixels | `640`, `480` |
| `--fps` | Playback frame rate of the output video | `20` |

**Sweeping `--lift_gain` and `--timescale` together:** both parameters
affect the same thing from different directions. `--lift_gain` sets how
strongly the closed-loop lift controller pulls each leg toward its target
angle; `--timescale` sets how fast the CTRNN's internal clock runs
relative to the real body's physical response. If the lift gain is too
weak, the leg never reaches its target in time regardless of timing; if
the timing is off, even a well-tuned lift controller is asked to reach
its target on the wrong schedule. Because they interact, it's worth
searching over both together rather than one at a time — `torque_scale`
(sweep torque only) is left fixed. `hexapod_lift_sweep.py` runs several
repetitions per (lift_gain, timescale) combination (different seeds, so a
single lucky/unlucky run doesn't skew the result) and plots average
displacement (direction-agnostic distance from start) as a heatmap:

```bash
python hexapod_lift_sweep.py --controller ctrnn --genome ctrnn_best.npy \
    --lift_gains 1.0 1.5 2.0 2.5 3.0 --timescales 1.0 2.0 4.0 8.0 16.0 \
    --reps 5 --out lift_sweep.png
```

| Option | Description | Default |
|--------|-------------|---------|
| `--controller`, `--genome`, `--hidden`, `--activation`, `--topology`, `--module_size`, `--interneurons`, `--mode`, `--torque_scale`, `--duration` | Same meaning as `hexapod_bridge.py` (`--genome` is required here) | see above |
| `--lift_gains` | List of lift_gain values to try | `1.0 1.5 2.0 2.5 3.0` |
| `--timescales` | List of timescale values to try (ignored, held at `1.0`, for `--controller feedforward`) | `1.0 2.0 4.0 8.0 16.0` |
| `--reps` | Repetitions per (lift_gain, timescale) combination, each a different seed | `5` |
| `--seed` | Base seed — repetition `r` uses `seed + r`, the same across every combination, so the comparison is fair | `0` |
| `--out` | Output plot path | `lift_sweep.png` |
| `--data_out` | Optional `.npz` path to also save raw per-cell/per-rep results | `None` |

---

## Understanding Fitness Scores

Because fitness here is a single quantity (average forward speed), it helps
to have a rough sense of what different values look like in practice —
this table applies to either controller, since it's about the physics and
balance constraint, not the architecture:

| Fitness (avg. forward speed) | What's typically happening |
|---|---|
| ~0 | The controller's leg commands never produce a stable 3+ leg tripod, so `vx` gets zeroed out almost every tick — legs may be moving, but the body isn't. This is where an untrained/newly-initialized controller usually starts. |
| Small positive, well under `MAX_VELOCITY` (1.0) | The body is supported and inching forward some of the time, but stance/swing timing across legs isn't well coordinated yet. |
| A meaningful fraction of `MAX_VELOCITY` (1.0), sustained | Legs are alternating stance/swing in a way that keeps the tripod stable most of the time — this is roughly "walking, imperfectly." |
| Close to `MAX_VELOCITY` (1.0), sustained | Efficient, well-coordinated gait — the body is moving forward near its speed cap for most of the episode. |

**A grounded reference point:** a controller that never plants a foot
(`foot` output always ≤ 0.5 for every leg) never has a supported tripod, so
its fitness is exactly 0. A freshly-initialized random network usually
produces uncoordinated foot-down patterns that rarely form a 3-leg tripod, so
it also scores close to 0 — evolution's first job is discovering *any*
leg-timing pattern that keeps the body supported at all, before it can be
shaped into efficient forward progress.

---

## Choosing an Architecture

### Phase A: feedforward layer sizes

`--hidden`: a single hidden layer (`--hidden 64`) or multiple layers of
different sizes (`--hidden 64 64 64`). More/larger layers add
representational capacity, but every added weight is another dimension the
genetic algorithm has to search — genome size grows fast:

| `--hidden` | Genome size |
|---|---|
| `16` | 418 |
| `32` | 818 |
| `64` | 1,618 |
| `128` | 3,218 |
| `32 32` | 1,874 |
| `64 64` | 5,778 |

**Initial weight range:** generation 0's genomes are sampled uniformly
from `(-1, 1)` per weight/bias. With Tanh hidden layers and a Sigmoid
output, a much wider range would push nearly every activation to a
saturated extreme right from the start (outputs pinned near 0 or 1
almost regardless of the input), which leaves very little fitness
variation for selection to act on. `(-1, 1)` keeps initial activations
in their responsive range instead.

**Why feedforward evolution tends to be slower to get going than
CTRNN:** `WalkerController` is a pure feedforward function of the
current observation — it has *no* internal memory whatsoever. A
coordinated tripod gait needs rhythm (legs taking turns being planted
vs. swinging in a stable, repeating pattern), and a memoryless network
can only produce that rhythm by routing it entirely through the body
itself — via the one sensory loop it has (`angle` per leg) as an
implicit oscillator, discovered by trial and error. A CTRNN, by
contrast, has recurrent connections and persistent state
(`self.States`/`self.Outputs`, carried across ticks by
`reset_state()`/`act()`) — random recurrent weight matrices commonly
produce oscillatory dynamics on their own, so a meaningful fraction of
*randomly initialized* CTRNN genomes already have *some* rhythmic
output for evolution to shape, while a feedforward net starts with
none. This is an expected, reportable asymmetry between the two
architectures, and arguably part of the point of comparing them in
Parts 1-2 of the assignment. If you want to narrow the gap rather than
just document it, worth trying: a larger population/more generations
for feedforward specifically, or giving it more than 6 raw angle
values to react to (e.g. augmenting the observation with each leg's
angular velocity, so at least *some* rate-of-change information is
available without requiring the net to infer it from a memoryless
single-tick snapshot).

### Phase B: CTRNN topology and size

`--topology modular` with `--module_size`: genome size is
`sm² + 5·sm` where `sm` is the module size (default 5 → 50 genes,
expanded to 30 neurons total via weight-sharing across the 6 legs).
`--topology fully_connected` with `--interneurons`: genome size is
`n² + 2n + 18` where `n = 18 + interneurons` (default 0 interneurons →
`n=18` → 720 genes — much larger than the modular default, since nothing
is shared across legs).

| `--topology` | Settings | Genome size |
|---|---|---|
| `modular` | `--module_size 3` (default) | 24 |
| `modular` | `--module_size 8` | 104 |
| `fully_connected` | `--interneurons 0` | 720 |
| `fully_connected` | `--interneurons 6` | 1,296 |

A reasonable experiment protocol for either phase: pick 2-3
configurations, run each for the same population size, generation count,
and `--episodes_per_eval` (use `--seed` so runs are comparable), and
compare best fitness reached and how quickly it got there.

**There's no `study.py` in this project (unlike Projects 1–3)** — same
reason as Projects 4 and 5: Parts 2 and 5 below ask you to compare several
configurations (hidden sizes, topologies, modes), and running that kind of
sweep and plotting the result is left as an exercise. Use `--fitness_output`
to save each run's fitness-over-generations curve
to disk instead of (or in addition to) just eyeballing `--vizperf`, so you
can reload several runs afterward and plot them together:

```bash
python evolve.py --controller feedforward --hidden 16 --fitness_output ff16.npz
python evolve.py --controller feedforward --hidden 64 --fitness_output ff64.npz
```

```python
import numpy as np
import matplotlib.pyplot as plt

for label, path in [("hidden=16", "ff16.npz"), ("hidden=64", "ff64.npz")]:
    data = np.load(path)
    plt.plot(data["best"], label=label)
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.legend()
plt.show()
```

---

## Performance

**The bottleneck is the per-tick Euler physics loop, not the neural
network's math.** Each fitness evaluation runs one walker episode
(`duration / 0.1` ticks, e.g. 2,200 ticks for the default 220s duration) per
genome, per episode. The controller itself is tiny by comparison (well under
2,000 parameters at typical settings, one 6-dimensional observation at a
time), so its forward pass is essentially free next to the physics loop.

**What actually helps: evaluating multiple genomes in parallel across CPU
cores.** `evolve.py` uses EvoTorch's `Problem(num_actors=...)` for this. Use
`--workers max` (the default) to use every available core, `--workers N` for
a fixed number, or `--workers none` for a single process (useful for
debugging).

**Also helps a lot for Phase A: not spending the full episode on genomes
that are obviously going nowhere.** Especially in early generations, most
feedforward genomes never get a stable tripod gait going at all and just
sit near `cx=0` for the entire episode — that's wasted compute. `--patience`
(default 10s, feedforward only) ends an episode early once the walker
hasn't advanced `--min_progress` over the trailing `--patience`-second
window, instead of always running the full `--duration`. This is a
sliding-window check, not just a one-time check at the start, so it also
catches a gait that walks for a while and then stalls partway through.
Fitness for an early-terminated episode is still computed as
`cx / duration` using the full nominal `duration` in the denominator (not
however many ticks actually ran) — this treats the rest of the episode as
if the walker had simply stayed put.

Set `--patience 0` to disable this and always run the full `--duration`.
The defaults (`--patience 10 --min_progress 0.5`) are a starting point,
not a validated setting. This mechanism is **not** used for
`--controller ctrnn` — see the CTRNN options table above for why.

**A GPU will not speed this up, and isn't used here.** The controller
network is small (a few thousand parameters at most) and `step()` calls it
once per tick on a single 6-value observation — there's no batch of
observations to hand a GPU. If you want to speed up evolution, more CPU
workers (`--workers`) is the lever that actually matters here, not a GPU.

**A note on warnings/log output when you run `evolve.py`:** with more than
one worker, EvoTorch parallelizes fitness evaluation using
[Ray](https://www.ray.io/), and you may still see a one-time `Started a
local Ray instance` message the first time it spins up its worker pool —
that's an informational status line, not a warning, and is safe to ignore.
A few `FutureWarning`s that Ray/PyTorch print about their own internal
bookkeeping are silenced automatically; none of them are about this
project's code. If you ever want to see everything Ray logs in full
detail, run with `--workers none` to skip Ray entirely, or set
`RAY_DEDUP_LOGS=0` in your environment before running.

---

## Understanding the Components

### Controllers (`neural_controller.py`)

`WalkerController` (feedforward):
- **Input layer**: 6 neurons (one per leg's angle sensor)
- **Hidden layer(s)**: one or more layers, each with a configurable number of
  neurons and a configurable activation (`tanh`/`relu`/`sigmoid`, default
  `tanh` — see `--activation`). Pass a single int (`hidden_sizes=64`) for
  one hidden layer, or a sequence (`hidden_sizes=(64, 64, 64)`) for a
  deeper network.
- **Output layer**: 18 neurons (forward/backward/foot per leg) with Sigmoid
  activation, so actions come out in [0, 1].
- The genome is a flat vector containing all weights and biases in
  PyTorch's parameter order. For `hidden_sizes=64` with the 6-observation
  input: Input→Hidden is 6 × 64 = 384 weights + 64 biases; Hidden→Output is
  64 × 18 = 1,152 weights + 18 biases — **1,618 parameters total**.

`WalkerCTRNNController` (CTRNN):
- Each neuron integrates its inputs (weighted recurrent connections from
  every other neuron, plus optional sensory input) over time via a
  per-neuron time constant, and its output is a sigmoid of its internal
  state plus bias — this internal state is what persists tick to tick.
- `reset_state()` must be called once per episode (randomizes starting
  neuron states); `act(observation, sensors_on=...)` advances the circuit
  by one tick and returns the 18 command-neuron outputs.
- The genome layout differs by topology — see `neural_controller.py`'s
  module docstring for the exact `[W | WS | CBW | ISW | T | B]` /
  `[W | WS | T | B]` layouts.

### Environment (`walker.py`)

`WalkerEnv` implements the walker's per-tick force → velocity →
balance-check physics, through a `reset()`/`step()` interface either
controller can drive identically.

### Evolution (`evolve.py`)

The `run_evolution()` function:
1. Creates a fitness function that evaluates a controller (feedforward or
   CTRNN, per `--controller`) over multiple episodes
2. Sets up EvoTorch's `GeneticAlgorithm` with SBX crossover and Gaussian
   mutation
3. Runs for the specified number of generations, optionally in parallel
   across CPU cores
4. Returns fitness trajectories and the best genome

### Simulation & Visualization (`sim.py`, `visualize_walker.py`)

`sim.py` loads a controller (evolved or random, feedforward or CTRNN) and
runs it for one or more repetitions, with three optional outputs:
- Scoring (`--scores`): average forward speed across repetitions
- Trajectory plots (`--viztraces`): body x-position over time
- Trace files (`--trace`): per-tick data for `visualize_walker.py`

`visualize_walker.py` turns a trace file into a gait diagram (stance/swing
per leg over time, with body speed underneath), a grid of the actuator or
neuron outputs over time, and a top-down GIF animation of the legs and
body moving — unmodified between the two controller types, since it infers
neuron count from the trace file's column count rather than assuming it.

---

## Tips

- Start with smaller networks/circuits (Phase A: 16-32 hidden neurons;
  Phase B: default `modular` topology) and fewer generations (40-75) to
  test your setup before running large experiments
- Use `--seed` for reproducibility when debugging or comparing configurations
- Visualize fitness curves (`--vizperf`) to monitor evolutionary progress
- Use `--workers max` (the default) so you're not waiting on a single CPU core
- Look at gait diagrams and animations (`sim.py --trace` +
  `visualize_walker.py`) early and often — a fitness number alone won't tell
  you *why* a controller is or isn't walking well, but a stance/swing diagram
  usually will
- For Phase B, try `--mode cpg` on an already-evolved `rpg` genome (or vice
  versa) — comparing the same genome's gait diagram under both sensor
  conditions is a quick, concrete way to see how much the evolved circuit
  actually depends on live feedback vs. its own internal rhythm

---

## Assignment

### Part 1 – Understand the Controllers

Answer these questions before running any experiments:

**Feedforward:**
- How many total parameters (weights + biases) does a network with a single
  32-neuron hidden layer have? Count them by layer. What about two hidden
  layers of 32 neurons each?
- Why is Sigmoid used for the output layer here, rather than Tanh?
- What would happen if we used Tanh instead of Sigmoid for the output layer,
  without changing anything else about how actions are interpreted (recall
  `foot > 0.5` means stance)?
- Looking at how fitness (average forward speed) and the balance constraint
  interact, what kind of controller would maximize "legs never leave the
  ground" behavior without necessarily producing a good walking gait?

**CTRNN:**
- What does a CTRNN have that a feedforward network doesn't, and what
  behavior does that make possible in principle that a feedforward network
  cannot produce, no matter how it's evolved?
- Concretely, why *can't* a feedforward controller run in `cpg` mode
  (zero sensory input) and still produce a moving gait, while a CTRNN can?
- What is `reset_state()` for, and why does a feedforward controller not
  need an equivalent step?

### Part 2 – Evolve and Analyze

**Phase A (feedforward):**
1. Run evolution with a few different `--hidden` configurations (e.g. a
   single layer of 16, 64, and 128, and at least one multi-layer network
   like `64 64`)
2. Compare final fitness values, and how many generations it took to reach
   them

**Phase B (CTRNN):**
1. Run evolution with the default `modular` topology, then with
   `fully_connected` (pick a comparable-effort configuration, e.g. a few
   interneurons)
2. Run at least one configuration in each of `cpg`, `rpg`, and `mpg` mode
3. Compare final fitness values and convergence speed across topology and
   mode

Questions to consider (answer for both phases, and compare across them):
- Does a larger network/circuit always achieve higher fitness?
- Is there a point of diminishing returns as you add capacity?
- What is the relationship between parameter count and convergence speed?
- Does the CTRNN's `cpg` mode reach meaningfully lower, similar, or (surprisingly) higher fitness than `rpg`? What does that tell you about how much this task actually needs live sensory feedback?

### Part 3 – Behavioral Analysis

1. Run simulations with your best evolved controller from each phase
2. Visualize body x-position over time (`--viztraces`) for both
3. Generate at least one gait diagram and animation (`--trace` +
   `visualize_walker.py`) for your best feedforward controller and your
   best CTRNN controller
4. For your best CTRNN controller, also generate a gait diagram running
   under the *other* sensory mode than it was evolved with (e.g. an
   `rpg`-evolved genome run with `sim.py --mode cpg`) and compare
5. Analyze walking patterns and stability

Questions:
- Does the body walk steadily forward, or does progress stall out repeatedly?
- What leg-timing pattern (tripod gait, wave gait, something else) does each
  gait diagram show?
- What strategies emerge from evolution, and do they differ between the
  feedforward and CTRNN controllers?
- Using the fitness ranges described above, where do your best controllers
  land, and does their behavior match what that range predicts?
- How does your CTRNN controller's gait change (if at all) when run under
  the sensory mode it *wasn't* evolved for?

### Part 4 – Sim-to-Real Transfer Test (Optional / Advanced)

Take your best feedforward controller and your best CTRNN controller from
Part 3, and run each through `hexapod_bridge.py` against the real MuJoCo
hexapod.

1. Run both controllers against `hexapod_env.py`. Report whether each
   stays upright, and its total displacement from its start position.
2. Sweep `--lift_gain` and `--timescale` together with `hexapod_lift_sweep.py`
   (try at least 3-4 values of each) for your better-transferring
   controller. Is there a clear sweet spot, or a broad region that works
   about equally well? Does one parameter matter more than the other?
3. Generate a video with `hexapod_render.py` of your best-transferring
   run.

Questions:
- What, concretely, does `walker.py`'s physics leave out that
  `hexapod_env.py` actually has (contact forces, friction, inertia,
  torque limits)? Which of these do you think matters most for why a
  gait would or wouldn't transfer?
- `hexapod.xml`'s legs were deliberately designed to match `walker.py`'s
  one-angle-per-leg abstraction (a single sweep+lift pivot, not a
  hip+knee chain), specifically so the sweep half of the bridge is a
  direct correspondence rather than an invented mapping. Given that, why
  does the transfer still fail (or at least not fully succeed)? What does
  that tell you about how much of the gap is really about leg *geometry*
  versus how much is genuinely about idealized-vs-real *physics*?
- The `lift` torque (from `foot`) is still an interpretation, unlike
  `sweep`'s direct correspondence — even a closed-loop controller reading
  real sensor state is still translating a signal `walker.py` itself
  never treated as a torque. Does that asymmetry show up in what you
  observe — e.g. does the walker's sweep timing look intact while its
  stance/swing footing looks off?
- This is real sim-to-(more-)real transfer, one level removed from actual
  hardware. What would the next gap be, going from `hexapod_env.py` to an
  actual physical hexapod robot?

### Part 5 – Quantitative Analysis

Collect data from multiple independent runs:

1. Run evolution with the same parameters 5 times for **one** configuration
   of your choice from each phase (use `--seed` to keep other sources of
   randomness controlled while varying the run)
2. Record best fitness, convergence generation, and genome size
3. Analyze variance across runs, and compare variance between the
   feedforward and CTRNN configurations you chose

Questions:
- Does evolution always find high-fitness solutions?
- How does population size affect success rate?
- What is the typical convergence pattern (early rapid progress vs. late
  refinement), and does it look different for the CTRNN than the
  feedforward controller?

---

## Optional / Advanced Challenge

Parts 1, 2, 3, and 5 are required. Part 4 above (Sim-to-Real Transfer Test) is already this project's most involved optional direction, worth up to 1 point of extra credit. Beyond that — instead of, or in addition to, Part 4 — here are three more optional directions that don't carry Part 4's explicit extra-credit point but are exactly the kind of "go one step further" work the Creativity & Critical Thinking rubric category rewards. Each is open-ended: form a hypothesis, run the experiment, and report what you found. Pick **one**; go as deep as you like on it.

**1. CTRNN topology at matched genome size, not matched neuron count.** Part 2 compares `--topology modular` against `--topology fully_connected` at whatever genome size each naturally lands on (24 genes vs. 720+ by default) — a very different-sized search space either way. Pick `--interneurons` for `fully_connected` so its genome size roughly matches a `modular` configuration's, and re-run the topology comparison under that matched-size condition. Hypothesis: does weight-sharing (modular) still win once you've controlled for genome size, or was its earlier advantage really just "smaller search space," not "better structure"?

**2. Damaged-leg robustness.** Take your best evolved controller from Part 3 and, without retraining, force one leg's `foot` output to always read as "swing" (clamp that leg's action before it reaches `WalkerEnv.step()`) — simulating a leg that never plants. Hypothesis: does a CTRNN, with its own internal rhythm that could in principle reorganize around the loss, degrade more gracefully than a feedforward controller — whose output is a fixed function of the current observation with no equivalent to "adapt" — when one leg stops contributing?

**3. CPG frequency and gait diversity across seeds.** Evolve several CTRNN genomes in `cpg` mode from different seeds, and for each one, measure the oscillation period of a representative command neuron — time between successive peaks in its output trace, readable from a `sim.py --trace` file's per-neuron columns. Hypothesis: does evolution reliably converge to one canonical gait frequency for this body, or are there multiple qualitatively different rhythmic solutions, at different frequencies, that are all roughly equally good?

You're encouraged to explore your own idea beyond these as well, as long as it's a genuine extension (not just a parameter change already covered in Parts 2, 3, or 5).

---

## What to Submit to Moodle

Submit a single **written report as a PDF** to Moodle.

### Title Page

The first page of your report should include:

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 6: Embodied NeuroEvolution III — Six-Legged Walker)
- Date submitted
- Amount of time spent on this project
- A self-assessment of your confidence in your understanding of the
  concepts, the code, and the insights gained from this project (a number
  between 1 and 10)

### Report Body

Organize the body of your report into one section per assignment part. Each
section should combine the relevant figures with a written discussion — a
plot with no interpretation, or an interpretation with no supporting plot,
is incomplete.

**Part 1 — Understand the Controllers**

- Your answers to the conceptual questions posed in Part 1, for both the
  feedforward network and the CTRNN.

**Part 2 — Evolve and Analyze**

- Fitness-over-generations plots (`--vizperf`) for the `--hidden`
  configurations you compared in Phase A, and for the topology/mode
  configurations you compared in Phase B.
- A summary plot or table comparing final best fitness across all the
  configurations you tried, across both phases.
- Answers to the guiding questions from Part 2, supported directly by your
  results.

**Part 3 — Behavioral Analysis**

- Trajectory plots (`--viztraces`) and at least one gait diagram/animation
  (from `visualize_walker.py`) for your best feedforward controller and
  your best CTRNN controller, plus the cross-mode gait diagram described
  above.
- A discussion of walking patterns, stability, and any strategies you
  observed, including how the two architectures compare.
- Answers to the guiding questions from Part 3.

**Part 4 — Sim-to-Real Transfer Test (Optional / Advanced)**

- Results (upright/fell, final displacement) for your best feedforward
  and CTRNN controllers run against the real hexapod, both before and
  after your `--lift_gain`/`--timescale` sweep.
- The video generated with `hexapod_render.py`.
- Answers to the guiding questions from Part 4.

**Part 5 — Quantitative Analysis**

- Results from 5 independent evolutionary runs for one feedforward
  configuration and one CTRNN configuration: best fitness, convergence
  generation, and genome size for each.
- A plot or discussion of the variance across runs, and how it compares
  between the two configurations.
- Answers to the guiding questions from Part 5.

**Optional / Advanced Challenge** *(if attempted, beyond Part 4)*: a section naming which of the three directions you chose (or your own idea), what you changed, your results (with supporting figures), and your interpretation. Omit this section if you didn't attempt one.

### Reminder of General Guidelines

- Figures should have readable axis labels, legends, and captions.
- Reference and discuss every figure in the text — don't paste a plot
  without commentary.
- Be concise: prioritize insight over volume. A focused paragraph beats a
  page of restated code output.

---

## Rubric

This project is worth **10 points**, broken down as follows:

### Assignment Completion (5 pts)

Parts 1, 2, 3, and 5 (see *Assignment* above) are weighted roughly
equally and together make up this score. Credit is based on whether the
part was genuinely completed — code implemented and working for both
phases, questions answered with reasoning, experiments actually run — not
just attempted. **Part 4 (sim-to-real transfer test) is optional and
extra credit** — up to 1 additional point, on top of the 10-point total,
for a genuine attempt at all three of its steps.

### Report Quality (5 pts)

- **Title page (1 pt)** — includes all required information: name, course
  title, assignment name, date submitted, time spent, and self-assessment
  (1–10).
- **Figures (2 pts)** — figures are easy to read, meaningful (they show what
  the text claims), properly labeled (axes, legend, caption), and each is
  paired with an interpretation in the text. A plot with no discussion, or
  discussion with no supporting plot, does not receive full credit.
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of
  open-ended reasoning, and evidence of genuine exploration beyond the
  minimum required to answer each question — especially in comparing the
  two architectures rather than treating them as two disconnected exercises.

---

## Further Reading

- Floreano, D., Dürr, P., & Mattiussi, C. (2008). *Neuroevolution: from
  architectures to learning.* Evolutionary Intelligence.
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks
  through Augmenting Topologies.* Evolutionary Computation.
- Beer, R. D., & Gallagher, J. C. (1992). *Evolving dynamical neural
  networks for adaptive behavior.* Adaptive Behavior. (the tripod-gait
  insect body plan this project's walker is modeled on, and the origin of
  the CPG/RPG/MPG sensory-mode framing used in Phase B)

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary
Robotics** at Rose-Hulman Institute of Technology.
