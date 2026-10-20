# Project 6: Embodied NeuroEvolution III — Six-Legged Walker

## Overview

Every brain you've evolved so far has been a **function**: give it an input, it gives you an output, and it remembers nothing in between. When those robots needed rhythm (the soft robots in Project 5), we handed it to them as a clock input.

In this project the brain makes its own rhythm. You'll evolve a **continuous-time recurrent neural network (CTRNN)**: a network whose neurons have internal state that changes continuously in time, and whose connections can loop back on themselves. A CTRNN is not a function, it's a *dynamical system*. With no input at all, it can settle down, switch, or oscillate, the way the central pattern generators in an insect's nervous system produce a walking rhythm on their own.

The body is a six-legged insect, inspired by Beer & Gallagher (1992). It only moves forward while the legs that are down form a stable tripod, so before evolution can make it walk *well*, it has to discover *any* coordinated stepping pattern at all.

The pipeline is the one you know: a flat genome, EvoTorch, evolve → simulate → analyze. What's new is the controller. Unit 6 (`PythonBasics/Unit6`) introduces the CTRNN on its own, with no body attached; start there.

This is the **most open-ended project of the term**. There are only two required parts: evolve a CTRNN that walks, and then design and run an experiment of your own. Optionally, try to get your walker to walk in a far more realistic physics simulator (MuJoCo).

## Checkpoint, Due Date, and Presentations

- **Checkpoint — TBA, during class.** Covers Required Part 1. No reporting necessary for the checkpoint; it will involve running code on your laptop to demonstrate, and showing figures and animations.
- **Report due — TBA.** Includes Required Part 2.
- **Presentations — TBA, during class.**

---

## Learning Objectives

By completing this project, you will learn how to:

1. Describe a CTRNN as a dynamical system: what its states, biases, time constants, and recurrent weights each do, and why it can produce behavior (like rhythm) that a feedforward network cannot.

2. Evolve a CTRNN to control a multi-legged body, and analyze *how* the evolved circuit produces a gait, not just how fast it goes.

3. Design, run, and interpret an experiment of your own: a question, a prediction, a fair comparison, and enough runs to trust the answer.

4. (Optional) Reason about the gap between an idealized simulation and a more realistic one, and what it takes for an evolved behavior to cross it.

---

## IMPORTANT NOTE

From Project 1 to now, each project has given you a little less to follow and a little more to decide. This one gives you the least. There is no "traditional path" of numbered questions: the required parts below say *what* to achieve, and almost nothing about *how*. Everything in this README beyond the Assignment section is reference material (how the code works, what the options are), not a list of things you're expected to do.

---

## Background

### From functions to dynamical systems: the CTRNN

Each neuron `i` in a CTRNN has a **state** `y_i` that changes continuously in time:

```
tau_i * dy_i/dt = -y_i + sum_j w_ji * o_j + I_i
o_i = sigmoid(y_i + theta_i)
```

| Symbol | Name | What it does |
|---|---|---|
| `y_i` | state | the neuron's internal "membrane potential"; it's what persists over time |
| `o_i` | output | the neuron's "firing rate", always between 0 and 1 |
| `tau_i` | time constant | how fast the neuron responds; bigger is slower |
| `theta_i` | bias | shifts where on its input range the neuron turns on |
| `w_ji` | weight | connection from neuron `j` to neuron `i`; any neuron can connect to any other, including itself |
| `I_i` | external input | e.g. a sensor reading |

Without inputs, the `-y_i` term pulls each state back toward zero, the recurrent connections push it around, and the time constants set how fast all of this happens. Depending on the parameters, a circuit can settle to a fixed point, switch between states, or oscillate forever. To simulate it, we take small Euler steps of size `dt`: `y <- y + dt * dy/dt`.

Most randomly chosen CTRNNs aren't very interesting: with large weights and biases, most neurons end up stuck fully on or fully off. A **center-crossing** circuit (Mathayomchan & Beer, 2002) sets each bias to `theta_i = -sum_j w_ji / 2`, which puts the middle of each neuron's input range right on the steepest part of its sigmoid. Oscillations become much more common: in Unit 6's `exp.py`, about 1–4% of random circuits oscillate, against roughly 10–40% of center-crossing ones (more for bigger circuits).

**Unit 6** (`PythonBasics/Unit6`) has a short, plain NumPy CTRNN (`ctrnn.py`), a script to simulate one and look at it (`sim.py`), and the random vs. center-crossing experiment (`exp.py`). The CTRNN in this project (`WalkerCTRNNController` in `neural_controller.py`) uses exactly the same equations; it just adds sensory inputs, wiring to the body, and a way to build it from a genome.

### The six-legged walker (`walker.py`)

`WalkerEnv` is an idealized, top-down insect body: six legs in two rows of three, each pivoting at one point and swinging forward and backward between angle limits.

- **Observation (6 values):** each leg's current angle.
- **Action (18 values in [0, 1]):** `(forward, backward, foot)` for each leg. `forward` and `backward` drive that leg's swing force; `foot > 0.5` means the foot is planted (stance), otherwise it's swinging.
- **Physics:** a planted leg pushes the body; a swinging leg moves freely. Every tick, the body only moves if the planted feet form a stable support under the body's center (at least three feet down, on both sides). Otherwise its forward velocity is set to zero.
- **Fitness:** average forward speed over the episode, `cx / duration`. The speed is capped at `MAX_VELOCITY = 1.0`.
- **Episodes:** each starts with random leg angles and runs for `--duration` simulated seconds (default 220), in ticks of `TS = 0.1` s.

### The CTRNN controller (`neural_controller.py`)

`WalkerCTRNNController` has 18 **command neurons**: a forward, backward, and foot neuron for each leg. Their outputs are the 18 actions. Each command neuron also receives its own leg's angle as sensory input, through an evolved sensory weight.

Two topologies (`--topology`):

- **`modular`** (default): one small circuit of `--module_size` neurons (at least 3: forward, backward, foot) is copied onto every leg, sharing the same weights. A few fixed connections link neighboring legs: across the body (`CBW`, contralateral) and along each side (`ISW`, ipsilateral). The genome is tiny: 24 genes at the default module size of 3, for an 18-neuron network.
- **`fully_connected`**: one network of `18 + --interneurons` neurons, every one connected to every other with its own weight. No sharing: 378 genes with no interneurons.

| `--topology` | Settings | Neurons | Genome size |
|---|---|---|---|
| `modular` | `--module_size 3` (default) | 18 | 24 |
| `modular` | `--module_size 5` | 30 | 50 |
| `modular` | `--module_size 8` | 48 | 104 |
| `fully_connected` | `--interneurons 0` | 18 | 378 |
| `fully_connected` | `--interneurons 6` | 24 | 642 |

Every gene is a number in [0, 1], decoded into weights in [-16, 16], biases in [-16, 16], and time constants in [1, 20]. `reset_state()` randomizes the neuron states once at the start of each episode; `act()` then advances the circuit by one Euler step (`dt = TS`) per tick of the body.

Three sensory modes (`--mode`):

- **`rpg`** (reflexive pattern generator, default): sensors on. The rhythm can come from the circuit, from the body (through feedback), or from both.
- **`cpg`** (central pattern generator): sensors off. The rhythm has to come entirely from the circuit's own dynamics.
- **`mpg`** (mixed): each genome's evaluation episodes alternate between sensors off and sensors on, and fitness is the average, so evolution has to find a circuit that walks either way. (In `sim.py`, `--mode mpg` alternates between the two across `--reps`; to see one condition, use `cpg` or `rpg` explicitly.)

`--center_crossing` seeds evolution's starting population with center-crossing biases (the same idea as in Unit 6, extended to include the sensory and inter-leg weights). Evolution is free to move the biases away from there afterward.

A **feedforward** controller (`--controller feedforward`, with `--hidden` and `--activation` as in earlier projects) is also available, in case you want something to compare against.

### A more realistic body: MuJoCo (optional)

`walker.py`'s physics is idealized: no real contact forces, friction, mass, or inertia, and a hand-written stability check. `hexapod.xml` is a MuJoCo model of a six-legged body with all of those. Each leg pivots at a single point with two joints, **sweep** (forward/backward, matching `walker.py`'s one leg angle) and **lift** (up/down), so the leg geometry matches as closely as possible and physics is the main thing that changes.

`hexapod_env.py` wraps it as a Gymnasium environment, and `hexapod_bridge.py` translates between the two worlds: each leg's sweep torque is `forward - backward`, each leg's sweep angle becomes the CTRNN's sensory input, and `foot` becomes a target lift angle held by a PD controller (down while planted, up while swinging). The hexapod's physics step is 0.05 s rather than 0.1 s, and its legs respond differently, so `--timescale` (how fast the CTRNN runs relative to the body) and `--lift_gain` (how hard the lift controller pulls) both matter, and interact.

Don't be surprised if a genome that walks well in `walker.py` goes nowhere in MuJoCo. That gap is the challenge.

---

## Project Structure

| File | Purpose |
|------|---------|
| `neural_controller.py` | The controllers: `WalkerCTRNNController` (CTRNN, modular or fully connected) and `WalkerController` (feedforward), plus the center-crossing seeding |
| `walker.py` | The idealized six-legged body (`WalkerEnv`) |
| `evolve.py` | Evolves a controller with EvoTorch (GA or CMA-ES), in parallel across CPU cores |
| `sim.py` | Runs a saved genome: scores, trajectories, and trace files for `visualize_walker.py` |
| `visualize_walker.py` | Turns a trace file into a gait diagram, neuron-output plots, and a top-down animation |
| `hexapod.xml` | [Optional] MuJoCo model of the six-legged body |
| `hexapod_env.py` | [Optional] Gymnasium `MujocoEnv` wrapper around `hexapod.xml` |
| `hexapod_bridge.py` | [Optional] Runs a `walker.py`-evolved genome on the MuJoCo hexapod |
| `hexapod_render.py` | [Optional] Renders a MuJoCo run to a video, on any machine (no display needed) |
| `hexapod_lift_sweep.py` | [Optional] Sweeps `--lift_gain` and `--timescale` together and plots displacement as a heatmap |
| `requirements.txt` | Pinned dependency versions |

---

## Installation

The project requires Python 3.9 or newer, NumPy, Matplotlib, PyTorch, EvoTorch, and Pillow, plus Gymnasium and MuJoCo for the optional part. All are listed with tested versions in `requirements.txt`. As in earlier projects, install them into a fresh virtual environment from inside this folder:

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Activate the environment again (`source venv/bin/activate`) in every new terminal. `conda` works too: create an environment with a matching Python version and run the same `pip install -r requirements.txt` inside it.

---

## Running Evolution

```bash
python evolve.py --controller ctrnn --verbose --vizperf --output best.npy --fitness_output best_fit.npz
```

Start small to get a feel for how long things take on your machine:

```bash
python evolve.py --controller ctrnn --popsize 20 --gens 20 --verbose --vizperf
```

**CTRNN options**

| Option | Description | Default |
|--------|-------------|---------|
| `--topology` | `modular` or `fully_connected` | `modular` |
| `--module_size` | Neurons per leg module (`modular` only; at least 3) | `3` |
| `--interneurons` | Extra neurons beyond the 18 command neurons (`fully_connected` only) | `0` |
| `--mode` | `rpg` (sensors on), `cpg` (sensors off), or `mpg` (both, averaged) | `rpg` |
| `--center_crossing` | Seed the starting population with center-crossing biases | off |

**Evolution options**

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithm` | `ga` (genetic algorithm: SBX crossover + Gaussian mutation) or `es` (CMA-ES, which adapts its own step size and search directions) | `ga` |
| `--popsize` | Population size | `50` |
| `--gens` | Number of generations | `100` |
| `--episodes_per_eval` | Episodes averaged per fitness evaluation (each starts from different random leg angles) | `3` |
| `--duration` | Simulated seconds per episode | `220.0` |
| `--mut_stdev` | GA: mutation standard deviation. ES: initial step size | `0.05` |
| `--tournament_size`, `--eta`, `--no-elitism` | GA only: tournament size, SBX distribution index, disable elitism | `3`, `20`, elitism on |
| `--seed_genome FILE` | Start from copies of a saved genome (perturbed by `--seed_noise`) instead of from scratch | `None` |
| `--seed_noise` | Perturbation applied to `--seed_genome` copies | `0.05` |
| `--workers` | Parallel workers: `max`, an integer, or `none` | `max` |
| `--seed` | Random seed, for reproducibility | `None` |
| `--verbose`, `--vizperf` | Print per-generation stats; plot fitness over generations | off |
| `--output FILE` | Save the best genome (`.npy`) | `None` |
| `--fitness_output FILE` | Save best/average/worst fitness per generation (`.npz`), to plot several runs together later | `None` |

Feedforward options (`--controller feedforward`): `--hidden` (one or more layer sizes, default `64`), `--activation` (`tanh`, `relu`, `sigmoid`), and `--patience`/`--min_progress` (end an episode early when the walker stops making progress).

There's no `study.py`: as in Projects 4 and 5, writing the code that runs a set of configurations and plots them together is part of the job. `--fitness_output` makes it easy:

```python
import numpy as np
import matplotlib.pyplot as plt

for label, path in [("rpg", "rpg_fit.npz"), ("cpg", "cpg_fit.npz")]:
    plt.plot(np.load(path)["best"], label=label)
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.legend()
plt.show()
```

---

## Running Simulations

`sim.py` loads a saved genome and runs it. Pass the same controller settings you evolved with (`--controller`, `--topology`, `--module_size`, `--interneurons`, or `--hidden`); `sim.py` will tell you if the genome's size doesn't match.

```bash
python sim.py --controller ctrnn --genome best.npy --scores              # average speed over 5 episodes
python sim.py --controller ctrnn --genome best.npy --viztraces --reps 10 # body position over time
python sim.py --controller ctrnn --genome best.npy --mode cpg --scores   # the same genome with its sensors off
```

| Option | Description | Default |
|--------|-------------|---------|
| `--controller`, `--topology`, `--module_size`, `--interneurons`, `--hidden`, `--activation` | Must match how the genome was evolved | as in `evolve.py` |
| `--genome FILE` | Saved genome (`.npy`). Omit for a random controller | `None` |
| `--mode` | Sensory condition to run under (CTRNN only) | `rpg` |
| `--duration` | Simulated seconds per episode | `220.0` |
| `--reps` | Number of episodes | `5` |
| `--scores` | Print the average forward speed | off |
| `--viztraces` | Plot body position over time | off |
| `--trace FILE` | Write one episode's per-tick data for `visualize_walker.py` | `None` |
| `--seed` | Random seed | `None` |

**Look at what your walker does, not just its fitness.** Write a trace, then turn it into a gait diagram (which legs are down when), a plot of every neuron's output, and an animation:

```bash
python sim.py --controller ctrnn --genome best.npy --trace walk.dat
python visualize_walker.py walk.dat --out_prefix walk --topology modular --module_size 3
```

This produces `walk_gait.png`, `walk_neurons.png`, and `walk_animation.gif`. `visualize_walker.py` options: `--topology`/`--module_size` (must match the genome; they only affect how `_neurons.png` groups neurons), `--max_time` (seconds shown, default 30; `0` for all), `--stride` and `--fps` (animation only).

---

## Understanding Fitness Scores

| Fitness (average forward speed) | What's typically happening |
|---|---|
| ~0 | The planted feet never form a stable tripod, so the body never moves. Where most random controllers start. |
| A few hundredths | Often **not walking**. A common early trick: plant all six feet and push, gliding forward until the feet are dragged too far behind and the body gets stuck for good. Check the gait diagram. |
| A meaningful fraction of 1.0, sustained | Legs alternate between stance and swing, keeping a tripod under the body most of the time: walking, imperfectly. |
| Close to 1.0, sustained | An efficient, well-coordinated gait near the speed cap. |

Evolution can get stuck in the "plant everything and push" trick: it's easy to find and better than standing still, but it's a dead end. If your fitness plateaus at a few hundredths, look at the gait diagram before you trust it, and try a different seed, more generations, center-crossing seeding, or a different mode.

---

## Performance

The cost is the physics loop: each fitness evaluation runs `--episodes_per_eval` episodes of `duration / 0.1` ticks (2,200 ticks at the default 220 s). The CTRNN itself is tiny by comparison.

`evolve.py` evaluates the population in parallel across CPU cores (via Ray). With the defaults (50 × 100 generations, 3 episodes each), a run takes about **3 minutes on a 16-core laptop**; expect longer on fewer cores. A GPU won't help. Running several evolutions at once just splits the same cores between them, so each takes proportionally longer. Use `--workers none` to skip Ray entirely (slower, but easiest for debugging).

You may see a `Started a local Ray instance` message, and a few `FutureWarning`s from Ray or PyTorch about their own internals; they're safe to ignore.

---

## Running on MuJoCo (Optional)

`hexapod_bridge.py` runs a genome evolved on `walker.py` on the MuJoCo hexapod, and reports whether it stayed upright and how far it got from its start (straight-line displacement, in any direction):

```bash
python hexapod_bridge.py --controller ctrnn --genome best.npy --duration 50
python hexapod_bridge.py --controller ctrnn --genome best.npy --duration 50 --render   # live 3D viewer (needs a display)
python hexapod_render.py --controller ctrnn --genome best.npy --duration 15 --out hexapod.mp4   # save a video
```

| Option | Description | Default |
|--------|-------------|---------|
| `--controller`, `--genome`, `--topology`, `--module_size`, `--interneurons`, `--mode`, ... | Same as `sim.py`; must match the genome | as in `sim.py` |
| `--timescale` | CTRNN only. How fast the CTRNN's clock runs relative to the hexapod's (`dt = TS * timescale` per 0.05 s physics step) | `4.0` |
| `--lift_gain` | Strength of the PD controller that raises and lowers each leg | `2.0` |
| `--torque_scale` | Scale on the sweep torque | `1.0` |
| `--duration` | Simulated seconds (MuJoCo steps are 0.05 s) | `50.0` |

`hexapod_render.py` takes the same options plus `--out`, `--width`, `--height`, and `--fps`. `hexapod_lift_sweep.py` sweeps `--lift_gain` and `--timescale` together, several seeds per combination, and plots the average displacement as a heatmap:

```bash
python hexapod_lift_sweep.py --controller ctrnn --genome best.npy \
    --lift_gains 1.0 1.5 2.0 2.5 3.0 --timescales 1.0 2.0 4.0 8.0 16.0 --reps 5 --out lift_sweep.png
```

---

## Tips

- Do Unit 6 first. Ten minutes with `sim.py` there will make everything in `neural_controller.py` easier to read.
- Look at gait diagrams and animations early and often. A fitness number tells you how far, not how.
- Evolution is stochastic: one run is an anecdote. Use `--seed` to make runs reproducible, and several seeds before you trust a conclusion.
- Save `--output` and `--fitness_output` for every run you might want later.
- Write down your prediction before you run an experiment.

---

## Assignment

### REQUIRED #1: Evolve a CTRNN that walks

Evolve a CTRNN that makes the walker in `walker.py` walk: a real stepping gait, with legs taking turns between stance and swing.

Show that it walks (fitness over generations, a gait diagram, an animation), and explain *how* it walks. What gait did evolution find? What are the neurons doing? Where does its rhythm come from: the circuit itself, the body through the sensors, or both? Any settings you like; say what you chose and why.

---

### REQUIRED #2: An experiment of your own

Think of something interesting to find out in this world, and find it out. State your question and your prediction, decide what to vary and what to hold fixed, run enough seeds to trust the answer, and explain what you learned, whether or not it matched your prediction.

Some directions, only to get you started:

- **Where does the rhythm come from?** `cpg` vs. `rpg` vs. `mpg`. What happens when you cut the sensors of a walker that evolved with them, or connect them to one that evolved without?
- **Inside the circuit.** Remove (lesion) neurons or connections from an evolved walker, or freeze a neuron's output, and see what breaks. Does an isolated leg module oscillate on its own?
- **Center-crossing.** Unit 6 shows it makes random circuits rhythmic more often. Does it make evolution find walkers faster, or better ones?
- **Structure.** Modular vs. fully connected, module size, interneurons. Does sharing weights across legs help because of the structure, or just because the genome is smaller?
- **Time.** What do evolved walkers do with their time constants? What happens to the gait if you make every neuron faster or slower after evolution?
- **Robustness.** A leg that never plants, noisy sensors, a body pushed mid-episode, episodes much longer than the ones it evolved on. Does the walker recover?
- **Diversity.** Do different seeds find the same gait at the same frequency, or several different solutions?
- **Fitness.** Change what's being rewarded: penalize effort, reward smoothness, or make the "plant everything and push" trick impossible. What gaits appear?
- **Search.** GA vs. CMA-ES on this problem, or a population-size or mutation-rate sweep.
- **Memoryless vs. dynamical.** Evolve a feedforward controller on the same task. What can it do and not do?

Pick one of these, combine several, or come up with something completely different.

---

### OPTIONAL ADVANCED CHALLENGE: Walk in MuJoCo

Get an evolved walker to walk on the MuJoCo hexapod. Take a genome evolved on `walker.py` and try it there with `hexapod_bridge.py`: what happens, and why? Then try to close the gap, any way you like. Some options: tune `--timescale` and `--lift_gain` (`hexapod_lift_sweep.py`); evolve on `walker.py` in ways that might make a walker more robust (noise, varied conditions); change the bridge; or evolve directly in MuJoCo. Show a video of your best result.

A genuine attempt is worth up to **1 point of extra credit**, on top of the 10 points for the project, whether or not you get it fully walking: what counts is a careful attempt and a clear explanation of what you tried and what you learned.

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
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body of your report into one section per part. Each section should combine the relevant figures with a written discussion — a plot with no interpretation, or an interpretation with no supporting plot, is incomplete. Animations can't go in a PDF, so include static frames and a link to the animations themselves.

**Required Part 1 — Evolve a CTRNN that walks**

- The settings you used, and why.
- Evidence that it walks: fitness over generations, a gait diagram, and frames from (and a link to) an animation.
- Your explanation of how it walks.

**Required Part 2 — Your experiment**

- Your question, your prediction, and why it's interesting.
- What you varied and what you held fixed, and results against a baseline, with enough seeds to support your conclusion.
- Most importantly, what you learned.

**Optional Advanced Challenge — Walk in MuJoCo** *(if attempted)*

- What happened when you first transferred your walker, what you tried, and what worked and didn't (with a link to a video).
- Your explanation of the gap between the two simulations.

### Reminder of General Guidelines

- Figures should have readable axis labels, legends, and captions.
- Reference and discuss every figure in the text — don't paste a plot without commentary.
- Be concise: prioritize insight over volume. A focused paragraph beats a page of restated code output.

---

## Rubric

This project is worth **10 points**, plus up to 1 point of extra credit, broken down as follows:

### Assignment Completion (5 pts)

The two required parts are weighted roughly equally. Credit is based on whether the part was genuinely completed — a CTRNN that really walks, an experiment that was really designed, run, and interpreted — not just attempted. **The MuJoCo challenge is optional and worth up to 1 point of extra credit.**

### Report Quality (5 pts)

- **Title page (1 pt)** — includes all required information: name, course title, assignment name, date submitted, time spent, and self-assessment (1–10).
- **Figures (2 pts)** — figures are easy to read, meaningful (they show what the text claims), properly labeled (axes, legend, caption), and each is paired with an interpretation in the text. A plot with no discussion, or discussion with no supporting plot, does not receive full credit.
- **Creativity & critical thinking (2 pts)** — the quality of your question and your experiment, depth of insight, and evidence of genuine exploration — especially in explaining *how* your walkers walk, not just how fast.

---

## Readings

**Required:**

- **[R6A]** Beer, R. D., & Gallagher, J. C. (1992). *Evolving dynamical neural networks for adaptive behavior.* Adaptive Behavior, 1(1). Evolves CTRNNs to make a six-legged insect walk, with and without sensory feedback. This project's body and its CPG/RPG/MPG modes come from here.
- **[R6B]** Beer, R. D. (1995). *On the dynamics of small continuous-time recurrent neural networks.* Adaptive Behavior, 3(4). What CTRNNs can do and why, starting from one and two neurons. The best introduction to thinking about a CTRNN as a dynamical system.

**Further reading:**

- Mathayomchan, B., & Beer, R. D. (2002). *Center-crossing recurrent neural networks for the evolution of rhythmic behavior.* Neural Computation, 14(9).
- Chiel, H. J., Beer, R. D., & Gallagher, J. C. (1999). *Evolution and analysis of model CPGs for walking: I. Dynamical modules.* Journal of Computational Neuroscience, 7(2).
- Jakobi, N., Husbands, P., & Harvey, I. (1995). *Noise and the reality gap: The use of simulation in evolutionary robotics.* European Conference on Artificial Life (ECAL).
- Floreano, D., Dürr, P., & Mattiussi, C. (2008). *Neuroevolution: from architectures to learning.* Evolutionary Intelligence, 1(1).

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.
