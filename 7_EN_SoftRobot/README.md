# Project 7: Embodied Evolution IV — Soft Robots: Evolving Bodies, Not Just Brains

## Overview

In Projects 4–6 the body was always given to you — a two-wheeled vehicle, a
cart with a pole, a six-legged insect — and evolution's only job was to find
a controller for it. In this project, **the body itself becomes part of the
genome.**

You'll work with soft robots in [Evolution Gym](https://evolutiongym.github.io)
(EvoGym), a widely-used soft-body physics simulator and benchmark. An EvoGym
robot is a small 2D grid of *voxels*, and each voxel is made of one of five
materials: empty, rigid, soft, or one of two kinds of muscle-like actuator
that stretch and squeeze on command. Change the grid, and you've changed the
animal.

You'll run the same evolve → simulate → analyze pipeline as before — the same
EvoTorch `GeneticAlgorithm`, the same SBX crossover and Gaussian mutation, the
same flat genome — in three modes:

- **Control:** a fixed, hand-designed body; evolve its controller (a neural
  network, as in Projects 4–6, or a much simpler open-loop oscillator).
- **Morphology:** a fixed controller; evolve the body.
- **Co-design:** evolve the body *and* its controller together, in a single
  genome.

This is also your first time on a physics engine you didn't see built from
scratch in this course — a real soft-body simulator used in research — and your
first time in a setting where the *interface* between the controller and the
body (how many sensors, how many actuators) isn't fixed in advance, but
depends on what evolution builds.

---

## Learning Objectives

By completing this project, you will learn how to:

- Encode a robot's *body* as a genome, and handle genomes that decode into
  bodies that can't be built
- Co-evolve morphology and control in a single fixed-length genome
- Compare a closed-loop neural controller against an open-loop rhythm
  generator on the same body and task
- Reason about how much of a behavior is "in the brain" versus "in the body"
  (morphological computation)
- Work with a standard Gymnasium-based physics benchmark (EvoGym) that you
  didn't write yourself

---

## Background

### From neuroevolution to body–brain co-design

Everything you've evolved so far has been a *controller*: a vector of numbers
decoded into network weights. Nothing stops us from decoding some of those
numbers into a *body* instead. That's the idea behind Karl Sims' 1994 *Evolved
Virtual Creatures*, and behind a large body of later work on evolving robot
morphology: if intelligent behavior comes from the interaction of body, brain,
and environment (the thesis of Project 1), then there's no reason to hold
the body fixed while only optimizing the brain.

Two consequences you'll run into immediately:

1. **Not every genome is a robot.** A random grid of materials may fall apart
   into disconnected pieces, or contain no actuators at all. These genomes
   can't be simulated, so they get a fixed, very low fitness
   (`INVALID_FITNESS = -1` in `soft_robot.py`) and are removed by selection.
2. **The controller has to fit whatever body evolution builds.** A neural
   network's input and output sizes depend on the body's sensors and
   actuators, so a network evolved for one body can't even be *loaded* onto
   another. Part of the design problem is choosing a controller that scales
   with the body.

### Soft robots in EvoGym

Each voxel in a robot's grid holds one of five materials:

| Code | Material | Symbol | Color in GIFs |
|---|---|---|---|
| 0 | empty | `.` | — |
| 1 | rigid | `R` | black |
| 2 | soft | `S` | gray |
| 3 | horizontal actuator | `H` | orange |
| 4 | vertical actuator | `V` | blue |

Row 0 of the grid is the top of the robot; the last row rests on the ground.
The simulator models each voxel as point masses at its corners, connected by
springs. Rigid voxels have stiff springs, soft voxels have compliant ones, and
actuator voxels have springs whose rest length you control.

**Actions:** one number per actuator voxel, every simulation step: the target
length as a fraction of the voxel's rest length, in `[0.6, 1.6]` (0.6 squeezes
it, 1.0 leaves it at rest, 1.6 stretches it). Horizontal actuators change
width, vertical ones change height. Actuators are listed in row-major order
(top-left to bottom-right), skipping every voxel that isn't an actuator.

**Observations (Walker-v0):** the robot's center-of-mass velocity (2 values)
plus the position of every point mass relative to the center of mass (2 values
per point mass). The size depends on the body: it's 74 for the `biped` preset.

**Reward (Walker-v0):** how far the center of mass moved to the right on that
step, so an episode's total reward (the fitness) is the distance walked.
Distances are in simulator units, where **one voxel is 0.1 wide**: a fitness
of 1.0 means 10 voxel widths, about two body lengths for a 5×5 robot. The
track ends at x = 9.9. A robot that gets there ends its episode early and
earns a +1 bonus, so **Walker-v0 fitness tops out around 10.5**, much as
CartPole tops out at 500.

**Episode length:** 500 steps for Walker-v0, adjustable with `--duration`.
The simulation is **deterministic**: the same robot always does exactly the
same thing. That's why fitness uses one episode per evaluation instead of
averaging several as Project 5 does. (CartPole starts every episode from
a random state; EvoGym doesn't.)

### Two controllers

**`NeuralController` (closed-loop):** the feedforward network from Projects
4–6: observation → hidden layer(s) → one command per actuator, with a Tanh
output rescaled to `[0.6, 1.6]`. It reacts to what the body is doing right now.
Because its input and output sizes depend on the body, it's only used in
`--mode control`.

**`OscillatorController` (open-loop):** every actuator follows the same sine
wave, shifted by its own evolved phase:

```
command_i(t) = 1.1 + 0.5 * sin(2π t / period + phase_i)
```

It never looks at the observation. It's a pure rhythm generator: the
simplest possible version of the central pattern generator (CPG) you evolved
with a CTRNN in Project 6. Its genome is one number per actuator, which is
exactly what makes it convenient for evolving bodies: add an actuator voxel,
add one phase.

### Three evolution modes

| `--mode` | Body | Controller | Genome (for a 5×5 grid) |
|---|---|---|---|
| `control` | fixed (`--body`) | evolved: `--controller mlp` or `oscillator` | mlp: all weights (1,574 for `biped`, `--hidden 16`); oscillator: 1 phase per actuator (22 for `biped`) |
| `morphology` | evolved | fixed traveling wave | 5 material genes per voxel = 125 |
| `codesign` | evolved | evolved oscillator | 5 material genes + 1 phase per voxel = 150 |

**How a body is decoded:** each voxel gets five genes, one per material, and
takes whichever material's gene is largest (an *argmax over logits*). A small
mutation usually leaves a voxel's material unchanged. Only when two of its
genes swap order does the material flip, and any material can always be
reached from any other.

**The fixed controller in `morphology` mode** is a traveling wave: each
actuator's phase is set by its column, `2π · column / grid`, so a wave of
contraction sweeps across the body at the same speed whatever body it's
attached to. Evolution's job is to find a body that turns that wave into
forward motion.

**In `codesign` mode**, *every* voxel gets a phase gene, but only the voxels
that decode as actuators use theirs. The rest are carried along silently and
can become active later if a mutation turns that voxel into an actuator.

---

## Project Structure

| File | Purpose |
|------|---------|
| `soft_robot.py` | Everything about bodies and the world: material codes, preset bodies, loading/decoding/validating bodies, creating the EvoGym environment, running one episode, plotting a body |
| `neural_controller.py` | The two controllers, `NeuralController` (closed-loop MLP) and `OscillatorController` (open-loop sine waves), plus the fixed traveling-wave phases used in `morphology` mode |
| `evolve.py` | Runs evolution with EvoTorch in any of the three modes. Also defines how a genome decodes into a body and controller (`decode_body_from_genome`, `decode_controller`) |
| `sim.py` | Replays a saved genome: fitness, body plot, center-of-mass and actuator traces, GIF, or a live window |
| `example_body.txt` | A hand-designed body in the plain-text format `--body` accepts. Copy it to design your own |
| `requirements.txt` | Pinned dependency versions for this project's environment |
| `README.md` | This documentation |

---

## Installation

The project requires **Python 3.10** (not newer), plus:

- EvoGym
- NumPy
- Matplotlib
- PyTorch
- EvoTorch
- Gymnasium
- imageio and Pillow (for saving GIFs)

**Why 3.10 exactly:** EvoGym's simulator is written in C++ and distributed as
precompiled packages ("wheels"), which exist only for Python 3.7–3.10. On
Python 3.11 or newer, `pip` will try to compile EvoGym from source and almost
certainly fail. So unlike earlier projects, you can't just reuse whatever
`python3` you already have. You need to create this project's environment
from a Python 3.10 interpreter.

**Option A — conda (easiest if you have Anaconda/Miniconda):**

```bash
conda create -n evogym python=3.10
conda activate evogym
pip install -r requirements.txt
```

Run `conda activate evogym` each time you come back to the project.

**Option B — venv**, if you have Python 3.10 installed separately (e.g. from
python.org or `brew install python@3.10`):

macOS / Linux:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
py -3.10 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Check it worked** (you should see the body printed, then a fitness score):

```bash
python sim.py --body biped
```

**Known quirks** (already handled, but good to know if you see them):

- `requirements.txt` pins `setuptools<81`. EvoGym 2.0 still imports the
  `pkg_resources` module, which newer setuptools removed. Without the pin,
  `import evogym` fails with `ModuleNotFoundError: No module named
  'pkg_resources'`.
- EvoGym prints `Using Evolution Gym Simulator v2.2.5` from its C++ core the
  first time it runs. That's normal.
- Occasionally an extreme body or controller makes the solver unstable, and
  EvoGym prints `SIMULATION UNSTABLE... TERMINATING`. It ends that episode
  and subtracts 3 from the reward, so the robot scores badly and evolution
  carries on. During evolution these messages come from worker processes and
  are hidden, but you may see one in `sim.py`.

---

## Running Evolution

### Mode 1: Control (fixed body)

```bash
# Neural network controller on the biped preset
python evolve.py --mode control --controller mlp --body biped --verbose --vizperf --output mlp_biped.npy

# Open-loop oscillator on the same body
python evolve.py --mode control --controller oscillator --body biped --verbose --vizperf --output osc_biped.npy

# Your own hand-designed body (see "Designing a body by hand" below)
python evolve.py --mode control --controller oscillator --body my_body.txt --verbose --output osc_mine.npy
```

Preset bodies: `biped` (two legs and a torso, all horizontal actuators),
`worm` (a flat, two-row strip), and `block` (a solid soft block with a rigid
top and actuators along the bottom).

### Mode 2: Morphology (fixed controller)

```bash
python evolve.py --mode morphology --grid 5 --verbose --vizperf --output morph.npy
```

### Mode 3: Co-design (body + controller)

```bash
python evolve.py --mode codesign --grid 5 --verbose --vizperf --output codesign.npy
```

For `morphology` and `codesign`, the best body is printed at the end as a text
picture (`R` rigid, `S` soft, `H`/`V` actuators, `.` empty).

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `control` | `control`, `morphology`, or `codesign` |
| `--body` | `biped` | [control] preset name, or path to a `.txt`/`.npy` body |
| `--grid` | `5` | [morphology/codesign] side length of the voxel grid |
| `--controller` | `mlp` | [control] `mlp` or `oscillator` |
| `--hidden` | `16` | [mlp] hidden layer size(s), e.g. `--hidden 32` or `--hidden 16 16` |
| `--activation` | `tanh` | [mlp] hidden-layer activation |
| `--period` | `25` | [oscillator] simulation steps per oscillation cycle |
| `--env` | `Walker-v0` | EvoGym task |
| `--duration` | task default | episode length in steps (500 for Walker-v0) |
| `--popsize` | `50` | population size |
| `--gens` | `50` | generations |
| `--mut_stdev` | `0.1` mlp / `0.2` otherwise | Gaussian mutation stdev |
| `--tournament_size`, `--eta`, `--no-elitism` | `3`, `20`, elitism on | GA settings, as in Projects 5–6 |
| `--seed_genome`, `--seed_noise` | — | continue from a saved genome |
| `--workers` | `max` | parallel workers (`max`, an integer, or `none`) |
| `--seed` | — | random seed |
| `--verbose`, `--vizperf` | off | print per-generation stats / plot fitness curves |
| `--output` | — | save best genome (`.npy`) **and its settings** (`.json`, same name) |
| `--fitness_output` | — | save per-generation `best`/`avg`/`worst` fitness to `.npz` |

### Designing a body by hand

A body file is plain text, one row per line, material codes separated by
spaces, with row 0 at the top. Lines starting with `#` are comments. See
`example_body.txt`:

```
1 1 1 1 1
2 3 3 3 2
2 4 0 4 2
2 4 0 4 2
3 3 0 3 3
```

A body must be **connected** (all non-empty voxels touching edge to edge) and
contain **at least one actuator** (3 or 4). Preview any body before evolving
it:

```bash
python sim.py --body my_body.txt --showbody --gif my_body.gif
```

---

## Running Simulations

`evolve.py --output best.npy` also writes `best.json` next to it with every
setting needed to rebuild the robot (mode, body, task, controller, period,
hidden sizes, ...). `sim.py` reads it automatically, so unlike Projects 5 and
6 you don't need to re-type matching flags.

```bash
# Fitness, body picture, and behavior traces
python sim.py --genome codesign.npy --showbody --viztraces

# Save an animation (the easiest way to see what evolution actually built)
python sim.py --genome codesign.npy --gif codesign.gif

# Watch it live in a window
python sim.py --genome codesign.npy --render

# Test a robot on a task or episode length it was NOT evolved for
python sim.py --genome codesign.npy --env UpStepper-v0 --gif upstepper.gif
python sim.py --genome codesign.npy --duration 1000
```

`--viztraces` plots, over time: the center of mass's x position (distance
walked), its y position (height, where hops and falls show up), and a heatmap
of every actuator's command (blue = contracting, red = expanding). For an
oscillator the heatmap shows the phase pattern directly. For an MLP it shows
what the network actually chose to do.

Without `--genome`, `sim.py` runs `--body` (default `biped`) with random
oscillator phases (`--seed` to fix them). Use this to see what a body does
before evolution touches it.

### Other EvoGym tasks

`--env` accepts any of EvoGym's 32 tasks. Some that work unchanged with this
project's code (the robot just needs to move right, over different terrain):
`BridgeWalker-v0` (soft rope bridge), `UpStepper-v0` and `DownStepper-v0`
(stairs), `ObstacleTraverser-v0` and `-v1` (rough ground), `Hurdler-v0`,
`PlatformJumper-v0`, `GapJumper-v0`, `CaveCrawler-v0`. Others reward
something different, such as jumping (`Jumper-v0`), climbing (`Climber-v0`),
or carrying a box (`Carrier-v0`). Read the [EvoGym task
docs](https://evolutiongym.github.io/) before interpreting fitness numbers on
those, because the reward isn't "distance walked" anymore.

---

## Understanding Fitness Scores

On Walker-v0, **fitness ≈ distance walked in 500 steps, in units of 10 voxel
widths**. A 5×5 robot is 5 voxels long, so each 1.0 of fitness is about two
body lengths.

- **≈ 0 or negative:** twitching in place, or drifting backwards.
- **1–3:** a slow shuffle or crawl.
- **4–9:** a real gait: 40–90 voxel widths in 500 steps. Look at the GIF:
  there's usually a clear repeating stride.
- **Around 10.5:** the robot reached the end of the track before time ran
  out. Walker-v0 can't score higher, so once several runs get close to this,
  compare *how fast* they get there instead of their final fitness.

`INVALID_FITNESS = -1` marks a genome whose body couldn't be built. The
fitness statistics `evolve.py` reports are for the population *after*
selection, so unbuildable bodies rarely show up there. They're culled in
the same generation they appear (see Part 1).

**What to expect at the defaults** (`--popsize 50 --gens 50`): most runs in
every mode reach a real gait (fitness above 4) within the budget. The spread
across seeds is large for some modes and small for others, and that spread is
itself one of the things worth reporting. A run that's still below 1 after 50
generations isn't necessarily a bug. It can be a seed where evolution got
stuck. Run another seed before you start debugging.

---

## Performance

One Walker-v0 episode (500 steps) takes roughly half a second on a laptop
core, so a generation of 50 robots takes a few seconds with
`--workers max`. A default run (`--popsize 50 --gens 50`) takes about 5–6
minutes on a 16-core laptop, and proportionally longer with fewer cores (so
roughly 10–12 minutes on 8). Parts 2–4 add up to a few dozen runs, so start
early and let them run while you work on the analysis.
Start with `--gens 20` while you're getting a feel for things.

The physics dominates the cost. The MLP's forward pass is negligible next
to it, and bigger bodies (`--grid 7`) cost more per step because they have
more point masses.

---

## Understanding the Components

### `soft_robot.py`

Material codes, the three preset bodies, `load_body` (preset name or file),
`decode_body` (genes → body via argmax), `is_valid_body` (connected + has an
actuator), `make_env`, `run_episode`, and `plot_body`. Note that
`make_env` imports `evogym.envs` *inside* the function: EvoGym registers its
tasks with Gymnasium as a side effect of that import, and each of Ray's worker
processes needs its own registration.

### `neural_controller.py`

`NeuralController` and `OscillatorController` share one interface: `reset()`
once per episode, then `act(obs)` once per step, returning a NumPy array of
commands in `[0.6, 1.6]`. `run_episode` doesn't know or care which controller
it's driving.

### `evolve.py`

The GA setup is the same as Projects 5–6. What's new is the decoding:
`decode_body_from_genome` reads the body out of the genome (or returns the
fixed body in `control` mode), and `decode_controller` builds the controller
*for that body*. The fitness function checks the body is buildable before
creating an environment for it.

### `sim.py`

Rebuilds the robot from `genome.npy` + `genome.json`, runs one episode with
recording on, and produces the plots/GIF you ask for.

---

## Tips

- **Look at the GIFs.** A fitness number tells you *how far*, not *how*.
  Evolution is very good at finding gaits you wouldn't have designed:
  dragging, hopping, rolling, inch-worming. Many are only obvious once
  you watch them.
- Use `--seed` when debugging and comparing, then run several seeds before
  drawing conclusions. Evolved bodies in particular vary a lot from seed to
  seed.
- Save `--fitness_output` for every run you might want to plot later, so you
  won't need to re-run evolution to make a comparison figure.
- A body in `morphology`/`codesign` mode is only as good as the controller
  it's paired with. Before concluding that a body "doesn't work", ask whether
  it could with different phases.

---

## Assignment

### Part 1 – Understand Bodies and Genomes

Answer these before running evolution:

- For a 5×5 grid, how many genes does each of these genomes have: `morphology`,
  `codesign`, `control --controller oscillator --body biped`, and
  `control --controller mlp --body biped --hidden 16`? For the MLP, count
  layer by layer, starting from the observation size (74 for `biped`). Check
  your answers against the first line `evolve.py` prints.
- Why can't a `NeuralController` evolved for `biped` be used on `worm`? Be
  specific about which dimension(s) mismatch.
- Estimate what fraction of *random* genomes decode to bodies that can't be
  built. Write a few lines of Python using `decode_body` and `is_valid_body`
  from `soft_robot.py` on a few thousand uniformly random genomes in
  `[-1, 1]` (the range evolution starts from). Repeat for grids of 3, 5, and
  7. Before you run it: do you expect bigger grids to be more or less likely
  to be invalid, and why?
- In `codesign` mode, every voxel carries a phase gene, but only actuator
  voxels use it. What role might those unused genes play over the course of
  evolution?

### Part 2 – Evolving Control for a Fixed Body

1. On the `biped` preset, evolve both an `mlp` controller and an `oscillator`
   controller with otherwise matched settings, 3 seeds each. Plot best-fitness
   curves for both on one figure.
2. Design your own body by hand (copy `example_body.txt` and edit it). Use
   what you learned from watching Part 2.1's GIFs. Evolve an oscillator for
   it, and compare against `biped`.

**Note:** a good oscillator on `biped` gets close to Walker-v0's ceiling (about
10.5, see *Understanding Fitness Scores*). If your curves converge near there,
compare them by how quickly they get there and by their generation-0
fitness, not just by final fitness.

Questions:
- The oscillator has about 70× fewer genes than the MLP and can't sense
  anything. Which evolves faster, and which reaches higher fitness within
  the budget? Why might the "dumber" controller do so well here? Connect this
  to Project 6's CPG vs. RPG results.
- What can a closed-loop controller do that an open-loop one fundamentally
  can't? Is Walker-v0 on flat ground a task that needs that?
- How did your hand-designed body do? What did you learn from designing it,
  and what does that suggest about how much of a gait lives in the body?

### Part 3 – Evolving the Body

1. Run `--mode morphology` (3 seeds). Save GIFs of the best body from each seed.
2. Run `--mode codesign` (3 seeds). Save GIFs of the best body from each seed.
3. Plot best-fitness curves from `morphology`, `codesign`, and your best
   Part 2 configuration on one figure.

Questions:
- Do different seeds converge on similar bodies, or very different ones? Are
  there recurring motifs, like legs, a rigid spine, actuators concentrated in
  one place, or empty space?
- Is evolving the body (with a fixed controller) better or worse than
  evolving the controller (with a fixed body)? Is co-design better than
  either alone? Explain your results in terms of what each genome gives
  evolution control over.
- Take your best `codesign` genome and look at its body with `--showbody`.
  Which materials did evolution use most, and which least? Why?

### Part 4 – Behavioral and Quantitative Analysis

1. Use `sim.py --viztraces` on your best robot from each mode. Describe the gait
   in words, and relate it to the COM traces and the actuator heatmap. What
   does the height trace tell you that the distance trace doesn't?
2. Pick one mode and vary one parameter that changes the *body*: either
   `--grid` (3, 4, 5, 6) or, for `codesign`, `--period` (e.g. 10, 25, 50).
   Run 3+ seeds per setting and summarize final best fitness (mean ± spread)
   in a plot.
3. **Transfer test:** take your best Walker-v0 robot from any mode and run it,
   without re-evolving, on at least two other terrains with `sim.py --env`
   (e.g. `UpStepper-v0`, `ObstacleTraverser-v0`, `BridgeWalker-v0`).

Questions:
- For Part 4.2: before running, predict the effect of your parameter. Were you
  right? What's the trade-off that parameter controls?
- For Part 4.3: which of your robots generalizes best to terrain it never
  saw? Is it the one with the highest Walker-v0 fitness? What features of
  the body or gait seem to matter for robustness?

---

## Optional / Advanced Challenge

Parts 1–4 are required. Beyond that, pick **one** of the following to
investigate further. Each is open-ended, and the point is to form a
hypothesis, run the experiment, and report what you found.

**1. Evolve for a harder task.** Pick a different EvoGym task with `--env`
(e.g. `UpStepper-v0`, `BridgeWalker-v0`, `ObstacleTraverser-v0`, `Climber-v0`,
`Jumper-v0`) and run `codesign` on it directly. How do the evolved bodies
differ from those evolved on flat ground? Does a body evolved for the new
task still walk on Walker-v0?

**2. Closing the loop in co-design.** The oscillator can't sense anything. Design a
controller that works with *any* body but still uses feedback. One option:
modulate each actuator's sine wave with a small network whose inputs are a
fixed-size summary of the observation (e.g. just the 2 center-of-mass
velocities) and whose weights are shared across actuators. Add it as a new
`codesign` variant and compare against the pure oscillator, on flat ground
and on a rough-terrain task.

**3. Material costs.** Real robots pay for actuators (weight, power, money).
Add a penalty to the fitness for each actuator voxel (or a bonus for each
empty one), and see how the evolved bodies change as the penalty grows. Is
there a point where evolution starts building mostly passive bodies with a
few well-placed actuators?

**4. A second algorithm.** Swap the GA for EvoTorch's `CMAES`, as Project 6
did, and compare on `control --controller mlp` (a large, smooth genome) and
on `morphology` (a genome whose argmax decoding makes fitness piecewise
constant). Does one algorithm suit one genome better than the other? Why
might that be?

You're encouraged to explore your own idea as well, as long as it's a genuine
extension (not a parameter change already covered in Parts 2–4).

---

## What to Submit to Moodle

Submit a single **written report as a PDF** to Moodle.

### Title Page

The first page of your report should include:

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 7: Embodied Evolution IV — Soft Robots)
- Date submitted
- Amount of time spent on this project
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body of your report into one section per assignment part. Each
section should combine the relevant figures with a written discussion — a
plot with no interpretation, or an interpretation with no supporting plot, is
incomplete. Static frames or a link to your GIFs are welcome wherever
behavior matters.

**Part 1 — Understand Bodies and Genomes**

- Gene counts for all four configurations (MLP counted layer by layer).
- Your explanation of the MLP/body mismatch.
- Your invalid-body fractions for grids 3, 5, and 7 (the code you used, the numbers, your prediction, and an explanation).
- Your answer on the role of unused phase genes.

**Part 2 — Evolving Control for a Fixed Body**

- Best-fitness curves, MLP vs. oscillator on `biped`, 3 seeds each.
- Your hand-designed body (picture from `--showbody`), its evolved fitness, and a comparison to `biped`.
- Answers to the Part 2 questions.

**Part 3 — Evolving the Body**

- Pictures of the best evolved bodies from each seed of `morphology` and `codesign`.
- The combined best-fitness figure (morphology vs. codesign vs. your best Part 2 configuration).
- Answers to the Part 3 questions.

**Part 4 — Behavioral and Quantitative Analysis**

- `--viztraces` figures for your best robot from each mode, with a written description of each gait.
- The parameter-sweep figure (mean ± spread over seeds), with your prediction stated before the results.
- Transfer-test results (a table of fitness per terrain per robot is fine).
- Answers to the Part 4 questions.

**Optional / Advanced Challenge** *(if attempted)*: which direction you chose (or your own idea), what you changed, your results with supporting figures, and your interpretation.

### Reminder of General Guidelines

- Figures should have readable axis labels, legends, and captions.
- Reference and discuss every figure in the text — don't paste a plot without commentary.
- Be concise: prioritize insight over volume. A focused paragraph beats a page of restated code output.

---

## Rubric

This project is worth **10 points**, broken down as follows:

### Assignment Completion (5 pts)

Each part of the assignment (see *Assignment* above) is weighted roughly equally. Credit is based on whether the part was genuinely completed — code implemented and working, questions answered with reasoning, experiments actually run — not just attempted.

### Report Quality (5 pts)

- **Title page (1 pt)** — includes all required information: name, course title, assignment name, date submitted, time spent, and self-assessment (1–10).
- **Figures (2 pts)** — figures are easy to read, meaningful (they show what the text claims), properly labeled (axes, legend, caption), and each is paired with an interpretation in the text. A plot with no discussion, or discussion with no supporting plot, does not receive full credit.
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in connecting what you *see* the robots doing (GIFs, traces) to the fitness numbers, and in reasoning about how behavior is divided between body and controller.

---

## Further Reading

- Bhatia, J., Jackson, H., Tian, Y., Xu, J., & Matusik, W. (2021). *Evolution
  Gym: A Large-Scale Benchmark for Evolving Soft Robots.* NeurIPS.
  ([project page](https://evolutiongym.github.io))
- Sims, K. (1994). *Evolving Virtual Creatures.* SIGGRAPH.
- Cheney, N., MacCurdy, R., Clune, J., & Lipson, H. (2013). *Unshackling
  Evolution: Evolving Soft Robots with Multiple Materials and a Powerful
  Generative Encoding.* GECCO.
- Pfeifer, R., & Bongard, J. (2006). *How the Body Shapes the Way We Think.*
  MIT Press.

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary
Robotics** at Rose-Hulman Institute of Technology.
