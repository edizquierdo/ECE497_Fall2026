# Project 5: Embodied Evolution II — Soft Robots: Evolving Bodies, Not Just Brains

## Overview

In Project 4 the body was given to you: a two-wheeled vehicle with two light sensors, designed by hand back in Project 1. Evolution's only job was to find a brain for it. In this project, **the body itself becomes part of the genome.**

You'll work with soft robots in [Evolution Gym](https://evolutiongym.github.io) (EvoGym), a soft-body physics simulator widely used in research. An EvoGym robot is a small 2D grid of *voxels*, and each voxel is made of one of five materials: empty, rigid, soft, or one of two kinds of muscle-like actuator that stretch and squeeze on command. Change the grid, and you've changed the animal. The task is simple: walk to the right, as far as you can.

The pipeline is the one you already know from Project 4: a flat genome, EvoTorch's `GeneticAlgorithm` with SBX crossover and Gaussian mutation, evolve → simulate → analyze. What's new is what the genome is allowed to describe. The project is organized around one comparison:

- **(a) Evolve the brain for a fixed body.** A hand-designed body, and evolution searches only for its controller.
- **(b) Evolve the brain and the body together (co-design).** Evolution searches for a body *and* the controller that drives it, in a single genome.

and around two very different kinds of brain:

- **A neural network**, closed-loop: it senses what the body is doing and reacts. On a fixed body this is the feedforward network from Project 4, one network for the whole body (`--controller neural`). When the body is evolving, it's a small network copied into every actuator voxel (`--controller local`), for reasons you'll see below.
- **An oscillator** (`--controller oscillator`), an open-loop rhythm generator: every actuator follows a sine wave with its own evolved phase, and it never senses anything at all.

Your goal is not just to get a robot that walks, but to reason about *where the walking comes from*: how much of a gait lives in the brain, and how much in the body.

## Checkpoints and Final Due Date

- **Checkpoint — TBD, during class.** Covers Required Parts 1 and 2. No reporting necessary for the checkpoint; it will involve running code on your laptop to demonstrate and showing figures (and GIFs!).
- **Final due date — TBD, before class.** Includes Required Part 3 and the report. There will be demos during class.

---

## Learning Objectives

By completing this project, you will learn how to:

1. Encode a robot's *body* as part of a genome, and co-evolve body and controller in a single genome.

2. Compare evolving a controller for a fixed, hand-designed body against evolving the body and controller together, and reason about what each one gives evolution control over.

3. Compare a closed-loop neural controller against an open-loop rhythm generator, and a centralized network against a distributed one.

4. Reason about how behavior is divided between brain and body (*morphological computation*), using what you see the robots doing, not just their fitness.

---

## IMPORTANT NOTE

What follows below is ONE possible path for this project. However, you do NOT have to take this path. The required learning goals are fixed; the implementation and experiments are flexible. Take your own path. You are just as welcome to explore on your own, or to follow along.

As in Project 4, this assignment has only three REQUIRED components, so that you have time to explore. The REQUIRED components are labeled clearly — everything else is OPTIONAL.

---

## Background

### From evolving brains to evolving bodies

Everything you've evolved so far has been a *controller*: a vector of numbers decoded into network weights. Nothing stops us from decoding some of those numbers into a *body* instead. That's the idea behind Karl Sims' 1994 *Evolved Virtual Creatures*, and behind a large body of later work on evolving robot morphology. If intelligent behavior comes from the interaction of body, brain, and environment (the lesson of the Braitenberg vehicles in Project 1), then there's no reason to hold the body fixed while only optimizing the brain.

Two consequences you'll run into immediately:

1. **Not every genome is a robot.** A random grid of materials may fall apart into disconnected pieces, or contain no actuators at all. These genomes can't be simulated, so they get a fixed, very low fitness (`INVALID_FITNESS = -1` in `soft_robot.py`) and are removed by selection.
2. **The brain has to fit whatever body evolution builds.** A network that reads the whole body and drives every actuator has input and output sizes that depend on the body, so a network evolved for one body can't even be *loaded* onto another. There are two ways around this, and this project uses both. The oscillator needs exactly one phase per actuator, so adding an actuator voxel just means adding one phase. The local network is one small network shared by every actuator voxel, so its size doesn't depend on the body at all.

### Soft robots in EvoGym

Each voxel in a robot's grid holds one of five materials:

| Code | Material | Symbol | Color in GIFs |
|---|---|---|---|
| 0 | empty | `.` | — |
| 1 | rigid | `R` | black |
| 2 | soft | `S` | gray |
| 3 | horizontal actuator | `H` | orange |
| 4 | vertical actuator | `V` | blue |

Row 0 of the grid is the top of the robot; the last row rests on the ground. The simulator models each voxel as point masses at its corners, connected by springs. Rigid voxels have stiff springs, soft voxels have compliant ones, and actuator voxels have springs whose rest length the controller sets.

- **Actions:** one number per actuator voxel, every simulation step: the target length as a fraction of the voxel's rest length, in `[0.6, 1.6]` (0.6 squeezes it, 1.0 leaves it at rest, 1.6 stretches it). Horizontal actuators change width, vertical ones change height.
- **Observations:** the robot's center-of-mass velocity (2 values) plus the position of every point mass relative to the center of mass (2 values per point mass). The size depends on the body: 74 for the `biped` preset.
- **Fitness:** how far the center of mass moved to the right over the episode (500 steps by default). One voxel is 0.1 wide, so a fitness of 1.0 means 10 voxel widths, about two body lengths for a 5×5 robot.
- **No noise.** The simulation is deterministic: the same robot always does exactly the same thing. Unlike Project 4, there's no need to average over several episodes or to re-evaluate the best genome after evolution. The fitness `evolve.py` reports is the fitness.

### The brains

**`NeuralController` (closed-loop, centralized):** the same kind of feedforward network as in Project 4. It's one network for the whole body: it reads the entire observation (for `biped`, all 74 numbers) and outputs one command per actuator (22 for `biped`), with a Tanh output rescaled to `[0.6, 1.6]`. Its genome is all of its weights and biases. Because its size depends on the body, it can only be used on a fixed body.

**`LocalController` (closed-loop, distributed):** one small network, copied into every actuator voxel, with the same weights in every copy. Each copy sees only local information and drives only its own voxel:

| Input to each copy | Values |
|---|---|
| how stretched its own voxel is (width, height) | 2 |
| the same for its neighbors above, below, left, and right (0 if no voxel there) | 8 |
| whether it's a horizontal or vertical actuator | 2 |
| a shared clock, sin and cos of 2π t / period | 2 |

Its genome is the one shared network (129 genes with the default 8 hidden units), whatever the body. The clock is there because without it, every copy sees nearly the same thing when the robot is at rest, so they all do nearly the same thing and no rhythm ever gets started.

**`OscillatorController` (open-loop):** every actuator follows the same sine wave, shifted by its own evolved phase:

```
command_i(t) = 1.1 + 0.5 * sin(2π t / period + phase_i)
```

It never looks at the observation. It's the simplest possible *central pattern generator* (CPG): a source of rhythm that doesn't need any sensory input. Its genome is one phase per actuator.

### How a body is encoded

Each voxel gets five genes, one per material, and takes whichever material's gene is largest (an *argmax*). Small mutations usually leave a voxel's material unchanged, and only when two of its genes swap order does the material flip. In co-design mode with the oscillator, every voxel also gets a phase gene, but only voxels that decode as actuators use theirs; the rest are carried along silently. With the local network, the material genes are followed by the shared network's weights.

### The evolution modes

| `--mode` | Body | Brain | Genome |
|---|---|---|---|
| `control` | fixed (`--body`) | evolved: `--controller neural`, `oscillator`, or `local` | the controller's parameters |
| `codesign` | evolved (on a `--grid` × `--grid` grid) | evolved: `--controller oscillator` or `local` | 5 material genes per voxel, then the controller's parameters |
| `morphology` | evolved | *fixed*: a traveling wave (phase set by each actuator's column) | 5 material genes per voxel |

Modes `control` and `codesign` are the core of this project. `morphology`, which evolves only the body, is there for you to explore in Part 3 if you want to.

---

## Project Structure

| File | Purpose |
|------|---------|
| `soft_robot.py` | Everything about bodies and the world: material codes, preset bodies, decoding and validating bodies, creating the EvoGym environment, running one episode, plotting a body |
| `neural_controller.py` | The brains: `NeuralController`, `OscillatorController`, and `LocalController` |
| `evolve.py` | Runs evolution in any mode. `decode_body_from_genome()` and `decode_controller()` turn a genome into a robot |
| `sim.py` | Replays a saved genome: fitness, body picture, behavior traces, GIF, or a live window |
| `example_body.txt` | A hand-designed body in the plain-text format `--body` accepts. Copy it to design your own |
| `requirements.txt` | Pinned dependency versions for this project's environment |

---

## Installation

This project requires **Python 3.10** (not newer). EvoGym's simulator is written in C++ and distributed as precompiled packages that only exist for Python 3.7–3.10. On a newer Python, `pip` will try to compile EvoGym from source and almost certainly fail. So unlike earlier projects, you need to create this project's environment from a Python 3.10 interpreter.

**Option A — conda (easiest if you have Anaconda/Miniconda):**

```bash
conda create -n evogym python=3.10
conda activate evogym
pip install -r requirements.txt
```

Run `conda activate evogym` each time you come back to the project.

**Option B — venv**, if you have Python 3.10 installed separately (e.g. from python.org or `brew install python@3.10`):

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

**Known quirks** (already handled, but good to know):

- `requirements.txt` pins `setuptools<81`, because EvoGym still imports `pkg_resources`, which newer setuptools removed.
- EvoGym prints `Using Evolution Gym Simulator v2.2.5` the first time it runs. That's normal.
- Occasionally an extreme body or controller makes the simulation unstable, and EvoGym prints `SIMULATION UNSTABLE... TERMINATING`. It ends that episode with a penalty, so the robot scores badly and evolution carries on.

---

## Running Evolution

```bash
# (a) Fixed body, evolved brain: neural network, then oscillator
python evolve.py --mode control --controller neural --body biped --verbose --vizperf --output neural_biped.npy
python evolve.py --mode control --controller oscillator --body biped --verbose --vizperf --output osc_biped.npy

# (b) Co-design on a 5x5 grid: body + oscillator, then body + local network
python evolve.py --mode codesign --controller oscillator --verbose --vizperf --output codesign_osc.npy
python evolve.py --mode codesign --controller local --verbose --vizperf --output codesign_local.npy
```

The first line `evolve.py` prints tells you what's being evolved and how many genes the genome has. For `codesign` (and `morphology`), the best body is printed at the end as a text picture (`R` rigid, `S` soft, `H`/`V` actuators, `.` empty).

Preset bodies for `--body`: `biped` (two legs and a torso, all horizontal actuators), `worm` (a flat, two-row strip), and `block` (a soft block with a rigid top and actuators along the bottom). `--body` also accepts a path to your own body file (see below).

Evaluations run in parallel across your CPU cores (`--workers max`), each robot in its own process.

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `control` | `control`, `codesign`, or `morphology` |
| `--body` | `biped` | [control] preset name, or path to a `.txt`/`.npy` body |
| `--grid` | `5` | [codesign/morphology] side length of the voxel grid |
| `--controller` | `neural` (control) / `oscillator` (codesign) | `neural` (control only), `oscillator`, or `local` |
| `--hidden` | `16` neural / `8` local | [neural/local] hidden layer size(s), e.g. `--hidden 32` or `--hidden 16 16` |
| `--activation` | `tanh` | [neural/local] hidden-layer activation |
| `--period` | `25` | [oscillator/local] simulation steps per oscillation (or clock) cycle |
| `--env` | `Walker-v0` | EvoGym task |
| `--duration` | `500` | episode length in steps |
| `--popsize`, `--gens` | `50`, `50` | population size, generations |
| `--mut_stdev` | `0.1` neural / `0.2` otherwise | Gaussian mutation stdev |
| `--tournament_size`, `--eta`, `--no-elitism` | `3`, `20`, elitism on | GA settings, as in Project 4 |
| `--seed_genome`, `--seed_noise` | — | start from a saved genome |
| `--workers` | `max` | parallel workers (`max`, an integer, or `none`) |
| `--seed` | — | random seed |
| `--verbose`, `--vizperf` | off | print per-generation stats / plot fitness curves |
| `--output` | — | save best genome (`.npy`) **and its settings** (`.json`, same name) |
| `--fitness_output` | — | save per-generation `best`/`avg`/`worst` fitness to `.npz` |

### Designing a body by hand

A body file is plain text, one row per line, material codes separated by spaces, row 0 at the top. Lines starting with `#` are comments. A body must be **connected** (all non-empty voxels touching edge to edge) and contain **at least one actuator**. See `example_body.txt`:

```
1 1 1 1 1
2 3 3 3 2
2 4 0 4 2
2 4 0 4 2
3 3 0 3 3
```

---

## Running Simulations

`evolve.py --output best.npy` also writes `best.json` with every setting needed to rebuild the robot, and `sim.py` reads it automatically, so you don't need to re-type matching flags.

```bash
python sim.py --genome codesign_osc.npy --showbody --viztraces   # fitness, body picture, behavior traces
python sim.py --genome codesign_osc.npy --gif codesign_osc.gif    # save an animation
python sim.py --genome codesign_osc.npy --render                  # watch it live
python sim.py --genome codesign_osc.npy --env UpStepper-v0        # try it on a task it wasn't evolved for
python sim.py --body my_body.txt --showbody --gif my_body.gif     # no genome: a body with random phases
```

`--viztraces` plots, over time, the center of mass's x position (distance walked), its y position (height, where hops and falls show up), and a heatmap of every actuator's command. For the oscillator the heatmap shows the phase pattern directly; for either network (`neural` or `local`) it shows what the network actually chose to do.

---

## Understanding Fitness Scores

- **≈ 0 or negative:** twitching in place, or drifting backwards.
- **1–3:** a slow shuffle or crawl.
- **4–9:** a real gait, usually with a clear repeating stride in the GIF.
- **Around 10.5:** the robot reached the end of the track (x = 9.9) before time ran out and earned a +1 bonus. Walker-v0 can't score higher, so once several runs get close to this, compare *how fast* they get there (generations to reach a threshold, or `--duration`) rather than their final fitness.

At the defaults (`--popsize 50 --gens 50`), most runs reach a real gait. The spread across seeds is large in some modes and small in others, and that spread is itself worth reporting. A run that's still below 1 after 50 generations isn't necessarily a bug: run another seed before you start debugging.

## Performance

One episode takes roughly half a second on one core, so a default run takes about 5–6 minutes on a 16-core laptop, and roughly 10–12 on 8 cores. Use `--gens 20` while you're getting a feel for things, and let longer runs work in the background while you analyze earlier ones. The physics dominates the cost; the neural network's forward pass is negligible next to it.

## Performing Parameter Studies

As in Project 4, you're expected to write your own sweep-and-plot scripts. `run_evolution(cfg, ...)` in `evolve.py` runs one evolution and returns the fitness curves and best genome; `make_config()` shows what goes in `cfg`. For quick comparisons, save each run's curve with `--fitness_output` and plot them together afterwards, the same way you did in Project 4.

---

## Tips

- **Look at the GIFs.** A fitness number tells you *how far*, not *how*. Evolution is very good at finding gaits you wouldn't have designed: dragging, hopping, rolling, inch-worming. Many are only obvious once you watch them.
- **Before running each experiment, write down your prediction for the result.**
- Use `--seed` when debugging, then run several seeds before drawing conclusions. Evolved bodies in particular vary a lot from seed to seed.
- Save `--output` and `--fitness_output` for every run you might want to look at later, so you won't need to re-run evolution to make a figure.
- A body is only as good as the brain it's paired with. Before concluding that a body "doesn't work", ask whether it could with a different controller.

---

## Assignment

### REQUIRED #1: Understand the Genomes

Read `neural_controller.py` and the genome section of `evolve.py` (`genome_length()`, `decode_body_from_genome()`, `decode_controller()`), and answer the following before running long experiments:

- How many genes does each of these genomes have: `control --controller neural --body biped`, `control --controller oscillator --body biped`, `codesign --controller oscillator`, and `codesign --controller local`? Work them out yourself, then check against the first line `evolve.py` prints. For each one, describe in a sentence *what evolution gets to decide*.
- Why can't co-design use the centralized `neural` controller? Be specific about what would break when a mutation changes the body. How does the local network get around this, and what does it give up in exchange?
- What happens to genomes that decode into bodies that can't be built? Do you expect this to be a big or a small problem for evolution, and why?

Then run a short neural-controller evolution on `biped` and watch the result:

```bash
python evolve.py --mode control --controller neural --body biped --verbose --vizperf --output neural_biped.npy
python sim.py --genome neural_biped.npy --viztraces --gif neural_biped.gif
```

Describe the gait in words. Compare to what `biped` does with random phases (`python sim.py --body biped --gif biped_random.gif`).

#### OPTIONAL: How often is a random genome a robot?

Use `decode_body` and `is_valid_body` from `soft_robot.py` to estimate what fraction of random genomes (uniform in `[-1, 1]`, where evolution starts) decode to bodies that can't be built, for a few grid sizes.

---

### REQUIRED #2: Evolving the Brain vs. Evolving the Brain and Body

This is the heart of the project.

1. **Fixed body, evolved brain.** On `biped`, evolve a `neural` controller and an `oscillator` controller, with at least 3 seeds each.
2. **Co-design.** Evolve body and brain together with `--mode codesign`, once with `--controller oscillator` and once with `--controller local`, at least 3 seeds each.
3. Plot the best-fitness curves of all four configurations on one figure, and save a GIF (and a `--showbody` picture) of the best robot from each run.

Together, these four make a 2×2 comparison: fixed vs. evolved body, crossed with closed-loop network vs. open-loop oscillator.

Questions to consider:

- **Brain vs. brain-and-body.** Does co-design beat evolving only a controller for a hand-designed body? Think about what each genome gives evolution control over, and how big a space it has to search. Is the comparison fair? (Who designed `biped`, and what did they already know about the task?)
- **Neural network vs. oscillator.** On `biped`, the oscillator has roughly 70× fewer genes than the network and can't sense anything. Which does better, and why might the "dumber" controller do so well? Does the answer change when the body is evolving too? What can a closed-loop controller do that an open-loop one fundamentally can't, and does walking on flat ground need it?
- **Where does the gait live?** Look at the co-designed bodies across seeds. Do they converge on similar shapes or very different ones? Are there recurring motifs (legs, a rigid spine, actuators concentrated in one place, empty space)? What does that suggest about how much of a gait is in the body rather than the brain?

**Note:** good runs on Walker-v0 get close to its ceiling (about 10.5). If your curves converge near there, compare them by how quickly they get there, not just by final fitness.

---

### REQUIRED #3: Pick ONE aspect to explore in depth

For the last part, pick **one** of the directions below, or come up with your own. Whatever you pick, state the question you're trying to answer, write down your prediction, and run enough seeds to trust the answer. Each of these is open-ended — there's no single right answer.

**A. Evolve only the body.** `--mode morphology` holds the brain fixed (a traveling wave whose phases depend only on each actuator's column) and evolves only the body. Together with Part 2, that completes the picture: brain only, body only, both. Which matters more for this task? What kind of bodies does evolution build to make use of a brain it can't change?

**B. Centralized vs. distributed brains.** `--controller local` also works on a fixed body. On `biped`, compare one network for the whole body (`neural`) against a small network copied into every voxel (`local`). How do their genome sizes, fitness, and gaits compare? Then look harder at the local network's clock. How much of its success comes from the clock, and how much from sensing? (Try removing the clock, or the local sensing, from `LocalController.local_inputs()`.)

A step further: can you get rid of the clock altogether with a **hybrid** of central and distributed control? Replace each copy's clock inputs with a small signal broadcast by one *central* network. To keep the hybrid co-designable, the central network should only read quantities whose size doesn't depend on the body, such as the center-of-mass velocity and the voxels' average strain. Two versions are worth comparing: a feedforward central network, and a recurrent one that also reads its own previous broadcast. Where does the rhythm come from when there's no clock? What timescale does evolution choose for it? Watch the GIFs every step, not every 4th (`frame_every=1` in `run_episode()`), or you may miss what's going on.

**C. Human designer vs. evolution.** Design your own body by hand (copy `example_body.txt`), using what you learned from watching the GIFs in Part 2, and evolve a controller for it. Can you beat what co-design finds? What did designing it teach you about which body features matter?

**D. Generalization to new terrain.** Take your best robots from Part 2 and test them, without re-evolving, on other terrains with `sim.py --env` (e.g. `UpStepper-v0`, `ObstacleTraverser-v0`, `BridgeWalker-v0`, `Hurdler-v0`). Which generalizes best? Is it the one with the highest Walker-v0 fitness? Or evolve on a harder terrain directly and compare the bodies. Does the closed-loop network have an advantage once the ground isn't flat? (Some EvoGym tasks reward something other than distance, such as `Jumper-v0` or `Carrier-v0`; read the [task docs](https://evolutiongym.github.io/) before interpreting those numbers.)

**E. Material costs.** Real robots pay for actuators in weight, power, and money. Add a penalty to the fitness for each actuator voxel (or a bonus for each empty one) and see how the evolved bodies change as the penalty grows. Is there a point where evolution builds mostly passive bodies with a few well-placed actuators?

**F. Sweep one parameter.** Pick one parameter that shapes the body or its rhythm, such as `--grid` (3–7) or `--period` (10–50), and sweep it in `codesign` mode (mean ± spread across seeds). What trade-off does it control? Or sweep an evolutionary-algorithm parameter, as in Projects 2–4, and ask whether bodies and brains respond to it differently.

You're encouraged to explore your own idea beyond these, as long as it's a genuine extension.

---

## What to Submit to Moodle

Submit a single **written report as a PDF** to Moodle.

### Title Page

The first page of your report should include:

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 5: Embodied Evolution II — Soft Robots)
- Date submitted
- Amount of time spent on this project
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body of your report into one section per assignment part. Each section should combine the relevant figures with a written discussion — a plot with no interpretation, or an interpretation with no supporting plot, is incomplete. Static frames from your GIFs, or a link to them, are welcome wherever behavior matters.

In what follows, I am going to mention the traditional path of required components. However, keep in mind that if you chose to meet the learning objectives in a different way, then your required parts might look different.

**Required Part 1 — Understand the Genomes**

- Your gene counts and what each genome lets evolution decide.
- Your answers on why co-design can't use the centralized network (and how the local one gets around it), and on unbuildable bodies.
- A fitness curve and a description of the evolved neural-controller gait on `biped`.

**Required Part 2 — Evolving the Brain vs. Evolving the Brain and Body**

- Best-fitness curves for all four configurations (at least 3 seeds each) on one figure.
- Pictures of the best robots, especially the co-designed bodies across seeds.
- Your discussion of the three questions: brain vs. brain-and-body, neural vs. oscillator, and where the gait lives.

**Required Part 3 — Explore One Aspect in Depth**

- State what you chose to investigate and the question you were trying to answer.
- Describe what you held fixed and what you varied, and show results against a baseline, with enough seeds to support your conclusion.
- Most importantly, explain what you learned. Did the results match your prediction?

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
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in connecting what you *see* the robots doing to the fitness numbers, and in reasoning about how behavior is divided between body and brain.

---

## Further Reading

- Bhatia, J., Jackson, H., Tian, Y., Xu, J., & Matusik, W. (2021). *Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots.* NeurIPS. ([project page](https://evolutiongym.github.io))
- Sims, K. (1994). *Evolving Virtual Creatures.* SIGGRAPH.
- Cheney, N., MacCurdy, R., Clune, J., & Lipson, H. (2013). *Unshackling Evolution: Evolving Soft Robots with Multiple Materials and a Powerful Generative Encoding.* GECCO.
- Pfeifer, R., & Bongard, J. (2006). *How the Body Shapes the Way We Think.* MIT Press.

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.
