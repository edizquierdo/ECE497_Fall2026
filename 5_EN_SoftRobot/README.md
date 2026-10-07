# Project 5: Embodied Evolution II — Soft Robots: Evolving Bodies, Not Just Brains

## Overview

In Project 4 the body was given to you: a two-wheeled vehicle with two light sensors, designed by hand back in Project 1. Evolution's only job was to find a brain for it. In this project, **the body itself becomes part of the genome.**

You'll work with soft robots in [Evolution Gym](https://evolutiongym.github.io) (EvoGym), a soft-body physics simulator widely used in research. An EvoGym robot is a small 2D grid of *voxels*, and each voxel is made of one of five materials: empty, rigid, soft, or one of two kinds of muscle that stretch and squeeze on command. Change the grid, and you've changed the animal. The task is simple: walk to the right, as far as you can.

The pipeline is the one you already know from Project 4: a flat genome, EvoTorch's `GeneticAlgorithm` with SBX crossover and Gaussian mutation, evolve → simulate → analyze. What's new is what the genome is allowed to describe. The project is organized around two problems:

- **(a) Evolve the brain for a fixed body.** The body is given (one of ours, or one you design by hand), and evolution searches only for its brain.
- **(b) Evolve the brain and the body together (co-design).** Evolution searches for a body *and* the brain that drives it, in a single genome.

Co-design raises a problem you haven't had to face before: **the brain has to work for whatever body evolution builds.** This project gives you four brains that solve it in different ways, along two choices: whether one network runs the whole body or a small copy runs each muscle (*global* vs. *local*), and whether the brain can feel what the body is doing (*closed-loop* vs. *open-loop*).

Your goal is not just to get a robot that walks, but to reason about *where the walking comes from*: how much of a gait lives in the brain, and how much in the body.

## Checkpoint, Due Date, and Presentations

- **Checkpoint — Friday, October 16, during class (2 PM).** Covers Required Parts 1 and 2. No reporting necessary for the checkpoint; it will involve running code on your laptop to demonstrate and showing figures (and animations!).
- **Report due — Monday, October 19, before class (2 PM).** Includes Required Part 3.
- **Presentations — Monday, October 19, during class (2 PM).**

---

## Learning Objectives

By completing this project, you will learn how to:

1. Encode a robot's *body* as part of a genome, and co-evolve body and brain in a single genome.

2. Compare evolving a brain for a fixed body (designed by you or by someone else) against evolving the body and brain together.

3. Design brains that work for any body, and compare global vs. local and open-loop vs. closed-loop control.

4. Reason about how behavior is divided between brain and body (*morphological computation*), using what you see the robots doing, not just their fitness.

---

## IMPORTANT NOTE

What follows below is ONE possible path for this project. However, you do NOT have to take this path. The required learning goals are fixed; the implementation and experiments are flexible. Take your own path. You are just as welcome to explore on your own, or to follow along.

As in Project 4, this assignment has only three REQUIRED components, and they are deliberately open-ended: you've done this kind of project a few times now, so the choices of what to run and what to compare are mostly yours.

---

## Background

### From evolving brains to evolving bodies

Everything you've evolved so far has been a *brain*: a vector of numbers decoded into network weights. Nothing stops us from decoding some of those numbers into a *body* instead. That's the idea behind Karl Sims' 1994 *Evolved Virtual Creatures*, and behind a large body of later work on evolving robot morphology. If intelligent behavior comes from the interaction of body, brain, and environment (the lesson of the Braitenberg vehicles in Project 1), then there's no reason to hold the body fixed while only optimizing the brain.

Two consequences you'll run into immediately:

1. **Not every genome is a robot.** A random grid of materials may fall apart into disconnected pieces, or contain no muscles at all. These genomes can't be simulated, so they get a fixed, very low fitness (`INVALID_FITNESS = -1` in `soft_robot.py`) and are removed by selection.
2. **The brain has to fit whatever body evolution builds.** The obvious brain, the one you'd write in Project 4, would take the simulator's observation of the whole body as input and give one command per muscle as output. But both of those sizes depend on the body: add a muscle and the network needs another output; add a voxel and the observation gets longer. A network evolved for one body can't even be *loaded* onto another, so it can't be co-evolved with a body that keeps changing. The four brains below are designed around this problem.

### Soft robots in EvoGym

Each voxel in a robot's grid holds one of five materials:

| Code | Material | Symbol | Color in animations |
|---|---|---|---|
| 0 | empty | `.` | — |
| 1 | rigid | `R` | black |
| 2 | soft | `S` | gray |
| 3 | horizontal muscle | `H` | orange |
| 4 | vertical muscle | `V` | blue |

Row 0 of the grid is the top of the robot; the last row rests on the ground. The simulator models each voxel as point masses at its corners, connected by springs. Rigid voxels have stiff springs, soft voxels have compliant ones, and muscle voxels have springs whose rest length the brain sets.

- **Commands:** one number per muscle, every simulation step: its target length as a fraction of its rest length, in `[0.6, 1.6]` (0.6 squeezes it, 1.0 leaves it at rest, 1.6 stretches it). Horizontal muscles change width, vertical ones change height.
- **Fitness:** how far the robot's center of mass moved to the right in one episode of **300 steps**. One voxel is 0.1 wide, so a fitness of 1.0 means 10 voxel widths, about two body lengths for a 5×5 robot.
- **No noise.** The simulation is deterministic: the same robot always does exactly the same thing. Unlike Project 4, there's no need to average over several episodes or to re-evaluate the best genome after evolution. The fitness `evolve.py` reports is the fitness.

### The four brains

Every brain is a small feedforward network, like the ones in Project 4. All four get a shared **clock** as input, the sin and cos of 2πt / period (one cycle every 25 steps by default), as a source of rhythm. They differ along two choices:

- **Local or global.** A *local* brain is one small network copied into every muscle, with the same weights in every copy; each copy computes the command for its own muscle. A *global* brain is one network for the whole body, which computes every muscle's command at once.
- **Open-loop or closed-loop.** An *open-loop* brain can't feel the body: its inputs don't depend on what the body is doing. A *closed-loop* brain also senses **strain**, how stretched each voxel is: the voxel's (width, height) relative to its rest size, minus 1, so 0 at rest, positive when stretched, negative when squeezed.

| | **Open-loop** | **Closed-loop** |
|---|---|---|
| **Local** (`--controller local`) | clock + the muscle's own position → its command | + strain of its own voxel and its four neighbors |
| **Global** (`--controller global`) | clock → one command per grid cell | + strain of every cell in the grid |

Choose the brain with `--controller local|global` and `--loop open|closed` (closed is the default). In more detail, from simplest to most complex:

**1. Local, open-loop** (`--controller local --loop open`). Each muscle's copy of the network gets 4 inputs: the clock (2), and the muscle's own (x, y) position in the body, each scaled to [0, 1] (x from left to right, y from bottom to top). It outputs that muscle's command. Every muscle runs the same function of time and position, so the only reason two muscles do different things is that they sit in different places. *Genome: 49 genes (8 hidden units), whatever the body.*

**2. Local, closed-loop** (`--controller local --loop closed`). The same, plus 10 sensory inputs: the strain of its own voxel and of the voxels above, below, left, and right of it (0 where there's no voxel). Now each muscle can react to what its neighborhood is doing. *Genome: 129 genes, whatever the body.*

**3. Global, open-loop** (`--controller global --loop open`). One network, whose only input is the clock (2), with **one output per cell of the grid** (25 on a 5×5 grid). Each muscle takes the output of the cell it sits in; outputs for cells that aren't muscles are ignored. Because every cell has its own output weights, neighboring muscles can do completely different things. *Genome: 473 genes on a 5×5 grid (16 hidden units).*

**4. Global, closed-loop** (`--controller global --loop closed`). The same, plus the strain of every cell in the grid as input (2 per cell, 0 for empty cells): 52 inputs on a 5×5 grid. It sees the whole body's shape at once. *Genome: 1,273 genes on a 5×5 grid.*

**Why these fit any body.** A local brain doesn't care how many muscles there are: there's just one more copy. A global brain is wired to the *grid*, not the body: input and output number *k* always mean "cell *k*", whatever happens to be in that cell. Adding, removing, or changing a voxel only changes which outputs get used and which strain inputs are 0. (The size of a global brain does depend on the size of the grid, so it's different for a 4×6 body than for a 5×5 one.)

### How a body is encoded

Each voxel gets five genes, one per material, and takes whichever material's gene is largest (an *argmax*). Small mutations usually leave a voxel's material unchanged, and only when two of its genes swap order does the material flip. In co-design, a 5×5 grid has 125 material genes, followed by the brain's weights.

### The evolution modes

| `--mode` | Body | Brain | Genome |
|---|---|---|---|
| `control` | fixed (`--body`) | evolved | the brain's weights |
| `codesign` | evolved, on a `--grid` × `--grid` grid | evolved | 5 material genes per voxel, then the brain's weights |
| `morphology` | evolved | *fixed*: a traveling wave (each muscle follows a sine wave, with a phase set by its column) | 5 material genes per voxel |

Modes `control` and `codesign` are the core of this project. `morphology`, which evolves only the body, is there for you to explore in Part 3 if you want to.

---

## Project Structure

| File | Purpose |
|------|---------|
| `soft_robot.py` | Everything about bodies and the world: material codes, preset bodies, decoding and validating bodies, measuring voxel strain, creating the EvoGym environment, running one episode, plotting a body |
| `neural_controller.py` | The brains: `LocalController` and `GlobalController` (each open- or closed-loop), and the fixed `TravelingWaveController` used by `morphology` mode |
| `evolve.py` | Runs evolution in any mode. `decode_body_from_genome()` and `decode_controller()` turn a genome into a robot |
| `sim.py` | Replays a saved genome: fitness, body picture, behavior traces, animation, or a live window |
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
- Occasionally an extreme body or brain makes the simulation unstable, and EvoGym prints `SIMULATION UNSTABLE... TERMINATING`. It ends that episode with a penalty, so the robot scores badly and evolution carries on.

---

## Bodies

### The preset bodies

| `--body` | Description |
|---|---|
| `biped` | Two legs and a torso, every voxel a horizontal muscle |
| `worm` | A flat, two-row strip: vertical muscles on top of horizontal ones |
| `block` | A solid soft block with a rigid top and muscles along the bottom |
| `tripod` | Three legs hanging from a rigid back, each leg mixing both kinds of muscle |

See any of them with `python sim.py --body worm --showbody --gif worm.gif` (with a random, unevolved brain).

### Designing a body by hand

A body file is plain text, one row per line, material codes separated by spaces, row 0 at the top. Lines starting with `#` are comments. A body must be **connected** (all non-empty voxels touching edge to edge) and contain **at least one muscle** (3 or 4). It doesn't have to be 5×5. See `example_body.txt`:

```
1 1 1 1 1
2 3 3 3 2
2 4 0 4 2
2 4 0 4 2
3 3 0 3 3
```

Preview a body before evolving a brain for it:

```bash
python sim.py --body my_body.txt --showbody --gif my_body.gif
```

---

## Running Evolution

```bash
# (a) Fixed body, evolved brain
python evolve.py --mode control --body biped --controller local --loop closed --verbose --vizperf --output biped_local.npy
python evolve.py --mode control --body my_body.txt --controller global --loop open --verbose --vizperf --output mine_global.npy

# (b) Co-design on a 5x5 grid
python evolve.py --mode codesign --controller local --loop closed --verbose --vizperf --output codesign.npy
```

The first line `evolve.py` prints tells you what's being evolved and how many genes the genome has. For `codesign` (and `morphology`), the best body is printed at the end as a text picture (`R` rigid, `S` soft, `H`/`V` muscles, `.` empty). Evaluations run in parallel across your CPU cores (`--workers max`), each robot in its own process.

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `control` | `control`, `codesign`, or `morphology` |
| `--body` | `biped` | [control] preset name, or path to a `.txt`/`.npy` body |
| `--grid` | `5` | [codesign/morphology] side length of the voxel grid |
| `--controller` | `local` | `local` or `global` |
| `--loop` | `closed` | `open` or `closed` |
| `--hidden` | `8` local / `16` global | hidden layer size(s), e.g. `--hidden 32` or `--hidden 16 16` |
| `--activation` | `tanh` | hidden-layer activation |
| `--period` | `25` | simulation steps per clock cycle |
| `--env` | `Walker-v0` | EvoGym task |
| `--duration` | `300` | episode length in steps |
| `--popsize`, `--gens` | `50`, `50` | population size, generations |
| `--mut_stdev` | `0.2` | Gaussian mutation stdev |
| `--tournament_size`, `--eta`, `--no-elitism` | `3`, `20`, elitism on | GA settings, as in Project 4 |
| `--seed_genome`, `--seed_noise` | — | start from a saved genome |
| `--workers` | `max` | parallel workers (`max`, an integer, or `none`) |
| `--seed` | — | random seed |
| `--verbose`, `--vizperf` | off | print per-generation stats / plot fitness curves |
| `--output` | — | save best genome (`.npy`) **and its settings** (`.json`, same name) |
| `--fitness_output` | — | save per-generation `best`/`avg`/`worst` fitness to `.npz` |

---

## Running Simulations

`evolve.py --output best.npy` also writes `best.json` with every setting needed to rebuild the robot, and `sim.py` reads it automatically, so you don't need to re-type matching flags.

```bash
python sim.py --genome codesign.npy --showbody --viztraces   # fitness, body picture, behavior traces
python sim.py --genome codesign.npy --gif codesign.gif       # save an animation
python sim.py --genome codesign.npy --render                 # watch it live
python sim.py --genome codesign.npy --env UpStepper-v0       # try it on a task it wasn't evolved for
```

`--viztraces` plots, over time, the center of mass's x position (distance walked), its y position (height, where hops and falls show up), and a heatmap of every muscle's command, which shows what the brain actually chose to do.

---

## Understanding Fitness Scores

On Walker-v0 with the default 300 steps, fitness is the distance walked in units of 10 voxel widths:

- **≈ 0 or negative:** twitching in place, or drifting backwards.
- **1–3:** a slow shuffle or crawl.
- **3–7:** a real gait, usually with a clear repeating stride in the animation.

At the defaults (`--popsize 50 --gens 50`), most runs end up somewhere between 3 and 7, and the fitness curves are usually **still rising at generation 50**: if you have the time, more generations will help. The spread across seeds can be large, and that spread is itself worth reporting. A run that's still below 1 after 50 generations isn't necessarily a bug: run another seed before you start debugging.

Walker-v0 also has a ceiling: a robot that reaches the end of the track (x = 9.9) gets a +1 bonus and its episode ends early, for a fitness of about 10.5. At 300 steps that's out of reach for anything we've seen evolve. If you raise `--duration`, good robots start hitting it, and then you'll need to compare *how fast* they get there rather than their final fitness.

## Performance

A default run (`--popsize 50 --gens 50`, 300 steps) takes about **3.5 minutes on a 16-core laptop**, and roughly twice that on 8 cores. Parts 1 and 2 together are 15 runs, so budget about an hour of computer time on 16 cores (two on 8), and start early. Use `--gens 10` while you're getting a feel for things, and let longer runs work in the background while you analyze earlier ones. The physics dominates the cost; the brains are negligible next to it, so all four take about the same time.

## Performing Parameter Studies

As in Project 4, you're expected to write your own sweep-and-plot scripts. `run_evolution(cfg, ...)` in `evolve.py` runs one evolution and returns the fitness curves and best genome; `make_config()` shows what goes in `cfg`. For quick comparisons, save each run's curve with `--fitness_output` and plot them together afterwards, the same way you did in Project 4.

---

## Tips

- **Look at the animations.** A fitness number tells you *how far*, not *how*. Evolution is very good at finding gaits you wouldn't have designed: dragging, hopping, rolling, inch-worming. Many are only obvious once you watch them.
- **Before running each experiment, write down your prediction for the result.**
- Use `--seed` when debugging, then run several seeds before drawing conclusions. Evolved bodies in particular vary a lot from seed to seed.
- Save `--output` and `--fitness_output` for every run you might want to look at later, so you won't need to re-run evolution to make a figure.
- A body is only as good as the brain it's paired with. Before concluding that a body "doesn't work", ask whether it could with a different brain.

---

## Assignment

For Parts 1 and 2, pick whichever brain you want (see *The four brains*). You can stick with one throughout, or compare several. If you compare, say why you chose the ones you did.

### REQUIRED #1: Evolve a Brain for a Fixed Body

1. Pick one of the preset bodies, and evolve a brain for it.
2. Design a body of your own by hand, and evolve a brain for it.
3. Run **5 evolutionary runs** (5 different seeds) for each of the two bodies, with the same settings, and compare.

Show the evolutionary progress (fitness over generations) for both bodies, and what the best robots actually do (animations or frames from them). What did you design your body to do, and did evolution find a brain that does it? Why do you think one body did better than the other?

---

### REQUIRED #2: Evolve the Brain and the Body Together

Co-design a body and its brain with `--mode codesign`, at least **5 evolutionary runs** (5 different seeds).

Show the evolutionary progress of all five runs, and the five final designs with an animation of each. Compare the results and discuss. Some things you might think about: do different runs converge on similar bodies or very different ones? Are there recurring motifs? How do the co-designed robots compare with the fixed bodies from Part 1, including the one you designed? What does that suggest about how much of a gait lives in the body rather than the brain?

---

### REQUIRED #3: Pick ONE aspect to explore in depth

For the last part, pick **one** of the directions below, or come up with your own. Whatever you pick, state the question you're trying to answer, write down your prediction, and run enough seeds to trust the answer. Each of these is open-ended — there's no single right answer.

**A. Evolve only the body.** `--mode morphology` holds the brain fixed (a traveling wave whose phases depend only on each muscle's column) and evolves only the body. Together with Parts 1 and 2, that completes the picture: brain only, body only, both. Which matters more for this task? What kind of bodies does evolution build to make use of a brain it can't change?

**B. Explore the four brains.** Compare global vs. local, or open-loop vs. closed-loop, or all four, on a fixed body, in co-design, or both. Does sensing help? Does it help more in some settings than others? Does a brain with far more genes do better or worse, and why? Then look harder at the clock. How much of a closed-loop brain's success comes from the clock, and how much from sensing? (Try removing the clock inputs in `neural_controller.py`.)

A step further: can you get rid of the clock altogether with a **hybrid** of global and local control? Replace each local copy's clock inputs with a small signal broadcast by one *central* network. To keep the hybrid co-designable, the central network should only read quantities whose size doesn't depend on the body, such as the center-of-mass velocity and the voxels' average strain. Two versions are worth comparing: a feedforward central network, and a recurrent one that also reads its own previous broadcast. Where does the rhythm come from when there's no clock? What timescale does evolution choose for it? Watch the animations every step, not every 4th (`frame_every=1` in `run_episode()`), or you may miss what's going on.

**C. Designer and evolution together.** Use what co-design found in Part 2 to design a better body by hand, and evolve a brain for it. Can you beat co-design? Can you beat your Part 1 body? What did co-design teach you about which body features matter?

**D. Generalization to new terrain.** Take your best robots from Parts 1 and 2 and test them, without re-evolving, on other terrains with `sim.py --env` (e.g. `UpStepper-v0`, `ObstacleTraverser-v0`, `BridgeWalker-v0`, `Hurdler-v0`). Which generalizes best? Is it the one with the highest Walker-v0 fitness? Or evolve on a harder terrain directly and compare the bodies. Does a closed-loop brain have an advantage once the ground isn't flat? (Some EvoGym tasks reward something other than distance, such as `Jumper-v0` or `Carrier-v0`; read the [task docs](https://evolutiongym.github.io/) before interpreting those numbers.)

**E. Material costs.** Real robots pay for muscles in weight, power, and money. Add a penalty to the fitness for each muscle voxel (or a bonus for each empty one) and see how the evolved bodies change as the penalty grows. Is there a point where evolution builds mostly passive bodies with a few well-placed muscles?

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

Organize the body of your report into one section per assignment part. Each section should combine the relevant figures with a written discussion — a plot with no interpretation, or an interpretation with no supporting plot, is incomplete. Animations can't go in a PDF, so include static frames, and a link to the animations themselves.

In what follows, I am going to mention the traditional path of required components. However, keep in mind that if you chose to meet the learning objectives in a different way, then your required parts might look different.

**Required Part 1 — Evolve a Brain for a Fixed Body**

- Which preset you chose, the body you designed (picture from `--showbody`) and what you designed it to do, and which brain(s) you used.
- Fitness over generations for 5 runs on each body.
- What the best robots do, and your comparison of the two bodies.

**Required Part 2 — Evolve the Brain and the Body Together**

- Fitness over generations for at least 5 co-design runs.
- The five final designs, with frames from (and a link to) an animation of each.
- Your comparison and discussion, including against Part 1.

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
