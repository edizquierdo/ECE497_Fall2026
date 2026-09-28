# Project 4: Embodied NeuroEvolution I — Braitenberg Phototaxis

## Overview

In this project you will put together the three pieces you have built so far: the **Braitenberg vehicle** from Project 1, the **evolutionary algorithm** from Project 2, and the **evolvable neural network** from Project 3.

In Project 1 you designed the vehicle's "brain" by hand: two wires, crossed. Here, you will replace those wires with a small neural network, and let evolution find the network's weights. The network receives the two sensor readings as input and produces the two motor commands as output. Nobody tells it how to steer — the only thing evolution sees is how close to the light each vehicle ends up.

The pipeline is the same one you used for XOR in Project 3 (flat genome of weights → EvoTorch genetic algorithm → best network). What changes is where fitness comes from. It is no longer a lookup in a truth table, but the result of letting a body with sensors and motors interact with its environment for a couple of thousand steps. That change has consequences you will get to explore: fitness becomes noisy, there is no known "perfect" score, and evolution finds strategies you would probably never have hand-designed.

Your goal is not simply to run the code, but to understand what the evolved controllers are doing, how they compare to the one you designed yourself, and what the evolutionary process is (and isn't) telling you.

## Checkpoints and Final Due Date

- **Checkpoint — Friday, October 2, during class (2 PM).** Covers Required Parts 1 and 2. No reporting necessary for the checkpoint; it will involve running code on your laptop to demonstrate and showing figures.
- **Final due date — Tuesday, October 6, before class (2 PM).** Includes Required Part 3 and the report. There will be demos during class.

---

## Learning Objectives

By completing this project, you will learn how to:

1. Apply neuroevolution to an embodied robot control problem, where fitness comes from behavior in a closed sensor–motor loop rather than from a fixed dataset.

2. Reason about noisy fitness evaluations: why they arise, how they affect the numbers evolution reports, and how to measure an evolved controller's performance reliably.

3. Compare an evolved controller against a hand-designed one on the same body and task, and identify the strategies each one uses.

4. Design and run a systematic experiment on one aspect of the evolutionary or embodied setup, across multiple independent runs.

---

## IMPORTANT NOTE

What follows below is ONE possible path for this project. However, you do NOT have to take this path. The required learning goals are fixed; the implementation and experiments are flexible. Take your own path. You are just as welcome to explore on your own, or to follow along.

As in Project 3, this assignment has only three REQUIRED components, so that you have time to explore. The REQUIRED components are labeled clearly — everything else is OPTIONAL.

---

## Background

### From XOR to a Body

In Project 3, evaluating a genome meant running four inputs through the network and counting correct signs. Here, evaluating a genome means:

1. Load the genome into the network.
2. Place a vehicle at a random spot 10 units away from the light, facing a random direction.
3. Repeat **sense → think → move** for 2000 steps, with the network doing the "think".
4. Score the episode by how close the vehicle stayed to the light.
5. Repeat for a few episodes (`--episodes_per_eval`, default 5) and average.

The network never sees the fitness function, and the fitness function never looks inside the network. Everything evolution learns about steering, it learns through the body.

### Fitness

At each time step, the vehicle is rewarded for proximity to the light as `1 / (1 + d)`, where `d` is its distance to the light. Fitness is that reward averaged over all steps and all episodes. This is bounded in (0, 1], with higher always better, like in Project 3. But there are two important differences from XOR:

- **There is no reachable "perfect" score.** A vehicle would have to start on top of the light to score 1.0. Every vehicle starts 10 units away and needs time to get there, so the best achievable fitness is well below 1.0. This is why, unlike Project 3, `evolve.py` has no early stopping and every run uses all of its generations.
- **Fitness is noisy.** Starting positions, headings, and motion noise are random, so evaluating the *same* genome twice gives two different numbers. The best fitness reported in a generation is partly a measure of how good that genome is, and partly a measure of how lucky it was. EvoTorch re-evaluates the surviving parents every generation, so a lucky genome does not stay on top forever — which also means that, unlike Project 3, the best fitness curve can go *down* from one generation to the next. After evolution, `evolve.py` re-evaluates the best genome on 100 new episodes (`--final_evals`) to give you a more trustworthy number.

To give you a sense of scale, with the default settings: a vehicle that never moves scores about 0.09; the hand-wired crossed vehicle from Project 1 scores about 0.2; a typical evolved controller scores about 0.6.

### The Neural Vehicle

`NeuralVehicle` (in `braitenberg.py`) extends the Project 1 `Vehicle` and replaces only its `think()` method: the two sensor readings go through a `NeuralController` (in `neural_controller.py`) and its two outputs become the motor commands. Sensing, moving, and noise are inherited unchanged.

The `NeuralController` is the same network you used for XOR in Project 3, with two changes: it has 2 outputs (left and right motor) instead of 1, and its output layer goes through Tanh, so motor commands are bounded to [-1, 1].

**Note on motor range:** The hand-wired `Vehicle` sets its motors directly from its sensors, which are always in (0, 1]. That means it can never drive backwards, and it can never stop — it is always moving. The `NeuralVehicle`'s motors range over [-1, 1], so an evolved controller *can* reverse, stop, spin in place, drive faster than the hand-wired vehicle usually does, and turn up to twice as sharply. Keep this in mind in Part 2: when the evolved controller does better, some of that may be a better strategy, and some may just be a wider range of available actions.

#### Vehicle Configuration

| Parameter | Description |
|-----------|-------------|
| `angle_offset` | Angular separation between sensors (radians). π/2 places sensors at 90° on each side. |
| `turn_gain` | How strongly the motor difference steers the vehicle. Larger values produce sharper turns. |
| `noise_stdev` | Standard deviation of Gaussian noise added to orientation at each step. |

These are passed as command-line arguments (`--angle_offset`, `--turn_gain`, `--noise`) to both `sim.py` and `evolve.py`.

### What Changed from Project 1

The vehicle is the same, but the environment is set up differently:

| | Project 1 | Project 4 |
|---|---|---|
| Light position | `(distance, 0)` | origin `(0, 0)` |
| Starting position | origin, same every repetition | random point on a circle of radius `distance` around the light |
| Starting heading | facing the light, same every repetition | random |
| Steps per episode | 5000 | 2000 |

The random starts are deliberate. If every evaluation started from the same position and heading, evolution could "solve" phototaxis by memorizing one good path, rather than learning to steer toward the light from wherever it happens to be.

**Note on sensor labels.** In Project 1's `braitenberg.py`, the left and right labels were mixed up in two places. The sensor called "right" was actually mounted on the vehicle's *left* (angles are measured counterclockwise, so `orientation + angle_offset` is to the left of the heading), and the turning equation was mirrored the same way (a faster *left* wheel turned the vehicle left). The two mistakes cancel out, so every behavior you saw in Project 1 was correct — the crossed vehicle really did seek the light — but the names were backwards. This project fixes both: the left sensor is on the left, and a faster left wheel turns the vehicle right, as with a real differential-drive robot. `move()` now uses `turn_gain × (right_motor − left_motor)`. If you bring code over from Project 1, keep this in mind.

---

## Project Structure

| File | Purpose |
|------|---------|
| `neural_controller.py` | The neural network controller (2 sensors → hidden → 2 motors), plus `genome_size()` and a batched forward pass. |
| `braitenberg.py` | The Project 1 `Vehicle` (with both wirings), `NeuralVehicle`, `Light`, and `simulate_population()` — a batched version of the sense → think → move loop used during evolution. |
| `evolve.py` | Defines the fitness function and runs neuroevolution with EvoTorch. |
| `sim.py` | Simulates an evolved controller, or a hand-wired one, and plots its behavior. |
| `requirements.txt` | Pinned dependency versions for this project's virtual environment. |
| `README.md` | Project documentation. |

---

## Installation

The project requires:

- Python 3.8 or newer
- NumPy
- Matplotlib
- PyTorch
- EvoTorch

All four are listed with tested version ranges in `requirements.txt`, so install them together into a project-specific virtual environment rather than into your system Python.

**Create and activate a virtual environment**, from inside this project's folder:

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

You'll know it worked if your terminal prompt now starts with `(venv)`. Do this every time you come back to work on the project, before running any of the commands below — you'll need to `activate` again in each new terminal session (no need to `venv` again; that step is one-time).

**Install the pinned dependencies:**

```bash
pip install -r requirements.txt
```

**To leave the environment** when you're done:

```bash
deactivate
```

If you'd rather use `conda`, that's fine too — just create an environment with a matching Python version and run the same `pip install -r requirements.txt` inside it.

---

## Running the Neuroevolution

To run the default neuroevolution experiment and save the best controller,

```bash
python evolve.py --verbose --vizperf --output best_genome.npy
```

A default run takes about a minute on a recent laptop. At the end, it prints two numbers: the best fitness in the final generation (from only 5 episodes, and usually a little lucky), and that same genome's fitness re-evaluated on 100 new episodes. **Use the re-evaluated number when you report how good a controller is.**

Useful command-line options include:

| Option | Description | Default |
|--------|-------------|---------|
| `--hidden` | Number of hidden neurons in a single hidden layer | `8` |
| `--hidden_sizes N [N ...]` | List of hidden-layer sizes, e.g. `--hidden_sizes 16 16`. Overrides `--hidden` when given. Pass `--hidden_sizes` with no numbers for a network with no hidden layer at all | `None` |
| `--activation` | Hidden layer activation (`tanh`, `sigmoid`, or `relu`). The output layer is always Tanh | `tanh` |
| `--popsize` | Population size | `50` |
| `--gens` | Number of generations | `100` |
| `--mut_stdev` | Gaussian mutation standard deviation | `0.5` |
| `--tournament_size` | Tournament size for SBX crossover | `3` |
| `--eta` | Distribution index for SBX crossover | `20` |
| `--no-crossover` | Disable SBX crossover, running a mutation-only GA | crossover on |
| `--no-elitism` | Disable elitism | elitism on |
| `--init_bounds LOW HIGH` | Initial genome sampling bounds | `-1.0 1.0` |
| `--episodes_per_eval` | Episodes (random starts) averaged per fitness evaluation | `5` |
| `--duration` | Simulation steps per episode | `2000` |
| `--distance` | Starting distance from the light | `10.0` |
| `--angle_offset` | Angular separation between sensors (radians) | `pi/2` |
| `--turn_gain` | Turn gain for steering | `0.1` |
| `--noise` | Motion noise standard deviation | `0.1` |
| `--final_evals` | Episodes used to re-evaluate the best genome after evolution | `100` |
| `--vizperf` | Plot fitness over generations | `False` |
| `--verbose` | Print per-generation statistics to the console | `False` |
| `--seed` | Random seed for reproducibility | `None` |
| `--output FILE` | Save the best evolved genome, e.g. `best_genome.npy` | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves later without re-running evolution | `None` |
| `--seed_genome FILE` | Seed the initial population around a genome saved by a previous `--output` run (must match `--hidden`/`--hidden_sizes`) | `None` |
| `--seed_noise` | Stdev of the perturbation applied to `--seed_genome` copies | `0.05` |

For example,

```bash
python evolve.py --hidden 16 --popsize 100 --gens 200 --vizperf
python evolve.py --episodes_per_eval 1 --verbose
python evolve.py --noise 0.3 --output noisy_genome.npy
```

---

## Running Simulations

`sim.py` lets you watch a controller drive. It can run either an evolved network or the hand-wired Project 1 vehicle:

```bash
python sim.py --genome best_genome.npy --viztraces --vizdist --scores
python sim.py --controller crossed --viztraces --vizdist --scores
```

When you give both commands the same `--seed`, the two controllers get exactly the same starting positions, headings, and motion noise, so you can compare their trajectories side by side:

```bash
python sim.py --genome best_genome.npy --seed 7 --reps 20 --viztraces --scores
python sim.py --controller crossed      --seed 7 --reps 20 --viztraces --scores
```

The fitness `sim.py` prints (`--scores`) is the same measure `evolve.py` uses, so the numbers are directly comparable.

Useful command-line options include:

| Option | Description | Default |
|--------|-------------|---------|
| `--controller` | `neural` (evolved network), or the hand-wired `crossed` or `direct` wiring from Project 1 | `neural` |
| `--genome` | Path to a `.npy` genome from `evolve.py --output`. If omitted with `--controller neural`, a random, unevolved network is used | `None` |
| `--hidden` / `--hidden_sizes` / `--activation` | Network architecture — must match what the genome was evolved with | `8` / `None` / `tanh` |
| `--duration` | Number of simulation steps | `2000` |
| `--reps` | Number of independent repetitions | `5` |
| `--distance` | Starting distance from the light | `10.0` |
| `--angle_offset` | Angular separation between sensors (radians) | `pi/2` |
| `--turn_gain` | Turn gain for steering | `0.1` |
| `--noise` | Motion noise standard deviation | `0.1` |
| `--viztraces` | Plot vehicle trajectories | off |
| `--vizdist` | Plot distance to the light over time (mean ± 1 std across reps) | off |
| `--scores` | Print the average fitness | off |
| `--seed` | Random seed for reproducibility | `None` |
| `--save DIR` | Save figures (PNG) and the recorded distances (`distances.npy`, shape `reps × duration`) to `DIR` instead of opening windows | off |

**Important:** the network architecture must match what the genome was evolved with, or it won't load. The vehicle settings (`--angle_offset`, `--turn_gain`, `--noise`, `--distance`, `--duration`) don't *have* to match — but a controller tested in a different environment than the one it evolved in may behave very differently. (That can be an interesting experiment in itself.)

---

## Performing Parameter Studies

There's no `study.py` in this project. In Project 3, `study.py` was a worked example of the pattern: sweep one parameter, repeat several times per value to average out randomness, save the results, and plot the mean ± std. From here on, you are expected to write that kind of script yourself. Project 3's `study.py` is a good starting point to adapt: `run_neuroevolution()` in this project's `evolve.py` has the same name and a very similar signature.

For a quick comparison without a full script, save each run's fitness curve with `--fitness_output`, then reload and plot them together:

```bash
python evolve.py --hidden 2 --seed 1 --fitness_output hidden2_s1.npz
python evolve.py --hidden 8 --seed 1 --fitness_output hidden8_s1.npz
```

```python
import numpy as np
import matplotlib.pyplot as plt

for label, path in [("hidden=2", "hidden2_s1.npz"), ("hidden=8", "hidden8_s1.npz")]:
    plt.plot(np.load(path)["best"], label=label)
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.legend()
plt.show()
```

When you compare evolved controllers, remember that the best fitness *during* evolution is noisy. For a fair comparison, re-evaluate each final genome with `evaluate_genome()` from `evolve.py` (that is what `evolve.py` prints at the end), or with `sim.py --scores --reps 50`.

### IMPORTANT REMINDER

When you are doing a parameter sweep, remember to run it first with a small number of repetitions: 3–5 seeds per value or so. That way you can get an idea of the general shape. But remember that the shape will be very noisy. That noise is likely NOT REAL. Then, give yourself some time to repeat the same experiment with more repetitions — 10 or more if you can. A default run takes about a minute, so a sweep of 5 values × 10 seeds is under an hour. Let your laptop sit and work on it!

---

## Understanding the Components

### Neural Controller (`neural_controller.py`)

`NeuralController` is a feedforward network:

```
Input (2 sensors)  →  Linear(2, hidden)  →  Tanh  →  Linear(hidden, 2)  →  Tanh  →  Output (2 motors)
```

The genome is a flat vector of all weights and biases, in PyTorch's parameter order (`W1`, `b1`, `W2`, `b2`) — the same layout as in Project 3. `genome_size(hidden=...)` computes the genome length directly from the architecture.

`batched_forward()` runs the whole population's networks at once, using the same trick as Project 3's `make_fitness_fn`: it slices each individual's weight matrices out of the batch of genomes and applies them with one batched matrix multiplication per layer.

### Vehicles and Simulation (`braitenberg.py`)

`Vehicle`, `NeuralVehicle`, and `Light` simulate one vehicle at a time, with the same `sense()` → `think()` → `move()` methods as in Project 1. `sim.py` uses these.

`simulate_population()` is the same physics, written with tensors so that every vehicle of every individual in a generation advances together in one step. `evolve.py` uses it; without it, a single evolution run would take about 30 times longer. If you change the physics (for example, add sensor noise or a second light), make the same change in both places — `Vehicle` so that `sim.py` shows it, and `simulate_population()` so that evolution sees it.

### Evolution (`evolve.py`)

- `make_fitness_fn()` returns the fitness function: it simulates each individual for `--episodes_per_eval` episodes and computes the average proximity reward. The reward is computed from the distances returned by `simulate_population()` in one clearly marked line — this is the place to change if you want a different fitness function.
- `run_neuroevolution()` sets up EvoTorch's `GeneticAlgorithm` with SBX crossover and Gaussian mutation, runs it for `--gens` generations, and returns the fitness curves and the best genome.
- `evaluate_genome()` re-evaluates one genome on many new episodes.

---

## Tips

- Start by running the default configuration before changing any parameters, then watch the result:
  ```bash
  python evolve.py --verbose --vizperf --output best_genome.npy
  python sim.py --genome best_genome.npy --viztraces --vizdist --scores
  ```
- Read the source code carefully before making modifications.
- **Before running each experiment, write down your prediction for the result.** A written prediction makes the genuinely surprising results — usually the most interesting ones to discuss — much easier to spot.
- Use `--seed` for reproducibility when debugging.
- Change one parameter at a time to isolate its effect.
- Always look at the trajectories (`sim.py --viztraces`), not just the fitness number. Two controllers with similar fitness can behave very differently.

---

## Assignment

### REQUIRED #1: Understand the Neural Controller and the Embodied Fitness

Read `neural_controller.py`, `braitenberg.py`, and `evolve.py`, and answer the following questions before running any experiments:

- How many total parameters (weights and biases) does a network with 8 hidden neurons have? Count them by layer. Then check your count against `genome_size(hidden=8)`.
- `neural_controller.py` applies Tanh after *both* the hidden layer and the output layer. These two uses of Tanh serve different purposes — what is each one doing? (Hint: one is about being able to represent nonlinear functions at all; the other is about the physical range of a motor command.)
- What fitness would a vehicle get if it never moved at all? Roughly what fitness would a vehicle get if it drove straight to the light at top speed and stopped there? (The top speed is `vel_gain × 1 = 1/50` units per step.) What does this tell you about the range of fitness values you should expect to see, and why `evolve.py` has no early stopping?
- In Project 3, "converged" meant reaching fitness 1.0. How would you define convergence for this task?
- Why does fitness average over several episodes with random starting positions and headings? What do you think would happen with `--episodes_per_eval 1`?

Then run the default configuration, and watch the result:

```bash
python evolve.py --verbose --vizperf --output best_genome.npy
python sim.py --genome best_genome.npy --viztraces --vizdist --scores
```

Compare the best fitness reported during evolution with the re-evaluated fitness printed at the end. Which one is higher, and why? Does the best fitness curve ever go down? Why is that possible here, when it wasn't in Project 3?

#### OPTIONAL: Watch an unevolved network

Run `sim.py` without `--genome` to see what a random network does, and compare its fitness against your answer above for a vehicle that never moves.

---

### REQUIRED #2: Evolved vs. Hand-Designed, Head to Head

This is the heart of the project: the same body, the same task, two very different design processes.

1. Evolve controllers with at least 5 different seeds (all other settings the same), saving each genome with `--output`. Record each one's re-evaluated fitness.
2. Simulate the hand-wired crossed vehicle with `sim.py --controller crossed --scores --reps 50`, and compare its fitness against the distribution of your evolved controllers.
3. Pick at least one evolved controller and compare its trajectories and distance-over-time plots against the crossed vehicle's, **using the same `--seed`** so both start from identical conditions.

Questions to consider:

- Does evolution reliably find a controller better than the hand-designed one? How much do the 5 runs vary?
- Do evolved controllers follow paths similar to the crossed vehicle? What "strategies" can you identify — e.g., driving backwards, stopping at the light, turning sharply at the start? Do different seeds find different strategies?
- What happens when each vehicle reaches the light?
- How much of the evolved controller's advantage comes from a smarter strategy, and how much from having a wider motor range (see *Note on motor range* above)? How could you design a comparison that separates the two?

---

### REQUIRED #3: Pick ONE aspect to explore in depth

For the last part, pick **one** of the directions below, or come up with your own. Whatever you pick, formulate the question you are trying to answer explicitly, write down your prediction, and run enough independent seeds to trust the answer. Each of these is open-ended — there's no single right answer.

**A. Sweep one parameter systematically.** Vary one thing, holding everything else fixed, and measure its effect on the re-evaluated fitness of the evolved controllers (mean ± std across seeds). Some candidates:

- `--episodes_per_eval`: how does the number of evaluation episodes affect the quality of the controller evolution finds? Is there a trade-off with runtime?
- `--noise`: are controllers evolved with more motion noise more robust? What happens when you test a controller in a noise level different from the one it evolved in?
- `--hidden`: does a bigger network evolve a better controller for this task? (Compare with what you found for XOR in Project 3.) Could a network with *no* hidden layer (`--hidden_sizes` with no numbers) do the job?
- `--distance` or `--duration`: does a controller evolved at one distance still work at another?
- An evolutionary-algorithm parameter (`--popsize`, `--mut_stdev`, `--no-crossover`, ...), as in Projects 2 and 3.

**B. Bring back your Project 1 fitness function.** In Project 1 you designed your own fitness function for this vehicle. Reimplement it in `make_fitness_fn()` (the reward line near the end of `fitness_fn`, which has access to the full `(pop, episodes, duration)` array of distances) and evolve with it. Do the two fitness functions lead to visibly different strategies? Does one converge faster or more reliably? A controller can score well on one fitness measure while behaving quite differently under another — evaluate each evolved controller under *both* fitness functions.

**C. Sensor noise and robustness.** Add noise to the sensor readings (in `Vehicle.sense()` and in the sense block of `simulate_population()`), separately from the existing motion noise. Evolve with it, and compare the evolved controller's robustness against the crossed vehicle's under the same corruption. Does the evolved network learn some implicit filtering that the fixed wiring cannot?

**D. Evolve the hand-wired controller's own parameters.** Instead of evolving a full network, evolve just the few parameters a crossed-wiring vehicle could use (e.g., a gain on each sensor-to-motor connection, plus maybe a bias) — a genome of 2–4 numbers instead of 42. You'll need a different `think` function inside `make_fitness_fn()`. Compare its performance against the full `NeuralController`. How much of the neural controller's advantage comes from having many more parameters, versus more expressive structure? **Careful:** make sure both controllers have comparable motor ranges (e.g., clip or Tanh-squash the gain-scaled motor commands to [-1, 1]); otherwise evolution can win simply by driving the gains up to go faster.

**E. Co-evolve sensor placement.** Add `angle_offset` as an extra gene, evolved alongside the network's weights, instead of holding it fixed at π/2. Does evolution discover a better sensor placement? Does it still make behavioral sense?

**F. Two light sources.** Add a second light (positions randomized each episode) and a reward that accounts for both. Does the evolved network learn a strategy the crossed vehicle could never produce, or does it just pick one light and ignore the other?

You're encouraged to explore your own idea beyond these, as long as it's a genuine extension.

---

## What to Submit to Moodle

Submit a single **written report as a PDF** to Moodle.

### Title Page

The first page of your report should include:

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 4: Embodied NeuroEvolution I — Braitenberg Phototaxis)
- Date submitted
- Amount of time spent on this project
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body of your report into one section per assignment part. Each section should combine the relevant figures with a written discussion — a plot with no interpretation, or an interpretation with no supporting plot, is incomplete.

In what follows, I am going to mention the traditional path of required components. However, keep in mind that if you chose to meet the learning objectives in a different way, then your required parts might look different.

**Required Part 1 — Understand the Neural Controller and the Embodied Fitness**

- Your answers to the conceptual questions posed in Part 1 (parameter count, the two uses of Tanh, the expected range of fitness values and why there is no early stopping, your definition of convergence, and why fitness averages over random episodes).
- The fitness-over-generations plot (`--vizperf`) and trajectory plot (`--viztraces`) from a default run, with a brief caption, and your explanation of the difference between the best fitness during evolution and the re-evaluated fitness.

**Required Part 2 — Evolved vs. Hand-Designed**

- The re-evaluated fitness of your evolved controllers (at least 5 seeds) alongside the crossed vehicle's fitness, as a plot or table.
- Trajectory and distance-over-time plots comparing an evolved controller and the crossed vehicle under the same starting conditions.
- Your discussion of the strategies you observed, and of how fair the comparison is.

**Required Part 3 — Explore One Aspect in Depth**

- State what you chose to investigate and the question you were trying to answer.
- Describe what you held fixed and what you varied, and show plots/results against a baseline configuration, with enough seeds to support your conclusion.
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
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in directly comparing the evolved controller against the hand-designed one, rather than describing each in isolation.

---

## Further Reading

- Braitenberg, V. (1984). *Vehicles: Experiments in Synthetic Psychology.* MIT Press.
- Floreano, D., Dürr, P., & Mattiussi, C. (2008). *Neuroevolution: from architectures to learning.* Evolutionary Intelligence.
- Nolfi, S., & Floreano, D. (2000). *Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines.* MIT Press.

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.
