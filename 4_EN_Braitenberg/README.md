# Project 4: Embodied NeuroEvolution I — Braitenberg Phototaxis

## Overview

In this project, you will combine the concepts from the previous projects:
evolutionary algorithms and neural networks to evolve a controller for the
Braitenberg vehicle's phototaxis (light-seeking) behavior.

Instead of using hand-engineered sensor-motor wiring as in Project 1, we'll
use an evolutionary algorithm to optimize the weights of a neural network
controller. The network receives sensor readings as input and produces motor
commands as output, and evolution finds the best weight configuration through
selection and mutation.

---

## Learning Objectives

By completing this project, you will learn how to:

- Combine neural networks with evolutionary algorithms (neuroevolution)
- Encode a neural network's weights as a genome for evolutionary search
- Apply neuroevolution to an embodied robot control problem
- Compare neural controllers with hand-engineered Braitenberg controllers
- Investigate how network architecture affects evolved behavior

---

## Background

### Neuroevolution

**Neuroevolution** is the application of evolutionary algorithms to optimize
neural network weights. Instead of using gradient descent and backpropagation,
the entire weight vector (the "genome") is treated as a solution to be evolved:

1. **Encode**: Flatten all network weights into a single real-valued vector
2. **Evaluate**: Load genome into the network, run in environment, compute fitness
3. **Evolve**: Apply selection, crossover, and mutation to produce next generation

This approach works even when the objective has no usable gradient.

### Braitenberg Vehicles Revisited

In Project 1, we used a simple crossed-wiring scheme:
- Left sensor → Right motor
- Right sensor → Left motor

This produces light-seeking behavior through direct sensorimotor connections.
Now we replace this fixed wiring with a neural network that can learn more complex
sensor-to-motor mappings.

### The Neural Vehicle

The key change in this project is the `NeuralVehicle` class (defined in `braitenberg.py`), which extends the Project 1 `Vehicle` and swaps out its `think()` method. In the original `Vehicle`, `think()` implements the fixed crossed wiring described above — sensors connect directly to motors with no adjustable parameters. In `NeuralVehicle`, `think()` instead passes the two sensor readings through a `NeuralController` (see `neural_controller.py`) and uses its two outputs as the motor commands. Everything else about the vehicle — sensing, moving, accumulating noise — is unchanged; only the sensor-to-motor mapping itself is now a network whose weights evolution can shape, rather than a mapping you hand-designed.

**Note on motor output range:** The original crossed-wiring `Vehicle` sets motor commands directly from sensor readings, which are always in [0, 1] — so that vehicle can never drive backward. `NeuralVehicle`'s motors, by contrast, come out of the network's Tanh output layer and range over [-1, 1], so an evolved controller *can* learn to reverse or spin in place. With the same `turn_gain`, this also roughly doubles the vehicle's maximum turning rate compared to the crossed-wiring controller (since `left_motor - right_motor` can range over [-2, 2] instead of [-1, 1]). Keep this in mind when comparing turning behavior between the two controllers in Part 3 — a sharper turn doesn't necessarily mean a "smarter" controller, it may just reflect the wider motor range available to it.

#### Vehicle Configuration

The `Vehicle` class has three key configurable parameters:

| Parameter | Description |
|-----------|-------------|
| `angle_offset` | Angular separation between sensors (radians). π/2 places sensors at 90° on each side. |
| `turn_gain` | How strongly sensor difference steers the vehicle. Larger values produce sharper turns. |
| `noise_stdev` | Standard deviation of Gaussian noise added to orientation at each step. |

These parameters are passed as command-line arguments to both `sim.py` and `evolve.py`.

---

## Project Structure

| File | Purpose |
|------|---------|
| `neural_controller.py` | Defines the neural network controller (2→hidden→2) |
| `braitenberg.py` | Core vehicle classes including shared `NeuralVehicle` |
| `evolve.py` | Core neuroevolution using EvoTorch |
| `sim.py` | Run simulations with evolved controllers |
| `requirements.txt` | Pinned dependency versions for this project's virtual environment. |
| `README.md` | This documentation |

---

## Installation

The project requires:

- Python 3.7 or newer
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

## Running Neuroevolution

### Simulation Setup

The evolution and simulation environment uses the following configuration:

- **Light source**: Always positioned at the origin (0, 0)
- **Agent starting position**: At the specified `distance` from the light, but with a random angle and orientation for each episode
- **Fitness evaluation**: The agent's neural controller is evaluated over multiple episodes with different random initial conditions

This setup ensures that evolved controllers learn to seek light regardless of starting position and orientation.

To evolve a neural network controller with default parameters:

```bash
python evolve.py
```

Useful command-line options:

| Option | Description | Default |
|--------|-------------|---------|
| `--hidden` | Number of hidden neurons | `8` |
| `--popsize` | Population size | `50` |
| `--gens` | Number of generations | `100` |
| `--mut_stdev` | Mutation strength | `0.5` |
| `--tournament_size` | Tournament size for crossover | `3` |
| `--eta` | Distribution index for SBX crossover | `20` |
| `--no-elitism` | Disable elitism | elitism on |
| `--vizperf` | Visualize fitness over generations | off |
| `--verbose` | Print progress to console | off |
| `--seed` | Random seed for reproducibility | `None` |
| `--distance` | Distance to light source | `10.0` |
| `--angle_offset` | Angular separation between sensors (radians) | `pi/2` |
| `--turn_gain` | Turn gain for steering | `0.1` |
| `--noise` | Motion noise standard deviation | `0.1` |
| `--duration` | Number of simulation steps per episode | `500` |
| `--output` | File path to save the best evolved genome, e.g. `best_genome.npy` | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves across configurations without re-running evolution | `None` |
| `--seed_genome FILE` | Seed the initial population around a genome saved by a previous `--output` run (must match the current `--hidden`) instead of starting from scratch — one exact copy plus the rest perturbed by `--seed_noise` | `None` |
| `--seed_noise` | Stdev of the Gaussian perturbation applied to `--seed_genome` copies | `0.05` |
| `--episodes_per_eval` | Number of episodes to average per fitness evaluation | `5` |

For example:

```bash
python evolve.py --hidden 16 --popsize 100 --gens 200 --vizperf --distance 15.0
```

To run with non-default vehicle parameters:

```bash
python evolve.py --angle_offset 1.0 --turn_gain 0.2 --noise 0.1
```

**Note on the distance parameter:** The `--distance` argument specifies how far from the origin (where the light is) the agent starts, not the light's position. The light is always at (0, 0), and agents start at a random angle around this circle.

---

## Running Simulations

After evolution, you can simulate the best evolved controller:

```bash
python sim.py
```

To use a saved genome from evolution:

```bash
# First, save the genome during evolution:
python evolve.py --output best_genome.npy

# Then simulate with the evolved controller:
python sim.py --genome best_genome.npy --viztraces
```

To simulate with the same vehicle configuration as evolution, pass matching arguments:

```bash
python sim.py --genome best_genome.npy --viztraces --angle_offset 1.57 --turn_gain 0.1 --noise 0.1
```

By default, both scripts use `angle_offset=pi/2`, `turn_gain=0.1`, and `noise_stdev=0.1`.

Useful command-line options for `sim.py`:

| Option | Description | Default |
|--------|-------------|---------|
| `--genome` | Path to a saved `.npy` genome file; if omitted, uses a randomly initialized (untrained) controller | `None` |
| `--duration` | Number of simulation steps (should match the `--duration` used during evolution) | `500` |
| `--reps` | Number of independent repetitions to average over | `5` |
| `--distance` | Distance to light source | `10.0` |
| `--hidden` | Number of hidden neurons; must match the network used during evolution | `8` |
| `--angle_offset` | Angular separation between sensors (radians) | `pi/2` |
| `--turn_gain` | Turn gain for steering | `0.1` |
| `--noise` | Motion noise standard deviation | `0.1` |
| `--viztraces` | Plot vehicle trajectories in the x-y plane | off |
| `--vizdist` | Plot average distance to light over time | off |
| `--scores` | Print the average fitness achieved per repetition | off |
| `--seed` | Random seed for reproducibility | `None` |

**Important:** `--hidden`, `--angle_offset`, `--turn_gain`, `--noise`, and `--distance` must match the values used when the genome was evolved — a genome evolved with one network size or vehicle configuration will not load correctly into a differently-shaped network, and will behave unpredictably in a mismatched environment.

---

## Parameter Studies

There's no `study.py` in this project (unlike Projects 1–3) — Part 2 of
the assignment asks you to compare hidden-layer sizes, and running the
sweep and plotting the result is left as an exercise.

The quickest way to do this without writing a full sweep script: run
`evolve.py` once per hidden size with `--fitness_output` to save each
run's fitness-over-generations curve, then reload and plot them together.

```bash
python evolve.py --hidden 4 --fitness_output hidden4.npz
python evolve.py --hidden 8 --fitness_output hidden8.npz
python evolve.py --hidden 16 --fitness_output hidden16.npz
python evolve.py --hidden 32 --fitness_output hidden32.npz
```

```python
import numpy as np
import matplotlib.pyplot as plt

for label, path in [("hidden=4", "hidden4.npz"), ("hidden=8", "hidden8.npz"),
                     ("hidden=16", "hidden16.npz"), ("hidden=32", "hidden32.npz")]:
    data = np.load(path)
    plt.plot(data["best"], label=label)
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.legend()
plt.show()
```

---

## Understanding the Components

### Neural Controller (`neural_controller.py`)

The `NeuralController` class defines a feedforward network:
- **Input layer**: 2 neurons (left and right sensor readings)
- **Hidden layer**: configurable number of neurons with Tanh activation
- **Output layer**: 2 neurons (left and right motor commands), also passed through Tanh so motor commands are bounded to [-1, 1]

The genome is a flat vector containing all weights and biases in PyTorch's
parameter order.

### Evolution (`evolve.py`)

The `run_evolution()` function:
1. Creates a fitness function that evaluates the controller in simulation, averaging bounded proximity reward over `--episodes_per_eval` episodes with randomized starting angle and orientation
2. Sets up EvoTorch's GeneticAlgorithm with SBX crossover and Gaussian mutation
3. Runs for the specified number of generations, stopping early if fitness reaches near-perfect (≥ 0.99)
4. Returns fitness trajectories and the best genome

### Simulation (`sim.py`)

The `NeuralVehicle` class (defined in `braitenberg.py`) extends the Braitenberg `Vehicle`:
- Uses a neural network instead of direct sensor-motor wiring
- Applies noise to orientation during movement

---

## Tips

- Start with small populations (25-50) and fewer generations (50-100) to test
  your setup before running large experiments
- Use `--seed` for reproducibility when debugging
- Visualize fitness curves (`--vizperf`) to monitor evolutionary progress
- Compare with the Braitenberg controller from Project 1 as a baseline

**Note on Fitness:** At each timestep, the fitness function rewards proximity to the light as `1 / (1 + distance)`, averaged over the episode and over `--episodes_per_eval` repetitions. This bounds fitness in (0.0, 1.0]: a vehicle sitting on top of the light the whole episode scores 1.0, and fitness approaches 0.0 as the vehicle stays far from the light. Higher fitness always means better performance.

---

## Assignment

### Part 1 – Understand the Neural Controller

Answer these questions before running any experiments:

- How many total parameters (weights + biases) does a network with 8 hidden
  neurons have? Count them by layer.
- Why is Tanh used for the hidden layer but not for outputs?
- What would happen if we used ReLU instead of Tanh in the hidden layer?

**Optional / Advanced Challenge:** The provided `NeuralController` (in `neural_controller.py`) is a single hidden layer, `2 → hidden → 2`, with Tanh activations throughout. Try modifying the architecture itself and re-run evolution to see how it affects evolvability:
- **Depth**: add a second hidden layer (e.g. `2 → hidden → hidden → 2`) and compare against the single-layer network of similar total parameter count. Does the deeper network evolve as easily, or does it struggle more (deeper genomes can be harder for a genetic algorithm to search)?
- **Activation function**: swap Tanh for another transfer function (e.g. ReLU, Sigmoid, or a mix) on the hidden layer(s), keeping Tanh on the output layer so motor commands stay bounded to [-1, 1]. How does the choice of hidden activation affect the smoothness of evolved trajectories or the final fitness reached?

You don't need to change anything outside `neural_controller.py` — `evolve.py` computes genome length directly from whatever architecture `NeuralController` defines, so a modified network will evolve and load correctly as-is.

### Part 2 – Evolve and Analyze

Evolve controllers with different hidden sizes:

1. Run evolution with 4, 8, 16, and 32 hidden neurons
2. Compare final fitness values across architectures
3. How does genome size affect evolutionary dynamics?

Questions to consider:
- Does a larger network always achieve higher fitness?
- Is there diminishing returns beyond a certain size?
- What is the relationship between parameter count and convergence speed?

**Optional / Advanced Challenge:** In Project 1, you designed your own fitness function for the Braitenberg vehicle, measuring something other than raw distance to the light. Bring that fitness function back here: reimplement it as an alternative to `make_fitness_fn()`'s `1 / (1 + distance)` reward (swap out the line inside the `fitness_fn` closure, or write a second version of `make_fitness_fn`), and run evolution with it under otherwise matched settings (same hidden size, popsize, generations, seed). Then compare the two fitness functions on their evolvability, not just their final scores:
- Does your fitness function converge faster, slower, or about as fast as the distance-based one?
- Does it produce a higher success rate across repeated runs (see Part 4), or is it noisier/more prone to getting stuck?
- Do the two fitness functions lead to visibly different strategies when you look at trajectories (Part 3)? A controller can score well on one fitness measure while behaving quite differently under another.

### Part 3 – Behavioral Comparison

Compare neural controllers with Braitenberg controllers:

1. Run simulations with evolved neural controllers using `sim.py`.
2. **Write your own small script (or adapt a copy of `sim.py`) to simulate the original crossed-wiring `Vehicle` from Project 1** under the same starting conditions (same `distance`, `angle_offset`, `turn_gain`, `noise_stdev`, and random seed) and log its trajectory the same way. There is no `--baseline` option provided in this project's code — `sim.py` only drives `NeuralVehicle`, so producing the comparison trajectories is part of the exercise. This should only take a few lines: instantiate a `Vehicle` instead of a `NeuralVehicle` (no controller needed, since `Vehicle.think()` already implements the crossed wiring), and reuse the same sensing/thinking/moving loop.
3. Compare trajectories between the two controllers.
4. Analyze differences in path smoothness, speed, and directness.

Questions:
- Do neural controllers follow similar paths to Braitenberg vehicles?
- Are there noticeable differences in turning behavior?
- Can you identify any "strategies" employed by evolved networks?

### Part 4 – Quantitative Analysis

Collect data from multiple independent runs:

1. Run evolution with the same parameters 10 times
2. Record best fitness, convergence generation, and genome size
3. Analyze variance across runs

Questions:
- Does evolution always find high-fitness solutions?
- How does population size affect success rate?
- What is the typical convergence pattern (early rapid progress vs. late refinement)?

---

## Optional / Advanced Challenge

Parts 1–4 are required (Parts 1 and 2 already offer smaller optional callouts of their own). Beyond that, pick **one** of the following four directions to investigate further. Each is open-ended — there's no single right answer, and the point is to form a hypothesis, run the experiment, and report what you found. Only attempt one; go as deep as you like on it.

**1. Multiple light sources.** Evolve under two light sources (positions randomized each episode) instead of one, modifying `make_fitness_fn`'s episode setup and reward to account for both. Hypothesis: does the evolved network learn a genuinely different strategy than the hand-wired crossed vehicle could ever produce — which can only track one simple gradient at a time — or does it just learn to pick one light and ignore the other?

**2. Sensor noise and robustness comparison.** Evolve with corrupted sensor readings (add noise directly to `left_sensor`/`right_sensor` inside `sense()`, separate from the vehicle's existing orientation noise) and compare the evolved controller's robustness against the hand-wired crossed-wiring `Vehicle`'s, tested under the same corruption. Hypothesis: does the evolved network learn some implicit filtering that the fixed wiring structurally cannot?

**3. Evolve the hand-wired controller's own parameters.** Instead of evolving a full neural network, evolve just a couple of scalar parameters the crossed-wiring scheme itself could use (e.g., a per-sensor gain applied before crossing) — a 2-parameter genome instead of the full network's dozens of weights. Compare its evolved performance against the full `NeuralController`. Hypothesis: how much of the neural controller's apparent advantage over Project 1's hand-wiring comes from having many more free parameters to tune, versus genuinely more expressive structure?

**4. Co-evolve sensor placement.** Evolve `angle_offset` (currently fixed at π/2) alongside the network's weights, instead of holding it fixed for the whole population. Hypothesis: does evolution discover a different sensor placement that performs better than the one you were given — and if so, does that placement still make behavioral sense (e.g., still roughly symmetric left-right)?

You're encouraged to explore your own idea beyond these four as well, as long as it's a genuine extension (not just one of the smaller optional callouts Part 1 or Part 2 already cover, and not just a parameter change already covered elsewhere).

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

**Part 1 — Understand the Neural Controller**

- Your answers to the conceptual questions posed in Part 1 (parameter count for 8 hidden neurons by layer, why Tanh is used on the hidden layer vs. the output layer, and what would change if ReLU replaced Tanh in the hidden layer).
- *(Optional)* If you attempted the Part 1 advanced architecture challenge (extra layers and/or alternate activation functions), briefly describe what you changed and how it affected evolvability.

**Part 2 — Evolve and Analyze**

- Fitness-over-generations plots (`--vizperf`) for hidden sizes 4, 8, 16, and 32, run with matched settings otherwise.
- A summary plot or table comparing final best fitness across the four architectures.
- Answers to the guiding questions from Part 2, supported directly by your results (does a larger network always win, is there a point of diminishing returns, how does genome size relate to convergence speed).
- *(Optional)* If you attempted the Part 2 custom fitness function challenge, describe the fitness function from Project 1 you brought back, and compare its evolvability against the provided distance-based fitness (convergence speed, success rate, resulting strategies).

**Part 3 — Behavioral Comparison**

- Trajectory plots (`--viztraces`) for at least one evolved neural controller, alongside trajectories from the original crossed-wiring Braitenberg controller under matching starting conditions.
- A discussion of differences in path smoothness, directness, and turning behavior between the two controllers.
- Answers to the guiding questions from Part 3, including any distinctive strategies you observed in the evolved networks (e.g., use of reverse motion, which the neural controller can do but the original crossed-wiring vehicle cannot).

**Part 4 — Quantitative Analysis**

- Results from 10 independent evolutionary runs with the same parameters: best fitness, convergence generation, and genome size for each.
- A plot or discussion of the variance across runs.
- Answers to the guiding questions from Part 4 (does evolution always succeed, effect of population size on success rate, typical shape of the convergence curve).

**Optional / Advanced Challenge** *(if attempted)*: a section naming which of the four directions you chose (or your own idea), what you changed, your results (with supporting figures), and your interpretation. Omit this section if you didn't attempt a challenge.

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
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in directly comparing the evolved controller against your own hand-designed one from Project 1, rather than describing each in isolation.

---

## Further Reading

- Floreano, D., Dürr, P., & Mattiussi, C. (2008). *Neuroevolution: from
  architectures to learning.* Evolutionary Intelligence.
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through
  Augmenting Topologies.* Evolutionary Computation.

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.