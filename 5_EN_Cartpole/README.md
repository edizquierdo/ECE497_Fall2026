# Project 5: Embodied NeuroEvolution II — CartPole Balancing

## Overview

In this project, you will combine the concepts from the previous projects:
evolutionary algorithms and neural networks to evolve a controller for the
classic CartPole balancing task.

Instead of using hand-designed control rules, we'll use an evolutionary
algorithm to optimize the weights of a neural network controller. The network
receives cart and pole measurements as input and produces motor commands as
output, and evolution finds the best weight configuration through selection
and mutation.

This is your second pass at putting the full pipeline together (evolutionary
algorithms + neural networks), now on a new platform. Beyond the core
assignment, you'll also get to choose your own open-ended challenge to
investigate in more depth — see [Optional / Advanced Challenge](#optional--advanced-challenge).

---

## Learning Objectives

By completing this project, you will learn how to:

- Combine neural networks with evolutionary algorithms (neuroevolution)
- Encode a neural network's weights as a genome for evolutionary search
- Apply neuroevolution to a standard reinforcement learning benchmark
- Compare continuous vs. discrete action spaces in neuroevolution
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

### CartPole Environment Revisited

The **CartPole** task is a classic control problem:

- A pole is attached to a cart that moves along a frictionless track
- The goal is to keep the pole balanced upright for as long as possible (500 steps)
- The system is unstable—without control, the pole falls quickly

#### Environment Configuration

The `CartPole` wrapper class provides a simplified interface to Gymnasium's
CartPole-v1 environment.

| Observation | Description |
|-------------|-------------|
| 0 | Cart position (x) |
| 1 | Cart velocity |
| 2 | Pole angle (radians from vertical) |
| 3 | Pole angular velocity |

**Actions:**
- `0`: Push cart to the left
- `1`: Push cart to the right

**Reward:** +1 per timestep (maximum 500)

**Termination conditions:**
- Pole falls too far (> 12 degrees from vertical, ≈0.209 radians)
- Cart moves beyond ±2.4 units
- Episode reaches the step limit (`--duration`, default 500) without falling — this is a **truncation**, not a failure; a controller that survives to the limit every time has effectively solved the task

---

## Project Structure

| File | Purpose |
|------|---------|
| `neural_controller.py` | Defines the neural network controller (4→hidden→2) |
| `cartpole.py` | Wrapper around Gymnasium's CartPole-v1 environment |
| `evolve.py` | Core neuroevolution using EvoTorch |
| `sim.py` | Run simulations with evolved controllers, with optional live rendering |
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
- Gymnasium

All five are listed with tested version ranges in `requirements.txt`, so install them together into a project-specific virtual environment rather than into your system Python.

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

**Note:** The neural controller uses a Tanh activation in the hidden layer and
linear outputs (logits) for discrete action selection. The simulation in
`sim.py` is consistent with `evolve.py`.

---

## Running Neuroevolution

### Simulation Setup

The evolution and simulation environment uses the following configuration:

- **Fitness evaluation**: The agent's neural controller is evaluated over multiple
  episodes with different random initial conditions
- **Episode termination**: Pole falls, cart moves beyond limits, or the episode
  reaches `--duration` steps (up to 500 by default)

#### A note on `--duration`

CartPole-v1's maximum reward of 500 comes from Gymnasium's own built-in
500-step limit on the environment, not from anything specific to this code.
Both `evolve.py` and `sim.py` expose this as an explicit `--duration`
argument (default 500 in both), so it's a visible, tunable parameter rather
than something baked in invisibly. **The two values should match** — if you
evolve with a shorter duration, the controller was never evaluated on (and
never selected for) surviving past that point, so testing it afterward with
a longer duration isn't a fair comparison. A shorter duration during
evolution makes each fitness evaluation cheaper (useful while iterating on
other settings), at the cost of not confirming the controller generalizes
to the full 500-step task; conversely, there's no benefit to raising
`--duration` past 500, since the pole either stays up or it doesn't — you
aren't buying additional learning signal, only extra compute per evaluation.

To evolve a neural network controller with default parameters:

```bash
python evolve.py
```

Useful command-line options:

| Option | Description | Default |
|--------|-------------|---------|
| `--hidden` | Number of hidden neurons | `8` |
| `--popsize` | Population size | `80` |
| `--gens` | Number of generations | `40` |
| `--episodes_per_eval` | Episodes to average per fitness evaluation | `5` |
| `--duration` | Max steps per episode during training | `500` |
| `--mut_stdev` | Gaussian mutation standard deviation | `0.4` |
| `--tournament_size` | Tournament size for SBX crossover | `3` |
| `--eta` | Distribution index for SBX crossover | `20` |
| `--no-elitism` | Disable elitism | elitism on |
| `--vizperf` | Visualize fitness over generations | off |
| `--verbose` | Print progress to console | off |
| `--seed` | Random seed for reproducibility | `None` |
| `--output FILE` | File path to save best genome | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves across configurations without re-running evolution | `None` |
| `--seed_genome FILE` | Seed the initial population around a genome saved by a previous `--output` run (must match the current `--hidden`) instead of starting from scratch — one exact copy plus the rest perturbed by `--seed_noise` | `None` |
| `--seed_noise` | Stdev of the Gaussian perturbation applied to `--seed_genome` copies | `0.05` |

For example:

```bash
python evolve.py --hidden 16 --popsize 100 --gens 50 --vizperf
```

---

## Running Simulations

After evolution, you can simulate the best evolved controller:

```bash
python sim.py --genome best_genome.npy --scores
```

To visualize pole angle and cart position over time:

```bash
python sim.py --genome best_genome.npy --viztraces --reps 10
```

To watch the controller balance the pole live, add `--render`:

```bash
python sim.py --genome best_genome.npy --render --reps 3
```

**Important:** `--hidden` must match the value used when the genome was
evolved. `evolve.py` prints a reminder of the exact `--hidden` (and
`--duration`) to use when it saves a genome. This isn't just for correctness
of results — if the value doesn't match, the genome will still load without
any error and will produce a controller that looks evolved but behaves
essentially randomly, because PyTorch's `vector_to_parameters` doesn't check
that the genome's length matches the network's parameter count; it just
consumes values positionally. `sim.py` includes an explicit length check
that raises a clear error in this situation, but it's still worth
double-checking your `--hidden` value directly rather than relying on it.

Useful command-line options for `sim.py`:

| Option | Description | Default |
|--------|-------------|---------|
| `--genome` | Path to a saved `.npy` genome file; if omitted, uses a randomly initialized (untrained) controller | `None` |
| `--duration` | Number of simulation steps (should match the duration used during evolution) | `500` |
| `--reps` | Number of independent repetitions | `5` |
| `--hidden` | Number of hidden neurons; must match the network used during evolution | `8` |
| `--viztraces` | Plot pole angle and cart position over time | off |
| `--scores` | Print average reward per repetition | off |
| `--render` | Render episodes live with Gymnasium's human renderer | off |
| `--seed` | Random seed for reproducibility | `None` |

---

## Parameter Studies

There's no `study.py` in this project (unlike Projects 1–3) — sweeping a
parameter and plotting the result is left as an exercise. `study_arch.py`
is one reasonable name if you want to build a reusable script for it.

Useful considerations for parameter studies:
- **Hidden layer size**: Try values from 4 to 64 neurons
- **Population size**: Larger populations explore more but are slower
- **Generations**: More generations allow more exploration but take longer

The quickest way to compare configurations without writing a full sweep
script: run `evolve.py` once per configuration with `--fitness_output` to
save each run's fitness-over-generations curve, then reload and plot them
together.

```bash
python evolve.py --hidden 4 --gens 30 --fitness_output hidden4.npz
python evolve.py --hidden 16 --gens 30 --fitness_output hidden16.npz
python evolve.py --hidden 32 --gens 30 --fitness_output hidden32.npz
```

```python
import numpy as np
import matplotlib.pyplot as plt

for label, path in [("hidden=4", "hidden4.npz"), ("hidden=16", "hidden16.npz"), ("hidden=32", "hidden32.npz")]:
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
- **Input layer**: 4 neurons (cart position, cart velocity, pole angle, pole angular velocity)
- **Hidden layer**: configurable number of neurons with Tanh activation
- **Output layer**: 2 neurons (action logits for left/right)

The genome is a flat vector containing all weights and biases in PyTorch's
parameter order.

### Evolution (`evolve.py`)

The `run_evolution()` function:
1. Creates a fitness function that evaluates the controller over multiple episodes
2. Sets up EvoTorch's GeneticAlgorithm with SBX crossover and Gaussian mutation
3. Runs for the specified number of generations, stopping early if fitness reaches the maximum possible episode reward (500)
4. Returns fitness trajectories and the best genome

### Simulation (`sim.py`)

The `CartPole` wrapper class provides:
- Reset and step methods matching the evolution environment
- Support for visualization of pole angle and cart position over time
- Tracking of episode lengths and rewards across repetitions
- Optional live rendering via `--render`, so you can watch a controller balance the pole

---

## Tips

- Start with small populations (25-50) and fewer generations (20-30) to test
  your setup before running large experiments
- Use `--seed` for reproducibility when debugging
- Visualize fitness curves (`--vizperf`) to monitor evolutionary progress
- For faster evolution, reduce `episodes_per_eval` but this gives noisier fitness

**Note on Fitness:** The fitness function averages episode returns over multiple
stochastic rollouts. Higher values are better (more time balancing the pole).

---

## Assignment

### Part 1 – Understand the Neural Controller

Answer these questions before running any experiments:

- How many total parameters (weights + biases) does a network with 8 hidden
  neurons have? Count them by layer.
- Why is Tanh used for the hidden layer but not for outputs?
- What would happen if we used ReLU instead of Tanh in the hidden layer?

### Part 2 – Evolve and Analyze

Evolve controllers with different hidden sizes:

1. Run evolution with 4, 8, 16, and 32 hidden neurons
2. Compare final fitness values across architectures
3. How does genome size affect evolutionary dynamics?

Questions to consider:
- Does a larger network always achieve higher fitness?
- Is there diminishing returns beyond a certain size?
- What is the relationship between parameter count and convergence speed?

### Part 3 – Behavioral Analysis

1. Run simulations with evolved controllers
2. Visualize pole angle and cart position over time
3. Analyze stability patterns

Questions:
- How does the controller handle different initial conditions?
- Are there noticeable oscillations or stable balancing?
- What strategies emerge from evolution?

### Part 4 – Quantitative Analysis

Collect data from multiple independent runs:

1. Run evolution with the same parameters 5 times
2. Record best fitness, convergence generation, and genome size
3. Analyze variance across runs

Questions:
- Does evolution always find high-fitness solutions?
- How does population size affect success rate?
- What is the typical convergence pattern (early rapid progress vs. late refinement)?

---

## Optional / Advanced Challenge

Parts 1–4 are required. Beyond that, pick **one** of the following four
directions to investigate further. Each is open-ended — there's no single
right answer, and the point is to form a hypothesis, run the experiment, and
report what you found. Only attempt one; go as deep as you like on it.

**1. Reward shaping.** The default fitness is the flat `+1` per timestep
built into CartPole-v1. Design an alternative fitness function that instead
rewards proximity to the ideal state directly — for example, penalizing the
magnitude of pole angle and/or cart position at each step rather than just
rewarding survival. Implement it as an alternative to the reward-accumulation
line inside `fitness_fn` (or write a second version of `make_fitness_fn`),
and compare it against the default under matched settings (same hidden size,
popsize, generations, seed). Does it converge faster or slower? Does it
produce visibly different balancing strategies?

**2. Continuous vs. discrete action spaces.** The provided controller
outputs two logits and picks a discrete left/right push via `argmax`.
Modify `NeuralController` so its output layer is a single Tanh-bounded
value representing a continuous force, and find a way to apply that force
directly to the cart instead of going through the discrete action wrapper
(you'll need to either use a continuous CartPole variant, or extend
`cartpole.py` to apply your continuous output directly to the underlying
physics rather than mapping it to one of the two discrete actions). Compare
evolvability and resulting behavior against the discrete controller from
Part 2.

**3. Architecture depth and activation functions.** `NeuralController` is
currently a single hidden layer, `4 → hidden → 2`, with Tanh in the hidden
layer. Try modifying the architecture itself: add a second hidden layer
(e.g. `4 → hidden → hidden → 2`) and compare it against a single-layer
network of similar total parameter count — does the deeper network evolve
as easily, or does it struggle more? Separately (or additionally), try
swapping Tanh for another hidden-layer activation (e.g. ReLU or Sigmoid).
You don't need to change anything outside `neural_controller.py` —
`evolve.py` computes genome length directly from whatever architecture
`NeuralController` defines, so a modified network will evolve and load
correctly as-is.

**4. Robustness and generalization.** Evolve a controller under the default
termination limits (24° pole angle, ±2.4 cart position), then — without
retraining — test it under stricter or looser limits (e.g. tighter angle
tolerance, or a narrower track). Does the evolved controller generalize
gracefully, or does performance collapse outside the conditions it was
evolved under? This gets at a question the Braitenberg project couldn't
easily ask, since CartPole's termination conditions are simple to vary
independently of the fitness function.

You're encouraged to explore your own idea beyond these four as well, as
long as it's a genuine extension (not just a parameter change already
covered in Parts 2–4).

---

## What to Submit to Moodle

Submit a single **written report as a PDF** to Moodle.

### Title Page

The first page of your report should include:

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 5: Embodied NeuroEvolution II — CartPole Balancing)
- Date submitted
- Amount of time spent on this project
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body of your report into one section per assignment part. Each section should combine the relevant figures with a written discussion — a plot with no interpretation, or an interpretation with no supporting plot, is incomplete.

**Part 1 — Understand the Neural Controller**

- Your answers to the conceptual questions posed in Part 1 (parameter count for 8 hidden neurons by layer, why Tanh is used on the hidden layer vs. the output layer, and what would change if ReLU replaced Tanh in the hidden layer).

**Part 2 — Evolve and Analyze**

- Fitness-over-generations plots (`--vizperf`) for hidden sizes 4, 8, 16, and 32, run with matched settings otherwise.
- A summary plot or table comparing final best fitness across the four architectures.
- Answers to the guiding questions from Part 2, supported directly by your results (does a larger network always win, is there a point of diminishing returns, how does genome size relate to convergence speed).

**Part 3 — Behavioral Analysis**

- Trajectory plots (`--viztraces`, pole angle and cart position over time) for at least one evolved controller.
- A discussion of stability patterns, oscillations, and any strategies you observed.
- Answers to the guiding questions from Part 3.

**Part 4 — Quantitative Analysis**

- Results from 5 independent evolutionary runs with the same parameters: best fitness, convergence generation, and genome size for each.
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
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in relating the stability patterns you observe in Part 3 to the quantitative convergence data in Part 4, rather than treating them as separate reports.

---

## Further Reading

- Floreano, D., Dürr, P., & Mattiussi, C. (2008). *Neuroevolution: from
  architectures to learning.* Evolutionary Intelligence.
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through
  Augmenting Topologies.* Evolutionary Computation.

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.