# Project 3: Neuroevolution (XOR)

## Overview

In this project you will explore one of the foundational ideas in modern AI: **Artificial Neural Networks (ANNs)** and how to optimize them using **evolutionary algorithms** — a technique called *neuroevolution*.

Instead of training a network with gradient descent and backpropagation, you will evolve the network's weights directly using a Genetic Algorithm. This approach works even when the objective has no usable gradient, and gives you direct insight into how the structure and parameters of a neural network affect its ability to solve a problem.

Your goal is not simply to run the code, but to understand how a neural network represents a function, why hidden layers are necessary for certain tasks, and how neuroevolution finds good network weights without any gradient information.

---

## Learning Objectives

By completing this project, you will learn how to:

- Understand the structure of a feedforward neural network: inputs, weights, activations, and outputs.
- Implement and use PyTorch's `nn.Module` to define a neural network.
- Encode a neural network's parameters as a flat genome for evolutionary search.
- Apply neuroevolution (EA-driven weight optimization) to solve a classification task.
- Investigate how network architecture and evolutionary parameters affect performance.
- Compare evolutionary optimization to gradient-based training.

---

## Background

### Neural Networks

A **feedforward neural network** transforms an input vector into an output by passing it through one or more layers of weighted sums followed by a non-linear activation function.

The network in this project has three layers:

```
Input (2 neurons)  →  Hidden (h neurons)  →  Output (1 neuron)
```

At each layer, the computation is:

```
hidden  = activation( W₁ · input + b₁ )
output  = W₂ · hidden + b₂
```

where `W` and `b` are the weights and biases that determine the network's behavior.

### Why XOR?

The **XOR function** is a classic benchmark for neural networks:

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

XOR cannot be solved by a linear classifier — it requires a non-linear decision boundary. A network with at least one hidden layer and a non-linear activation can represent this boundary, making it an ideal minimal test for network expressiveness.

### Neuroevolution

Instead of computing gradients to update weights, **neuroevolution** treats the entire set of network weights as a genome and optimizes it with an evolutionary algorithm:

1. **Encode**: Flatten all weights and biases into a single real-valued vector (the genome).
2. **Evaluate**: Load the genome into the network, run a forward pass, compute fitness.
3. **Evolve**: Apply selection, crossover, and mutation to produce the next generation.
4. **Repeat** until a high-fitness solution is found.

This approach is gradient-free, works on discontinuous or non-differentiable objectives, and directly connects to the evolutionary methods from Project 2.

---

## Project Structure

| File | Purpose |
|------|---------|
| `neural.py` | Defines the network, fitness function, problem generators, and runs neuroevolution. |
| `study.py` | Performs systematic parameter sweeps and generates plots. |
| `requirements.txt` | Pinned dependency versions for this project's virtual environment. |
| `README.md` | Project documentation. |

Most of your modifications will involve understanding and extending the code in `neural.py`.

### Alternative Problem Generators

In addition to the standard XOR task, `neural.py` includes generators for custom classification problems:

- **Random problem**: Points sampled uniformly from `[-0.5, 1.5]²` with random labels (+1/-1).
- **Convex problem**: Points placed on a circle at equal angles with alternating labels.

These generalize the XOR task and allow investigation of how network architecture affects performance on different classification challenges.

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

## Running the Neuroevolution

To run the default neuroevolution experiment,

```bash
python neural.py
```

Useful command-line options include:

| Option | Description | Default |
|--------|-------------|---------|
| `--hidden` | Number of hidden neurons | `3` |
| `--popsize` | Population size | `100` |
| `--gens` | Number of generations | `200` |
| `--mut_stdev` | Gaussian mutation standard deviation | `0.5` |
| `--tournament_size` | Tournament size for SBX crossover | `3` |
| `--eta` | Distribution index for SBX crossover | `20` |
| `--no-elitism` | Disable elitism (elitism is enabled by default) | elitism on |
| `--activation` | Hidden layer activation function (`tanh`, `sigmoid`, or `relu`) | `tanh` |
| `--vizperf` | Plot fitness over generations | `False` |
| `--vizbound` | Plot the decision boundary of the best evolved network | `False` |
| `--verbose` | Print per-generation statistics to the console | `False` |
| `--seed` | Random seed for reproducibility | `None` |
| `--random` | Number of random data points instead of XOR | `None` |
| `--convex` | Number of convex-position data points on a circle | `None` |
| `--device` | Hardware device for execution (`auto`, `cpu`, `cuda`, `mps`) | `auto` |
| `--output FILE` | File path to save the best evolved genome, e.g. `best_genome.npy` | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves across configurations without re-running evolution | `None` |
| `--seed_genome FILE` | Seed the initial population around a genome saved by a previous `--output` run (must match the current `--hidden`/`--activation`) instead of starting from scratch — one exact copy plus the rest perturbed by `--seed_noise` | `None` |
| `--seed_noise` | Stdev of the Gaussian perturbation applied to `--seed_genome` copies | `0.05` |

> **Note on `--device`:** For a network as tiny as `XORNet` (10–20 weights),
> the GPU will typically be *slower* than the CPU — the whole population is
> now evaluated as one batched matrix operation, but that operation is so
> small that data-transfer and kernel-launch overhead dominates the runtime.
> GPU speedups only start to show up once `--popsize` and/or `--hidden` are
> scaled up significantly (e.g. `--popsize 500+`, `--hidden 20+`). Use
> `--device` to confirm your code runs correctly on a workstation GPU, not
> as a way to make the default XOR run faster.

For example,

```bash
python neural.py --vizperf --vizbound
```

or

```bash
python neural.py --hidden 5 --popsize 50 --gens 300 --verbose
```

Or run on GPU workstations or custom devices:

```bash
python neural.py --device cuda --popsize 500 --gens 1000
python neural.py --random 8 --vizbound
python neural.py --convex 6 --activation sigmoid --vizperf
```

---

## Performing Parameter Studies

The file `study.py` automates experiments in which a single parameter is varied over many values. For each value, neuroevolution is run multiple times and the average final fitness is recorded.

`study.py` can sweep any one of four parameters — chosen with `--param` — while holding everything else fixed. The script supports three problem types: XOR, random, and convex; for random/convex problems, one problem instance is generated up front and reused for every run in the study, so the variance you see reflects only the algorithm's own stochasticity, not a different random problem on every repetition.

By default (`--param hidden`, the default), `study.py` studies the number of hidden neurons on the XOR problem:

```bash
python study.py --problem xor --reps 10 --output xor_study.png
```

To sweep a different parameter, pass `--param`:

```bash
python study.py --param popsize --min 20 --max 200 --steps 10 --reps 5 --output popsize_study.png
python study.py --param mut_stdev --min 0.1 --max 2.0 --steps 10 --reps 5 --output mutstdev_study.png
python study.py --param gens --min 20 --max 300 --steps 10 --reps 5 --output gens_study.png
```

Or run on a custom problem:

```bash
python study.py --problem random --points 8 --gens 300 --reps 10
python study.py --problem convex --points 6 --output convex_study.png
```

Useful command-line options include:

| Option | Description | Default |
|--------|-------------|---------|
| `--param` | Parameter to sweep: `hidden`, `popsize`, `mut_stdev`, or `gens` | `hidden` |
| `--min` | Minimum value for the swept parameter | depends on `--param` |
| `--max` | Maximum value for the swept parameter | depends on `--param` |
| `--steps` | Number of values to test between min and max | `10` |
| `--gens` | Generations per run (ignored when `--param gens`, which sweeps this instead) | `200` |
| `--reps` | Repetitions per parameter value | `5` |
| `--problem` | Problem type: `xor`, `random`, or `convex` | `xor` |
| `--points` | Number of points for random/convex problems | `4` |
| `--output` | Output filename for plot | `study_arch.png` |
| `--seed` | Base random seed for reproducibility | `None` |
| `--verbose` | Print progress during the study | `False` |
| `--device` | Hardware device for execution (`auto`, `cpu`, `cuda`, `mps`) | `auto` |

Each experiment generates a figure showing how the chosen parameter influences the final fitness, including a shaded ±1 standard deviation band across repetitions.

---

## Understanding the Network

The class `XORNet` in `neural.py` implements the network using PyTorch's `nn.Sequential`:

```
Input (2)  →  nn.Linear(2, hidden)  →  Activation  →  nn.Linear(hidden, 1)  →  Output (1)
```

All of the network's weights and biases are collected into a **flat genome vector** using PyTorch's utility:

```python
genome = torch.nn.utils.parameters_to_vector(net.parameters())
```

This genome layout (all of `W1`, then `b1`, then `W2`, then `b2`, flattened in the order PyTorch's `parameters()` iterator visits them) is what EvoTorch evolves.

The fitness function evaluates the *entire population at once* rather than one genome at a time: it slices a `(popsize, n_genes)` batch of genomes back into per-individual `W1, b1, W2, b2` matrices and runs one batched matrix multiplication across the whole population (see `make_fitness_fn` in `neural.py`). For each individual, it then:

1. Runs a forward pass on all four XOR inputs.
2. Counts how many of the four outputs have the correct sign.
3. Returns that count divided by 4 as the fitness (0.0 to 1.0).

This batched evaluation is also what makes `--device cuda`/`mps` meaningful (see the note under `--device` above) — a single large batched operation is real GPU work, unlike evaluating one tiny network at a time.

---

## Tips

- Start by running the default configuration before changing any parameters:
  ```bash
  python neural.py --verbose --vizperf --vizbound
  ```
- Read the source code carefully before making modifications.
- Use `--seed` for reproducibility when debugging.
- Change one parameter at a time to isolate its effect.
- Use `study.py` with `--reps 10` or more for reliable estimates of average behavior.
- The decision boundary plot (`--vizbound`) is a powerful diagnostic: a failed run will show an incorrect or degenerate boundary.
- For custom problems, combine options as needed:
  ```bash
  python neural.py --convex 8 --activation sigmoid --gens 300 --verbose
  python study.py --problem random --points 6 --reps 10
  ```

---

## Assignment

### Part 1 – Understand the Network and Fitness Function

Read `neural.py` carefully and answer the following questions before running any experiments:

- How many total weights and biases does a network with 3 hidden neurons have? Count them by layer.
- Why does the fitness function check the *sign* of the output rather than its exact value?
- What fitness score would a completely random genome achieve on average? Why?
- Why can a network with zero hidden neurons (i.e., a linear classifier) never solve XOR?
- The code stops early once perfect fitness (1.0) is reached, and reports the generation this happened at (`convergence_gen`, also printed after each run). Why is it important that this generation be recorded *before* the remaining generations are backfilled with 1.0 for plotting? What would be lost if you tried to recover "generations needed to converge" from the fitness-over-generations plot alone?

Run the default configuration and verify that the network reaches perfect fitness:

```bash
python neural.py --verbose --vizperf --vizbound
```

---

### Part 2 – Explore Neuroevolution Parameters

Investigate how performance changes as you vary:

- number of hidden neurons,
- population size,
- mutation standard deviation,
- number of generations.

Generate plots using `study.py` that illustrate these relationships and explain the observed behavior.

Questions to consider include:

- Does a larger network always converge faster? Is there a point of diminishing returns?
- What happens when the population is very small (e.g., 5 individuals)?
- What is the effect of a very high mutation standard deviation? And a very low one?
- Is there a minimum number of generations needed to reliably find a perfect solution?
- How does variance across independent runs change with population size?

---

### Part 3 – Modify the Network

Extend or alter the code by implementing one or more of the following:

- **Change the activation function**: Compare `tanh`, `sigmoid`, and `relu` using `study.py`. Does the choice of activation affect how quickly the network converges?
- **Add a second hidden layer**: Modify `XORNet` to include two hidden layers. Does this help or hurt performance on XOR? Why?
- **Change the fitness function**: Instead of counting correct signs, implement a smooth fitness that rewards outputs closer to the correct value (e.g., mean squared error against ±1 targets). How does this change evolutionary dynamics?
- **Change the task**: Modify the input/output table to represent a different Boolean function (e.g., AND, OR, XNOR). Is XOR harder or easier than these alternatives?
- **Test on custom problems**: Use `--random` and `--convex` options to evaluate network performance on alternative classification challenges. How does architecture affect performance on convex vs. random problems?

---

### Part 4 – Quantitative Analysis

Collect data from multiple independent runs and analyze:

- The distribution of convergence generation across 20 or more independent seeds.
- Whether the algorithm always converges or sometimes fails entirely.
- The relationship between genome length (number of weights) and convergence speed.
- The tradeoff between population size and number of generations for a fixed evaluation budget.

Support your conclusions with appropriate plots and discussion.

---

## Optional / Advanced Challenge

Parts 1–4 are required (Part 3's modifications already give you several ways to extend the network itself). Beyond that, pick **one** of the following four directions to investigate further. Each is open-ended — there's no single right answer, and the point is to form a hypothesis, run the experiment, and report what you found. Only attempt one; go as deep as you like on it.

**1. Neuroevolution vs. backprop, head to head.** Train the exact same `XORNet` architecture with PyTorch's own gradient descent instead of EvoTorch — a few lines: define a loss (e.g., MSE against ±1 targets), call `loss.backward()`, and step an optimizer. Compare convergence speed and reliability against neuroevolution across many random seeds. Hypothesis: for a network this tiny, does gradient descent actually converge faster and more reliably, or does XOR's well-known symmetric, non-convex loss landscape trip up gradient descent in ways evolution's population-based search avoids?

**2. Genome initialization range.** `run_neuroevolution` samples generation-0 weights uniformly from `initial_bounds=(-1, 1)`. Sweep this range itself (e.g., try `(-3, 3)` or `(-0.1, 0.1)`) instead of any of the Part 2 parameters. Hypothesis: does starting weights too large or too small stall evolution the same way an oversized weight range stalled Project 6's feedforward controller — saturating Tanh and leaving little fitness variation for selection to act on?

**3. Crossover ablation.** Turn off `SimulatedBinaryCrossOver` (mutation-only GA — remove it from the `operators` list) and compare convergence against the default crossover-plus-mutation setup. Hypothesis: for a genome this small (10–20 genes for a few hidden neurons), does crossover actually help, or is the problem small enough that mutation alone finds solutions just as fast?

**4. Robustness of an evolved solution to weight noise.** After evolving a network that reaches perfect fitness, inject increasing amounts of Gaussian noise into its weights (post-hoc, not during evolution) and measure how much noise it takes before it stops solving XOR. Hypothesis: do networks evolved with a larger population or more generations end up more "robust" (a flatter fitness peak) than ones that just barely converged, or is robustness unrelated to how easily a genome converged in the first place?

You're encouraged to explore your own idea beyond these four as well, as long as it's a genuine extension (not just one of Part 3's five modification options, and not just a parameter change already covered in Part 2).

---

## What to Submit to Moodle

Submit a single **written report as a PDF** to Moodle.

### Title Page

The first page of your report should include:

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 3: Neuroevolution (XOR))
- Date submitted
- Amount of time spent on this project
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body of your report into one section per assignment part. Each section should combine the relevant figures with a written discussion — a plot with no interpretation, or an interpretation with no supporting plot, is incomplete.

**Part 1 — Understand the Network and Fitness Function**

- Your answers to the conceptual questions posed in Part 1 (weight/bias counts, why fitness checks sign rather than exact value, expected fitness of a random genome, why zero hidden neurons can't solve XOR, and what's lost by only keeping the backfilled fitness curve instead of the explicit `convergence_gen`).
- Verification that the default configuration reaches perfect fitness — include the fitness-over-generations plot (`--vizperf`) and decision boundary plot (`--vizbound`) from the default run, with a brief caption.

**Part 2 — Explore Neuroevolution Parameters**

- Plots from `study.py` illustrating the effect of each of the four swept parameters: number of hidden neurons, population size, mutation standard deviation, and number of generations.
- For each plot, state what was varied and what was held fixed, and interpret the trend (e.g., where performance plateaus, where it degrades).
- Answers to the five guiding questions from Part 2, supported directly by your plots (diminishing returns from network size, small-population behavior, mutation stdev extremes, minimum generations for reliable convergence, variance vs. population size).

**Part 3 — Modify the Network**

- State which modification(s) you chose to implement (activation function comparison, second hidden layer, smooth/MSE-based fitness function, alternative Boolean function, and/or custom `--random`/`--convex` problems). You only need to implement one or more, not all five.
- For each modification you made: a short description of the change, and plots/results comparing it against the unmodified baseline.
- A discussion of what changed and why — e.g., did the modification affect convergence speed, final fitness, robustness across seeds, or the shape of the decision boundary?

**Part 4 — Quantitative Analysis**

- The distribution of convergence generation across 20+ independent seeds (e.g., a histogram), and whether the algorithm ever fails to converge at all.
- A plot or table relating genome length (number of weights, which grows with hidden neurons) to convergence speed.
- An analysis of the population-size vs. generations tradeoff under a fixed evaluation budget (e.g., popsize × generations held constant).
- Conclusions that are clearly tied to the evidence you present, not just restated observations.

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
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in connecting your Part 1 reasoning about why XOR needs a hidden layer to what you actually observe once you start modifying the network in Part 3.

---

## Further Reading

- Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. (2015). *Deep Learning.* Nature.
- Floreano, D., Dürr, P., & Mattiussi, C. (2008). *Neuroevolution: from architectures to learning.* Evolutionary Intelligence.
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies.* Evolutionary Computation.

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.