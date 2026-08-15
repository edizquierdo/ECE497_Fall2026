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
| `evolve.py` | Defines the network, fitness function, problem generators, and runs neuroevolution. |
| `study.py` | Performs systematic parameter sweeps and generates plots. |
| `requirements.txt` | Pinned dependency versions for this project's virtual environment. |
| `README.md` | Project documentation. |

Most of your modifications will involve understanding and extending the code in `evolve.py`.

### Alternative Problem Generators

In addition to the standard XOR task, `evolve.py` includes generators for custom classification problems:

- **Random problem**: Points sampled uniformly from `[-0.5, 1.5]²` with random labels (+1/-1).
- **Convex problem**: Points placed on a circle at equal angles with alternating labels.

These generalize the XOR task and allow investigation of how network architecture affects performance on different classification challenges.

> **Note on `--seed`:** it controls both the evolutionary algorithm's randomness and the *random* problem generator's point placement. The *convex* problem generator is fully deterministic (points sit at fixed angles on a circle) — `--seed` has no effect on which points a `--convex N` run uses, only on the EA.

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
python evolve.py
```

Useful command-line options include:

| Option | Description | Default |
|--------|-------------|---------|
| `--hidden` | Number of hidden neurons in a single hidden layer | `3` |
| `--hidden_sizes N [N ...]` | List of hidden-layer sizes, e.g. `--hidden_sizes 4 4` for two hidden layers of 4 neurons each. Overrides `--hidden` when given | `None` |
| `--popsize` | Population size | `100` |
| `--gens` | Number of generations | `200` |
| `--mut_stdev` | Gaussian mutation standard deviation | `0.5` |
| `--tournament_size` | Tournament size for SBX crossover | `3` |
| `--eta` | Distribution index for SBX crossover | `20` |
| `--no-crossover` | Disable SBX crossover, running a mutation-only GA | crossover on |
| `--no-elitism` | Disable elitism (elitism is enabled by default) | elitism on |
| `--init_bounds LOW HIGH` | Initial genome sampling bounds | `-1.0 1.0` |
| `--activation` | Hidden layer activation function (`tanh`, `sigmoid`, or `relu`) | `tanh` |
| `--fitness_mode` | Fitness function: `sign` (fraction of points classified with the correct sign) or `mse` (smooth 1 − normalized-MSE against the ±1 targets) | `sign` |
| `--task` | Boolean truth table to solve when `--random`/`--convex` aren't given (`xor`, `and`, `or`, `xnor`) | `xor` |
| `--vizperf` | Plot fitness over generations | `False` |
| `--vizbound` | Plot the decision boundary of the best evolved network | `False` |
| `--verbose` | Print per-generation statistics to the console | `False` |
| `--seed` | Random seed for reproducibility | `None` |
| `--random` | Number of random data points instead of XOR | `None` |
| `--convex` | Number of convex-position data points on a circle | `None` |
| `--device` | Hardware device for execution (`auto`, `cpu`, `cuda`, `mps`) | `auto` |
| `--output FILE` | File path to save the best evolved genome, e.g. `best_genome.npy` | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves across configurations without re-running evolution | `None` |
| `--seed_genome FILE` | Seed the initial population around a genome saved by a previous `--output` run (must match the current `--hidden`/`--hidden_sizes`/`--activation`) instead of starting from scratch — one exact copy plus the rest perturbed by `--seed_noise` | `None` |
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
python evolve.py --vizperf --vizbound
```

or

```bash
python evolve.py --hidden 5 --popsize 50 --gens 300 --verbose
```

or, using a couple of the other options above together:

```bash
python evolve.py --hidden_sizes 4 4 --activation relu --fitness_mode mse --vizperf --verbose
python evolve.py --task xnor --no-crossover --init_bounds -3 3 --verbose
```

Or run on GPU workstations or custom devices:

```bash
python evolve.py --device cuda --popsize 500 --gens 1000
python evolve.py --random 8 --vizbound
python evolve.py --convex 6 --activation sigmoid --vizperf
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
| `--activation` | Fixed (non-swept) hidden layer activation function | `tanh` |
| `--fitness_mode` | Fixed (non-swept) fitness function (`sign` or `mse`) | `sign` |
| `--task` | Fixed (non-swept) boolean task when `--problem xor` | `xor` |
| `--hidden_sizes N [N ...]` | Fixed (non-swept) hidden-layer-sizes override. Meaningless when `--param hidden`, since that sweeps the single `--hidden` value `hidden_sizes` would override | `None` |

Each experiment generates a figure showing how the chosen parameter influences the final fitness, including a shaded ±1 standard deviation band across repetitions.

> **Note:** unlike `evolve.py`'s `--vizperf`/`--vizbound` (which open an interactive plot window and don't save to disk), `study.py` always writes its plot straight to the `--output` file and never opens a window — look for the saved PNG rather than expecting a popup.

### Building Your Own Experiment Scripts

`study.py` is a worked example of a pattern you'll rely on throughout this
course: sweep one parameter, repeat several times per value to average out
randomness, save the results systematically, and produce a labeled figure
with error bars rather than a single noisy run. Read it before you need to
write something like it yourself — later projects increasingly expect you to
build this kind of tooling on your own rather than have it handed to you, so
it's worth understanding *why* `study.py` is structured the way it is (fixed
problem instance held constant across repetitions, `reps` independent seeds
per value, mean ± std recorded and plotted) and not just *that* it works.

---

## Understanding the Network

The class `XORNet` in `evolve.py` implements the network using PyTorch's `nn.Sequential`:

```
Input (2)  →  nn.Linear(2, hidden)  →  Activation  →  nn.Linear(hidden, 1)  →  Output (1)
```

All of the network's weights and biases are collected into a **flat genome vector** using PyTorch's utility:

```python
genome = torch.nn.utils.parameters_to_vector(net.parameters())
```

This genome layout (all of `W1`, then `b1`, then `W2`, then `b2`, flattened in the order PyTorch's `parameters()` iterator visits them) is what EvoTorch evolves. If you want to check a by-hand parameter count from Part 1 programmatically, `genome_size(hidden=..., hidden_sizes=...)` computes it directly from the layout, without building an actual network.

The fitness function evaluates the *entire population at once* rather than one genome at a time: it slices a `(popsize, n_genes)` batch of genomes back into per-individual `W1, b1, W2, b2` matrices and runs one batched matrix multiplication across the whole population (see `make_fitness_fn` in `evolve.py`). For each individual, it then:

1. Runs a forward pass on all four XOR inputs.
2. Counts how many of the four outputs have the correct sign.
3. Returns that count divided by 4 as the fitness (0.0 to 1.0).

This batched evaluation is also what makes `--device cuda`/`mps` meaningful (see the note under `--device` above) — a single large batched operation is real GPU work, unlike evaluating one tiny network at a time.

---

## Tips

- Start by running the default configuration before changing any parameters:
  ```bash
  python evolve.py --verbose --vizperf --vizbound
  ```
- Read the source code carefully before making modifications.
- Use `--seed` for reproducibility when debugging.
- Change one parameter at a time to isolate its effect.
- Use `study.py` with `--reps 10` or more for reliable estimates of average behavior.
- Most default-configuration runs (`hidden=3`, `popsize=100`) converge in well under 50 generations — the default `gens=200` is intentionally generous headroom, not a tight budget, so a run that takes longer isn't necessarily a sign something's wrong.
- The decision boundary plot (`--vizbound`) is a powerful diagnostic: a failed run will show an incorrect or degenerate boundary.
- For custom problems, combine options as needed:
  ```bash
  python evolve.py --convex 8 --activation sigmoid --gens 300 --verbose
  python study.py --problem random --points 6 --reps 10
  ```

---

## Assignment

### Part 1 – Understand the Network and Fitness Function

Read `evolve.py` carefully and answer the following questions before running any experiments:

- How many total weights and biases does a network with 3 hidden neurons have? Count them by layer. Then check your count against `genome_size(hidden=3)` — do they agree?
- Why does the fitness function check the *sign* of the output rather than its exact value?
- What fitness score would a completely random genome achieve on average? Why?
- Why can a network with zero hidden neurons (i.e., a linear classifier) never solve XOR?
- The code stops early once perfect fitness (1.0) is reached, and reports the generation this happened at (`convergence_gen`, also printed after each run). Why is it important that this generation be recorded *before* the remaining generations are backfilled with 1.0 for plotting? What would be lost if you tried to recover "generations needed to converge" from the fitness-over-generations plot alone?

Run the default configuration and verify that the network reaches perfect fitness:

```bash
python evolve.py --verbose --vizperf --vizbound
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

### Part 3 – Explore the Rest of the Neural Controller

`evolve.py` already ships working support for every direction below via CLI
flags — this part is about designing a fair comparison and interpreting what
you see, not about writing new code (Parts 1–2 already exercised `--hidden`;
this part is where you exercise everything else the controller can do).
Investigate at least two of the following:

- **Activation function**: Compare `--activation tanh`, `sigmoid`, and `relu`, holding everything else fixed. Does the choice of activation affect how quickly the network converges, or how reliably it does?
- **Network depth**: Compare a single hidden layer (`--hidden`) against two hidden layers of the same width (`--hidden_sizes h h`) at *similar total parameter count* — not matching neuron counts: two hidden layers of size 4 have 37 weights and biases total, while a single hidden layer needs roughly 8 neurons to reach a similar count, since the hidden-to-hidden connection alone costs `h²` weights (`genome_size(hidden=..., hidden_sizes=...)` computes either count directly). Does depth help or hurt performance on XOR? Why?
- **Fitness function**: Compare `--fitness_mode sign` (today's default: fraction of points with the correct sign) against `--fitness_mode mse` (a smooth alternative: tanh-squashed output vs. the ±1 target, mean squared error). How does the shape of the fitness landscape change evolutionary dynamics — smoother convergence, different failure modes, more or less sensitivity to `--mut_stdev`?
- **Task**: Compare `--task xor` against `--task and`, `or`, or `xnor` (same four corner points, different labels). Is XOR harder or easier than these alternatives? What does that say about which of the four is/isn't linearly separable?
- **Custom problems**: Use `--random` and `--convex` to evaluate network performance on alternative classification challenges. How does architecture affect performance on convex vs. random problems?

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

Parts 1–4 are required (Part 3's comparisons already give you several ways to explore the network itself). Beyond that, pick **one** of the following four directions to investigate further. Each is open-ended — there's no single right answer, and the point is to form a hypothesis, run the experiment, and report what you found. Only attempt one; go as deep as you like on it. Note: #1 and #4 require you to write new code; #2 and #3 are pre-built CLI options where the real work is in the experimental design and interpretation, not new code — pick whichever kind of challenge you'd rather spend your time on.

**1. Neuroevolution vs. backprop, head to head.** Train the exact same `XORNet` architecture with PyTorch's own gradient descent instead of EvoTorch — a few lines: define a loss (e.g., MSE against ±1 targets), call `loss.backward()`, and step an optimizer. Compare convergence speed and reliability against neuroevolution across many random seeds. Hypothesis: for a network this tiny, does gradient descent actually converge faster and more reliably, or does XOR's well-known symmetric, non-convex loss landscape trip up gradient descent in ways evolution's population-based search avoids?

**2. Genome initialization range.** `--init_bounds LOW HIGH` controls the range generation-0 weights are sampled from (default `-1.0 1.0`) — no code change needed, just the flag. Sweep this range itself (e.g., `--init_bounds -3 3` or `--init_bounds -0.1 0.1`) instead of any of the Part 2 parameters. Hypothesis: does starting weights too large or too small stall evolution — e.g. by saturating Tanh and leaving little fitness variation for selection to act on?

**3. Crossover ablation.** `--no-crossover` runs a mutation-only GA (`SimulatedBinaryCrossOver` removed from the operator list) — no code change needed, just the flag. Compare convergence against the default crossover-plus-mutation setup. Hypothesis: for a genome this small (10–20 genes for a few hidden neurons), does crossover actually help, or is the problem small enough that mutation alone finds solutions just as fast?

**4. Robustness of an evolved solution to weight noise.** After evolving a network that reaches perfect fitness, inject increasing amounts of Gaussian noise into its weights (post-hoc, not during evolution) and measure how much noise it takes before it stops solving XOR. Hypothesis: do networks evolved with a larger population or more generations end up more "robust" (a flatter fitness peak) than ones that just barely converged, or is robustness unrelated to how easily a genome converged in the first place?

You're encouraged to explore your own idea beyond these four as well, as long as it's a genuine extension (not just one of Part 3's five comparison options, and not just a parameter change already covered in Part 2).

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

**Part 3 — Explore the Rest of the Neural Controller**

- State which comparison(s) you chose to investigate (activation function, network depth, fitness function, alternative Boolean task, and/or custom `--random`/`--convex` problems). You only need at least two, not all five.
- For each comparison: a short description of what you held fixed and what you varied, and plots/results comparing it against the baseline configuration.
- A discussion of what changed and why — e.g., did the comparison affect convergence speed, final fitness, robustness across seeds, or the shape of the decision boundary?

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