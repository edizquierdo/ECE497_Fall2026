# Project 2: Evolutionary Algorithms

## Overview

In this project you will explore one of the fundamental approaches to optimization and search: **evolutionary algorithms**. These bio-inspired algorithms mimic natural selection to solve complex problems without requiring gradient information or exhaustive search.

Your goal is not simply to run the evolutionary algorithm, but to understand how evolution discovers high-fitness solutions and to experimentally investigate how different algorithmic choices affect its performance.

---

## Learning Objectives

By completing this project, you will learn how to:

- Understand the core components of evolutionary algorithms: selection, mutation, and fitness-based survival.
- Analyze how population size, mutation rate, and genome length affect evolutionary search.
- Design quantitative measures of evolutionary progress.
- Experiment with different algorithm parameters (GA vs. ES).
- Perform systematic parameter studies.
- Interpret fitness trajectories over generations.

---

## Background

Evolutionary algorithms are a class of optimization algorithms inspired by biological evolution. They operate on a population of candidate solutions, iteratively improving them through operations that mimic natural selection:

1. **Fitness evaluation**: Each individual is scored based on how well it solves the problem.
2. **Selection**: Higher-fitness individuals are more likely to be selected as parents.
3. **Recombination/Mutation**: New offspring are created through genetic operators.
4. **Replacement**: The population is updated with new individuals.

In this project, we implement two classic evolutionary algorithms:

- **Genetic Algorithm (GA)**: Uses a population of binary/real-coded individuals with mutation and elitism.
- **Separable Natural Evolution Strategy (SNES)**: A gradient-free optimization method that evolves the parameters of a search distribution.

Both approaches will be applied to a simple task: maximizing the number of "ON" genes in a genome.

---

## Project Structure

The project consists of two core Python files.

| File | Purpose |
|------|---------|
| `evolve.py` | Core evolutionary algorithm implementation using EvoTorch. |
| `study.py` | Performs systematic parameter sweeps and generates plots. |
| `requirements.txt` | Pinned dependency versions for this project's virtual environment. |
| `README.md` | Project documentation. |

Most of your work will involve understanding and modifying the code in `evolve.py`.

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

## Running the Evolutionary Algorithm

To run the default evolutionary simulation,

```bash
python evolve.py
```

Useful command-line options include:

| Option | Description | Default |
|---------|-------------|---------|
| `--popsize` | Size of the population | `10` |
| `--gens` | Number of generations to evolve | `10` |
| `--algo` | Algorithm to use ("GA" or "ES") | `GA` |
| `--genesize` | Length of each individual's genome | `10` |
| `--minbound` | Minimum gene value | `0.0` |
| `--maxbound` | Maximum gene value | `1.0` |
| `--mut_stdev` | Standard deviation for Gaussian mutation | `0.1` |
| `--no-elitism` | Disable elitism (elitism is enabled by default) | elitism on |
| `--vizperf` | Visualize fitness over generations | off |
| `--verbose` | Print progress to console | off |
| `--seed` | Random seed for reproducibility | `None` |
| `--device` | Hardware device to use ("auto", "cuda", "cuda:0", "mps", "cpu") | "auto" |
| `--output FILE` | File path to save the best genome, e.g. `best_genome.npy` (the raw continuous genome, not the thresholded 0/1 genotype) | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves across configurations without re-running evolution | `None` |
| `--seed_genome FILE` | Seed the initial population around a genome saved by a previous `--output` run (must match `--genesize`) instead of starting from scratch — one exact copy plus the rest perturbed by `--seed_noise`. Only supported with `--algo GA`, not `--algo ES` | `None` |
| `--seed_noise` | Stdev of the Gaussian perturbation applied to `--seed_genome` copies | `0.05` |

For example,

```bash
python evolve.py --popsize 20 --gens 100 --vizperf --device auto
```

or

```bash
python evolve.py --algo ES --mut_stdev 0.05 --verbose --device cuda
```

---

## GPU Acceleration & Hardware Support

The evolutionary implementation leverages PyTorch and EvoTorch tensor operations with vectorized evaluation (`vectorized=True`) for high throughput across GPU devices.

- **Auto Detection (`--device auto`)**: Automatically detects and selects CUDA (`cuda`) on NVIDIA GPU workstations, MPS (`mps`) on Apple Silicon, or falls back to CPU (`cpu`).
- **Explicit Device (`--device cuda:0`, `--device mps`, `--device cpu`)**: Allows targeting a specific GPU index or processor device.

Running on GPU accelerates large population runs and parameter studies by batching candidate evaluations directly in GPU memory.

You are welcome to ask your instructor for details on logging in to our local GPU workstation. Although the problems in this project are not computationally intensive enough to show a noticeable speedup, it may be useful to get comfortable logging into the GPU workstation for the more advanced projects we will work on later in the course.

---

## Performing Parameter Studies

The file `study.py` automates experiments in which the mutation standard deviation (`mut_stdev`) is varied over a specified range.

For example:

```bash
python study.py --min 0.01 --max 0.5 --steps 21 --device auto
```

or with custom arguments:

```bash
python study.py --min 0.01 --max 0.2 --steps 10 --algo ES --popsize 20 --reps 10 --device cuda --output study_mut_stdev_es.png
```

Each experiment runs multiple repetitions of the evolutionary algorithm for each `mut_stdev` value and saves a visualization plot showing the final best fitness as a function of the parameter.

Useful command-line options for `study.py` include:

| Option | Description | Default |
|--------|-------------|---------|
| `--min` | Minimum `mut_stdev` value | `0.01` |
| `--max` | Maximum `mut_stdev` value | `0.5` |
| `--steps` | Number of steps between min and max | `21` |
| `--gens` | Number of generations per run | `50` |
| `--reps` | Number of repetitions per parameter value | `5` |
| `--algo` | Evolutionary algorithm to use ("GA" or "ES") | `GA` |
| `--popsize` | Population size | `10` |
| `--genesize` | Gene size / genome length | `10` |
| `--output` | Output filename for plot | `study_mut_stdev.png` |
| `--seed` | Random seed for reproducibility | `None` |
| `--verbose` | Print progress information | off |
| `--device` | Hardware device to use ("auto", "cuda", "cuda:0", "mps", "cpu") | "auto" |

---

## Understanding the Evolutionary Process

The evolutionary algorithm in `evolve.py` implements:

- A population of individuals, each with a genome of **real-valued genes** (initialized uniformly in [0, 1])
- A fitness function that counts genes with values > 0.5 as "ON"
- Gaussian mutation to introduce variation
- Elitism to preserve the best solution across generations

**Important note:** Genes are continuous during evolution; the 0.5 threshold is only applied when computing fitness and producing the final genotype output.

At each generation:
1. Individuals are evaluated and scored based on their fitness
2. The best, average, and worst fitness are recorded
3. New individuals are created through mutation of parents

By monitoring these metrics over time, you can observe how the population converges toward higher-fitness solutions.

---

## Tips

- Start by running the default configuration before changing any parameters.
- Read the source code carefully to understand how EvoTorch works.
- Change one parameter at a time to isolate effects.
- Use random seeds when you need reproducible experiments.
- Visualize both individual runs (via `--vizperf`) and parameter studies.

---

## Assignment

### Part 1 – Understand the Fitness Function

Analyze the `count_ones` function in `evolve.py`:

- Why does it divide by genome length?
- What would happen if you removed this normalization?
- How would fitness change if genes were truly binary (0 or 1)?

**Note:** Genes are real-valued during evolution and only thresholded at 0.5 for fitness evaluation. This means mutation works on continuous values, but the fitness function treats them as binary after thresholding.

---

### Part 2 – Explore Algorithm Parameters

Investigate how behavior changes as you vary:

- population size
- mutation standard deviation
- number of generations
- genome length
- algorithm choice (GA vs. ES)

Generate plots that illustrate these relationships and explain the observed performance.

Questions to consider include:

- What happens when the population is too small?
- Can a very high mutation rate prevent convergence?
- Is there an optimal mutation rate? Why or why not?
- How does GA compare to ES in terms of speed and quality?

---

### Part 3 – Change the Fitness Function

Maximizing ones is a relatively trivial problem. Implement one or more alternative fitness functions:

- **Step function**: Reward specific patterns (e.g., alternating ON/OFF genes)
- **Rastrigin-like**: Create a rugged landscape with many local optima
- **Sparse reward**: Only reward individuals with > X% of genes ON

Evaluate how these changes affect evolutionary dynamics.

---

### Part 4 – Quantitative Analysis

Collect data from multiple runs and analyze:

- The shape of fitness trajectories (early rapid progress vs. late-stage refinement)
- Variance in final fitness across independent runs
- Correlation between population diversity and performance
- The impact of elitism on convergence

Support your conclusions using appropriate plots and statistics.

---

## Optional / Advanced Challenge

Parts 1–4 are required. Beyond that, pick **one** of the following four directions to investigate further. Each is open-ended — there's no single right answer, and the point is to form a hypothesis, run the experiment, and report what you found. Only attempt one; go as deep as you like on it.

**1. Solve a genuinely hard combinatorial problem.** Pick one: the Traveling Salesperson Problem (find the shortest route visiting a list of cities and returning home), the Knapsack Problem (choose which valuable items to pack without exceeding a weight limit), or a Scheduling/Routing problem (assign jobs to workers or trucks to delivery stops under time and capacity constraints). None of these fit the fixed-length real-valued genome `evolve.py` currently uses — you'll need a different genome representation (e.g., a permutation of city indices for TSP, decoded inside a custom `objective_func`) and possibly different operators. How much of `run_evolution`'s structure (population, elitism, generation loop) can you reuse unchanged, and how much has to change because the representation itself changed?

**2. Self-adaptive mutation.** Instead of a fixed `--mut_stdev`, make each individual's own mutation strength part of its genome — a classic self-adaptation trick: extend the genome by one gene encoding that individual's own stdev, and mutate the solution and its stdev gene together each generation. Hypothesis: does self-adaptation reach solution quality comparable to your best hand-tuned `--mut_stdev` from Part 2, without you having to sweep it manually?

**3. A simple diversity-preservation mechanism.** Add fitness sharing (penalize individuals too similar to others in the population) or a restart-on-stagnation rule (reinitialize part of the population if best fitness hasn't improved in N generations) to `run_evolution`. Hypothesis: does this change the "population too small" failure mode you characterized in Part 2 — does a small population with diversity preservation behave more like a larger one, or does it just fail differently?

**4. GA vs. ES on a genuinely multimodal landscape.** Take your Rastrigin-like fitness function from Part 3 and run a controlled head-to-head between GA and ES specifically on *how many local optima each escapes*, not just final fitness or speed. Does whichever algorithm "won" on the trivial `count_ones` task in Part 2 still win here, or does a rugged landscape change which algorithm is actually better suited to the problem?

You're encouraged to explore your own idea beyond these four as well, as long as it's a genuine extension (not just a parameter change already covered in Parts 2–4).

---

## What to Submit to Moodle

Submit a single **written report as a PDF** to Moodle.

### Title Page

The first page of your report should include:

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 2: Evolutionary Algorithms)
- Date submitted
- Amount of time spent on this project
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body of your report into one section per assignment part. Each section should combine the relevant figures with a written discussion — a plot with no interpretation, or an interpretation with no supporting plot, is incomplete.

**Part 1 — Understand the Fitness Function**

- Your answers to the questions posed in Part 1.
- A brief explanation, in your own words, of what the `count_ones` function computes and why.

**Part 2 — Explore Algorithm Parameters**

- Plots illustrating the effect of population size, mutation standard deviation, number of generations, genome length, and algorithm choice (GA vs. ES).
- For each plot, describe the experimental setup (what was varied, what was held fixed) and interpret the trend you observe.
- Answers to the guiding questions from Part 2, supported by your results.

**Part 3 — Change the Fitness Function**

- A description of the alternative fitness function(s) you implemented.
- Plots comparing evolutionary dynamics under the new fitness function(s) to the baseline `count_ones` task.
- A discussion of how and why the dynamics changed (e.g., convergence speed, final fitness, sensitivity to parameters).

**Part 4 — Quantitative Analysis**

- Plots and summary statistics addressing each of the four questions in Part 4 (trajectory shape, variance across runs, diversity vs. performance, effect of elitism).
- Conclusions that are clearly tied to the evidence you present.

**Optional / Advanced Challenge** *(if attempted)*: a section naming which of the four directions you chose (or your own idea), what you changed, your results (with supporting figures), and your interpretation. Omit this section if you didn't attempt a challenge.

### General Guidelines

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
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in connecting the parameter intuitions from Part 2 to how they hold up (or don't) on the harder fitness landscapes in Part 3.

---

## Further Reading

- Fogel, D. B. (2006). *Evolutionary Computation: The Fuzzy Theory of Evolution.*
- Mitchell, M. (1998). *An Introduction to Genetic Algorithms.*
- Beyer, H. G., & Schwefel, H. P. (2002). *Evolution Strategies: A Comprehensive Introduction.*

---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.
