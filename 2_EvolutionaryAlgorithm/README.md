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

- **Genetic Algorithm (GA)**: Uses a population of binary/real-coded individuals with crossover, mutation, and elitism.
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
| `--algorithm` | Algorithm to use ("GA" or "ES") | `GA` |
| `--genesize` | Length of each individual's genome | `10` |
| `--minbound` | Minimum gene value (also the initial sampling range's lower bound) | `0.0` |
| `--maxbound` | Maximum gene value (also the initial sampling range's upper bound) | `1.0` |
| `--mut_stdev` | Standard deviation for Gaussian mutation | `0.1` |
| `--tournament_size` | Tournament size for SBX crossover (GA only) | `3` |
| `--eta` | Distribution index for SBX crossover (GA only) | `20` |
| `--no-crossover` | Disable SBX crossover, running a mutation-only GA (GA only) | crossover on |
| `--no-elitism` | Disable elitism (elitism is enabled by default) | elitism on |
| `--vizperf` | Visualize fitness over generations | off |
| `--verbose` | Print progress to console | off |
| `--seed` | Random seed for reproducibility | `None` |
| `--device` | Hardware device to use ("auto", "cuda", "cuda:0", "mps", "cpu") | "auto" |
| `--output FILE` | File path to save the best genome, e.g. `best_genome.npy` (the raw continuous genome, not the thresholded 0/1 genotype) | `None` |
| `--fitness_output FILE` | Save per-generation best/avg/worst fitness to this `.npz` file (keys `best`/`avg`/`worst`), so you can reload and compare fitness curves across configurations without re-running evolution | `None` |
| `--seed_genome FILE` | Seed the initial population around a genome saved by a previous `--output` run (must match `--genesize`) instead of starting from scratch — one exact copy plus the rest perturbed by `--seed_noise`. Only supported with `--algorithm GA`, not `--algorithm ES` | `None` |
| `--seed_noise` | Stdev of the Gaussian perturbation applied to `--seed_genome` copies | `0.05` |
| `--fitness` | Fitness function to optimize: `count_ones`, `step`, `rastrigin`, or `sparse`. Only `count_ones` is implemented out of the box — the other three are stubs you'll implement in Part 3 (see below) | `count_ones` |
| `--sparse_threshold` | Fraction of genes that must be ON before the `sparse` fitness function gives any reward. Ignored unless `--fitness sparse` | `0.7` |

For example,

```bash
python evolve.py --popsize 20 --gens 100 --vizperf --device auto
```

or

```bash
python evolve.py --algorithm ES --mut_stdev 0.05 --verbose --device cuda
```

or, comparing crossover on vs. off (GA only):

```bash
python evolve.py --tournament_size 5 --eta 10 --vizperf
python evolve.py --no-crossover --vizperf
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
python study.py --min 0.01 --max 0.2 --steps 10 --algorithm ES --popsize 20 --reps 10 --device cuda --output study_mut_stdev_es.png
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
| `--algorithm` | Evolutionary algorithm to use ("GA" or "ES") — held fixed across the sweep | `GA` |
| `--popsize` | Population size | `10` |
| `--genesize` | Gene size / genome length | `10` |
| `--output` | Output filename for plot | `study_mut_stdev.png` |
| `--data_output` | Save the raw sweep data (mut_stdev values and their scores) to this `.npz` file (keys `mut_stdev_values`/`scores`), so you can reload and re-plot a sweep without re-running it | `None` |
| `--seed` | Random seed for reproducibility | `None` |
| `--verbose` | Print progress information | off |
| `--device` | Hardware device to use ("auto", "cuda", "cuda:0", "mps", "cpu") | "auto" |
| `--fitness` | Fitness function to sweep over (see `evolve.py`'s `--fitness`) | `count_ones` |
| `--sparse_threshold` | Threshold for the `sparse` fitness function | `0.7` |

**`study.py` only sweeps `mut_stdev`.** Part 2 below asks you to investigate five things — population
size, mutation standard deviation, number of generations, genome length, and algorithm choice — but
the shipped `study.py` only automates one of them. For the other four, you'll write your own small
sweep script (or extend a copy of `study.py`) following the same pattern it uses: loop over a range
of values, repeat several times per value to average out randomness, and plot the result with error
bars. This is a pattern you'll rely on throughout the course, in increasingly self-directed form as
you go — `study.py` here is a fully worked example of it, worth reading closely before you write your
own version.

---

## Understanding the Evolutionary Process

The evolutionary algorithm in `evolve.py` implements:

- A population of individuals, each with a genome of **real-valued genes** (initialized uniformly in [0, 1])
- A fitness function that counts genes with values > 0.5 as "ON"
- SBX crossover (GA only, on by default — disable with `--no-crossover`) and Gaussian mutation to introduce variation
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
- crossover on vs. off (GA only, `--no-crossover`)

Generate plots that illustrate these relationships and explain the observed performance.

Questions to consider include:

- What happens when the population is too small?
- Can a very high mutation rate prevent convergence?
- Is there an optimal mutation rate? Why or why not?
- How does GA compare to ES in terms of speed and quality?
- Does disabling crossover change convergence speed, final fitness, or both? For a genome this
  small, is crossover doing much work at all?

**Note on comparing GA and ES:** look closely at how `run_evolution` constructs the `Problem` for each algorithm. GA keeps `bounds=(minbound, maxbound)` enforced every generation, but ES's `bounds` is set to `None` — genes are only constrained to `[minbound, maxbound]` at generation-0 initialization, and nothing stops ES's search distribution from drifting outside that range afterward. Keep this in mind when interpreting a GA-vs-ES comparison: a difference in final fitness might reflect this asymmetry rather than (or in addition to) a genuine difference in search quality between the two algorithms.

---

### Part 3 – Change the Fitness Function

Maximizing ones is a relatively trivial problem. Implement one or more alternative fitness functions:

- **Step function**: Reward specific patterns (e.g., alternating ON/OFF genes)
- **Rastrigin-like**: Create a rugged landscape with many local optima
- **Sparse reward**: Only reward individuals with > X% of genes ON

`evolve.py` already has the plumbing to select a fitness function at the command line (`--fitness {count_ones,step,rastrigin,sparse}`, plus `--sparse_threshold` for the sparse case — see the CLI options table above) and a `count_ones` implementation to use as a reference. What's missing is the actual logic: `step_pattern`, `rastrigin_like`, and `sparse_reward` in `evolve.py` are stubs (marked `TODO (Part 3)`, each currently raising `NotImplementedError`) — your job is to fill in their bodies. You don't need to implement all three; pick at least one to evaluate below.

Evaluate how these changes affect evolutionary dynamics.

---

### Part 4 – Quantitative Analysis

Collect data from multiple runs and analyze:

- The shape of fitness trajectories (early rapid progress vs. late-stage refinement)
- Variance in final fitness across independent runs
- Correlation between population diversity and performance
- The impact of elitism on convergence

Support your conclusions using appropriate plots and statistics.

**Note on population diversity:** `run_evolution` currently only tracks and returns best/average/worst *fitness* per generation — it doesn't expose anything about the population's spread of gene values. Unlike the other three bullets above, which can be answered by calling `run_evolution` from an external script and analyzing its existing return values, the diversity-vs-performance bullet requires you to modify `run_evolution` itself: inside the generation loop, after `algorithm.step()`, use `algorithm.population.access_values()` to read the current population's gene values and compute some measure of spread across individuals (e.g., the standard deviation of each gene across the population, averaged over genes), then record and return it alongside the fitness arrays.

---

## Optional / Advanced Challenge

Parts 1–4 are required. Beyond that, pick **one** of the following three directions to investigate further. Each is open-ended — there's no single right answer, and the point is to form a hypothesis, run the experiment, and report what you found. Only attempt one; go as deep as you like on it.

**1. Write a simple evolutionary algorithm from scratch.** Instead of relying on EvoTorch's built-in `GeneticAlgorithm`, implement your own minimal EA loop directly in plain Python/NumPy — no EvoTorch algorithm classes involved (you can still use an EvoTorch `Problem` for fitness evaluation if you find it convenient, or skip EvoTorch entirely). A good target is a **Microbial GA**: repeatedly pick two random individuals, have the loser copy (part of) the winner's genome with mutation, and put both back in the population — no explicit generations, no separate offspring population, just pairwise "infections" over time. Hypothesis: how does its behavior (convergence speed, final fitness, sensitivity to population size) compare to the `GeneticAlgorithm`-based results from Part 2?

**2. Try an existing algorithm from EvoTorch you haven't used yet.** `evolve.py` only uses `GeneticAlgorithm` and `SNES`, but EvoTorch's `evotorch.algorithms` module ships several others, including `CEM` (Cross-Entropy Method), `CMAES`/`PyCMAES` (Covariance Matrix Adaptation ES), `Cosyne` (Cooperative Synapse Neuroevolution), `MAPElites` (a quality-diversity algorithm — note this one optimizes for a diverse archive of solutions rather than a single best one, so it needs a bit more setup), `PGPE` (Policy Gradients with Parameter-based Exploration), and `SteadyStateGA`. Swap one in for GA/ES in a copy of `run_evolution` and run it on the same `count_ones` task (and, if you like, your Part 3 fitness function too). Hypothesis: how does it compare to GA and ES on speed, final fitness, and sensitivity to its own hyperparameters? Check that algorithm's EvoTorch documentation for what those hyperparameters are and what they control.

**3. Explore an optimization paradigm beyond evolutionary algorithms.** EAs are one family within a broader landscape of derivative-free optimization methods. Pick one adjacent paradigm — for example **particle swarm optimization** (individuals are "particles" with velocity, pulled toward their own best-known position and the swarm's best-known position, rather than selected/mutated) or **multi-objective optimization** (optimizing for more than one fitness criterion at once, where there's no single "best" individual but a Pareto front of trade-offs) — and produce a short comparative write-up: how does its core mechanism differ from the GA/ES approach in this project, what kinds of problems is it particularly well- or poorly-suited for, and (if you have time to implement a small example, even a toy one) how does it perform on `count_ones` or your Part 3 fitness function? This option can be primarily a research/comparison exercise rather than a full implementation if your chosen paradigm doesn't have a convenient existing library to build on.

You're encouraged to explore your own idea beyond these three as well, as long as it's a genuine extension (not just a parameter change already covered in Parts 2–4).

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

- Plots illustrating the effect of population size, mutation standard deviation, number of generations, genome length, algorithm choice (GA vs. ES), and crossover on vs. off.
- For each plot, describe the experimental setup (what was varied, what was held fixed) and interpret the trend you observe.
- Answers to the guiding questions from Part 2, supported by your results.

**Part 3 — Change the Fitness Function**

- A description of the alternative fitness function(s) you implemented.
- Plots comparing evolutionary dynamics under the new fitness function(s) to the baseline `count_ones` task.
- A discussion of how and why the dynamics changed (e.g., convergence speed, final fitness, sensitivity to parameters).

**Part 4 — Quantitative Analysis**

- Plots and summary statistics addressing each of the four questions in Part 4 (trajectory shape, variance across runs, diversity vs. performance, effect of elitism).
- Conclusions that are clearly tied to the evidence you present.

**Optional / Advanced Challenge** *(if attempted)*: a section naming which of the three directions you chose (or your own idea), what you changed, your results (with supporting figures), and your interpretation. Omit this section if you didn't attempt a challenge.

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
