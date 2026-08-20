# Project 1: Braitenberg Vehicles

## Overview

In this project you will explore one of the classic ideas in autonomous robotics: **Braitenberg vehicles**. Although these robots have extremely simple control systems, they are capable of producing surprisingly complex behavior through the interaction of sensing, control, and the environment.

Your goal is not simply to run the simulator, but to understand how behavior emerges from the robot's design and to experimentally investigate how different design choices affect its performance.

---

## Learning Objectives

By completing this project, you will learn how to:

- Understand how simple sensorimotor connections produce complex behavior.
- Analyze the relationship between sensing, control, and movement.
- Design quantitative measures of behavioral performance.
- Experiment with different controller parameters.
- Perform systematic parameter studies.
- Modify a robotic controller and evaluate the consequences of those changes.

- Visualize data, interpret results, generate insights and new experiments, and archive results.
- Generate hypotheses and tests their validity through experiments. 
---

## Background

In his influential 1984 book *Vehicles: Experiments in Synthetic Psychology*, Valentino Braitenberg demonstrated that surprisingly sophisticated behaviors can emerge from extremely simple robots.

The robot in this project consists of only:

- two light sensors,
- two wheel motors, and
- a direct connection from sensors to motors.

There is no map, no planning algorithm, no neural network controller, no memory, and no learning. Nevertheless, the robot is capable of moving toward a light source simply because of how its sensors are wired to its motors.

This project illustrates one of the central ideas of embodied intelligence:

> Complex behavior does not necessarily require a complex controller.

Instead, intelligent-looking behavior can emerge from the interaction between the robot, its body, and its environment.

One of the cleverest aspects of Braitenberg's original design is *how* the sensors are wired to the motors. Braitenberg's Vehicles 2 and 3 combine two design choices: whether the connection is **ipsilateral** (same side) or **contralateral** (crossed), and whether it is **excitatory** or **inhibitory**. Together these give four combinations, each producing a qualitatively different behavior:

| Wiring | Excitatory | Inhibitory |
|---|---|---|
| **Ipsilateral (direct)** | Fear | Liking |
| **Contralateral (crossed)** | Aggression | Love |

These four behaviors are Braitenberg's names for what are, biologically, simple positive and negative taxes (approach and avoidance responses) found throughout the animal kingdom. In parallel, you have been asked to read the relevant chapters in *Vehicles*, so use this as complementary material.

The simulator supports the two excitatory wiring schemes:

- **Crossed (contralateral)**: the left sensor drives the right motor and the right sensor drives the left motor. (Aggression.)
- **Direct (ipsilateral)**: each sensor drives the motor on the same side of the body. (Fear.)

Neither is implemented yet — that wiring is exactly what you will write in Part 1 of the assignment, and it's what turns this bare-bones scaffolding into an actual light-seeking robot.

---

## Project Structure

The project consists of the following files.

| File | Purpose |
|------|---------|
| `braitenberg.py` | Defines the robot and the light source. |
| `sim.py` | Runs one or more simulations. |
| `study.py` | Performs systematic parameter sweeps and generates plots. |
| `requirements.txt` | Pinned dependency versions for this project's virtual environment. |
| `README.md` | Project documentation. |

Most of your modifications will involve understanding and extending the code in `braitenberg.py` and `sim.py`.

---

## Installation

The project requires:

- Python 3.7 or newer
- NumPy
- Matplotlib

Both packages are listed with tested version ranges in `requirements.txt`, so install them together into a project-specific virtual environment rather than into your system Python.

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

#### Notes for Windows 11 (Lenovo) Laptops

Most of you are working on Rose-Hulman-issued Lenovo laptops running Windows 11. A few tips to avoid common setup issues:

- **Use PowerShell or the Windows Terminal**, not the old `cmd.exe`, for the commands in this README.
- **Check that Python is installed and on your PATH** by running `python --version` in PowerShell. If Windows opens the Microsoft Store instead of printing a version number, Python isn't installed yet — install it from [python.org](https://www.python.org/downloads/) and make sure to check **"Add python.exe to PATH"** during setup, or install it from the Microsoft Store.
- If `python` isn't recognized but `py` is, use the `py` launcher instead (e.g. `py -m venv venv`, then `py sim.py --viztraces` after activating).
- **Plot windows**: `--viztraces` and `--vizdist` open Matplotlib windows. If a plot window doesn't appear, make sure it isn't opening behind your terminal window (check the taskbar), and confirm you're not running inside a headless terminal (e.g. some SSH sessions).
- If you run into a `ModuleNotFoundError`, double-check your terminal prompt starts with `(venv)` — it's easy to open a new terminal tab and forget to `activate` again before running `pip install` or `python sim.py`.

---

## Running the Simulator

To run the default simulation,

```bash
python sim.py
```

Useful command-line options include:

| Option | Description | Default |
|---------|-------------|---------|
| `--duration` | Number of simulation steps | 5000 |
| `--reps` | Number of independent simulation runs | 10 |
| `--distance` | Distance from the robot to the light source | 10 |
| `--noise` | Amount of random motion noise | 0.1 |
| `--turn_gain` | Turning sensitivity | 0.1 |
| `--angle_offset` | Angular separation between the two sensors | π/2 |
| `--wiring` | Sensor-to-motor wiring scheme (`crossed` or `direct`) — only takes effect once you implement the OPTIONAL runtime switch in `think()` | `crossed` |
| `--seed` | Random seed for reproducibility | None (random each run) |
| `--viztraces` | Display robot trajectories | off |
| `--vizdist` | Plot average distance from the light over time | off |
| `--scores` | Print fitness scores | off |
| `--save DIR` | Save `--viztraces`/`--vizdist` figures to `DIR` as PNGs instead of opening an interactive window (handy when generating many figures) | off (shows interactively) |

For example,

```bash
python sim.py --viztraces
```

or

```bash
python sim.py --noise 0.01 --vizdist
```

---

## Performing Parameter Studies

The file `study.py` automates experiments in which a single parameter is varied over many values.

For example,

```bash
python study.py --param noise
```

or

```bash
python study.py --param turn_gain --min 0.01 --max 1.0 --steps 20
```

Each experiment generates a figure showing how the chosen parameter influences the robot's performance.

`study.py` is a worked example of a pattern you'll rely on throughout this course: sweep one
parameter, repeat several times per value to average out randomness, and produce a labeled figure
rather than judging a single noisy run by eye. Read it before you need to write something like it
yourself — later projects increasingly expect you to build this kind of experiment-running,
data-saving, and figure-generating tooling on your own rather than have it handed to you, in
increasingly sophisticated forms as the term goes on.

---

## Understanding the Controller

The robot contains:

- two light sensors,
- two wheel motors,
- a simple controller connecting sensors to motors.

At every simulation step the robot repeatedly performs three operations:

1. **Sense** the environment.
2. **Compute** motor commands.
3. **Move** according to those commands.

Because these operations are repeated many times, the robot's behavior emerges naturally from the interaction between the controller and the environment.

One of your goals is to understand exactly how this process works by reading the source code.

---

## Tips

- Read the source code carefully before making modifications — start with `braitenberg.py`, since `python sim.py` won't run until Part 1 is done.
- Change one parameter at a time.
- Perform multiple trials because the simulator includes randomness.
- Use random seeds when you need reproducible experiments.

---

## Assignment

### Part 1 – Implement the Sensor-to-Motor Wiring

Before exploring parameters, get the robot working by implementing the piece that's currently missing: `Vehicle.think()` in `braitenberg.py`. As shipped, `think()` is unimplemented — it raises a `NotImplementedError`, and running `python sim.py` will fail until you fill it in. There is a `TODO` in the code marking exactly where this goes and what's expected.

Implement a **crossed (contralateral)** wiring scheme: the left sensor should drive the right motor, and the right sensor should drive the left motor. Before you run it, think through why this particular crossing should make the vehicle steer towards the light. Once it's in place,

```bash
python sim.py --viztraces
```

should show the vehicle approaching the light source.

**Now experiment.** Before changing anything, write down a prediction: how do you expect the vehicle to behave if you wire it the other way — **direct (ipsilateral)**, where each sensor drives the motor on the *same* side of the body (left sensor → left motor, right sensor → right motor)? Then hand-edit `think()` to use direct wiring, rerun `python sim.py --viztraces`, and compare. Report on whether the behavior matched your prediction, and explain *why* the two wiring schemes lead to such different behavior. Make sure to switch `think()` back to crossed wiring before moving on to the rest of the assignment — the parameter exploration below assumes light-seeking behavior.

**OPTIONAL (advanced):** rather than hand-editing `think()` every time you want to compare schemes, use the `self.wiring` attribute already set in `Vehicle.__init__` (default `"crossed"`) to select between the two schemes at runtime inside `think()`. A `--wiring` command-line flag is already wired up in both `sim.py` and `study.py` (it just constructs the `Vehicle` with whichever scheme you pass in) — once `think()` reads `self.wiring`, `python sim.py --wiring direct --viztraces` will switch schemes with no code changes needed.

---

### Part 2 – Understand the Behavior (Qualitative Exploration)

There are many components to this simulation. Of particular importance are the many parameters:

| Option | Description | Default |
|---------|-------------|---------|
| `--duration` | Number of simulation steps | 5000 |
| `--reps` | Number of independent simulation runs | 10 |
| `--distance` | Distance from the robot to the light source | 10 |
| `--noise` | Amount of random motion noise | 0.1 |
| `--turn_gain` | Turning sensitivity | 0.1 |
| `--angle_offset` | Angular separation between the two sensors | π/2 |
| `--wiring` | Sensor-to-motor wiring scheme (`crossed` or `direct`) — only takes effect once you implement the OPTIONAL runtime switch in `think()` | `crossed` |
| `--seed` | Random seed for reproducibility | None (random each run) |
| `--viztraces` | Display robot trajectories | off |
| `--vizdist` | Plot distance from the light over time | off |
| `--scores` | Print fitness scores | off |
| `--save DIR` | Save `--viztraces`/`--vizdist` figures to `DIR` as PNGs instead of opening an interactive window (handy when generating many figures) | off (shows interactively) |

> **Note:** every repetition starts from the exact same position and heading — the vehicle always
> begins at the origin facing directly toward the light. The only thing that varies between
> repetitions is the random noise. Keep that in mind as you experiment with `noise` in particular.

Experiment by varying **each one** of these six parameters: `duration`, `reps`, `distance`, `noise`,  `turn_gain`, and `angle_offset`. Observe the changes in the traces (`--viztraces`) and the distances (`--vizdist`). Reproduce figures in your report and explain your observations and insights about each of the components. 

Note that when your visualizing the distance (`--vizdist`), you are visualizing an average across N runs (`--reps`). Add one line to the code in the sim.py that allows you to see not just the average, but also the standard deviation across the different repetitions. This kind of detail is important for comparisons later on. You should also be able to save the average and standard deviation into a file, and then write a script that visualizes two different configurations in the same figure, as a way to compare them head to head. You can even add a line of code so that when `--save` is called, in addition to the figures, the averages and standard deviation is also saved it to a file. This will be useful for you to then import and visualize. 

Next, **pick one parameter** that you found of either `noise`,  `turn_gain`, or `angle_offset` that you found particularly interesting, and explore it more closely. Form a hypothesis, purely from watching the traces and distance plots, about how that parameter affects the robot's ability to reach the light (e.g. "performance should get worse past a certain noise level" or "there should be a best turning gain, with worse performance on either side"). You will check this hypothesis in Part 3.

Keep in mind that the fitness scores (`--scores`) will just produce a value of 0 for now, because you will be implementing that next.

---

### Part 3 – Define a Fitness Function, and Test Your Hypothesis

Design a quantitative measure that evaluates how successfully the robot approaches the light. `run_simulation()` in `sim.py` has a `TODO` marking exactly where this goes.

A useful fitness function should distinguish between robots that:

- move directly toward the light,
- wander randomly,
- fail to reach the light, or
- move away from it.

Think carefully about what aspect of behavior your fitness function rewards. Possible measurements to consider include:

- final distance to the light,
- closest distance reached,
- average distance,
- time required to approach the light,
- total path length,
- variability across repeated trials.

Write up a precise description of your fitness function. This should include a description of your evaluation: how many times is the agent ran, for how long, from what kind of starting positions, what is being measured, how is that measure being averaged to produced a single index. The fitness function can include a descriptive explanation and it can also include some formulas.

Note: Your fitness score calculation could be such that a higher score means better performance or the other way around a score of 0 means a perfect performance and a higher score means worst performance. Either works, but we recommend the former (higher score means better performance). More over, we highly recommend that you create a fitness that can be bounded, for example, between [0, 1]. That makes interpreting the fitness easier. Ultimately it is up to you. One way or another, keep in mind that `study.py` currently assumest that the higher score is better fitness. 

Once your fitness function is in place, use `study.py` to sweep the **same parameter you chose in Part 2**, for example:

```bash
python study.py --param noise
```

Before you set out to use the study.py program, provide a description of what parameters it can receive, what they stand for, what their defaults are, and what all you think you will be able to do with it. For example, can you change the number of `reps` and what exactly does that change in the code. Compare two studies, one with reps set to 1 and one with reps set to 100. What effect does reps have on the study of the parameters? 

Finally, compare the resulting plot to the hypothesis you wrote down in Part 2. Did the quantitative results match your qualitative expectations? If not, discuss why — it's common (and interesting!) for a hypothesis formed from watching a handful of traces to miss something a full sweep reveals.

---

### Part 4 – Explore the Rest of the Parameter Space

Now use your fitness function to investigate the parameters you *didn't* sweep in Part 3. Between sensor angle, turning gain, and motion noise, sweep whichever ones remain.

Generate plots that illustrate these relationships and explain the observed behavior.

Questions to consider include:

- What happens when the turning gain is very small?
- What happens when it is very large?
- What happens when noise is set to 0 and why does that happen? 
- Can noise ever improve performance?
- Is there a sweet spot for noise, if so what is it? And why? 
- Does the placement of the sensors (angle offset) affect the performance? If so, what is the best angle offset?
- Which parameter has the greatest influence on behavior?
- Is there one combination of all three parameters the makes the agent most optimal? 

---

## Optional / Advanced Challenge

Parts 1–4 are required; these are optional. If you would like, pick **one** of the following four directions to investigate further. Each is open-ended — there's no single right answer, and the point is to form a hypothesis, run the experiment, and report what you found. Go as deep as you like on it.

**1. Multiple light sources.** Add a second `Light` instance and modify `Vehicle.sense()` to combine readings from both (e.g., sum the inverse-distance intensities from each, or take the max). Before running anything, predict: will the vehicle settle at a point between the two lights, orbit between them, or commit to the nearer one? Then test it under a few different placements — lights close together vs. far apart, and symmetric vs. off-to-one-side — and see whether your prediction held.

**2. Sensor asymmetry and fault injection.** Give the two sensors different gains, or add sensor-specific noise (a new Gaussian noise term added directly to one sensor's reading in `sense()` — note that the existing `noise_stdev` only ever affects actuator/orientation noise in `move()`, so there's no existing per-sensor noise to "turn up" on one side; you're adding a new noise source, not scaling an existing one), and use `study.py`-style sweeps to find how much asymmetry the crossed-wiring controller can tolerate before it stops reliably reaching the light. Is the breakdown gradual (fitness degrades smoothly) or is there a sharp threshold?

**3. Add inertia to the movement model.** `Vehicle.move()` currently sets orientation and velocity instantaneously from the current motor commands every step. Modify it so turning rate and speed change gradually toward their commanded values (e.g., exponential smoothing) instead of jumping there immediately. Hypothesis: does this more realistic inertia make trajectories smoother, or does it interact badly with `noise_stdev` — since noise is now effectively integrated over time rather than applied fresh each step?

**4. Combine attraction and avoidance.** Add a second stimulus the vehicle should avoid (e.g., a second `Light`-like object wired with inhibitory-crossed connections instead of excitatory-crossed), and give the vehicle a controller that combines both influences — attraction to the real light, repulsion from the "obstacle." Does a simple linear combination of the two wiring schemes produce sensible trade-off behavior (e.g., approach the light while keeping distance from the obstacle), or does it need something more than direct summation to avoid the two signals canceling out?

You're encouraged to explore your own idea beyond these four as well, as long as it's a genuine extension of the simulator (not just a parameter change already covered in Parts 2–4).

---

## What to Submit to Moodle

Submit the code zipped (`p1_lastname.zip`) and the written report as a PDF (`p1_lastname.pdf`) to Moodle. 

The code should include everything you used and generated for this project (including code, scripts, data, figures). The code does not need to be perfectly organized. Simply compress your working folder as it is. 

The report should include:

### Title Page

- Your name
- Course title (ECE497: Evolutionary Robotics)
- Assignment name (Project 1: Braitenberg Vehicles)
- Date submitted
- Amount of time spent on this project 
- A self-assessment of your confidence in your understanding of the concepts, the code, and the insights gained from this project (a number between 1 and 10)

### Report Body

Organize the body into sections that mirror the assignment parts:

- **Part 1 – Sensor-to-Motor Wiring**: your prediction for direct (ipsilateral) wiring, what you actually observed, and your explanation of why crossed and direct wiring lead to such different behavior.
- **Part 2 – Understanding the Behavior**: figures from `--viztraces` and `--vizdist` for your exploration of each parameter, your observations and insights, and the hypothesis you formed for the parameter you chose to explore further.
- **Part 3 – Fitness Function**: a precise description of your fitness function (including formulas where helpful), and the sweep of your chosen parameter compared against your Part 2 hypothesis, with discussion of whether they matched.
- **Part 4 – Rest of the Parameter Space**: plots and explanations for the remaining parameters, and your answers to the questions posed in that section.
- **Optional / Advanced Challenge** *(if attempted)*: a section naming which of the four directions you chose (or your own idea), what you changed, your results (with supporting figures), and your interpretation. Omit this section if you didn't attempt a challenge.

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
- **Creativity & critical thinking (2 pts)** — depth of insight, quality of open-ended reasoning, and evidence of genuine exploration beyond the minimum required to answer each question — especially in whether your Part 3 results genuinely confirmed or overturned the hypothesis you formed in Part 2, not just whether the plots looked similar.

---
---

This project was developed by Eduardo Izquierdo for **ECE497 (Fall 2026): Evolutionary Robotics** at Rose-Hulman Institute of Technology.