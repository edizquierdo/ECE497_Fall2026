# Unit 5: Evolving the Body and Brain of a Soft Robot

Evolve a soft robot that walks to the right, with [Evolution Gym](https://evolutiongym.github.io) (EvoGym) as the physics engine, using the same `eas.py` and `fnn.py` as Unit 4. It mirrors Project 5 without PyTorch or EvoTorch: the same bodies, the same four brains, and the same two problems, set in `config.py` instead of on the command line.

## The idea

The robot is a small grid of voxels. Each voxel is one of five materials: empty, rigid, soft, horizontal muscle, or vertical muscle. Every step, each muscle gets a command between 0.6 (squeeze) and 1.6 (stretch).

There are two things to try (`mode` in `config.py`):

- **`"control"`: evolve the brain for a fixed body.** The body is one of the presets (`"biped"`, `"worm"`, `"block"`, `"tripod"`), or a text file you design, with one row of material codes per line (row 0 at the top): copy `example_body.txt` and edit it.
- **`"codesign"`: evolve the body and the brain together.** One genotype holds both. **Body:** 5 genes per voxel, and each voxel becomes whichever material's gene is largest (the same trick `fnn.py` uses to pick each layer's activation function). **Brain:** the network's weights, biases, and activation functions.

## The four brains

When evolution can change the body, the brain has to work for whatever body it builds. The four brains (`controller` and `loop` in `config.py`) all do, in different ways. All four get a clock (sin and cos of time) as a source of rhythm.

| | `loop = "open"` (no sensing) | `loop = "closed"` (+ sensing) |
|---|---|---|
| `controller = "local"`: one small network, copied into every muscle | clock + the muscle's own (x, y) position → its command | + how stretched its own voxel and its 4 neighbors are |
| `controller = "global"`: one network for the whole body | clock → one command per grid cell (each muscle takes its cell's) | + how stretched every cell of the grid is |

"How stretched" (the *strain*) is a voxel's width and height relative to its rest size, minus 1: 0 at rest, positive when stretched, negative when squeezed.

## Files

- `config.py`: settings shared by `evolve.py` and `sim.py`: the task (`envname`, `duration`), what to evolve (`mode`), the body (`body`, `grid`), and the brain (`controller`, `loop`, `hidden`, `period`, `weightrange`).
- `env.py`: the preset bodies, loading a body from a file (`loadBody`), building a body from genes (`makeBody`), checking it can be simulated (`isValid`), the size of each brain (`brainLayers`), and the robot itself (`SoftRobot`), with `think()` and `move()` steps like the vehicle in Unit 4.
- `example_body.txt`: a hand-designed body, to copy when designing your own.
- `fnn.py`: feedforward neural network, same as Unit 4.
- `eas.py`: evolutionary algorithms, same as Unit 4.
- `evolve.py`: evolves the brain (and, in co-design, the body). Fitness is the distance walked to the right. Bodies that fall apart or have no muscles get a fitness of -1.
- `sim.py`: runs the best robot in an animation window, and saves its behavior.
- `viz.py`: plots the body, the distance walked and the height of the center of mass over time, and every muscle's command over time.

## Setup

EvoGym only installs on **Python 3.10 or older**, so this unit needs its own environment:

```bash
conda create -n evogym python=3.10
conda activate evogym
pip install evogym "setuptools<81" matplotlib
```

(`setuptools<81` is needed because EvoGym still imports `pkg_resources`, which newer versions removed.)

## Run

```bash
python3 evolve.py   # evolve (~15 min); prints the best fitness (and body), saves best.npy, shows fitness over generations
python3 sim.py      # watch best.npy walk; saves sim.npz
python3 viz.py      # plot the body, distance and height over time, and muscle commands
```

Distances are in EvoGym's units, where one voxel is 0.1 wide: a fitness of 1 means the robot walked 10 voxels in 300 steps. At the defaults, expect a best fitness of around 2–3. That's lower than in Project 5, which uses a different evolutionary algorithm and evaluates many robots in parallel, so don't expect the numbers to match. EvoGym sometimes prints `SIMULATION UNSTABLE... TERMINATING` when an extreme body or brain makes the physics blow up. It ends that trial with a penalty, and evolution carries on.

Try changing what you evolve (`mode`, `body`), the body grid (`grid`), the brain (`controller`, `loop`, `hidden`, `period`, `weightrange`), the task (`duration`, or another EvoGym task for `envname`, such as `UpStepper-v0` or `BridgeWalker-v0`), or the evolutionary algorithm (`popsize`, `generations`, `mutatStd`, or swap `ea.Microbial` for another algorithm in `eas.py`).

Set `seed` in `evolve.py` to an integer to reproduce a run exactly.
