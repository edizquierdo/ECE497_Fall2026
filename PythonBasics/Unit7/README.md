# Unit 7: Evolving the Body and Brain of a Soft Robot

Evolve both the shape and the neural controller of a soft robot that walks to the right, using the same `eas.py` and `fnn.py` as Unit 4, with [Evolution Gym](https://evolutiongym.github.io) (EvoGym) as the physics engine. It mirrors Project 7 without PyTorch or EvoTorch.

## The idea

The robot is a small square grid of voxels (4x4 by default). Each voxel is one of five materials: empty, rigid, soft, horizontal muscle, or vertical muscle. One genotype holds both the body and the brain:

- **Body (5 genes per voxel):** each voxel becomes whichever material's gene is largest. This is the same trick `fnn.py` uses to pick each layer's activation function.
- **Brain (the network's weights, biases, and activation functions):** a single small network drives *every* muscle. Each muscle feeds it the same clock (sin and cos of time) plus its own (x, y) position in the body, and gets back its own command. The network doesn't change size when evolution adds or removes muscles, so the genotype always has the same length.

## Files

- `config.py`: settings shared by `evolve.py` and `sim.py` (`envname`, `duration`, `grid`, `period`, `layers`, `weightrange`).
- `env.py`: building a body from genes (`makeBody`), checking it can be simulated (`isValid`), and the robot itself (`SoftRobot`), with `think()` and `move()` steps like the vehicle in Unit 4.
- `fnn.py`: feedforward neural network, same as Unit 4.
- `eas.py`: evolutionary algorithms, same as Unit 4.
- `evolve.py`: evolves body and brain together. Fitness is the distance walked to the right. Bodies that fall apart or have no muscles get a fitness of -1.
- `sim.py`: runs the best robot in an animation window, and saves its behavior.
- `viz.py`: plots the evolved body, the distance walked over time, and every muscle's command over time.

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
python3 evolve.py   # evolve (~3 min); prints the best fitness and body, saves best.npy, shows fitness over generations
python3 sim.py      # watch best.npy walk; saves sim.npz
python3 viz.py      # plot the body, distance over time, and muscle commands
```

Distances are in EvoGym's units, where one voxel is 0.1 wide: a fitness of 1 means the robot walked 10 voxels. EvoGym sometimes prints `SIMULATION UNSTABLE... TERMINATING` when an extreme body or controller makes the physics blow up. It ends that trial with a penalty, and evolution carries on.

Try changing the body (`grid` in `config.py`), the brain (`layers`, `period`, `weightrange`), the task (`duration`, or another EvoGym task for `envname`, such as `UpStepper-v0` or `BridgeWalker-v0`), or the evolutionary algorithm (`popsize`, `generations`, `mutatStd`, or swap `ea.Generational` for another algorithm in `eas.py`).

Set `seed` in `evolve.py` to an integer to reproduce a run exactly.
