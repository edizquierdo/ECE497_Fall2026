# Unit 4: Evolving a Neural Controller for a Braitenberg Vehicle

Evolve a feedforward neural network that steers a two-wheeled Braitenberg vehicle toward a light, written from scratch in plain Python/NumPy. It mirrors Project 4 without PyTorch or EvoTorch.

## Files

- `config.py`: settings shared by `evolve.py` and `sim.py` (`duration`, `distance`, `layers`, `weightrange`).
- `env.py`: the vehicle (two light sensors, two motors) and a light at the origin. Each trial starts the vehicle 5 units from the light at a random position and heading.
- `fnn.py`: feedforward neural network, same as Unit 3. Maps [left, right] sensors to [left, right] motors.
- `eas.py`: evolutionary algorithms, same as Unit 3.
- `evolve.py`: evolves the network. Fitness is the average of `1/(1 + distance to light)` over every time step of several trials.
- `sim.py`: tests the best evolved network on new random trials.
- `viz.py`: plots the trajectories and distance to the light over time.

## Run

```bash
python3 evolve.py   # evolve (~15 s); prints recorded and fresh-trial fitness, saves best.npy, shows fitness over generations
python3 sim.py                       # test best.npy on 20 new trials; saves sim.npz
python3 viz.py                       # plot trajectories and distance over time
```

Try changing the task (`duration`, `distance` in `config.py`, `reps` in `evolve.py`), the network (`layers`, `weightrange` in `config.py`), or the evolutionary algorithm (`popsize`, `generations`, `mutatStd`, or swap `ea.Microbial` for another algorithm in `eas.py`).

Set `seed` in `evolve.py` to an integer to reproduce a run exactly.
