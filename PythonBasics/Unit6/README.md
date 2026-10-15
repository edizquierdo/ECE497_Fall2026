# Unit 6: Continuous-Time Recurrent Neural Networks (CTRNNs)

Simulate small CTRNNs with no inputs and see what they do on their own: settle down, switch on or off, or oscillate. Written from scratch in plain Python/NumPy. It introduces the network used in Project 6, where a CTRNN generates the rhythm that makes a six-legged body walk.

## Files

- `ctrnn.py`: the CTRNN. Each neuron has a state `y`, a bias `theta`, and a time constant `tau`, and follows `tau * dy/dt = -y + sum_j w_ji * sigmoid(y_j + theta_j) + I`. `step(dt)` advances it by one Euler step. `randomizeParameters()` picks random weights and biases in [-10, 10] and time constants in [0.1, 5]; `setCenterCrossing()` replaces the biases with center-crossing ones (see below).
- `sim.py`: simulates one circuit and plots each neuron's output over time, and the first two neurons against each other ("output space").
- `exp.py`: generates many random circuits of each size and counts how many oscillate, with random biases vs. center-crossing biases. Also plots how fast the oscillating ones oscillate.

## Run

```bash
python3 sim.py   # one circuit: outputs over time, and the trajectory in output space
python3 exp.py   # many circuits (~1 min): fraction that oscillate vs. network size; saves exp.npz
```

In `sim.py`, try changing:

- `circuit`: `"random"`, `"centercrossing"`, or `"oscillator"` (a hand-picked 2-neuron circuit that always oscillates).
- `size`: the number of neurons.
- `duration` and `dt`: how long to simulate, and how finely. What happens when `dt` gets close to (or bigger than) the smallest time constant?
- `seed`: set it to an integer to get the same circuit every time. Then switch between `"random"` and `"centercrossing"` with the same seed: same weights and time constants, only the biases change.

In `exp.py`, try changing `sizes`, `samples`, or the ranges in `randomizeParameters()` in `ctrnn.py`.

## Center-crossing

Neuron `i`'s total input from the other neurons always lies between the sum of its negative incoming weights and the sum of its positive ones. A **center-crossing** circuit sets each bias to `theta_i = -sum_j w_ji / 2`, so the middle of that range lands right on the steepest part of the neuron's sigmoid. Each neuron is then sensitive to changes in the others instead of being stuck fully on or fully off, which makes rich dynamics like oscillations much more likely (Mathayomchan & Beer, 2002). Project 6's `--center_crossing` option uses the same idea to seed evolution's starting population.
