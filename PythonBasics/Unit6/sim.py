##################################################################################
# Simulate one CTRNN with no inputs and plot what it does.
# Change the settings below and run it again.
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
import ctrnn

size = 3              # number of neurons
duration = 100.0      # how long to simulate (in time units)
dt = 0.01             # Euler time step: keep it well below the smallest time constant
circuit = "random"    # "random", "centercrossing", or "oscillator" (a known 2-neuron oscillator)
seed = None           # set to an integer to get the same circuit every time

np.random.seed(seed)

# Create the network and set its parameters
if circuit == "oscillator":
    # A hand-picked 2-neuron circuit that oscillates (Beer, 1995). Each neuron
    # excites itself; neuron 2 excites neuron 1, and neuron 1 inhibits neuron 2.
    size = 2
    nn = ctrnn.CTRNN(size)
    nn.setWeights(np.array([[4.5, -1.0],     # Weights[j][i]: from neuron j to neuron i
                            [1.0, 4.5]]))
    nn.setBiases(np.array([-2.75, -1.75]))
    nn.setTimeConstants(np.array([1.0, 1.0]))
else:
    nn = ctrnn.CTRNN(size)
    nn.randomizeParameters()          # weights and biases in [-10, 10], time constants in [0.1, 5]
    if circuit == "centercrossing":
        nn.setCenterCrossing()        # same weights and time constants, but center-crossing biases

print("Weights (row = from, column = to):\n", np.round(nn.Weights, 2))
print("Biases:", np.round(nn.Biases, 2))
print("Time constants:", np.round(nn.TimeConstants, 2))

# Start every neuron at a small random state
nn.initializeState(np.random.uniform(-1, 1, size=size))

# Run the simulation, keeping track of every neuron's output
steps = int(duration / dt)
time = np.arange(steps) * dt
outputs = np.zeros((steps, size))
for t in range(steps):
    nn.step(dt)
    outputs[t] = nn.Outputs

# Plot each neuron's output over time
for i in range(size):
    plt.plot(time, outputs[:, i], label="Neuron " + str(i + 1))
plt.xlabel("Time")
plt.ylabel("Output")
plt.ylim(-0.05, 1.05)
plt.title("CTRNN activity (" + circuit + ", " + str(size) + " neurons)")
plt.legend()
plt.show()

# Plot the first two neurons against each other: the trajectory in "output space".
# A circuit that settles down ends at a point; an oscillating one ends in a loop.
if size >= 2:
    plt.plot(outputs[:, 0], outputs[:, 1])
    plt.plot(outputs[0, 0], outputs[0, 1], "ko", label="Start")
    plt.plot(outputs[-1, 0], outputs[-1, 1], "r^", label="End")
    plt.xlabel("Output of neuron 1")
    plt.ylabel("Output of neuron 2")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title("Output space")
    plt.legend()
    plt.show()
