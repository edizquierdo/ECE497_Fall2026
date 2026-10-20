##################################################################################
# Evolve a CTRNN that oscillates.
# The genotype is a list of numbers between -1 and 1. We stretch them into the
# CTRNN's weights, biases, and time constants, run the circuit, and score it by
# how much its neurons' outputs keep changing over time. A circuit that settles
# down stops changing, so the only way to keep scoring is to keep moving:
# to oscillate.
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
import ctrnn
import eas

size = 10              # number of neurons
duration = 50.0       # how long to simulate each circuit
transient = 25.0      # ignore the first part, while the circuit settles into its long-term behavior
dt = 0.05             # Euler time step (smallest time constant is 1)
generations = 100
popsize = 20
seed = None           # set to an integer to get the same run every time

np.random.seed(seed)

genesize = size * size + size + size   # weights, biases, time constants

def build(genotype):
    # Turn a genotype (numbers in [-1, 1]) into a CTRNN
    nn = ctrnn.CTRNN(size)
    nn.setWeights(genotype[:size * size].reshape(size, size) * 10)                  # [-10, 10]
    nn.setBiases(genotype[size * size:size * size + size] * 10)                     # [-10, 10]
    nn.setTimeConstants((genotype[size * size + size:] + 1) / 2 * 4 + 1)           # [1, 5]
    return nn

def simulate(nn):
    # Run the circuit from the same starting state every time, and return
    # the outputs after the transient
    nn.initializeState(np.zeros(size))
    steps = int(duration / dt)
    skip = int(transient / dt)
    outputs = np.zeros((steps - skip, size))
    for t in range(steps):
        nn.step(dt)
        if t >= skip:
            outputs[t - skip] = nn.Outputs
    return outputs

def fitnessFunction(genotype):
    # Average rate of change of the outputs, across all neurons and time
    outputs = simulate(build(genotype))
    return np.mean(np.abs(np.diff(outputs, axis=0))) / dt

# Evolve
ga = eas.Microbial(fitnessFunction, genesize, generations,
                   popsize=popsize, recombProb=0.5, mutatStd=0.05, demeSize=2)
ga.run()
ga.showFitness()

# Show what the best circuit does
avgfit, bestfit, bestind = ga.fitStats()
print("Best fitness:", bestfit)
nn = build(bestind)
print("Weights (row = from, column = to):\n", np.round(nn.Weights, 2))
print("Biases:", np.round(nn.Biases, 2))
print("Time constants:", np.round(nn.TimeConstants, 2))

outputs = simulate(nn)
time = transient + np.arange(len(outputs)) * dt
for i in range(size):
    plt.plot(time, outputs[:, i], label="Neuron " + str(i + 1))
plt.xlabel("Time")
plt.ylabel("Output")
plt.ylim(-0.05, 1.05)
plt.title("Activity of the best evolved circuit")
plt.legend()
plt.show()
