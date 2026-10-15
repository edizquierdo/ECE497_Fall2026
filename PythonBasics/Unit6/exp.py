##################################################################################
# How common are rhythmic CTRNNs?
# For each network size, generate many random circuits and count how many of
# them end up oscillating. Do it twice: once with random biases, and once with
# the same kind of random weights but center-crossing biases.
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
import ctrnn

sizes = [1, 2, 3, 4, 5, 6, 8, 10]   # network sizes to test
samples = 100         # circuits per size and condition
duration = 150.0      # how long to simulate each circuit
transient = 100.0     # ignore the first part, while the circuit settles into its long-term behavior
dt = 0.02             # Euler time step (smallest time constant is 0.1)
threshold = 0.05      # a neuron must swing by at least this much to count as oscillating
seed = 0              # set to None for a different experiment every time

np.random.seed(seed)

def simulate(nn):
    # Run the circuit with no inputs from a random starting state, and
    # return the outputs after the transient
    nn.initializeState(np.random.uniform(-1, 1, size=nn.Size))
    steps = int(duration / dt)
    skip = int(transient / dt)
    outputs = np.zeros((steps - skip, nn.Size))
    for t in range(steps):
        nn.step(dt)
        if t >= skip:
            outputs[t - skip] = nn.Outputs
    return outputs

def period(outputs):
    # Is the circuit rhythmic? Take the neuron that swings the most. If it
    # barely moves, the circuit has settled to a fixed point. If it moves,
    # it must also cross its own average at least 4 times (2 full cycles);
    # otherwise it is just drifting slowly in one direction.
    # Returns the period of the oscillation, or 0 if there isn't one.
    swing = outputs.max(axis=0) - outputs.min(axis=0)
    i = np.argmax(swing)
    if swing[i] < threshold:
        return 0.0
    o = outputs[:, i] - outputs[:, i].mean()
    upward = np.where((o[:-1] < 0) & (o[1:] >= 0))[0]    # times it crosses its average going up
    if len(upward) < 2:
        return 0.0
    return (upward[-1] - upward[0]) * dt / (len(upward) - 1)

conditions = ["random", "centercrossing"]
fraction = {c: [] for c in conditions}
periods = {c: [] for c in conditions}

for size in sizes:
    for c in conditions:
        count = 0
        for s in range(samples):
            nn = ctrnn.CTRNN(size)
            nn.randomizeParameters()
            if c == "centercrossing":
                nn.setCenterCrossing()
            p = period(simulate(nn))
            if p > 0:
                count += 1
                periods[c].append(p)
        fraction[c].append(count / samples)
    print("Size", size, " random:", fraction["random"][-1], " center-crossing:", fraction["centercrossing"][-1])

np.savez("exp.npz", sizes=sizes, random=fraction["random"], centercrossing=fraction["centercrossing"],
         periods_random=periods["random"], periods_centercrossing=periods["centercrossing"])

# Fraction of rhythmic circuits for each size
plt.plot(sizes, fraction["random"], "o-", label="Random biases")
plt.plot(sizes, fraction["centercrossing"], "o-", label="Center-crossing biases")
plt.xlabel("Number of neurons")
plt.ylabel("Fraction of circuits that oscillate")
plt.ylim(-0.02, 1.02)
plt.title("How common are rhythmic CTRNNs? (" + str(samples) + " circuits per point)")
plt.legend()
plt.show()

# Periods of the rhythmic circuits found
bins = np.linspace(0, 50, 51)
plt.hist(periods["random"], bins=bins, alpha=0.6, label="Random biases")
plt.hist(periods["centercrossing"], bins=bins, alpha=0.6, label="Center-crossing biases")
plt.xlabel("Period of oscillation")
plt.ylabel("Number of circuits")
plt.title("How fast do they oscillate?")
plt.legend()
plt.show()
