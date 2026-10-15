"""
How far do RANDOM (unevolved) brains walk?  

Runs N random brains of each of the four kinds on one body, prints the
average distance each kind walked, and plots every single run.

Each run is exactly what you'd get from typing, for example,
    python sim.py --body example_body.txt --controller local --loop open --seed 3
in the terminal: sim.main() takes the same arguments as a list, and returns
the reward. With no --genome, sim.py gives the body a random brain, and
--seed picks which one, so any run can be replayed (add --gif to watch it).

Change N and BODY below, then run:  python random_brains_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt

import sim

N = 100                       # random brains of each kind (try 100: it takes about 2 minutes)
BODY = "example_body.txt"    # a preset ("biped", "worm", ...) or a body file

kinds = [("local", "open"), ("local", "closed"), ("global", "open"), ("global", "closed")]
distances = []   # one list of N distances for each kind of brain

for controller, loop in kinds:
    results = []
    for i in range(N):
        reward = sim.main(["--body", BODY, "--controller", controller, "--loop", loop,
                           "--seed", str(i), "--quiet"])
        results.append(reward)   # the reward = distance walked
    distances.append(results)

# Averages, in the terminal
print(f"\nDistance walked by {N} random brains on '{BODY}':")
for (controller, loop), results in zip(kinds, distances):
    label = f"{controller}, {loop}-loop:"
    print(f"  {label:22s} average {np.mean(results):6.3f}")

# Every individual run, in a figure (one column of points per kind of brain)
# Box: the middle half of the runs, with a line at the median; whiskers show the spread
plt.boxplot(distances, positions=range(4), widths=0.6, showfliers=False)
for k, results in enumerate(distances):
    x = k + np.random.uniform(-0.2, 0.2, N)   # spread the dots sideways so they don't overlap
    plt.plot(x, results, "o", markersize=4, alpha=0.4)
    plt.plot(k, np.mean(results), "kD", markersize=7)   # black diamond = average
plt.xticks(range(4), [f"{c}\n{l}-loop" for c, l in kinds])
plt.ylabel("Distance walked (reward)")
plt.title(f"{N} random brains on '{BODY}'")
plt.grid(True, alpha=0.3)
plt.show()
