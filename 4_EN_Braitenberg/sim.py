"""
Simulation runner for Braitenberg vehicles.

Simulates either an evolved neural controller (from evolve.py) or one of the
hand-wired Project 1 controllers, and visualizes the vehicles' behavior.
Running both with the same --seed gives them identical starting positions,
headings, and motion noise, so their trajectories can be compared directly.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from braitenberg import Vehicle, NeuralVehicle, Light, place_randomly
from neural_controller import NeuralController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate an evolved neural controller or a hand-wired Braitenberg vehicle."
    )
    parser.add_argument("--controller", type=str, default="neural", choices=["neural", "crossed", "direct"],
                        help="Which controller drives the vehicle: an evolved network ('neural') or the "
                             "hand-wired Project 1 wiring ('crossed' or 'direct') (default: neural)")
    parser.add_argument("--genome", type=str, default=None,
                        help="Path to a .npy genome saved by evolve.py --output. If omitted with "
                             "--controller neural, a random (unevolved) network is used.")
    parser.add_argument("--hidden", type=int, default=8,
                        help="Number of hidden neurons; must match the genome (default: 8)")
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=None,
                        help="List of hidden layer sizes, e.g. --hidden_sizes 16 16. Overrides --hidden; "
                             "must match the genome (default: None)")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "sigmoid"],
                        help="Hidden layer activation; must match the genome (default: tanh)")
    parser.add_argument("--duration", type=int, default=2000,
                        help="Number of simulation steps (default: 2000, matches evolve.py)")
    parser.add_argument("--reps", type=int, default=5, help="Number of independent repetitions (default: 5)")
    parser.add_argument("--distance", type=float, default=10.0,
                        help="Starting distance from the light (default: 10.0)")
    parser.add_argument("--angle_offset", type=float, default=np.pi / 2,
                        help="Angular separation between sensors (radians, default: pi/2)")
    parser.add_argument("--turn_gain", type=float, default=0.1, help="Turn gain for steering (default: 0.1)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Motion noise standard deviation (default: 0.1)")
    parser.add_argument("--viztraces", action="store_true", help="Plot vehicle trajectories")
    parser.add_argument("--vizdist", action="store_true",
                        help="Plot distance to light over time (mean ± 1 std across reps)")
    parser.add_argument("--scores", action="store_true", help="Print the average fitness")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save figures (PNG) and recorded distances (distances.npy) "
                             "instead of opening interactive windows")
    return parser.parse_args()


def run_simulation(controller="neural", genome_path=None, hidden=8, hidden_sizes=None, activation="tanh",
                   duration=2000, reps=5, distance=10.0, angle_offset=np.pi / 2, turn_gain=0.1,
                   noise=0.1, viztraces=False, vizdist=False, seed=None, save=None):
    """
    Run the vehicle simulation.

    Args:
        controller: 'neural' (evolved network), 'crossed' or 'direct' (hand-wired).
        genome_path: Path to saved genome file, or None for a random network.
        hidden, hidden_sizes, activation: Network architecture (neural controller only).
        duration: Number of simulation steps (should match evolve.py).
        reps: Number of independent repetitions.
        distance: Starting distance from the light.
        angle_offset: Angular separation between sensors (radians).
        turn_gain: Turn gain for steering.
        noise: Motion noise standard deviation.
        viztraces: Visualize trajectories if True.
        vizdist: Visualize distance over time if True.
        seed: Random seed.
        save: Directory to save figures and distances to, instead of showing them.

    Returns:
        (fitness, dist): average fitness (same measure as evolve.py; higher = better),
        and the (reps, duration) array of distances to the light.
    """
    if seed is not None:
        np.random.seed(seed)

    if controller == "neural":
        net = NeuralController(hidden=hidden, hidden_sizes=hidden_sizes, activation=activation)
        if genome_path is not None:
            genome = torch.tensor(np.load(genome_path), dtype=torch.float32)
            torch.nn.utils.vector_to_parameters(genome, net.parameters())
            print(f"Loaded genome from: {genome_path}")
        else:
            print("Using a randomly initialized (unevolved) network.")

    start_x = np.zeros(reps)
    start_y = np.zeros(reps)
    xpos = np.zeros((reps, duration))
    ypos = np.zeros((reps, duration))
    dist = np.zeros((reps, duration))

    light = Light()

    for r in range(reps):
        if controller == "neural":
            agent = NeuralVehicle(net, angle_offset=angle_offset, turn_gain=turn_gain,
                                  noise_stdev=noise, distance=distance)
        else:
            agent = Vehicle(angle_offset=angle_offset, turn_gain=turn_gain,
                            noise_stdev=noise, distance=distance, wiring=controller)

        place_randomly(agent, distance)
        start_x[r] = agent.x_pos
        start_y[r] = agent.y_pos

        for t in range(duration):
            agent.sense(light)
            agent.think()
            agent.move()

            xpos[r, t] = agent.x_pos
            ypos[r, t] = agent.y_pos
            dist[r, t] = agent.distance(light)

    # Bounded proximity reward, averaged over time and repetitions -- the same
    # fitness evolve.py uses, so the two are directly comparable.
    fitness = float(np.mean(1.0 / (1.0 + dist)))

    label = "Neural" if controller == "neural" else f"Hand-wired ({controller})"

    if save is not None:
        os.makedirs(save, exist_ok=True)
        np.save(os.path.join(save, "distances.npy"), dist)

    if viztraces:
        plt.figure(figsize=(8, 8))
        for r in range(reps):
            plt.scatter(xpos[r], ypos[r], s=0.5, c=range(duration), cmap="plasma")
        plt.plot(0.0, 0.0, "y^", markersize=12, label="Light (origin)")
        plt.scatter(start_x, start_y, c="red", s=20, marker="o", label="Start positions")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.title(f"{label} Vehicle Trajectories")
        plt.axis("equal")
        plt.tight_layout()
        if save is not None:
            plt.savefig(os.path.join(save, "traces.png"), dpi=150)
            plt.close()
        else:
            plt.show()

    if vizdist:
        mean, std = dist.mean(axis=0), dist.std(axis=0)
        steps = np.arange(duration)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, mean, label="Mean across reps")
        plt.fill_between(steps, mean - std, mean + std, alpha=0.3, label="± 1 std")
        plt.xlabel("Time step")
        plt.ylabel("Distance from light (lower = better)")
        plt.title(f"{label} Vehicle: Distance to Light Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save is not None:
            plt.savefig(os.path.join(save, "dist.png"), dpi=150)
            plt.close()
        else:
            plt.show()

    return fitness, dist


def main():
    args = parse_args()

    fitness, _ = run_simulation(
        controller=args.controller,
        genome_path=args.genome,
        hidden=args.hidden,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        duration=args.duration,
        reps=args.reps,
        distance=args.distance,
        angle_offset=args.angle_offset,
        turn_gain=args.turn_gain,
        noise=args.noise,
        viztraces=args.viztraces,
        vizdist=args.vizdist,
        seed=args.seed,
        save=args.save,
    )

    if args.scores:
        print(f"Average fitness: {fitness:.4f}")


if __name__ == "__main__":
    main()
