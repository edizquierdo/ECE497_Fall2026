"""
Simulation runner for evolved neural network controllers.

Runs simulations using the best evolved neural controller from evolve.py
and visualizes the robot's behavior. Also provides a reference simulation
using the original Braitenberg controller for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

from braitenberg import NeuralVehicle, Light
from neural_controller import NeuralController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate evolved neural network controller for Braitenberg vehicle."
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=None,
        help="Path to numpy file containing the evolved genome (optional)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Number of simulation steps (default: 500, matches evolve.py)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of independent repetitions (default: 5)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=10.0,
        help="Distance to light source (default: 10.0)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=8,
        help="Number of hidden neurons in network (default: 8)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="List of hidden layer sizes, e.g. --hidden_sizes 16 16. Overrides "
             "--hidden when given. Must match the architecture the genome was "
             "evolved with. (default: None, i.e. use the single --hidden layer)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "sigmoid"],
        help="Activation function applied on hidden layer(s); must match what the "
             "genome was evolved with. (default: tanh)",
    )
    parser.add_argument(
        "--angle_offset",
        type=float,
        default=np.pi / 2,
        help="Angular separation between sensors (radians, default: pi/2)",
    )
    parser.add_argument(
        "--turn_gain",
        type=float,
        default=0.1,
        help="Turn gain for steering (default: 0.1)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Motion noise standard deviation (default: 0.1, matches evolve.py)",
    )
    parser.add_argument(
        "--viztraces",
        action="store_true",
        help="Visualize individual vehicle trajectories",
    )
    parser.add_argument(
        "--vizdist",
        action="store_true",
        help="Visualize distance to light over time",
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Print average fitness per repetition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def run_simulation(
    genome_path=None,
    hidden_size=8,
    duration=500,
    reps=5,
    distance=10.0,
    angle_offset=np.pi / 2,
    turn_gain=0.1,
    noise=0.1,
    viztraces=False,
    vizdist=False,
    seed=None,
    hidden_sizes=None,
    activation="tanh",
):
    """
    Run the neural vehicle simulation.

    Args:
        genome_path: Path to saved genome file, or None for random initialization.
        hidden_size: Number of hidden neurons.
        duration: Number of simulation steps (should match evolve.py default).
        reps: Number of independent repetitions.
        distance: Distance to light source.
        angle_offset: Angular separation between sensors (radians).
        turn_gain: Turn gain for steering.
        noise: Motion noise standard deviation.
        viztraces: Visualize trajectories if True.
        vizdist: Visualize distance over time if True.
        seed: Random seed.

    Returns:
        Final score (average fitness, where higher = better).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create controller
    controller = NeuralController(hidden_size=hidden_size, hidden_sizes=hidden_sizes, activation=activation)

    if genome_path is not None:
        # Load evolved genome
        genome_data = np.load(genome_path)
        genome = torch.tensor(genome_data, dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(genome, controller.parameters())
        print(f"Loaded genome from: {genome_path}")
    else:
        print("Using randomly initialized controller.")

    # Setup tracking variables
    start_x = np.zeros(reps)
    start_y = np.zeros(reps)

    if viztraces:
        xpos = np.zeros((reps, duration))
        ypos = np.zeros((reps, duration))

    dist = np.zeros((reps, duration))
    total_reward = np.zeros(reps)

    for r in range(reps):
        if seed is not None:
            torch.manual_seed(seed + r)

        # Create vehicle with neural controller
        agent = NeuralVehicle(
            controller=controller,
            angle_offset=angle_offset,
            turn_gain=turn_gain,
            noise_stdev=noise,
            distance=distance,
        )

        # Set starting position at distance from origin (light) at random angle
        start_angle = np.random.uniform(0, 2 * np.pi)
        agent.x_pos = distance * np.cos(start_angle)
        agent.y_pos = distance * np.sin(start_angle)

        # Store start position for visualization
        start_x[r] = agent.x_pos
        start_y[r] = agent.y_pos

        # Random orientation
        agent.orientation = np.random.uniform(0, 2 * np.pi)

        # Initialize light source at origin (0, 0)
        light = Light()

        episode_reward = 0.0
        for t in range(duration):
            agent.sense(light)
            agent.think()
            agent.move()

            if viztraces:
                xpos[r, t] = agent.x_pos
                ypos[r, t] = agent.y_pos

            dist[r, t] = agent.distance(light)

            # Bounded proximity reward: 1.0 when on top of the light,
            # approaching 0.0 as distance grows -- matches evolve.py's
            # fitness function, so scores here are comparable to fitness
            # values reported during evolution.
            episode_reward += 1.0 / (1.0 + dist[r, t])

        total_reward[r] = episode_reward / duration  # Normalize by episode length

    # Calculate fitness: average normalized reward across repetitions
    final_avg = float(np.mean(total_reward))

    if viztraces:
        plt.figure(figsize=(8, 8))
        for r in range(reps):
            plt.scatter(xpos[r], ypos[r], s=0.5, c=range(duration), cmap="plasma")
        plt.plot(0.0, 0.0, "y^", markersize=12, label="Light (origin)")
        plt.scatter(start_x, start_y, c="red", s=20, marker="o", label="Start positions")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.title("Neural Vehicle Trajectories")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    if vizdist:
        avg_fitness_over_time = np.mean(dist, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(avg_fitness_over_time)
        plt.xlabel("Time step")
        plt.ylabel("Average distance from light (lower = better)")
        plt.title("Neural Controller Performance Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return final_avg


def main():
    args = parse_args()

    final_avg = run_simulation(
        genome_path=args.genome,
        hidden_size=args.hidden,
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
    )

    if args.scores:
        print(f"Average fitness: {final_avg:.4f}")

    return final_avg


if __name__ == "__main__":
    main()
