"""
Simulation runner for evolved neural network controllers on CartPole.

Runs simulations using the best evolved neural controller from evolve.py,
visualizes the robot's behavior, and can optionally render episodes live
with Gymnasium's human renderer.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.utils as utils

from cartpole import CartPole
from neural_controller import NeuralController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate evolved neural network controller for CartPole."
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
        help="Max steps per episode (default: 500, matches CartPole-v1's "
             "built-in limit and evolve.py's default --duration). Must "
             "match what was used during evolution for fitness numbers to "
             "be comparable.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of independent repetitions (default: 5)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=8,
        help="Number of hidden neurons in network (default: 8)",
    )
    parser.add_argument(
        "--viztraces",
        action="store_true",
        help="Visualize pole angle over time",
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Print average reward per repetition",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment using Gymnasium's human renderer, so you can watch the controller balance the pole",
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
    viztraces=False,
    render=False,
    seed=None,
):
    """
    Run the CartPole simulation.

    Args:
        genome_path: Path to saved genome file, or None for random initialization.
        hidden_size: Number of hidden neurons.
        duration: Number of simulation steps (should match evolve.py default).
        reps: Number of independent repetitions.
        viztraces: Visualize pole angle over time if True.
        render: Render the environment with Gymnasium's human renderer if True.
        seed: Random seed.

    Returns:
        Final score (average fitness, where higher = better).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create controller
    controller = NeuralController(hidden_size=hidden_size)

    if genome_path is not None:
        # Load evolved genome
        genome_data = np.load(genome_path)
        genome = torch.tensor(genome_data, dtype=torch.float32)

        expected_n = sum(p.numel() for p in controller.parameters())
        if genome.numel() != expected_n:
            raise ValueError(
                f"Genome has {genome.numel()} values, but a network with "
                f"hidden_size={hidden_size} expects {expected_n}. "
                f"This genome was almost certainly evolved with a different "
                f"--hidden value — pass the matching --hidden to sim.py. "
                f"(torch.nn.utils.vector_to_parameters will NOT raise an "
                f"error on a size mismatch; it silently loads garbage.)"
            )

        utils.vector_to_parameters(genome, controller.parameters())
        print(f"Loaded genome from: {genome_path}")
    else:
        print("Using randomly initialized controller.")

    # Setup tracking variables
    if viztraces:
        pole_angle = np.zeros((reps, duration))
        cart_pos = np.zeros((reps, duration))

    rewards = np.zeros(reps)
    episode_lengths = np.zeros(reps)

    for r in range(reps):
        if seed is not None:
            torch.manual_seed(seed + r)

        # Create environment
        env = CartPole(render_mode="human" if render else None, max_episode_steps=duration)

        # Reset the controller for each repetition
        controller.eval()

        total_reward = 0.0
        for t in range(duration):
            # Select action using neural network
            observation = env.observation
            action = controller.act(observation)

            # Step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if viztraces:
                pole_angle[r, t] = observation[2]  # Pole angle
                cart_pos[r, t] = observation[0]    # Cart position

            if terminated or truncated:
                break

        rewards[r] = total_reward
        episode_lengths[r] = t + 1

        env.close()

    # Calculate fitness: average reward across repetitions
    final_avg = float(np.mean(rewards))

    if viztraces:
        # Plot pole angle over time
        plt.figure(figsize=(10, 6))
        for r in range(reps):
            end_idx = min(int(episode_lengths[r]), duration)
            plt.plot(pole_angle[r, :end_idx], alpha=0.7, label=f"Rep {r+1}")
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label="Vertical")
        plt.xlabel("Time step")
        plt.ylabel("Pole angle (radians)")
        plt.title("Pole Angle Over Time (lower = more stable)")
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, alpha=0.3)

        # Plot cart position over time
        plt.figure(figsize=(10, 6))
        for r in range(reps):
            end_idx = min(int(episode_lengths[r]), duration)
            plt.plot(cart_pos[r, :end_idx], alpha=0.7, label=f"Rep {r+1}")
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label="Center")
        plt.axhline(y=2.4, color='r', linestyle=':', alpha=0.5, label="Limit")
        plt.axhline(y=-2.4, color='r', linestyle=':', alpha=0.5)
        plt.xlabel("Time step")
        plt.ylabel("Cart position")
        plt.title("Cart Position Over Time")
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Return both fitness and episode lengths
    return final_avg, episode_lengths


def main():
    args = parse_args()

    final_avg, episode_lengths = run_simulation(
        genome_path=args.genome,
        hidden_size=args.hidden,
        duration=args.duration,
        reps=args.reps,
        viztraces=args.viztraces,
        render=args.render,
        seed=args.seed,
    )

    if args.scores:
        print(f"\nAverage reward over {args.reps} episodes: {final_avg:.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")

    return final_avg


if __name__ == "__main__":
    main()
