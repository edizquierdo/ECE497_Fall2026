"""
Evolutionary algorithm for evolving neural network controllers on CartPole.

This module wraps the neuroevolution process using EvoTorch to evolve
a neural network controller for the CartPole balancing task.
"""

import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation

from neural_controller import NeuralController

# Suppress EvoTorch warnings
logging.getLogger("evotorch").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve a neural network controller for CartPole."
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=8,
        help="Number of hidden neurons (default: 8)",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=80,
        help="Population size (default: 80)",
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=40,
        help="Number of generations (default: 40)",
    )
    parser.add_argument(
        "--episodes_per_eval",
        type=int,
        default=5,
        help="Number of episodes to average per fitness evaluation (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Max steps per episode during training (default: 500, matches "
             "CartPole-v1's built-in limit and sim.py's default --duration)",
    )
    parser.add_argument(
        "--mut_stdev",
        type=float,
        default=0.4,
        help="Gaussian mutation standard deviation (default: 0.4)",
    )
    parser.add_argument(
        "--tournament_size",
        type=int,
        default=3,
        help="Tournament size for SBX crossover (default: 3)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=20,
        help="Distribution index for SBX crossover (default: 20)",
    )
    parser.add_argument(
        "--no-elitism",
        action="store_false",
        dest="elitism",
        help="Disable elitism (best individual always survives by default)",
    )
    parser.add_argument(
        "--vizperf",
        action="store_true",
        help="Visualize fitness over generations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-generation statistics to console",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to save the best genome (e.g., best_genome.npy)",
    )
    parser.add_argument(
        "--fitness_output",
        type=str,
        default=None,
        help="Save per-generation best/avg/worst fitness to this .npz file (keys: 'best', "
             "'avg', 'worst'), so runs from different configurations can be reloaded and "
             "compared later without re-running evolution.",
    )
    parser.add_argument(
        "--seed_genome",
        type=str,
        default=None,
        help="Path to a genome saved by a previous --output run (must match the current "
             "--hidden architecture). Seeds the initial population around this genome "
             "instead of starting from scratch, to continue/restart evolution from a "
             "previous best result.",
    )
    parser.add_argument(
        "--seed_noise",
        type=float,
        default=0.05,
        help="Stdev of the Gaussian perturbation applied to --seed_genome copies that fill "
             "out the rest of the initial population (one individual is kept as an exact, "
             "unperturbed copy of the seed). Ignored if --seed_genome isn't given.",
    )
    return parser.parse_args()


def make_fitness_fn(hidden_size, episodes_per_eval=5, duration=500):
    """
    Create a fitness function that evaluates a genome as a neural controller.

    Args:
        hidden_size: Number of hidden neurons in the network.
        episodes_per_eval: Number of episodes to average for each evaluation.
        duration: Max steps per episode (must match what sim.py uses to
            evaluate the same genome later, or fitness comparisons won't
            be apples-to-apples).

    Returns:
        A callable (genome -> float) suitable for EvoTorch.
    """

    def fitness_fn(genome: torch.Tensor) -> torch.Tensor:
        """Evaluate a genome by running it in the CartPole simulation."""
        from cartpole import CartPole

        total_reward = 0.0

        # Create controller and load genome
        controller = NeuralController(hidden_size=hidden_size)
        torch.nn.utils.vector_to_parameters(genome, controller.parameters())
        controller.eval()

        # Reuse a single environment across episodes; gymnasium's reset()
        # already randomizes the initial state each time, so a fresh
        # env per episode isn't needed and is far more expensive.
        env = CartPole(max_episode_steps=duration)

        for ep in range(episodes_per_eval):
            obs, _ = env.reset()

            episode_reward = 0.0
            while True:
                action = controller.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

                if terminated or truncated:
                    break

            total_reward += episode_reward

        env.close()

        return total_reward / episodes_per_eval  # Average reward per episode

    return fitness_fn


def run_evolution(
    hidden_size=8,
    popsize=80,
    gens=40,
    mut_stdev=0.4,
    tournament_size=3,
    eta=20,
    elitism=True,
    episodes_per_eval=5,
    duration=500,
    verbose=False,
    seed=None,
    seed_genome_path=None,
    seed_noise=0.05,
):
    """
    Run the neuroevolution process.

    Args:
        hidden_size: Number of hidden neurons.
        popsize: Population size.
        gens: Number of generations.
        mut_stdev: Mutation standard deviation.
        tournament_size: Tournament size for crossover.
        eta: Distribution index for SBX.
        elitism: Whether to use elitism.
        episodes_per_eval: Episodes to average per evaluation.
        duration: Max steps per episode during training.
        verbose: Print progress if True.
        seed: Random seed.
        seed_genome_path: Path to a genome saved by a previous --output run. If given,
            seeds the initial population around this genome (one exact copy plus the
            rest perturbed by `seed_noise`) instead of starting from scratch.
        seed_noise: Stdev of the Gaussian perturbation applied to seed_genome_path copies.

    Returns:
        best_fit, avg_fit, worst_fit, best_genome, best_fitness
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create fitness function and problem
    fitness_fn = make_fitness_fn(
        hidden_size=hidden_size,
        episodes_per_eval=episodes_per_eval,
        duration=duration,
    )

    # Compute genome length
    temp = NeuralController(hidden_size=hidden_size)
    n_genes = sum(p.numel() for p in temp.parameters())

    problem = Problem(
        objective_sense="max",
        objective_func=fitness_fn,
        solution_length=n_genes,
        initial_bounds=(-1.0, 1.0),
        dtype=torch.float32,
    )

    # Create algorithm
    algorithm = GeneticAlgorithm(
        problem,
        popsize=popsize,
        operators=[
            # eta (SBX's "distribution index"): higher = offspring cluster
            # closer to their parents, lower = offspring spread further
            # apart. tournament_size: how many individuals compete for
            # each parent slot; higher = stronger pressure toward
            # already-fit individuals.
            SimulatedBinaryCrossOver(problem, tournament_size=tournament_size, eta=eta),
            GaussianMutation(problem, stdev=mut_stdev),
        ],
        elitist=elitism,
    )

    # -- Seed the initial population, if requested --
    # GeneticAlgorithm builds its starting population in its constructor
    # (accessible as .population immediately, before the first .step()), so
    # overwriting it here applies before any evolution happens.
    if seed_genome_path is not None:
        seed_genome = torch.tensor(np.load(seed_genome_path), dtype=torch.float32)
        if seed_genome.numel() != n_genes:
            raise ValueError(
                f"Seed genome at '{seed_genome_path}' has {seed_genome.numel()} values, but "
                f"the current --hidden architecture expects {n_genes}. Make sure --hidden "
                f"matches what the seed genome was evolved with."
            )
        values = algorithm.population.access_values(keep_evals=False)
        values[0] = seed_genome  # one exact, unperturbed copy
        noise = torch.randn((popsize - 1, n_genes)) * seed_noise
        values[1:] = seed_genome.unsqueeze(0) + noise

    # Evolution loop
    best_fit = np.zeros(gens)
    avg_fit = np.zeros(gens)
    worst_fit = np.zeros(gens)

    for gen in range(gens):
        algorithm.step()

        fitnesses = algorithm.population.access_evals()[:, 0]

        # Handle NaN values
        fitnesses = torch.where(torch.isnan(fitnesses), torch.zeros_like(fitnesses), fitnesses)

        best_fit[gen] = fitnesses.max().item()
        avg_fit[gen] = fitnesses.mean().item()
        worst_fit[gen] = fitnesses.min().item()

        if verbose and (gen % 10 == 0 or gen == gens - 1):
            print(f"Gen {gen:>4d} | Best: {best_fit[gen]:.3f} | Mean: {avg_fit[gen]:.3f} | Worst: {worst_fit[gen]:.3f}")

        # Early stopping if perfect fitness found
        if best_fit[gen] >= duration:
            if verbose:
                print(f"Converged at generation {gen}!")
            best_fit[gen:] = duration
            avg_fit[gen:] = avg_fit[gen]
            worst_fit[gen:] = worst_fit[gen]
            break

    # Retrieve best genome
    fitnesses = algorithm.population.access_evals()[:, 0]
    fitnesses = torch.where(torch.isnan(fitnesses), torch.zeros_like(fitnesses), fitnesses)
    best_idx = fitnesses.argmax().item()
    best_genome = algorithm.population.access_values()[best_idx].clone()
    best_fitness = fitnesses[best_idx].item()

    return best_fit, avg_fit, worst_fit, best_genome, best_fitness


def main():
    """CLI entry point to evolve neural controllers for CartPole.

    Parses command-line arguments, runs neuroevolution for CartPole balancing,
    and optionally saves the best genome and visualizes fitness trajectories.
    """
    args = parse_args()

    print(
        f"Evolving neural controller for CartPole: "
        f"hidden={args.hidden}, popsize={args.popsize}, gens={args.gens}, "
        f"episodes_per_eval={args.episodes_per_eval}, duration={args.duration}"
    )

    if args.seed_genome:
        print(f"Seeding initial population from: {args.seed_genome} (seed_noise={args.seed_noise})")

    best_fit, avg_fit, worst_fit, best_genome, best_fitness = run_evolution(
        hidden_size=args.hidden,
        popsize=args.popsize,
        gens=args.gens,
        mut_stdev=args.mut_stdev,
        tournament_size=args.tournament_size,
        eta=args.eta,
        elitism=args.elitism,
        episodes_per_eval=args.episodes_per_eval,
        duration=args.duration,
        verbose=args.verbose,
        seed=args.seed,
        seed_genome_path=args.seed_genome,
        seed_noise=args.seed_noise,
    )

    print(f"\nBest fitness achieved: {best_fitness:.3f} (average reward per episode)")

    # Save best genome to file if output path specified
    if args.output:
        np.save(args.output, best_genome.numpy())
        print(f"Best genome saved to: {args.output}")
        print(
            f"IMPORTANT: to simulate this genome correctly, sim.py must be "
            f"run with matching --hidden {args.hidden} --duration {args.duration} "
            f"(a mismatched --hidden loads the genome with WRONG weights, "
            f"silently, with no error)"
        )

    if args.fitness_output:
        np.savez(args.fitness_output, best=best_fit, avg=avg_fit, worst=worst_fit)
        print(f"Fitness-over-generations saved to: {args.fitness_output}")

    if args.vizperf:
        plt.figure(figsize=(10, 6))
        gens = np.arange(len(best_fit))
        plt.plot(gens, best_fit, label="Best Fitness", color="green", linewidth=2)
        plt.plot(gens, avg_fit, label="Average Fitness", color="blue", linewidth=2)
        plt.plot(gens, worst_fit, label="Worst Fitness", color="red", linewidth=1, linestyle="--")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (average reward)")
        plt.title("Neuroevolution Performance: CartPole Balancing")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
