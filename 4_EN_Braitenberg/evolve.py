"""
Neuroevolution of a neural network controller for Braitenberg phototaxis.

The same pipeline as Project 3 (a flat genome of network weights, evolved
with EvoTorch's GeneticAlgorithm), with one big change: fitness is no longer
computed from a fixed truth table, but by putting each network in the
vehicle's body and letting it drive around.
"""

import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation

from braitenberg import simulate_population
from neural_controller import batched_forward, genome_size

logging.getLogger("evotorch").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────
#  1.  Fitness Function
# ─────────────────────────────────────────────────────────────

def make_fitness_fn(hidden=8, hidden_sizes=None, activation="tanh", episodes_per_eval=5,
                    duration=2000, distance=10.0, angle_offset=np.pi / 2, turn_gain=0.1,
                    noise_stdev=0.1):
    """Return a *vectorized* fitness function for a batch of controller genomes.

    Each individual drives `episodes_per_eval` vehicles, each starting at a
    random point `distance` away from the light and facing a random direction.
    All pop x episodes_per_eval vehicles are simulated together in one batch
    (see simulate_population in braitenberg.py).

    Fitness: at every step, reward proximity to the light as 1 / (1 + d), where
    d is the vehicle's distance to the light; average over the episode and over
    the episodes. This is bounded in (0, 1], but 1.0 is not reachable in
    practice: every vehicle starts `distance` away and needs time to get to
    the light (at top speed, about 500 steps for the default distance of 10).

    Because the starting conditions are random, fitness is *noisy*: evaluating
    the same genome twice gives two different values.

    Returns:
        A callable (genomes: Tensor[pop, n_genes]) -> Tensor[pop] suitable
        for EvoTorch's vectorized objective mode (Problem(..., vectorized=True)).
    """

    def fitness_fn(genomes: torch.Tensor) -> torch.Tensor:
        """genomes: (pop, n_genes) -> fitness: (pop,)."""
        # EvoTorch hands us its own read-only Tensor subclass, which makes
        # every tensor operation slower; a plain tensor runs ~3x faster.
        genomes = genomes.as_subclass(torch.Tensor)
        pop = genomes.shape[0]

        def think(sensors):
            return batched_forward(genomes, sensors, hidden=hidden,
                                   hidden_sizes=hidden_sizes, activation=activation)

        dist = simulate_population(think, pop, episodes_per_eval, duration=duration,
                                   distance=distance, angle_offset=angle_offset,
                                   turn_gain=turn_gain, noise_stdev=noise_stdev)  # (pop, episodes, duration)

        # Bounded proximity reward: 1.0 when on top of the light, approaching
        # 0.0 as distance grows. Averaged over time steps, then over episodes.
        reward = 1.0 / (1.0 + dist)
        return reward.mean(dim=(1, 2))

    return fitness_fn


def evaluate_genome(genome, episodes=100, **fitness_kwargs):
    """Re-evaluate one genome on many fresh episodes, for a less noisy fitness estimate.

    The fitness values reported *during* evolution are each based on only
    `episodes_per_eval` random starts, and the best of them is usually a lucky
    draw. Use this to measure how good an evolved genome really is.

    Args:
        genome: Flat genome tensor (n_genes,).
        episodes: Number of episodes to average over.
        **fitness_kwargs: Same keyword arguments as make_fitness_fn (hidden,
            activation, duration, distance, ...) -- must match evolution.

    Returns:
        (mean, std) of the per-episode fitness.
    """
    fitness_fn = make_fitness_fn(episodes_per_eval=1, **fitness_kwargs)
    with torch.no_grad():
        per_episode = fitness_fn(genome.unsqueeze(0).repeat(episodes, 1))
    return per_episode.mean().item(), per_episode.std().item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve a neural network controller for the Braitenberg vehicle."
    )
    # -- Network --
    parser.add_argument("--hidden", type=int, default=8,
                        help="Number of hidden neurons in a single hidden layer (default: 8)")
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=None,
                        help="List of hidden layer sizes, e.g. --hidden_sizes 16 16. "
                             "Overrides --hidden when given; pass it with no values for no hidden layer "
                             "(default: None)")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "sigmoid"],
                        help="Hidden layer activation. The output layer is always Tanh (default: tanh)")
    # -- Evolutionary algorithm --
    parser.add_argument("--popsize", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument("--mut_stdev", type=float, default=0.5,
                        help="Gaussian mutation standard deviation (default: 0.5)")
    parser.add_argument("--tournament_size", type=int, default=3,
                        help="Tournament size for SBX crossover (default: 3)")
    parser.add_argument("--eta", type=float, default=20,
                        help="Distribution index for SBX crossover (default: 20)")
    parser.add_argument("--no-crossover", action="store_false", dest="use_crossover",
                        help="Disable SBX crossover, running a mutation-only GA (default: crossover on)")
    parser.add_argument("--no-elitism", action="store_false", dest="elitism",
                        help="Disable elitism (default: elitism on)")
    parser.add_argument("--init_bounds", type=float, nargs=2, default=[-1.0, 1.0], metavar=("LOW", "HIGH"),
                        help="Initial genome sampling bounds (default: -1.0 1.0)")
    parser.add_argument("--seed_genome", type=str, default=None,
                        help="Seed the initial population around a genome saved by a previous --output "
                             "run (must match --hidden/--hidden_sizes)")
    parser.add_argument("--seed_noise", type=float, default=0.05,
                        help="Stdev of the perturbation applied to --seed_genome copies (default: 0.05)")
    # -- Task / vehicle --
    parser.add_argument("--episodes_per_eval", type=int, default=5,
                        help="Episodes (random starts) averaged per fitness evaluation (default: 5)")
    parser.add_argument("--duration", type=int, default=2000,
                        help="Number of simulation steps per episode (default: 2000)")
    parser.add_argument("--distance", type=float, default=10.0,
                        help="Starting distance from the light (default: 10.0)")
    parser.add_argument("--angle_offset", type=float, default=np.pi / 2,
                        help="Angular separation between sensors (radians, default: pi/2)")
    parser.add_argument("--turn_gain", type=float, default=0.1, help="Turn gain for steering (default: 0.1)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Motion noise standard deviation (default: 0.1)")
    # -- Output --
    parser.add_argument("--final_evals", type=int, default=100,
                        help="Episodes used to re-evaluate the best genome after evolution (default: 100)")
    parser.add_argument("--vizperf", action="store_true", help="Plot fitness over generations")
    parser.add_argument("--verbose", action="store_true", help="Print per-generation statistics")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="File path to save the best genome (e.g., best_genome.npy)")
    parser.add_argument("--fitness_output", type=str, default=None,
                        help="Save per-generation best/avg/worst fitness to this .npz file")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
#  2.  Neuroevolution Runner
# ─────────────────────────────────────────────────────────────

def run_neuroevolution(hidden=8, popsize=50, gens=100, mut_stdev=0.5, activation="tanh",
                       verbose=False, seed=None, tournament_size=3, eta=20, elitism=True,
                       seed_genome_path=None, seed_noise=0.05, hidden_sizes=None,
                       init_bounds=(-1.0, 1.0), use_crossover=True, episodes_per_eval=5,
                       duration=2000, distance=10.0, angle_offset=np.pi / 2, turn_gain=0.1,
                       noise_stdev=0.1):
    """Evolve a neural network controller for Braitenberg phototaxis.

    The genome encodes all weights and biases of NeuralController as a flat
    real-valued vector. A GeneticAlgorithm with SimulatedBinaryCrossOver and
    GaussianMutation maximizes the fitness function from make_fitness_fn.

    Unlike Project 3, there is no early stopping: there is no known "perfect"
    fitness to stop at, so every run uses all `gens` generations.

    Args:
        hidden: Number of hidden neurons (single hidden layer).
        popsize: Number of individuals in the population.
        gens: Number of generations to run.
        mut_stdev: Standard deviation for Gaussian mutation.
        activation: Hidden activation function ('tanh', 'sigmoid', 'relu').
        verbose: If True, print statistics every 10 generations.
        seed: Random seed for reproducibility.
        tournament_size: Tournament size for SBX crossover.
        eta: Distribution index for SBX crossover.
        elitism: If True, parents compete with their children for survival.
        seed_genome_path: Path to a genome saved by a previous --output run. If given,
            seeds the initial population around this genome (one exact copy plus the
            rest perturbed by `seed_noise`) instead of starting from scratch.
        seed_noise: Stdev of the Gaussian perturbation applied to seed_genome_path copies.
        hidden_sizes: Optional list of hidden-layer sizes (overrides `hidden`).
        init_bounds: (low, high) range for the initial genome sampling.
        use_crossover: If False, run a mutation-only GA.
        episodes_per_eval: Number of random-start episodes averaged per fitness evaluation.
        duration, distance, angle_offset, turn_gain, noise_stdev: Task / vehicle settings.

    Returns:
        best_fit     (np.ndarray): Best fitness at each generation.
        avg_fit      (np.ndarray): Average fitness at each generation.
        worst_fit    (np.ndarray): Worst fitness at each generation.
        best_genome  (torch.Tensor): Flat weight vector of the best individual in the final population.
        best_fitness (float): That individual's (noisy) fitness from its last evaluation.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    fitness_fn = make_fitness_fn(hidden=hidden, hidden_sizes=hidden_sizes, activation=activation,
                                 episodes_per_eval=episodes_per_eval, duration=duration,
                                 distance=distance, angle_offset=angle_offset,
                                 turn_gain=turn_gain, noise_stdev=noise_stdev)

    # Compute genome length analytically (see genome_size() in neural_controller.py).
    n_genes = genome_size(hidden=hidden, hidden_sizes=hidden_sizes)

    problem = Problem(
        objective_sense="max",              # maximise fitness
        objective_func=fitness_fn,
        solution_length=n_genes,            # one gene per weight/bias
        initial_bounds=tuple(init_bounds),  # weights initialised in [low, high]
        dtype=torch.float32,
        vectorized=True,                    # fitness_fn takes/returns whole-population batches
    )

    operators = []
    if use_crossover:
        # eta (SBX's "distribution index"): higher = offspring cluster
        # closer to their parents, lower = offspring spread further apart.
        # tournament_size: how many individuals compete for each parent
        # slot; higher = stronger pressure toward already-fit individuals.
        operators.append(SimulatedBinaryCrossOver(problem, tournament_size=tournament_size, eta=eta))
    operators.append(GaussianMutation(problem, stdev=mut_stdev))

    # With elitism, parents compete with their children for a place in the
    # next generation. Note that EvoTorch *re-evaluates* the parents every
    # generation (on fresh random starts), so -- unlike in Project 3 -- the
    # best fitness can go down from one generation to the next.
    algorithm = GeneticAlgorithm(
        problem,
        popsize=popsize,
        operators=operators,
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
                f"the current --hidden/--hidden_sizes architecture expects {n_genes}. Make sure "
                f"these match what the seed genome was evolved with."
            )
        values = algorithm.population.access_values(keep_evals=False)
        values[0] = seed_genome  # one exact, unperturbed copy
        noise = torch.randn((popsize - 1, n_genes)) * seed_noise
        values[1:] = seed_genome.unsqueeze(0) + noise

    best_fit  = np.zeros(gens)
    avg_fit   = np.zeros(gens)
    worst_fit = np.zeros(gens)

    for gen in range(gens):
        algorithm.step()

        fitnesses = algorithm.population.access_evals()[:, 0]
        # Replace any NaN values that may arise from bad mutations
        fitnesses = torch.where(torch.isnan(fitnesses), torch.zeros_like(fitnesses), fitnesses)

        best_fit[gen]  = fitnesses.max().item()
        avg_fit[gen]   = fitnesses.mean().item()
        worst_fit[gen] = fitnesses.min().item()

        if verbose and (gen % 10 == 0 or gen == gens - 1):
            print(f"Gen {gen:>4d} | Best: {best_fit[gen]:.4f} | "
                  f"Mean: {avg_fit[gen]:.4f} | Worst: {worst_fit[gen]:.4f}")

    # Retrieve the best genome of the final population
    fitnesses = algorithm.population.access_evals()[:, 0]
    fitnesses = torch.where(torch.isnan(fitnesses), torch.zeros_like(fitnesses), fitnesses)
    best_idx     = fitnesses.argmax().item()
    best_genome  = algorithm.population.access_values()[best_idx].clone()
    best_fitness = fitnesses[best_idx].item()

    return best_fit, avg_fit, worst_fit, best_genome, best_fitness


# ─────────────────────────────────────────────────────────────
#  3.  Visualisation
# ─────────────────────────────────────────────────────────────

def plot_fitness(best_fit, avg_fit, worst_fit):
    """Plot best/average/worst fitness over generations."""
    plt.figure(figsize=(10, 6))
    gens = np.arange(len(best_fit))
    plt.plot(gens, best_fit, label="Best Fitness", color="green", linewidth=2)
    plt.plot(gens, avg_fit, label="Average Fitness", color="blue", linewidth=2)
    plt.plot(gens, worst_fit, label="Worst Fitness", color="red", linewidth=1, linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Neuroevolution Performance: Braitenberg Vehicle Phototaxis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  4.  Main Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    """Command-line entry point: evolve a controller, re-evaluate it, optionally save and plot."""
    args = parse_args()

    print(f"Evolving neural controller for Braitenberg vehicle: hidden={args.hidden}, "
          f"hidden_sizes={args.hidden_sizes}, activation={args.activation}, popsize={args.popsize}, "
          f"gens={args.gens}, duration={args.duration}, episodes_per_eval={args.episodes_per_eval}")

    if args.seed_genome:
        print(f"Seeding initial population from: {args.seed_genome} (seed_noise={args.seed_noise})")

    task_kwargs = dict(duration=args.duration, distance=args.distance, angle_offset=args.angle_offset,
                       turn_gain=args.turn_gain, noise_stdev=args.noise)

    best_fit, avg_fit, worst_fit, best_genome, best_fitness = run_neuroevolution(
        hidden=args.hidden,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        popsize=args.popsize,
        gens=args.gens,
        mut_stdev=args.mut_stdev,
        tournament_size=args.tournament_size,
        eta=args.eta,
        use_crossover=args.use_crossover,
        elitism=args.elitism,
        init_bounds=tuple(args.init_bounds),
        seed_genome_path=args.seed_genome,
        seed_noise=args.seed_noise,
        verbose=args.verbose,
        seed=args.seed,
        episodes_per_eval=args.episodes_per_eval,
        **task_kwargs,
    )

    reeval_mean, reeval_std = evaluate_genome(best_genome, episodes=args.final_evals,
                                              hidden=args.hidden, hidden_sizes=args.hidden_sizes,
                                              activation=args.activation, **task_kwargs)

    print(f"\nBest fitness in final generation ({args.episodes_per_eval} episodes): {best_fitness:.4f}")
    print(f"Same genome re-evaluated on {args.final_evals} new episodes: "
          f"{reeval_mean:.4f} ± {reeval_std:.4f}")

    if args.output:
        np.save(args.output, best_genome.numpy())
        print(f"Best genome saved to: {args.output}")

    if args.fitness_output:
        np.savez(args.fitness_output, best=best_fit, avg=avg_fit, worst=worst_fit)
        print(f"Fitness-over-generations saved to: {args.fitness_output}")

    if args.vizperf:
        plot_fitness(best_fit, avg_fit, worst_fit)


if __name__ == "__main__":
    main()
