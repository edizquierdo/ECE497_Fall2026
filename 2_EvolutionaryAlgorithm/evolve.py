import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.algorithms import SNES
from evotorch.operators import GaussianMutation

logging.getLogger("evotorch").setLevel(logging.WARNING)

def resolve_device(device_str: str = "auto") -> str:
    """Resolve device parameter to a valid torch device string.

    Args:
        device_str: Target device requested ("auto", "cuda", "cuda:0", "mps", "cpu", etc.)

    Returns:
        String representing the resolved device.
    """
    if device_str.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str

def parse_args():
    parser = argparse.ArgumentParser(description="Perform an evolutionary run.")
    parser.add_argument("--popsize", type=int, default=10, help="Size of the population")
    parser.add_argument("--gens", type=int, default=10, help="Number of generations to evolve")
    parser.add_argument("--algo", type=str, default="GA", choices=["GA", "ES"], help="Algorithm to use: GA or ES")
    parser.add_argument("--genesize", type=int, default=10, help="Length of each individual's genome")
    parser.add_argument("--minbound", type=float, default=0.0, help="Minimum gene value")
    parser.add_argument("--maxbound", type=float, default=1.0, help="Maximum gene value")
    parser.add_argument("--mut_stdev", type=float, default=0.1, help="Standard deviation for Gaussian mutation")
    parser.add_argument("--no-elitism", action='store_false', dest='elitism', help="Disable elitism (best individual always survives by default)")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cuda, cuda:0, mps, cpu")
    parser.add_argument("--vizperf", action='store_true', help="Enable visualization of fitness over generations")
    parser.add_argument("--verbose", action='store_true', help="Print generations and fitness values to console")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="File path to save the best genome, e.g. best_genome.npy (saves "
                             "the raw continuous genome, not the thresholded 0/1 genotype)")
    parser.add_argument("--fitness_output", type=str, default=None,
                        help="Save per-generation best/avg/worst fitness to this .npz file "
                             "(keys: 'best', 'avg', 'worst'), so runs from different "
                             "configurations can be reloaded and compared later without "
                             "re-running evolution.")
    parser.add_argument("--seed_genome", type=str, default=None,
                        help="Path to a genome saved by a previous --output run (must match "
                             "the current --genesize). Seeds the initial population around this "
                             "genome instead of starting from scratch, to continue/restart "
                             "evolution from a previous best result. Only supported with "
                             "--algo GA, not --algo ES.")
    parser.add_argument("--seed_noise", type=float, default=0.05,
                        help="Stdev of the Gaussian perturbation applied to --seed_genome "
                             "copies that fill out the rest of the initial population (one "
                             "individual is kept as an exact, unperturbed copy of the seed). "
                             "Ignored if --seed_genome isn't given.")
    return parser.parse_args()

def count_ones(x: torch.Tensor) -> torch.Tensor:
    """Fitness function: count genes whose value is > 0.5 (i.e. 'ON').

    Note: Genes are real-valued during evolution; the 0.5 threshold is only applied
    for fitness evaluation and final genotype interpretation.
    """
    return ((x > 0.5).float().sum(dim=-1)) / x.shape[-1]  # Normalize by genome length

def run_evolution(popsize=10, gens=10, algo="GA", genesize=10,
                  minbound=0.0, maxbound=1.0, mut_stdev=0.1,
                  elitism=True, verbose=False, seed=None,
                  device="auto", seed_genome_path=None, seed_noise=0.05):
    """
    Run an Evolutionary Algorithm.

    Args:
        popsize: Size of the population
        gens: Number of generations
        algo: Algorithm to use ("GA" for Genetic Algorithm or "ES" for Evolution Strategy)
        genesize: Size of each individual's genome
        minbound: Minimum value for gene initialization
        maxbound: Maximum value for gene initialization
        mut_stdev: Standard deviation for Gaussian mutation
        elitism: If True, best individual always survives to next generation
        verbose: Print generations and fitness values to console
        seed: Random seed for reproducibility
        device: Hardware device to run evolution on ("auto", "cuda", "mps", "cpu", etc.)
        seed_genome_path: Path to a genome saved by a previous --output run. If given,
            seeds the initial population around this genome (one exact copy plus the
            rest perturbed by `seed_noise`) instead of starting from scratch. Only
            supported when algo="GA" — raises ValueError if algo="ES".
        seed_noise: Stdev of the Gaussian perturbation applied to seed_genome_path copies.
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    resolved_device = resolve_device(device)
    if verbose:
        print(f"Running evolution on device: {resolved_device}")

    # --- Prepare arrays to store fitness values  ---
    best_fit = np.zeros(gens)
    avg_fit = np.zeros(gens)
    worst_fit = np.zeros(gens)

    problem = Problem(
        objective_sense="max",       # we want to MAXIMISE fitness
        objective_func=count_ones,     # use the max-ones fitness function as intended
        solution_length=genesize,         # X genes per individual
        initial_bounds=(minbound, maxbound),   # genes initialised uniformly in [minbound, maxbound]
        bounds=None if algo == "ES" else (minbound, maxbound),    # genes kept strictly within bounds (None for ES)
        dtype=torch.float32,
        device=resolved_device,
        vectorized=True,
    )

    if algo == "GA":
        algorithm = GeneticAlgorithm(
            problem,
            popsize=popsize,
            operators=[GaussianMutation(problem, stdev=mut_stdev)],
            elitist=elitism,   # best individual always survives → fitness never drops
        )
    elif algo == "ES":
        # SNES uses stdev_init for mutation strength. Note bounds=None was
        # set above for this branch -- genes are only bounded to
        # [minbound, maxbound] at generation-0 initialization; nothing
        # stops SNES's search distribution from drifting outside that
        # range as it evolves (unlike GA, which keeps bounds=(minbound,
        # maxbound) enforced every generation).
        algorithm = SNES(
            problem,
            stdev_init=mut_stdev,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # -- Seed the initial population, if requested --
    # GeneticAlgorithm builds its starting population in its constructor
    # (accessible as .population immediately, before the first .step()), so
    # overwriting it here applies before any evolution happens. SNES (the
    # --algo ES path) doesn't expose an equivalent, directly-seedable
    # population — it resamples each generation from its own internal
    # search distribution — so seeding isn't supported there.
    if seed_genome_path is not None:
        if algo != "GA":
            raise ValueError(
                "--seed_genome is only supported with --algo GA, not --algo ES."
            )
        seed_genome = torch.tensor(np.load(seed_genome_path), dtype=torch.float32, device=resolved_device)
        if seed_genome.numel() != genesize:
            raise ValueError(
                f"Seed genome at '{seed_genome_path}' has {seed_genome.numel()} values, but "
                f"--genesize {genesize} expects {genesize}. Make sure --genesize matches what "
                f"the seed genome was evolved with."
            )
        values = algorithm.population.access_values(keep_evals=False)
        values[0] = seed_genome  # one exact, unperturbed copy
        noise = torch.randn((popsize - 1, genesize), device=resolved_device) * seed_noise
        values[1:] = seed_genome.unsqueeze(0) + noise
        values.clamp_(minbound, maxbound)

    for gen in range(gens):
        algorithm.step()

        # --- read and fitness values from the population every generation ---
        fitnesses = algorithm.population.access_evals()[:, 0]
        best_fit[gen] = fitnesses.max().item()
        avg_fit[gen] = fitnesses.mean().item()
        worst_fit[gen] = fitnesses.min().item()

        # --- console log every X generations ---
        if verbose and (gen % 10 == 0 or gen == gens - 1):
            print(f"Gen {gen:>4d} | Best: {best_fit[gen]:.2f} | Mean: {avg_fit[gen]:.2f} | Worst: {worst_fit[gen]:.2f}")

    # Best solution
    if "best" in algorithm.status:
        best_sol = algorithm.status["best"]
    elif "pop_best" in algorithm.status:
        best_sol = algorithm.status["pop_best"]
    else:
        best_idx_in_pop = algorithm.population.access_evals()[:, 0].argmax()
        best_sol = algorithm.population[best_idx_in_pop]

    best_values = best_sol.values
    # Threshold continuous gene values at 0.5 to produce a binary genotype for interpretation
    best_genotype = (best_values > 0.5).int()

    return best_fit, avg_fit, worst_fit, best_genotype, best_values

def main():
    """Entry point for the evolutionary algorithm CLI.

    Parses command-line arguments and runs an evolutionary optimization using
    the configured algorithm and parameters. Prints a short summary of results
    and (optionally) visualizes fitness curves.
    """
    args = parse_args()

    if args.seed_genome:
        print(f"Seeding initial population from: {args.seed_genome} (seed_noise={args.seed_noise})")

    best_fit, avg_fit, worst_fit, best_genotype, best_values = run_evolution(
        popsize=args.popsize,
        gens=args.gens,
        algo=args.algo,
        genesize=args.genesize,
        minbound=args.minbound,
        maxbound=args.maxbound,
        mut_stdev=args.mut_stdev,
        elitism=args.elitism,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        seed_genome_path=args.seed_genome,
        seed_noise=args.seed_noise,
    )

    if args.output:
        np.save(args.output, best_values.cpu().numpy())
        print(f"Best genome saved to: {args.output}")

    if args.fitness_output:
        np.savez(args.fitness_output, best=best_fit, avg=avg_fit, worst=worst_fit)
        print(f"Fitness-over-generations saved to: {args.fitness_output}")

    # Handle visualization
    if args.vizperf:
        plt.figure()
        plt.plot(best_fit, label="Best Fitness", color='green')
        plt.plot(avg_fit, label="Average Fitness", color='blue')
        plt.plot(worst_fit, label="Worst Fitness", color='red')
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Evolutionary Algorithm Performance")
        plt.tight_layout()
        plt.show()

    print(f"\nBest genotype: {best_genotype.tolist()}")
    print(f"Number of ones in best solution: {best_genotype.sum().item()}")

if __name__ == "__main__":
    main()
