#!/usr/bin/env python3
"""
Parameter study script for evolutionary algorithm.

This script runs the evolution iteratively while varying one parameter
and visualizes the final fitness as a function of that parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from evolve import run_evolution, resolve_device

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter study for mutation standard deviation (mut_stdev)."
    )
    parser.add_argument(
        "--min",
        type=float,
        default=0.01,
        help="Minimum mut_stdev value (default: 0.01)"
    )
    parser.add_argument(
        "--max",
        type=float,
        default=0.5,
        help="Maximum mut_stdev value (default: 0.5)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=21,
        help="Number of steps between min and max (default: 21)"
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=50,
        help="Number of generations per run (default: 50)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of repetitions per parameter value (default: 5)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="GA",
        choices=["GA", "ES"],
        dest="algo",
        help="Evolutionary algorithm to use (default: GA)"
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=10,
        help="Population size (default: 10)"
    )
    parser.add_argument(
        "--genesize",
        type=int,
        default=10,
        help="Gene size / genome length (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cuda, cuda:0, mps, cpu (default: auto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="study_mut_stdev.png",
        help="Output filename for plot (default: study_mut_stdev.png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )
    parser.add_argument(
        "--fitness",
        type=str,
        default="count_ones",
        choices=["count_ones", "step", "rastrigin", "sparse"],
        help="Fitness function to optimize (default: count_ones)"
    )
    parser.add_argument(
        "--sparse_threshold",
        type=float,
        default=0.7,
        help="Fraction of genes that must be ON before the 'sparse' fitness function "
             "gives any reward. Ignored unless --fitness sparse is used."
    )
    parser.add_argument(
        "--data_output",
        type=str,
        default=None,
        help="Save the raw sweep data (mut_stdev values and their scores) to this "
             ".npz file (keys: 'mut_stdev_values', 'scores'), so the sweep can be "
             "reloaded and re-plotted later without re-running evolution."
    )
    return parser.parse_args()


def run_evolution_reps(**kwargs):
    """
    Run evolution multiple times and return average final best fitness.

    Args:
        **kwargs: Parameters to pass to run_evolution

    Returns:
        Average final best fitness across repetitions
    """
    reps = kwargs.get("reps", 5)
    # Remove reps from kwargs since run_evolution doesn't accept it
    kwargs = {k: v for k, v in kwargs.items() if k != "reps"}

    final_fitnesses = []
    base_seed = kwargs.get("seed")

    for r in range(reps):
        # Use different seed for each repetition if seed is provided
        if base_seed is not None:
            kwargs["seed"] = base_seed + r

        best_fit, avg_fit, worst_fit, best_genotype, best_values = run_evolution(**kwargs)
        final_fitnesses.append(best_fit[-1])

    return np.mean(final_fitnesses)


def run_study(mut_stdev_values, study_verbose=False, **sim_kwargs):
    """
    Run evolution for each mut_stdev value and collect final fitness.

    Args:
        mut_stdev_values: List of mut_stdev values to test
        study_verbose: Print progress information for the study
        **sim_kwargs: Other simulation parameters (gens, reps, algo, popsize, genesize, etc.)

    Returns:
        List of final fitness values corresponding to each mut_stdev value
    """
    scores = []

    for i, val in enumerate(mut_stdev_values):
        val_str = f"{val:.4f}"
        if study_verbose:
            print(f"[{i+1}/{len(mut_stdev_values)}] Testing mut_stdev={val_str}")

        # Update the mut_stdev value in simulation kwargs
        sim_kwargs["mut_stdev"] = val

        # Run evolution and collect score
        score = run_evolution_reps(**sim_kwargs)
        scores.append(score)

    return scores


def main():
    """Run a parameter study over mutation standard deviation.

    Parses command-line arguments, sweeps mut_stdev over the requested range,
    runs multiple repetitions per value, and produces a plot summarizing final
    best fitness as a function of mutation strength.
    """
    args = parse_args()

    # Generate parameter values
    mut_stdev_values = np.linspace(args.min, args.max, args.steps)

    # Set up simulation kwargs from command line arguments
    sim_kwargs = {
        "gens": args.gens,
        "reps": args.reps,
        "algo": args.algo,
        "popsize": args.popsize,
        "genesize": args.genesize,
        "device": args.device,
        "seed": args.seed,
        "verbose": False,  # Suppress per-generation output during study
        "fitness": args.fitness,
        "sparse_threshold": args.sparse_threshold,
    }

    if args.verbose:
        resolved = resolve_device(args.device)
        print("Running parameter study for mut_stdev")
        print(f"Device: {resolved}")
        print(f"Range: {mut_stdev_values[0]:.4f} to {mut_stdev_values[-1]:.4f}, {len(mut_stdev_values)} steps")
        print()

    # Run the study
    scores = run_study(mut_stdev_values, study_verbose=args.verbose, **sim_kwargs)

    if args.verbose:
        print()
        print("Results summary:")
        for val, score in zip(mut_stdev_values, scores):
            print(f"  mut_stdev={val:.4f} -> fitness={score:.4f}")

    if args.data_output:
        np.savez(args.data_output, mut_stdev_values=mut_stdev_values, scores=scores)
        print(f"Raw sweep data saved to: {args.data_output}")

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(mut_stdev_values, scores, marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.xlabel("Mutation Standard Deviation (mut_stdev)", fontsize=12)
    plt.ylabel("Final Best Fitness", fontsize=12)
    plt.title("Evolutionary Algorithm: Final Fitness vs mut_stdev", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add summary statistics
    min_idx = np.argmin(scores)
    max_idx = np.argmax(scores)
    min_val_str = f"{mut_stdev_values[min_idx]:.3f}"
    max_val_str = f"{mut_stdev_values[max_idx]:.3f}"
    plt.scatter([mut_stdev_values[min_idx]], [scores[min_idx]],
                color='red', s=150, zorder=5, label=f'Min: {min_val_str}')
    plt.scatter([mut_stdev_values[max_idx]], [scores[max_idx]],
                color='green', s=150, zorder=5, label=f'Max: {max_val_str}')
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
