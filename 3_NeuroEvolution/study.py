#!/usr/bin/env python3
"""
Parameter study script for neural network neuroevolution.

This script runs the neuroevolution loop iteratively while varying one
architectural or algorithmic parameter, then visualises the effect of that
parameter on evolutionary performance.

Supported parameters (choose with --param):
    hidden       — Number of hidden neurons
    popsize      — Population size
    mut_stdev    — Gaussian mutation standard deviation
    gens         — Number of generations
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from evolve import run_neuroevolution, generate_random_problem, generate_convex_problem, XOR_INPUTS, XOR_SIGNS, TASKS


# ─────────────────────────────────────────────────────────────
#  Argument Parsing
# ─────────────────────────────────────────────────────────────

# Default sweep range + axis label for each supported parameter.
# "int": True rounds swept values to integers (hidden/popsize/gens);
# mut_stdev is swept as a continuous float.
PARAM_DEFAULTS = {
    "hidden":    {"min": 1,   "max": 10,  "steps": 10, "label": "Number of Hidden Neurons",   "int": True},
    "popsize":   {"min": 20,  "max": 200, "steps": 10, "label": "Population Size",             "int": True},
    "mut_stdev": {"min": 0.1, "max": 2.0, "steps": 10, "label": "Mutation Standard Deviation", "int": False},
    "gens":      {"min": 20,  "max": 300, "steps": 10, "label": "Number of Generations",       "int": True},
}

# Fixed defaults for hidden/popsize/mut_stdev when *not* the swept parameter.
# "gens" is deliberately excluded here — it's always threaded through
# separately (via the --gens flag or the swept values), never via this dict,
# to avoid passing it twice into run_neuroevolution.
FIXED_DEFAULTS = {
    "hidden": 3,
    "popsize": 100,
    "mut_stdev": 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a parameter study for neuroevolution of an XOR controller.",
    )
    parser.add_argument(
        "--param",
        type=str,
        default="hidden",
        choices=list(PARAM_DEFAULTS.keys()),
        help="Parameter to sweep: hidden, popsize, mut_stdev, or gens (default: hidden)",
    )
    parser.add_argument(
        "--min",
        type=float,
        default=None,
        help="Minimum value for the swept parameter (default depends on --param)",
    )
    parser.add_argument(
        "--max",
        type=float,
        default=None,
        help="Maximum value for the swept parameter (default depends on --param)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of values to test between min and max (default depends on --param)",
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=200,
        help="Generations per run. Ignored (overridden by --min/--max/--steps) when --param gens.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of independent repetitions per parameter value (default: 5)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="xor",
        choices=["xor", "random", "convex"],
        help="Problem to solve: xor, random, or convex (default: xor)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=4,
        help="Number of points for random/convex problems (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="study_arch.png",
        help="Output filename for the plot (default: study_arch.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility (each repetition uses seed+rep)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information during the study",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: auto, cpu, cuda, mps, etc. (default: auto)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["tanh", "sigmoid", "relu"],
        help="Fixed (non-swept) activation function (default: tanh).",
    )
    parser.add_argument(
        "--fitness_mode",
        type=str,
        default="sign",
        choices=["sign", "mse"],
        help="Fixed (non-swept) fitness mode (default: sign).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="xor",
        choices=["xor", "and", "or", "xnor"],
        help="Boolean task to study when --problem xor (default: xor).",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Fixed hidden-layer-sizes override, e.g. --hidden_sizes 4 4. Ignored/"
             "meaningless when --param hidden, since --param hidden sweeps the single "
             "--hidden value that hidden_sizes would override (default: None).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
#  Single Run with Repetitions
# ─────────────────────────────────────────────────────────────

def run_reps(reps, gens, base_seed, problem_inputs, problem_signs, **sim_kwargs):
    """Run neuroevolution multiple times and return average final best fitness.

    The same (problem_inputs, problem_signs) instance is used for every
    repetition — only the EA's random seed varies across reps. This keeps
    the measured variance attributable to the algorithm's own stochasticity
    (population init, selection, crossover, mutation), rather than mixing in
    variance from a different randomly-generated problem on each repetition.

    Args:
        reps:           Number of independent repetitions.
        gens:           Number of generations per run.
        base_seed:      Base random seed (each rep adds its index to this seed).
                        If None, no seed is set.
        problem_inputs: Input tensor for the problem, held fixed across reps.
        problem_signs:  Sign labels for the problem, held fixed across reps.
        **sim_kwargs:   Additional keyword arguments forwarded to run_neuroevolution.

    Returns:
        mean_fitness (float): Average final best fitness across all repetitions.
        std_fitness  (float): Standard deviation of final best fitness.
    """
    final_fitnesses = []

    for rep in range(reps):
        seed = (base_seed + rep) if base_seed is not None else None
        # run_neuroevolution returns: best_fit, avg_fit, worst_fit, best_genome,
        #                             best_fitness, inputs, signs, convergence_gen (8 values)
        best_fit, _, _, _, _, _, _, _ = run_neuroevolution(
            gens=gens,
            seed=seed,
            verbose=False,
            inputs=problem_inputs,
            signs=problem_signs,
            **sim_kwargs,
        )
        final_fitnesses.append(best_fit[-1])

    return float(np.mean(final_fitnesses)), float(np.std(final_fitnesses))


# ─────────────────────────────────────────────────────────────
#  Parameter Sweep
# ─────────────────────────────────────────────────────────────

def run_study(param_name, param_values, reps, gens, base_seed, verbose,
              problem_inputs, problem_signs, **fixed_kwargs):
    """Sweep one parameter and record the mean final best fitness for each value.

    Args:
        param_name:     Name of the parameter being swept.
        param_values:   Sequence of values to test.
        reps:           Repetitions per parameter value.
        gens:           Generations per run (overridden per-value when param_name == "gens").
        base_seed:      Base random seed.
        verbose:        Print progress if True.
        problem_inputs: Problem inputs, held fixed across every run in the study.
        problem_signs:  Problem sign labels, held fixed across every run in the study.
        **fixed_kwargs: All other sim parameters held constant.

    Returns:
        means (list[float]): Mean final fitness for each parameter value.
        stds  (list[float]): Std dev of final fitness for each parameter value.
    """
    means, stds = [], []

    for i, val in enumerate(param_values):
        if verbose:
            print(f"  [{i+1}/{len(param_values)}] {param_name}={val}")

        # Inject the swept parameter into the kwargs (or into `gens` directly,
        # since gens is a positional-ish argument to run_reps rather than a
        # run_neuroevolution kwarg passed through **sim_kwargs).
        kwargs = dict(fixed_kwargs)
        run_gens = gens
        if param_name == "gens":
            run_gens = val
        else:
            kwargs[param_name] = val

        mean_fit, std_fit = run_reps(
            reps=reps, gens=run_gens, base_seed=base_seed,
            problem_inputs=problem_inputs, problem_signs=problem_signs,
            **kwargs,
        )
        means.append(mean_fit)
        stds.append(std_fit)

        if verbose:
            print(f"              → mean fitness={mean_fit:.4f} ± {std_fit:.4f}")

    return means, stds


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    defaults = PARAM_DEFAULTS[args.param]

    # Build array of values to sweep (integers for discrete params, floats otherwise)
    p_min   = args.min   if args.min   is not None else defaults["min"]
    p_max   = args.max   if args.max   is not None else defaults["max"]
    p_steps = args.steps if args.steps is not None else defaults["steps"]

    if defaults["int"]:
        param_values = np.unique(
            np.round(np.linspace(p_min, p_max, p_steps)).astype(int)
        ).tolist()
    else:
        param_values = np.linspace(p_min, p_max, p_steps).tolist()

    # Generate the problem instance ONCE, up front, and hold it fixed across
    # every run in the study (all parameter values, all repetitions). This
    # avoids confounding "variance across repetitions" (which Part 2 asks
    # students to study) with "variance because each rep saw a different
    # random problem instance."
    if args.problem == "random":
        problem_inputs, problem_signs = generate_random_problem(num_points=args.points, seed=args.seed)
    elif args.problem == "convex":
        problem_inputs, problem_signs = generate_convex_problem(num_points=args.points)
    else:
        problem_inputs, problem_signs = TASKS[args.task]

    if args.verbose:
        print(f"Studying {defaults['label'].lower()}")
        print(f"Range: {param_values[0]} to {param_values[-1]}, {len(param_values)} values")
        print(f"Repetitions per value: {args.reps}"
              + (f",  Generations per run: {args.gens}" if args.param != "gens" else ""))
        print()
        problem_desc = {
            "xor": "XOR problem",
            "random": f"Random problem ({args.points} points, fixed across all runs)",
            "convex": f"Convex problem ({args.points} points on circle)",
        }
        print(f"Problem: {problem_desc.get(args.problem, 'Unknown')}")

    # Fixed simulation parameters (the swept one, if it's here, is removed)
    fixed = dict(FIXED_DEFAULTS)
    fixed["activation"] = args.activation
    fixed["device"] = args.device
    fixed["fitness_mode"] = args.fitness_mode
    fixed["hidden_sizes"] = args.hidden_sizes
    if args.param in fixed:
        del fixed[args.param]

    if args.verbose:
        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
        print(f"Fixed parameters: {fixed_str}")
        print()

    means, stds = run_study(
        param_name=args.param,
        param_values=param_values,
        reps=args.reps,
        gens=args.gens,
        base_seed=args.seed,
        verbose=args.verbose,
        problem_inputs=problem_inputs,
        problem_signs=problem_signs,
        **fixed,
    )

    if args.verbose:
        print("\nResults summary:")
        for val, m, s in zip(param_values, means, stds):
            print(f"  {args.param}={val} → fitness={m:.4f} ± {s:.4f}")

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(param_values, means, marker="o", linewidth=2, markersize=5, label="Mean final fitness")
    ax.fill_between(
        param_values,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2, label="±1 std dev",
    )
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.6, label="Perfect fitness (1.0)")

    # Mark best value
    best_idx = int(np.argmax(means))
    ax.scatter([param_values[best_idx]], [means[best_idx]],
               color="green", s=150, zorder=5,
               label=f"Best: {args.param}={param_values[best_idx]}")

    ax.set_xlabel(defaults["label"], fontsize=12)
    ax.set_ylabel("Final Best Fitness", fontsize=12)

    # Build problem description for title
    problem_names = {"xor": "XOR", "random": f"Random({args.points})", "convex": f"Convex({args.points})"}
    problem_title = problem_names.get(args.problem, "Unknown")

    gens_label = "swept" if args.param == "gens" else f"{args.gens} gens"
    ax.set_title(
        f"Neuroevolution: Final Fitness vs {defaults['label']}\n"
        f"{problem_title} | {args.reps} reps × {gens_label}\n"
        f"fixed: " + ", ".join(f"{k}={v}" for k, v in fixed.items() if k != "device"),
        fontsize=11,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
