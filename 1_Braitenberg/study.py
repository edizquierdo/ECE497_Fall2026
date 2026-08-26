#!/usr/bin/env python3
"""
Parameter study script for Braitenberg vehicle simulation.

This script runs the simulation iteratively while varying one parameter
and visualises the final score (average normalised distance from the light
source) as a function of that parameter.  Lower score = closer to the light
= better performance.

Run as:
    python study.py --param noise
    python study.py --param turn_gain --min 0.01 --max 1.0 --steps 20
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from sim import run_simulation

# Sensible default sweep ranges (min, max) for each sweepable parameter.
# Used whenever the user does not explicitly pass --min / --max.
DEFAULT_RANGES = {
    "noise":        (0.0, 1.0),
    "turn_gain":    (0.0, 40.0),   # wide enough to show BOTH regimes: fitness improves with
                                    # diminishing returns out past turn_gain=20, then breaks down
                                    # (a single step can overshoot by full rotations) -- a range
                                    # capped at 1.0 only ever shows the first half of that story.
    "distance":     (2.0, 20.0),   # distance=0 makes the sensor gain (1/distance) blow up
    "angle_offset": (0.0, np.pi),  # sensors range from facing forward (0) to facing backward (pi)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter study for Braitenberg vehicle simulation."
    )
    parser.add_argument(
        "--param",
        type=str,
        default="noise",
        choices=["noise", "turn_gain", "distance", "angle_offset"],  # standardized option names
        help="Parameter to sweep (default: noise)"
    )
    parser.add_argument(
        "--min",
        type=float,
        default=None,
        help="Minimum parameter value (default: depends on --param, see DEFAULT_RANGES)"
    )
    parser.add_argument(
        "--max",
        type=float,
        default=None,
        help="Maximum parameter value (default: depends on --param, see DEFAULT_RANGES)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=21,
        help="Number of steps between min and max (default: 21)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of repetitions per parameter value (default: 10)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5000,
        help="Simulation duration per run (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for plot (default: study_<param>.png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        # Note: with --seed set, np.random.seed(seed) is called fresh at
        # the start of every point in the sweep, so every parameter value
        # starts from the same RNG state (paired comparison / reduced
        # variance across the sweep) rather than being fully independent.
        help="Random seed for reproducibility (see note above re: how this "
             "affects independence between sweep points)"
    )
    parser.add_argument(
        "--wiring",
        type=str,
        default="crossed",
        choices=["crossed", "direct"],
        help="Sensor-to-motor wiring scheme, held fixed across the sweep (default: crossed). "
             "Only has an effect once you implement Vehicle.think()'s OPTIONAL runtime switch."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )
    return parser.parse_args()


def run_study(param_name, param_values, verbose=False, **sim_kwargs):
    """Run simulation for each parameter value and collect final scores.

    Args:
        param_name:   Name of the parameter to vary. Can be ANY keyword
                       argument accepted by sim.run_simulation() -- this
                       function itself doesn't restrict it to the fixed
                       list in --param's CLI choices above. That CLI
                       restriction exists only because DEFAULT_RANGES needs
                       a known range to fall back on; if you add a new
                       parameter to run_simulation() (e.g. for one of the
                       optional challenges) you can sweep it by calling
                       run_study() directly with your own param_values,
                       without touching this file's CLI at all.
        param_values: List of parameter values to test.
        verbose:      Print progress information if True.
        **sim_kwargs: Other simulation parameters (reps, duration, etc.).

    Returns:
        List of final scores corresponding to each parameter value.
    """
    scores = []

    for i, val in enumerate(param_values):
        if verbose:
            print(f"[{i+1}/{len(param_values)}] Testing {param_name}={val:.4f}")

        # Inject the swept parameter value and run the simulation
        sim_kwargs[param_name] = val
        score = run_simulation(**sim_kwargs)
        scores.append(score)

    return scores


def main():
    args = parse_args()

    # Fall back to this parameter's default range if --min/--max weren't given
    default_min, default_max = DEFAULT_RANGES[args.param]
    if args.min is None:
        args.min = default_min
    if args.max is None:
        args.max = default_max

    # Generate parameter values
    param_values = np.linspace(args.min, args.max, args.steps)

    # Set up simulation kwargs from command line arguments
    sim_kwargs = {
        "reps":     args.reps,
        "duration": args.duration,
        "seed":     args.seed,
        "wiring":   args.wiring,
    }

    # All supported parameters happen to share the same name in both the CLI
    # and the run_simulation() signature, so no mapping is needed.
    sim_param_name = args.param

    if args.verbose:
        print(f"Running parameter study for {args.param}")
        print(f"Range: {param_values[0]:.4f} to {param_values[-1]:.4f}, {len(param_values)} steps")
        print()

    # Run the study (do not pass verbose into run_simulation — it doesn't accept it)
    scores = run_study(sim_param_name, param_values, verbose=args.verbose, **sim_kwargs)

    if args.verbose:
        print()
        print("Results summary:")
        for val, score in zip(param_values, scores):
            print(f"  {args.param}={val:.4f} -> score={score:.4f}")

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, scores, marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.xlabel(f"{args.param}", fontsize=12)
    plt.ylabel("Final Score (avg. distance from light)", fontsize=12)
    plt.title(f"Braitenberg Vehicle: Score vs {args.param.capitalize()}", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Highlight best and worst parameter values.
    # Higher fitness = robot reached and stayed near the light = better performance.
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)
    plt.scatter([param_values[best_idx]], [scores[best_idx]],
                color='green', s=150, zorder=5, label=f'Best: {param_values[best_idx]:.3f}')
    plt.scatter([param_values[worst_idx]], [scores[worst_idx]],
                color='red', s=150, zorder=5, label=f'Worst: {param_values[worst_idx]:.3f}')
    plt.legend()

    # Save or show plot
    output_file = args.output if args.output else f"study_{args.param}.png"
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
