"""
Sweep hexapod_bridge.py's --timescale for a given evolved CTRNN genome,
running several repetitions per value, and plot average displacement
(direction-agnostic straight-line distance from the start position -- see
hexapod_bridge.py's run_transfer() docstring for why raw x-axis progress
isn't a fair metric on the real MuJoCo body) as a function of timescale.

CTRNN-only: a feedforward controller has no internal clock to rescale (it's
a pure function of the current observation), so `--controller feedforward`
is rejected here.

`timescale` is currently the only CLI-exposed knob controlling how the
CTRNN's internal integration step (dt passed to `act()`) relates to the
real hexapod's physical step -- see hexapod_bridge.py's `run_transfer()`
docstring for why that ratio isn't 1:1 by default (walker.py's TS=0.1s
per tick vs. hexapod_env.py's real dt=0.05s per step). This script sweeps
it the same way hexapod_torque_sweep.py sweeps --torque_scale; the two can
be composed by re-running this sweep at whatever --torque_scale that
script already found to be a good value.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from hexapod_bridge import run_transfer
from neural_controller import SM_DEFAULT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep --timescale for a walker.py-evolved CTRNN genome on the real hexapod."
    )
    parser.add_argument("--genome", type=str, required=True,
                         help="CTRNN genome evolved by evolve.py against walker.py (required -- "
                              "a random controller's timescale response isn't meaningful to sweep).")
    parser.add_argument("--topology", choices=["modular", "fully_connected"], default="modular",
                         help="Must match what --genome was evolved with")
    parser.add_argument("--module_size", type=int, default=SM_DEFAULT,
                         help="Must match what --genome was evolved with")
    parser.add_argument("--interneurons", type=int, default=0,
                         help="Must match what --genome was evolved with")
    parser.add_argument("--mode", choices=["cpg", "rpg", "mpg"], default="rpg",
                         help="Sensor condition to run under: cpg = no sensory input, rpg = "
                              "live leg-angle feedback, mpg = alternate by rep")
    parser.add_argument("--torque_scale", type=float, default=1.0,
                         help="Held fixed across the sweep -- run hexapod_torque_sweep.py first "
                              "if you don't already know a good value for this genome.")
    parser.add_argument("--duration", type=float, default=50.0,
                         help="Simulated seconds per repetition (same convention as "
                              "hexapod_bridge.py's --duration)")
    parser.add_argument("--timescales", type=float, nargs="+",
                         default=[0.25, 0.5, 1.0, 2.0, 4.0],
                         help="Timescale values to try (default spans a 16x range, same shape "
                              "as hexapod_torque_sweep.py's default)")
    parser.add_argument("--reps", type=int, default=5,
                         help="Repetitions per timescale value, each with a different seed")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed; repetition r at a given timescale uses seed + r, "
                              "so every timescale is tested against the *same* set of "
                              "reps/seeds for a fair comparison")
    parser.add_argument("--out", type=str, default="timescale_sweep.png",
                         help="Output plot path")
    parser.add_argument("--data_out", type=str, default=None,
                         help="Optional .npz path to also save the raw per-rep results "
                              "(keys: 'timescales', 'displacement', 'fell')")
    return parser.parse_args()


def sweep(
    genome_path, topology, module_size, n_interneurons, mode, torque_scale,
    duration, timescales, reps, base_seed,
):
    """Run reps repetitions at each timescale. Returns:
        displacement: (len(timescales), reps) total straight-line distance
                      from the start position per run (direction-agnostic --
                      see hexapod_bridge.py's run_transfer() docstring)
        fell:         (len(timescales), reps) bool, whether it fell
    """
    n_scales = len(timescales)
    displacement = np.zeros((n_scales, reps))
    fell = np.zeros((n_scales, reps), dtype=bool)

    for i, timescale in enumerate(timescales):
        for r in range(reps):
            seed = base_seed + r
            dist, did_fall, _ = run_transfer(
                controller="ctrnn",
                genome_path=genome_path,
                topology=topology,
                module_size=module_size,
                n_interneurons=n_interneurons,
                mode=mode,
                torque_scale=torque_scale,
                timescale=timescale,
                duration=duration,
                seed=seed,
                render=False,
            )
            displacement[i, r] = dist
            fell[i, r] = did_fall
            print(
                f"timescale={timescale:>6.3f}  rep={r}  "
                f"displacement={dist:.4f}  {'fell' if did_fall else 'upright'}"
            )

    return displacement, fell


def main():
    args = parse_args()
    timescales = sorted(args.timescales)

    displacement, fell = sweep(
        genome_path=args.genome,
        topology=args.topology,
        module_size=args.module_size,
        n_interneurons=args.interneurons,
        mode=args.mode,
        torque_scale=args.torque_scale,
        duration=args.duration,
        timescales=timescales,
        reps=args.reps,
        base_seed=args.seed,
    )

    mean_d = displacement.mean(axis=1)
    std_d = displacement.std(axis=1)
    fell_rate = fell.mean(axis=1)

    print("\nSummary (average displacement from start over "
          f"{args.reps} reps, +/- 1 std):")
    for ts, m, s, fr in zip(timescales, mean_d, std_d, fell_rate):
        print(f"  timescale={ts:>6.3f}: {m:.4f} +/- {s:.4f}  (fell {fr*100:.0f}% of reps)")

    best_idx = int(np.argmax(mean_d))
    print(
        f"\nBest timescale: {timescales[best_idx]:.3f} "
        f"(avg displacement {mean_d[best_idx]:.4f})"
    )

    if args.data_out:
        np.savez(
            args.data_out,
            timescales=np.array(timescales),
            displacement=displacement,
            fell=fell,
        )
        print(f"Raw per-rep data saved to: {args.data_out}")

    plt.figure(figsize=(8, 5))
    plt.errorbar(timescales, mean_d, yerr=std_d, marker="o", capsize=4, color="tab:green")
    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.axvline(1.0, color="lightgray", linewidth=1, linestyle=":")
    plt.xlabel("timescale (CTRNN internal dt = TS * timescale)")
    plt.ylabel("Displacement from start, direction-agnostic (avg over reps)")
    plt.title(f"Timescale sweep (ctrnn controller, torque_scale={args.torque_scale}, {args.genome})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Plot saved to: {args.out}")


if __name__ == "__main__":
    main()
