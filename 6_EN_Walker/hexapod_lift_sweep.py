"""
Sweep hexapod_bridge.py's --lift_gain and --timescale together for a given
evolved genome, running several repetitions per (lift_gain, timescale)
cell, and plot average displacement (direction-agnostic straight-line
distance from the start position -- see hexapod_bridge.py's
run_transfer() docstring for why raw x-axis progress isn't a fair metric
on the real MuJoCo body) as a heatmap over the two parameters.

Both parameters matter for the same underlying reason: the lift joint's
closed-loop PD controller needs enough gain to actually reach its target
angle within the time each gait phase allows, and that gait's timing
(relative to how fast the real body physically responds) is set by
`timescale`. Sweeping them together, rather than one at a time, is what
lets you see whether they trade off against each other or whether one
setting works well across a range of the other.

`--torque_scale` (sweep torque only -- see hexapod_bridge.py) is held
fixed across this sweep; if you want to search over that too, pass a
different fixed value and re-run, or extend this script yourself.

Repetitions per cell use different seeds (both the hexapod's small
reset-noise perturbation and, for a CTRNN controller, its randomized
initial neuron state -- see hexapod_bridge.py's run_transfer() -- are the
sources of run-to-run variability at a fixed setting), so a single run at
a given (lift_gain, timescale) isn't representative on its own -- and
with real physical noise, a handful of reps may still not be, either; the
more you can afford, the more you should trust the result.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hexapod_bridge import run_transfer
from neural_controller import SM_DEFAULT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep hexapod_bridge.py's --lift_gain and --timescale together."
    )
    parser.add_argument("--controller", choices=["feedforward", "ctrnn"], default="feedforward",
                         help="Same meaning as hexapod_bridge.py -- must match how --genome was evolved.")
    parser.add_argument("--genome", type=str, required=True,
                         help="Genome evolved by evolve.py against walker.py.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64],
                         help="[feedforward only] must match what --genome was evolved with")
    parser.add_argument("--topology", choices=["modular", "fully_connected"], default="modular",
                         help="[ctrnn only] must match what --genome was evolved with")
    parser.add_argument("--module_size", type=int, default=SM_DEFAULT,
                         help="[ctrnn only] must match what --genome was evolved with")
    parser.add_argument("--interneurons", type=int, default=0,
                         help="[ctrnn only] must match what --genome was evolved with")
    parser.add_argument("--mode", choices=["cpg", "rpg", "mpg"], default="rpg",
                         help="[ctrnn only] sensor condition to run under: cpg = no sensory "
                              "input, rpg = live leg-angle feedback, mpg = alternate by rep")
    parser.add_argument("--torque_scale", type=float, default=1.0,
                         help="Held fixed across the sweep -- scales sweep torque only "
                              "(see hexapod_bridge.py's --torque_scale)")
    parser.add_argument("--duration", type=float, default=50.0,
                         help="Simulated seconds per run (see hexapod_bridge.py's --duration)")
    parser.add_argument("--lift_gains", type=float, nargs="+",
                         default=[1.0, 1.5, 2.0, 2.5, 3.0],
                         help="lift_gain values to try")
    parser.add_argument("--timescales", type=float, nargs="+",
                         default=[1.0, 2.0, 4.0, 8.0, 16.0],
                         help="timescale values to try (ctrnn only -- ignored, held at 1.0, "
                              "for --controller feedforward)")
    parser.add_argument("--reps", type=int, default=5,
                         help="Repetitions per (lift_gain, timescale) cell, each a different seed")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed -- repetition r uses seed + r, the same across every "
                              "cell, so the comparison is fair")
    parser.add_argument("--out", type=str, default="lift_sweep.png", help="Output plot path")
    parser.add_argument("--data_out", type=str, default=None,
                         help="Optional .npz path to also save raw per-cell/per-rep results "
                              "(keys: 'lift_gains', 'timescales', 'displacement', 'fell')")
    return parser.parse_args()


def run_sweep(
    controller, genome_path, hidden_sizes, topology, module_size, n_interneurons, mode,
    torque_scale, duration, lift_gains, timescales, reps, base_seed,
):
    """Returns:
        displacement: (len(timescales), len(lift_gains), reps) total straight-line distance
        fell:         (len(timescales), len(lift_gains), reps) bool, whether it fell
    """
    n_ts, n_lg = len(timescales), len(lift_gains)
    displacement = np.zeros((n_ts, n_lg, reps))
    fell = np.zeros((n_ts, n_lg, reps), dtype=bool)

    for i, timescale in enumerate(timescales):
        for j, lift_gain in enumerate(lift_gains):
            for r in range(reps):
                seed = base_seed + r
                dist, did_fall, _ = run_transfer(
                    controller=controller,
                    genome_path=genome_path,
                    hidden_sizes=hidden_sizes,
                    topology=topology,
                    module_size=module_size,
                    n_interneurons=n_interneurons,
                    mode=mode,
                    torque_scale=torque_scale,
                    timescale=timescale,
                    lift_gain=lift_gain,
                    duration=duration,
                    seed=seed,
                )
                displacement[i, j, r] = dist
                fell[i, j, r] = did_fall
            mean_d = displacement[i, j].mean()
            fell_pct = 100 * fell[i, j].mean()
            print(f"timescale={timescale:>6.3f}  lift_gain={lift_gain:>5.2f}  "
                  f"mean_displacement={mean_d:.4f}  fell={fell_pct:.0f}%")

    return displacement, fell


def main():
    args = parse_args()
    hidden_sizes = args.hidden[0] if len(args.hidden) == 1 else tuple(args.hidden)
    timescales = sorted(args.timescales) if args.controller == "ctrnn" else [1.0]
    lift_gains = sorted(args.lift_gains)

    displacement, fell = run_sweep(
        controller=args.controller,
        genome_path=args.genome,
        hidden_sizes=hidden_sizes,
        topology=args.topology,
        module_size=args.module_size,
        n_interneurons=args.interneurons,
        mode=args.mode,
        torque_scale=args.torque_scale,
        duration=args.duration,
        lift_gains=lift_gains,
        timescales=timescales,
        reps=args.reps,
        base_seed=args.seed,
    )

    mean_grid = displacement.mean(axis=2)  # (n_ts, n_lg)
    fell_grid = fell.mean(axis=2)

    best_i, best_j = np.unravel_index(np.argmax(mean_grid), mean_grid.shape)
    print(f"\nBest: timescale={timescales[best_i]}, lift_gain={lift_gains[best_j]} "
          f"(mean displacement {mean_grid[best_i, best_j]:.4f}, "
          f"fell {100 * fell_grid[best_i, best_j]:.0f}% of reps)")

    if args.data_out:
        Path(args.data_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.data_out, lift_gains=lift_gains, timescales=timescales,
                  displacement=displacement, fell=fell)
        print(f"Raw per-cell/per-rep data saved to: {args.data_out}")

    fig, ax = plt.subplots(figsize=(1.4 * len(lift_gains) + 2, 1.1 * len(timescales) + 2))
    im = ax.imshow(mean_grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(lift_gains)))
    ax.set_xticklabels(lift_gains)
    ax.set_yticks(range(len(timescales)))
    ax.set_yticklabels(timescales)
    ax.set_xlabel("lift_gain")
    ax.set_ylabel("timescale")
    vmax = mean_grid.max()
    for i in range(len(timescales)):
        for j in range(len(lift_gains)):
            ax.text(j, i, f"{mean_grid[i, j]:.2f}", ha="center", va="center",
                     color="white" if mean_grid[i, j] < vmax * 0.6 else "black", fontsize=9)
    ax.set_title(f"Displacement vs. lift_gain x timescale ({args.controller}, "
                 f"torque_scale={args.torque_scale}, {args.reps} reps/cell)")
    fig.colorbar(im, ax=ax, label="mean displacement")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Plot saved to: {args.out}")


if __name__ == "__main__":
    main()
