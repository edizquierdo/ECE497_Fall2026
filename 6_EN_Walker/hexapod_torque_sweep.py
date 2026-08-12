"""
Sweep hexapod_bridge.py's --torque_scale for a given evolved genome,
running several repetitions per value, and plot average displacement
(direction-agnostic straight-line distance from the start position -- see
hexapod_bridge.py's run_transfer() docstring for why raw x-axis progress
isn't a fair metric on the real MuJoCo body) as a function of torque scale.

`torque_scale` is currently the *only* CLI-exposed parameter of
`walker_action_to_hexapod_action()` (see hexapod_bridge.py) -- it scales
both the sweep and lift torques uniformly. The lift-torque magnitudes
themselves (-0.3 for planted, 0.5 for swinging) are hardcoded constants,
not exposed here; sweeping those, or decoupling the sweep/lift scale, is
exactly the kind of "propose and implement a concrete change" Part 4.3
asks for -- this script only sweeps what's already tunable without
modifying hexapod_bridge.py.

Repetitions per torque value use different seeds (both the hexapod's
small reset-noise perturbation and, for a CTRNN controller, its randomized
initial neuron state -- see hexapod_bridge.py's run_transfer() -- are the
sources of run-to-run variability at a fixed torque scale), so a single
run at a given torque isn't representative on its own.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from hexapod_bridge import run_transfer
from neural_controller import SM_DEFAULT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep --torque_scale for a walker.py-evolved genome on the real hexapod."
    )
    parser.add_argument("--controller", choices=["feedforward", "ctrnn"], default="feedforward",
                         help="Same meaning as evolve.py/sim.py -- must match how --genome was evolved.")
    parser.add_argument("--genome", type=str, required=True,
                         help="Genome evolved by evolve.py against walker.py (required -- a "
                              "random controller's torque response isn't meaningful to sweep).")
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
    parser.add_argument("--duration", type=float, default=50.0,
                         help="Simulated seconds per repetition (same convention as "
                              "hexapod_bridge.py's --duration)")
    parser.add_argument("--torque_scales", type=float, nargs="+",
                         default=[0.25, 0.5, 1.0, 2.0, 4.0],
                         help="Torque-scale values to try (default spans a 16x range)")
    parser.add_argument("--reps", type=int, default=5,
                         help="Repetitions per torque-scale value, each with a different seed")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed; repetition r at a given torque scale uses seed + r, "
                              "so every torque scale is tested against the *same* set of "
                              "reps/seeds for a fair comparison")
    parser.add_argument("--out", type=str, default="torque_sweep.png",
                         help="Output plot path")
    parser.add_argument("--data_out", type=str, default=None,
                         help="Optional .npz path to also save the raw per-rep results "
                              "(keys: 'torque_scales', 'displacement', 'fell')")
    return parser.parse_args()


def sweep(
    controller, genome_path, hidden_sizes, topology, module_size, n_interneurons, mode,
    duration, torque_scales, reps, base_seed,
):
    """Run reps repetitions at each torque scale. Returns:
        displacement: (len(torque_scales), reps) total straight-line distance
                      from the start position per run (direction-agnostic --
                      see hexapod_bridge.py's run_transfer() docstring)
        fell:         (len(torque_scales), reps) bool, whether it fell
    """
    n_scales = len(torque_scales)
    displacement = np.zeros((n_scales, reps))
    fell = np.zeros((n_scales, reps), dtype=bool)

    for i, torque_scale in enumerate(torque_scales):
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
                duration=duration,
                seed=seed,
                render=False,
            )
            displacement[i, r] = dist
            fell[i, r] = did_fall
            print(
                f"torque_scale={torque_scale:>6.3f}  rep={r}  "
                f"displacement={dist:.4f}  {'fell' if did_fall else 'upright'}"
            )

    return displacement, fell


def main():
    args = parse_args()
    hidden_sizes = args.hidden[0] if len(args.hidden) == 1 else tuple(args.hidden)
    torque_scales = sorted(args.torque_scales)

    displacement, fell = sweep(
        controller=args.controller,
        genome_path=args.genome,
        hidden_sizes=hidden_sizes,
        topology=args.topology,
        module_size=args.module_size,
        n_interneurons=args.interneurons,
        mode=args.mode,
        duration=args.duration,
        torque_scales=torque_scales,
        reps=args.reps,
        base_seed=args.seed,
    )

    mean_d = displacement.mean(axis=1)
    std_d = displacement.std(axis=1)
    fell_rate = fell.mean(axis=1)

    print("\nSummary (average displacement from start over "
          f"{args.reps} reps, +/- 1 std):")
    for ts, m, s, fr in zip(torque_scales, mean_d, std_d, fell_rate):
        print(f"  torque_scale={ts:>6.3f}: {m:.4f} +/- {s:.4f}  (fell {fr*100:.0f}% of reps)")

    best_idx = int(np.argmax(mean_d))
    print(
        f"\nBest torque_scale: {torque_scales[best_idx]:.3f} "
        f"(avg displacement {mean_d[best_idx]:.4f})"
    )

    if args.data_out:
        np.savez(
            args.data_out,
            torque_scales=np.array(torque_scales),
            displacement=displacement,
            fell=fell,
        )
        print(f"Raw per-rep data saved to: {args.data_out}")

    plt.figure(figsize=(8, 5))
    plt.errorbar(torque_scales, mean_d, yerr=std_d, marker="o", capsize=4, color="tab:blue")
    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("torque_scale")
    plt.ylabel("Displacement from start, direction-agnostic (avg over reps)")
    plt.title(f"Torque-scale sweep ({args.controller} controller, {args.genome})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Plot saved to: {args.out}")


if __name__ == "__main__":
    main()
