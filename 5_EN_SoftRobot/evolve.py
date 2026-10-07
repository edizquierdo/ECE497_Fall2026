"""
Evolve soft robots in Evolution Gym with EvoTorch's GeneticAlgorithm (SBX
crossover + Gaussian mutation + elitism), parallelized across CPU cores via Ray.

Three things can be evolved, selected with `--mode`:

    control     The body is fixed (--body); evolve only its brain.
    codesign    Evolve the body AND its brain together, in a single genome,
                on a --grid x --grid voxel grid.
    morphology  The brain is fixed (a traveling sine wave); evolve only the
                body, on a --grid x --grid voxel grid.

The brain is chosen with --controller local|global and --loop open|closed
(see neural_controller.py). All four combinations fit any body on the grid,
so every one of them works in every mode.

The EA itself is identical in all modes. It only ever sees a flat vector of
real numbers -- what changes is how that vector is decoded into a robot
(`decode_body_from_genome()`, `decode_controller()`), and so how long it is
(`genome_length()`).

Genome layouts (G = grid*grid voxels):

    control      the brain's network weights and biases
    codesign     G*5 material genes (each voxel: argmax of its five), then
                 the brain's network weights and biases
    morphology   G*5 material genes
"""

import os
import json
import warnings
import logging
import argparse

# Ray worker processes inherit this, so they skip the same deprecation
# warnings (from torch and EvoGym's dependencies) the main process does.
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation

from soft_robot import (
    N_MATERIALS, INVALID_FITNESS, DEFAULT_ENV, DEFAULT_DURATION, PRESET_BODIES,
    load_body, is_valid_body, decode_body, make_env, run_episode,
    body_to_string,
)
from neural_controller import (
    CONTROLLERS, DEFAULT_HIDDEN, TravelingWaveController, DEFAULT_PERIOD,
)

# Silence start-up noise from Ray and EvoTorch that's about their own
# internals, not this project.
logging.getLogger("evotorch").setLevel(logging.WARNING)
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
logging.getLogger("ray").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Genome <-> robot
# ---------------------------------------------------------------------------
# A "config" is a plain dict holding every setting needed to turn a genome
# back into the same robot. evolve.py saves it next to the genome (as JSON)
# so sim.py can rebuild the robot without you re-typing matching flags.

def make_config(args):
    return {
        "mode": args.mode,
        "controller": args.controller,
        "loop": args.loop,
        "body": load_body(args.body).tolist() if args.mode == "control" else None,
        "grid": args.grid,
        "env": args.env,
        "duration": args.duration,
        "hidden": list(args.hidden),
        "activation": args.activation,
        "period": args.period,
    }


def _body_shape(cfg):
    """Shape of the grid the brain has to cover: the fixed body's, or the evolved grid's."""
    if cfg["mode"] == "control":
        return np.array(cfg["body"]).shape
    return (cfg["grid"], cfg["grid"])


def _n_material_genes(cfg):
    return 0 if cfg["mode"] == "control" else cfg["grid"] ** 2 * N_MATERIALS


def genome_length(cfg):
    if cfg["mode"] == "morphology":
        return _n_material_genes(cfg)
    brain = CONTROLLERS[cfg["controller"]].genome_size(
        _body_shape(cfg), cfg["loop"] == "closed", cfg["hidden"])
    return _n_material_genes(cfg) + brain


def decode_body_from_genome(genome, cfg):
    """The body this genome specifies (the fixed body, in control mode)."""
    if cfg["mode"] == "control":
        return np.array(cfg["body"])
    return decode_body(genome[:_n_material_genes(cfg)], cfg["grid"])


def decode_controller(genome, cfg, body, env):
    """The brain this genome specifies, for an already-decoded body."""
    if cfg["mode"] == "morphology":
        return TravelingWaveController(body, cfg["period"])
    return CONTROLLERS[cfg["controller"]](
        genome[_n_material_genes(cfg):], body, env, closed_loop=cfg["loop"] == "closed",
        hidden_sizes=cfg["hidden"], activation=cfg["activation"], period=cfg["period"])


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def make_fitness_fn(cfg):
    """Fitness = total reward from one episode of cfg['env'].

    EvoGym's physics is deterministic and every episode starts from the same
    state, so one episode per evaluation is enough -- a second one would
    return exactly the same number. (Contrast with Project 4, where random
    starting positions and motion noise made fitness noisy, so it averaged
    several episodes.)

    For Walker-v0 the reward is essentially how far the robot's center of
    mass moved to the right, so fitness ~ distance traveled.
    """

    def fitness_fn(genome: torch.Tensor) -> float:
        genome = genome.detach().cpu().numpy()
        body = decode_body_from_genome(genome, cfg)
        if not is_valid_body(body):
            return INVALID_FITNESS
        env = make_env(body, cfg["env"], cfg["duration"])
        controller = decode_controller(genome, cfg, body, env)
        reward = run_episode(env, controller)
        env.close()
        # EvoGym already ends an unstable simulation with a -3 penalty; this
        # only guards against a NaN ever reaching the GA's selection step.
        return reward if np.isfinite(reward) else INVALID_FITNESS

    return fitness_fn


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

def _parse_workers(workers):
    workers = str(workers).strip().lower()
    if workers == "none":
        return None
    if workers == "max":
        return "max"
    return int(workers)


def run_evolution(cfg, popsize=50, gens=50, mut_stdev=0.1, tournament_size=3, eta=20,
                  elitism=True, workers="max", seed=None, seed_genome_path=None,
                  seed_noise=0.05, verbose=False):
    """Run the GA and return (best_fit, avg_fit, worst_fit, best_genome, best_fitness)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_genes = genome_length(cfg)
    if _parse_workers(workers) is not None:
        # Start Ray ourselves (EvoTorch reuses a running instance) so its
        # start-up banner and the workers' own console output stay quiet.
        # Errors inside a worker are still raised here in the main process.
        import ray
        if not ray.is_initialized():
            ray.init(logging_level=logging.ERROR, log_to_driver=False)
    problem = Problem(
        objective_sense="max",
        objective_func=make_fitness_fn(cfg),
        solution_length=n_genes,
        initial_bounds=(-1.0, 1.0),
        dtype=torch.float32,
        num_actors=_parse_workers(workers),
    )
    ga = GeneticAlgorithm(
        problem,
        popsize=popsize,
        operators=[
            SimulatedBinaryCrossOver(problem, tournament_size=tournament_size, eta=eta),
            GaussianMutation(problem, stdev=mut_stdev),
        ],
        elitist=elitism,
    )

    if seed_genome_path is not None:
        seed_genome = torch.tensor(np.load(seed_genome_path), dtype=torch.float32)
        if seed_genome.numel() != n_genes:
            raise ValueError(
                f"Seed genome has {seed_genome.numel()} values, but these settings need {n_genes}."
            )
        values = ga.population.access_values(keep_evals=False)
        values[0] = seed_genome
        values[1:] = seed_genome.unsqueeze(0) + torch.randn((popsize - 1, n_genes)) * seed_noise

    best_fit, avg_fit, worst_fit = (np.zeros(gens) for _ in range(3))
    for gen in range(gens):
        ga.step()
        fitnesses = ga.population.access_evals()[:, 0]
        best_fit[gen] = fitnesses.max().item()
        avg_fit[gen] = fitnesses.mean().item()
        worst_fit[gen] = fitnesses.min().item()
        if verbose:
            print(f"Gen {gen:>4d} | Best: {best_fit[gen]:7.3f} | Mean: {avg_fit[gen]:7.3f} "
                  f"| Worst: {worst_fit[gen]:7.3f}")

    # Read the fitness BEFORE touching the genome: access_values() marks the
    # population's stored fitnesses as stale (NaN), since it hands you a
    # writable view of the genomes.
    fitnesses = ga.population.access_evals()[:, 0]
    best_idx = fitnesses.argmax().item()
    best_fitness = fitnesses[best_idx].item()
    best_genome = ga.population.values[best_idx].clone()
    return best_fit, avg_fit, worst_fit, best_genome, best_fitness


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evolve soft robots (body, controller, or both) in EvoGym.")
    parser.add_argument("--mode", choices=["control", "codesign", "morphology"], default="control",
                        help="What to evolve: the controller of a fixed body, the body under a fixed "
                             "controller, or both together (default: control)")

    # -- body --
    parser.add_argument("--body", type=str, default="biped",
                        help=f"[control only] Fixed body: a preset ({', '.join(PRESET_BODIES)}) or "
                             "a path to a .txt/.npy grid of material codes (default: biped)")
    parser.add_argument("--grid", type=int, default=5,
                        help="[morphology/codesign only] Side length of the voxel grid bodies are "
                             "evolved on (default: 5)")

    # -- controller --
    parser.add_argument("--controller", choices=["local", "global"], default="local",
                        help="local = one small network copied into every muscle; global = one network "
                             "for the whole body, with one output per grid cell (default: local). "
                             "Ignored in morphology mode, which uses a fixed traveling wave.")
    parser.add_argument("--loop", choices=["open", "closed"], default="closed",
                        help="open = the brain only sees a clock (and, if local, its position); closed = "
                             "it also senses how stretched the voxels are (default: closed)")
    parser.add_argument("--hidden", type=int, nargs="+", default=None,
                        help="Hidden layer size(s), e.g. --hidden 32 or --hidden 16 16 "
                             "(default: 8 for local, 16 for global)")
    parser.add_argument("--activation", choices=["tanh", "relu", "sigmoid"], default="tanh",
                        help="Hidden-layer activation (output is always Tanh, rescaled)")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD,
                        help=f"Steps per clock cycle (default: {DEFAULT_PERIOD})")

    # -- task --
    parser.add_argument("--env", type=str, default=DEFAULT_ENV,
                        help=f"EvoGym task (default: {DEFAULT_ENV}). See README for others.")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help=f"Episode length in simulation steps (default: {DEFAULT_DURATION})")

    # -- EA --
    parser.add_argument("--popsize", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--gens", type=int, default=50, help="Number of generations (default: 50)")
    parser.add_argument("--mut_stdev", type=float, default=0.2,
                        help="Gaussian mutation standard deviation (default: 0.2)")
    parser.add_argument("--tournament_size", type=int, default=3, help="Tournament size for SBX crossover")
    parser.add_argument("--eta", type=float, default=20, help="Distribution index for SBX crossover")
    parser.add_argument("--no-elitism", action="store_false", dest="elitism",
                        help="Disable elitism (best individual always survives by default)")
    parser.add_argument("--seed_genome", type=str, default=None,
                        help="Seed the initial population around a genome saved by a previous run "
                             "(must have been evolved with the same settings)")
    parser.add_argument("--seed_noise", type=float, default=0.05,
                        help="Stdev of the noise added to --seed_genome copies")
    parser.add_argument("--workers", type=str, default="max",
                        help="Parallel workers: 'max' (all CPU cores), an integer, or 'none'")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # -- output --
    parser.add_argument("--verbose", action="store_true", help="Print per-generation statistics")
    parser.add_argument("--vizperf", action="store_true", help="Plot fitness over generations")
    parser.add_argument("--output", type=str, default=None,
                        help="Save the best genome here (e.g. best.npy). Its settings are saved "
                             "alongside as best.json, which sim.py reads automatically.")
    parser.add_argument("--fitness_output", type=str, default=None,
                        help="Save per-generation fitness to this .npz file (keys: 'best', 'avg', "
                             "'worst')")
    args = parser.parse_args()
    if args.hidden is None:
        args.hidden = DEFAULT_HIDDEN[args.controller]
    return args


def config_path_for(genome_path):
    return os.path.splitext(genome_path)[0] + ".json"


def main():
    args = parse_args()
    cfg = make_config(args)

    n_genes = genome_length(cfg)
    brain = f"{cfg['loop']}-loop {cfg['controller']} brain"
    what = {"control": f"{brain} for fixed body '{args.body}'",
            "morphology": f"{args.grid}x{args.grid} body under a fixed traveling-wave brain",
            "codesign": f"{args.grid}x{args.grid} body + {brain} together"}[args.mode]
    print(f"Evolving {what} on {args.env}: {n_genes} genes, popsize={args.popsize}, "
          f"gens={args.gens}, mut_stdev={args.mut_stdev}")
    if args.mode == "control":
        print(body_to_string(cfg["body"]))

    best_fit, avg_fit, worst_fit, best_genome, best_fitness = run_evolution(
        cfg,
        popsize=args.popsize,
        gens=args.gens,
        mut_stdev=args.mut_stdev,
        tournament_size=args.tournament_size,
        eta=args.eta,
        elitism=args.elitism,
        workers=args.workers,
        seed=args.seed,
        seed_genome_path=args.seed_genome,
        seed_noise=args.seed_noise,
        verbose=args.verbose,
    )

    print(f"\nBest fitness achieved: {best_fitness:.3f}")
    if args.mode != "control":
        print("Best body (R=rigid, S=soft, H=horizontal actuator, V=vertical actuator, .=empty):")
        print(body_to_string(decode_body_from_genome(best_genome.numpy(), cfg)))

    if args.output:
        np.save(args.output, best_genome.numpy())
        with open(config_path_for(args.output), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Best genome saved to: {args.output} (settings in {config_path_for(args.output)})")

    if args.fitness_output:
        np.savez(args.fitness_output, best=best_fit, avg=avg_fit, worst=worst_fit)
        print(f"Fitness-over-generations saved to: {args.fitness_output}")

    if args.vizperf:
        gens = np.arange(len(best_fit))
        plt.figure(figsize=(10, 6))
        plt.plot(gens, best_fit, label="Best Fitness", color="green", linewidth=2)
        plt.plot(gens, avg_fit, label="Average Fitness", color="blue", linewidth=2)
        plt.plot(gens, worst_fit, label="Worst Fitness", color="red", linewidth=1, linestyle="--")
        plt.xlabel("Generation")
        plt.ylabel(f"Fitness (total reward on {args.env})")
        plt.title(f"Soft-Robot Evolution: {args.mode}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
