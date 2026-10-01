"""
Evolve soft robots in Evolution Gym with EvoTorch's GeneticAlgorithm (SBX
crossover + Gaussian mutation + elitism), parallelized across CPU cores via Ray.

Three things can be evolved, selected with `--mode`:

    control     The body is fixed (--body); evolve only its controller
                (--controller mlp or oscillator).
    morphology  The controller is fixed (a traveling sine wave); evolve only
                the body, on a --grid x --grid voxel grid.
    codesign    Evolve the body AND an oscillator controller together, in a
                single genome.

The EA itself is identical in all three modes. It only ever sees a flat
vector of real numbers -- what changes is how that vector is decoded into a
robot (see `decode_robot()`), and so how long it is (`genome_length()`).

Genome layouts (G = grid*grid voxels, A = actuators in the fixed body):

    control / mlp          all network weights and biases
    control / oscillator   A phases
    morphology             G*5 material genes (each voxel: argmax of its five)
    codesign               G*5 material genes, then G phases (one per voxel;
                           only voxels that end up as actuators use theirs)
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
    N_MATERIALS, INVALID_FITNESS, DEFAULT_ENV, PRESET_BODIES,
    load_body, is_valid_body, actuator_indices, decode_body, make_env, run_episode,
    body_to_string,
)
from neural_controller import (
    NeuralController, OscillatorController, traveling_wave_phases, DEFAULT_PERIOD,
)

# Silence start-up noise from Ray and EvoTorch that's about their own
# internals, not this project (same as Project 6).
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
        "controller": args.controller if args.mode == "control" else "oscillator",
        "body": load_body(args.body).tolist() if args.mode == "control" else None,
        "grid": args.grid,
        "env": args.env,
        "duration": args.duration,
        "hidden": list(args.hidden),
        "activation": args.activation,
        "period": args.period,
    }


def _mlp_sizes(cfg, body):
    """(obs_size, n_actuators) for an MLP on `body` in cfg's task."""
    env = make_env(body, cfg["env"], cfg["duration"])
    sizes = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    return sizes


def genome_length(cfg):
    n_voxels = cfg["grid"] ** 2
    if cfg["mode"] == "morphology":
        return n_voxels * N_MATERIALS
    if cfg["mode"] == "codesign":
        return n_voxels * N_MATERIALS + n_voxels
    body = np.array(cfg["body"])
    if cfg["controller"] == "oscillator":
        return len(actuator_indices(body))
    obs_size, n_act = _mlp_sizes(cfg, body)
    return NeuralController.genome_size(obs_size, n_act, cfg["hidden"])


def decode_body_from_genome(genome, cfg):
    """The body this genome specifies (the fixed body, in control mode)."""
    if cfg["mode"] == "control":
        return np.array(cfg["body"])
    n_material_genes = cfg["grid"] ** 2 * N_MATERIALS
    return decode_body(genome[:n_material_genes], cfg["grid"])


def decode_controller(genome, cfg, body, env):
    """The controller this genome specifies, for an already-decoded body."""
    period = cfg["period"]
    if cfg["mode"] == "morphology":
        return OscillatorController(traveling_wave_phases(body), period)
    if cfg["mode"] == "codesign":
        n_material_genes = cfg["grid"] ** 2 * N_MATERIALS
        voxel_phases = OscillatorController.genes_to_phases(genome[n_material_genes:])
        return OscillatorController(voxel_phases[actuator_indices(body)], period)
    if cfg["controller"] == "oscillator":
        return OscillatorController(OscillatorController.genes_to_phases(genome), period)
    net = NeuralController(env.observation_space.shape[0], env.action_space.shape[0],
                           cfg["hidden"], cfg["activation"])
    return net.load_genome(genome)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def make_fitness_fn(cfg):
    """Fitness = total reward from one episode of cfg['env'].

    EvoGym's physics is deterministic and every episode starts from the same
    state, so one episode per evaluation is enough -- a second one would
    return exactly the same number. (Contrast with CartPole, whose random
    initial state is why Project 5 averages several.)

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
    parser.add_argument("--mode", choices=["control", "morphology", "codesign"], default="control",
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
    parser.add_argument("--controller", choices=["mlp", "oscillator"], default="mlp",
                        help="[control only] mlp = closed-loop feedforward network; oscillator = "
                             "open-loop sine wave, one evolved phase per actuator (default: mlp). "
                             "morphology/codesign always use an oscillator.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[16],
                        help="[mlp only] Hidden layer size(s), e.g. --hidden 32 or --hidden 16 16")
    parser.add_argument("--activation", choices=["tanh", "relu", "sigmoid"], default="tanh",
                        help="[mlp only] Hidden-layer activation (output is always Tanh, rescaled)")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD,
                        help=f"[oscillator only] Steps per oscillation cycle (default: {DEFAULT_PERIOD})")

    # -- task --
    parser.add_argument("--env", type=str, default=DEFAULT_ENV,
                        help=f"EvoGym task (default: {DEFAULT_ENV}). See README for others.")
    parser.add_argument("--duration", type=int, default=None,
                        help="Episode length in simulation steps (default: the task's own, 500 for Walker-v0)")

    # -- EA --
    parser.add_argument("--popsize", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--gens", type=int, default=50, help="Number of generations (default: 50)")
    parser.add_argument("--mut_stdev", type=float, default=None,
                        help="Gaussian mutation standard deviation (default: 0.1 for mlp, 0.2 otherwise)")
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
    return parser.parse_args()


def config_path_for(genome_path):
    return os.path.splitext(genome_path)[0] + ".json"


def main():
    args = parse_args()
    cfg = make_config(args)
    if args.mut_stdev is None:
        args.mut_stdev = 0.1 if cfg["controller"] == "mlp" else 0.2

    n_genes = genome_length(cfg)
    what = {"control": f"{cfg['controller']} controller for fixed body '{args.body}'",
            "morphology": f"{args.grid}x{args.grid} body under a fixed traveling-wave controller",
            "codesign": f"{args.grid}x{args.grid} body + oscillator controller together"}[args.mode]
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
