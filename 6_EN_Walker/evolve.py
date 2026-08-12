"""
Evolutionary algorithm for evolving a controller -- feedforward network or
CTRNN, selected with `--controller` -- on the six-legged walker (WalkerEnv).
Uses EvoTorch's GeneticAlgorithm with SBX crossover and Gaussian mutation,
parallelized across CPU cores via Ray.

The EA mechanics (selection, crossover, mutation, generation loop) don't
care what kind of controller they're evolving -- only three things below
branch on `--controller`: genome length, how a genome is decoded into a
controller, and the inner per-episode simulation loop (a CTRNN needs
`reset_state()` once per episode and runs with sensors on/off; a
feedforward net is a pure function of the current observation and needs
neither).
"""

import os
import warnings
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm, CMAES
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation

from neural_controller import (
    WalkerController, WalkerCTRNNController, SM_DEFAULT, sample_center_crossing_genome,
)
from walker import WalkerEnv, OBS_SIZE

logging.getLogger("evotorch").setLevel(logging.WARNING)

# EvoTorch parallelizes fitness evaluation across CPU cores using Ray, and
# Ray/PyTorch print a few startup messages and warnings that are about their
# own internals (worker-process bookkeeping, how tensors move between
# processes), not about anything in this project. They're silenced here so
# the output you actually care about -- generation stats and any real
# errors -- isn't buried:
#   - A "Started a local Ray instance" INFO line, printed once whenever
#     Ray's worker pool spins up (i.e. whenever --workers isn't "none").
#   - A FutureWarning about Ray overriding accelerator-visibility env vars
#     when num_gpus=0 (this project always runs on CPU, so it doesn't apply).
#   - A FutureWarning from torch.load(weights_only=False), which Ray uses
#     internally to move genome tensors between worker processes -- this is
#     not the "loading an untrusted file" scenario that warning is about.
#   - A FutureWarning that Ray's own internals use Pydantic v1, which a
#     future Ray release will require upgrading to v2 for. This is about
#     Ray's dependencies, not this project's code -- upgrading Pydantic
#     yourself isn't necessary unless/until you also upgrade Ray and hit an
#     actual incompatibility.
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
logging.getLogger("ray").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*accelerator visible devices.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Pydantic v1 is deprecated.*", category=FutureWarning)


def compute_genome_length(
    controller="feedforward", hidden_sizes=64, obs_size=OBS_SIZE,
    topology="modular", module_size=SM_DEFAULT, n_interneurons=0,
) -> int:
    if controller == "feedforward":
        return WalkerController.genome_size(hidden_sizes=hidden_sizes, obs_size=obs_size)
    return WalkerCTRNNController.genome_size(
        topology=topology, module_size=module_size, n_interneurons=n_interneurons
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve a controller (feedforward net or CTRNN) for the six-legged walker."
    )
    parser.add_argument("--controller", choices=["feedforward", "ctrnn"], default="feedforward",
                         help="Controller architecture to evolve (default: feedforward)")

    # -- feedforward-only --
    parser.add_argument("--hidden", type=int, nargs="+", default=[64],
                         help="[feedforward only] Hidden layer size(s). Give one value for a "
                              "single hidden layer, or several for a deeper network, "
                              "e.g. --hidden 64 64")

    # -- ctrnn-only --
    parser.add_argument("--topology", choices=["modular", "fully_connected"], default="modular",
                         help="[ctrnn only] modular = one small circuit duplicated per leg "
                              "(weight-sharing); fully_connected = one big network, no weight-sharing")
    parser.add_argument("--module_size", type=int, default=SM_DEFAULT,
                         help="[ctrnn only] Neurons per leg module, modular topology only "
                              f"(default: {SM_DEFAULT}, minimum: 3)")
    parser.add_argument("--interneurons", type=int, default=0,
                         help="[ctrnn only] Extra free neurons beyond the 18 command neurons, "
                              "fully_connected topology only (default: 0)")
    parser.add_argument("--mode", choices=["cpg", "rpg", "mpg"], default="rpg",
                         help="[ctrnn only] cpg = no sensory input into the CTRNN; "
                              "rpg = live leg-angle feedback into the CTRNN every tick; "
                              "mpg = fitness averaged over separate cpg and rpg episodes")

    # -- shared EA / environment settings --
    parser.add_argument("--algorithm", choices=["ga", "es"], default="ga",
                         help="'ga' = GeneticAlgorithm (SBX crossover + Gaussian mutation, "
                              "the default used elsewhere in this course); 'es' = CMA-ES "
                              "(evotorch's CMAES), an evolution-strategies alternative with no "
                              "explicit crossover/mutation operators of its own -- --mut_stdev "
                              "is reused as its initial step size (stdev_init).")
    parser.add_argument("--popsize", type=int, default=50, help="Population size")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--episodes_per_eval", type=int, default=3,
                         help="Episodes to average per fitness evaluation")
    parser.add_argument("--mut_stdev", type=float, default=0.05,
                         help="--algorithm ga: Gaussian mutation standard deviation. "
                              "--algorithm es: reused as CMA-ES's initial step size "
                              "(stdev_init).")
    parser.add_argument("--tournament_size", type=int, default=3,
                         help="[--algorithm ga only] Tournament size for SBX crossover")
    parser.add_argument("--eta", type=float, default=20,
                         help="[--algorithm ga only] Distribution index for SBX crossover")
    parser.add_argument("--no-elitism", action="store_false", dest="elitism",
                         help="[--algorithm ga only] Disable elitism (best individual always "
                              "survives by default)")
    parser.add_argument("--seed_genome", type=str, default=None,
                         help="Path to a genome saved by a previous --output run (must match "
                              "the --controller/--hidden or --topology/--module_size/"
                              "--interneurons settings it was evolved with). Seeds the initial "
                              "population around this genome instead of from scratch, to "
                              "continue/restart evolution from a previous best result.")
    parser.add_argument("--seed_noise", type=float, default=0.05,
                         help="Stdev of the Gaussian perturbation applied to --seed_genome "
                              "copies that fill out the rest of the initial population (one "
                              "individual is kept as an exact, unperturbed copy of the seed). "
                              "Ignored if --seed_genome isn't given.")
    parser.add_argument("--center_crossing", action="store_true",
                         help="[ctrnn only] Seed the initial population's biases using the "
                              "center-crossing recipe (each neuron's steady-state input at 0, "
                              "so its output starts at 0.5) instead of drawing them uniformly "
                              "at random -- meant to start evolution in a region of richer "
                              "circuit dynamics. Off by default so the population starts "
                              "uniformly random unless you opt in. For --topology "
                              "fully_connected this uses a direct generalization of the same "
                              "idea -- see sample_center_crossing_genome()'s docstring in "
                              "neural_controller.py. Ignored if --seed_genome is also given "
                              "(the seed genome's own biases are used instead). Only affects "
                              "genome initialization, not decoding -- evolution can still move "
                              "biases away from this point afterward via ordinary mutation.")
    parser.add_argument("--duration", type=float, default=220.0,
                         help="Simulated seconds per episode")
    parser.add_argument("--patience", type=float, default=0.0,
                         help="[feedforward only] Early-terminate an episode if the walker "
                              "hasn't advanced --min_progress over the trailing --patience "
                              "seconds. 0 (default) disables it, matching walker.py's own "
                              "documented default -- pass e.g. --patience 10 to enable. Not "
                              "used for --controller ctrnn (a CTRNN's rhythm can take a while "
                              "to spin up, so early-terminating on early stillness would "
                              "unfairly penalize slow-starting gaits).")
    parser.add_argument("--min_progress", type=float, default=0.5,
                         help="[feedforward only] Minimum forward-position advance required "
                              "over the trailing --patience window to keep an episode going. "
                              "Only used if --patience > 0.")
    parser.add_argument("--workers", type=str, default="max",
                         help="Parallel workers for evaluating the population: 'max' (all CPU "
                              "cores), an integer, or 'none' for a single process")
    parser.add_argument("--vizperf", action="store_true",
                         help="Visualize fitness over generations")
    parser.add_argument("--verbose", action="store_true",
                         help="Print per-generation statistics to console")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                         help="File path to save the best genome (e.g. best_genome.npy)")
    parser.add_argument("--fitness_output", type=str, default=None,
                         help="Save per-generation best/avg/worst fitness to this .npz file "
                              "(keys: 'best', 'avg', 'worst'), so runs from different "
                              "configurations can be reloaded and compared later without "
                              "re-running evolution. Example: "
                              "np.load('run.npz')['best'] to get the best-fitness curve.")
    return parser.parse_args()


def _parse_workers(workers):
    if workers is None:
        return None
    if isinstance(workers, int):
        return workers
    workers = str(workers).strip().lower()
    if workers == "none":
        return None
    if workers == "max":
        return "max"
    return int(workers)


def _sensors_on_for_episode(mode: str, episode_index: int) -> bool:
    """[ctrnn only] Which sensor condition episode `episode_index` runs
    under, for the given mode. cpg/rpg are constant across episodes; mpg
    alternates so fitness reflects both conditions equally."""
    if mode == "cpg":
        return False
    if mode == "rpg":
        return True
    if mode == "mpg":
        return episode_index % 2 == 1
    raise ValueError(f"mode must be one of: cpg, rpg, mpg; got {mode!r}")


def make_fitness_fn(
    controller="feedforward",
    hidden_sizes=64,
    topology="modular", module_size=SM_DEFAULT, n_interneurons=0, mode="rpg",
    episodes_per_eval=3, seed=None, duration=220.0,
    patience=None, min_progress=0.5,
):
    """Fitness = average forward speed (cx / duration) over episodes_per_eval
    episodes, for either controller type.

    If `patience` is set (feedforward only), each episode can end early
    once the walker stalls (see WalkerEnv's docstring) -- but fitness is
    still divided by the full nominal `duration`, not by however many
    ticks actually ran, so a stalled-out episode is scored exactly as if
    it had kept sitting still for the rest of the episode.
    """

    def fitness_fn(genome: torch.Tensor) -> torch.Tensor:
        if controller == "feedforward":
            model = WalkerController(hidden_sizes=hidden_sizes, obs_size=OBS_SIZE)
            torch.nn.utils.vector_to_parameters(genome, model.parameters())
            model.eval()
        else:
            model = WalkerCTRNNController(
                topology=topology, module_size=module_size, n_interneurons=n_interneurons
            )
            model.set_genome(genome.numpy())

        env = WalkerEnv(
            duration=duration,
            patience=(patience if controller == "feedforward" else None),
            min_progress=min_progress,
        )

        total_fitness = 0.0
        for ep in range(episodes_per_eval):
            episode_seed = None if seed is None else seed + ep
            obs, _ = env.reset(seed=episode_seed)

            if controller == "ctrnn":
                rng = np.random.default_rng(episode_seed)
                model.reset_state(rng)
                sensors_on = _sensors_on_for_episode(mode, ep)

            while True:
                if controller == "feedforward":
                    action = model.act(obs)
                else:
                    action = model.act(obs, sensors_on=sensors_on)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            total_fitness += info["cx"] / duration  # avg forward speed this episode

        env.close()
        return total_fitness / episodes_per_eval

    return fitness_fn


def run_evolution(
    controller="feedforward",
    hidden_sizes=64,
    topology="modular", module_size=SM_DEFAULT, n_interneurons=0, mode="rpg",
    algo="ga",
    popsize=50,
    gens=100,
    mut_stdev=0.05,
    tournament_size=3,
    eta=20,
    elitism=True,
    seed_genome_path=None,
    seed_noise=0.05,
    center_crossing=False,
    episodes_per_eval=3,
    duration=220.0,
    patience=None,
    min_progress=0.5,
    workers="max",
    verbose=False,
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    fitness_fn = make_fitness_fn(
        controller=controller, hidden_sizes=hidden_sizes,
        topology=topology, module_size=module_size, n_interneurons=n_interneurons, mode=mode,
        episodes_per_eval=episodes_per_eval, seed=seed, duration=duration,
        patience=patience, min_progress=min_progress,
    )
    n_genes = compute_genome_length(
        controller=controller, hidden_sizes=hidden_sizes, obs_size=OBS_SIZE,
        topology=topology, module_size=module_size, n_interneurons=n_interneurons,
    )

    if controller == "feedforward":
        # Unbounded real-valued weights, sampled from a small initial
        # range: with Tanh hidden layers and a Sigmoid output, weights
        # drawn from a wide range (e.g. (-5, 5)) push nearly every
        # activation to its saturated extreme right at generation 0 --
        # outputs pinned near 0 or 1 regardless of the input, leaving
        # little fitness variation for gradient-free search to select on.
        # (-1, 1) keeps initial activations in their responsive range.
        # Nothing clips weights back into this range after mutation -- it
        # only affects generation-0 sampling.
        problem_kwargs = dict(initial_bounds=(-1.0, 1.0))
    else:
        # CTRNN genomes decode as raw [0,1] fractions (see neural_controller.py)
        # -- hard-clip after mutation too, same as the CTRNN literature's
        # microbial-GA convention.
        problem_kwargs = dict(initial_bounds=(0.0, 1.0), bounds=(0.0, 1.0))

    problem = Problem(
        objective_sense="max",
        objective_func=fitness_fn,
        solution_length=n_genes,
        dtype=torch.float32,
        num_actors=_parse_workers(workers),
        **problem_kwargs,
    )

    if algo == "ga":
        ea = GeneticAlgorithm(
            problem,
            popsize=popsize,
            operators=[
                SimulatedBinaryCrossOver(problem, tournament_size=tournament_size, eta=eta),
                GaussianMutation(problem, stdev=mut_stdev),
            ],
            elitist=elitism,
        )
    elif algo == "es":
        # CMA-ES has no separate crossover/mutation operators to configure
        # -- it adapts its own step size and covariance every generation.
        # --mut_stdev is reused here as its initial step size only.
        ea = CMAES(problem, popsize=popsize, stdev_init=mut_stdev)
    else:
        raise ValueError(f"algo must be 'ga' or 'es', got {algo!r}")

    # -- Seed the initial population, if requested --
    # Both GeneticAlgorithm and CMAES build their starting population in
    # their constructor (accessible as .population immediately, before the
    # first .step()), so overwriting it here applies before any evolution
    # happens.
    if seed_genome_path is not None:
        seed_genome = torch.tensor(np.load(seed_genome_path), dtype=torch.float32)
        if seed_genome.numel() != n_genes:
            raise ValueError(
                f"Seed genome at '{seed_genome_path}' has {seed_genome.numel()} values, but "
                f"the current --controller/--hidden or --topology/--module_size/--interneurons "
                f"settings expect {n_genes}. Make sure these match what the seed genome was "
                f"evolved with."
            )
        values = ea.population.access_values(keep_evals=False)
        values[0] = seed_genome  # one exact, unperturbed copy
        noise = torch.randn((popsize - 1, n_genes)) * seed_noise
        values[1:] = seed_genome.unsqueeze(0) + noise
        if controller != "feedforward":
            values.clamp_(0.0, 1.0)
    elif center_crossing and controller == "ctrnn":
        rng = np.random.default_rng(seed)
        values = ea.population.access_values(keep_evals=False)
        seeded = np.stack([
            sample_center_crossing_genome(topology, module_size, n_interneurons, rng)
            for _ in range(popsize)
        ])
        values[:] = torch.tensor(seeded, dtype=torch.float32)

    best_fit = np.zeros(gens)
    avg_fit = np.zeros(gens)
    worst_fit = np.zeros(gens)

    for gen in range(gens):
        ea.step()
        fitnesses = ea.population.access_evals()[:, 0]
        fitnesses = torch.where(torch.isnan(fitnesses), torch.zeros_like(fitnesses), fitnesses)

        best_fit[gen] = fitnesses.max().item()
        avg_fit[gen] = fitnesses.mean().item()
        worst_fit[gen] = fitnesses.min().item()

        if verbose and (gen % 10 == 0 or gen == gens - 1):
            print(f"Gen {gen:>4d} | Best: {best_fit[gen]:.4f} | Mean: {avg_fit[gen]:.4f} | Worst: {worst_fit[gen]:.4f}")

    fitnesses = ea.population.access_evals()[:, 0]
    fitnesses = torch.where(torch.isnan(fitnesses), torch.zeros_like(fitnesses), fitnesses)
    best_idx = fitnesses.argmax().item()
    best_genome = ea.population.access_values()[best_idx].clone()
    best_fitness = fitnesses[best_idx].item()

    return best_fit, avg_fit, worst_fit, best_genome, best_fitness


def main():
    args = parse_args()
    hidden_sizes = args.hidden[0] if len(args.hidden) == 1 else tuple(args.hidden)
    patience = args.patience if args.patience and args.patience > 0 else None

    if args.controller == "feedforward":
        print(
            f"Evolving feedforward controller for six-legged walker: "
            f"algorithm={args.algorithm}, hidden={hidden_sizes}, obs_size={OBS_SIZE}, "
            f"popsize={args.popsize}, gens={args.gens}, "
            f"episodes_per_eval={args.episodes_per_eval}, duration={args.duration}, "
            f"patience={patience}, min_progress={args.min_progress}, workers={args.workers}"
        )
    else:
        topology_info = (
            f"module_size={args.module_size}" if args.topology == "modular"
            else f"interneurons={args.interneurons}"
        )
        print(
            f"Evolving CTRNN controller for six-legged walker: "
            f"algorithm={args.algorithm}, topology={args.topology}, {topology_info}, "
            f"mode={args.mode}, popsize={args.popsize}, gens={args.gens}, "
            f"episodes_per_eval={args.episodes_per_eval}, duration={args.duration}, workers={args.workers}"
        )
    if args.seed_genome:
        print(f"Seeding initial population from: {args.seed_genome} (seed_noise={args.seed_noise})")
    elif args.center_crossing:
        if args.controller != "ctrnn":
            raise ValueError("--center_crossing only applies to --controller ctrnn")
        print("Seeding initial population with center-crossing biases")

    best_fit, avg_fit, worst_fit, best_genome, best_fitness = run_evolution(
        controller=args.controller,
        hidden_sizes=hidden_sizes,
        topology=args.topology,
        module_size=args.module_size,
        n_interneurons=args.interneurons,
        mode=args.mode,
        algo=args.algorithm,
        popsize=args.popsize,
        gens=args.gens,
        mut_stdev=args.mut_stdev,
        tournament_size=args.tournament_size,
        eta=args.eta,
        seed_genome_path=args.seed_genome,
        seed_noise=args.seed_noise,
        center_crossing=args.center_crossing,
        elitism=args.elitism,
        episodes_per_eval=args.episodes_per_eval,
        duration=args.duration,
        patience=patience,
        min_progress=args.min_progress,
        workers=args.workers,
        verbose=args.verbose,
        seed=args.seed,
    )

    print(f"\nBest fitness achieved: {best_fitness:.4f} (average forward speed)")

    if args.output:
        np.save(args.output, best_genome.numpy())
        print(f"Best genome saved to: {args.output}")

    if args.fitness_output:
        np.savez(args.fitness_output, best=best_fit, avg=avg_fit, worst=worst_fit)
        print(f"Fitness-over-generations saved to: {args.fitness_output}")

    if args.vizperf:
        plt.figure(figsize=(10, 6))
        gens = np.arange(len(best_fit))
        plt.plot(gens, best_fit, label="Best Fitness", color="green", linewidth=2)
        plt.plot(gens, avg_fit, label="Average Fitness", color="blue", linewidth=2)
        plt.plot(gens, worst_fit, label="Worst Fitness", color="red", linewidth=1, linestyle="--")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (average forward speed)")
        plt.title(f"Neuroevolution Performance: Six-Legged Walker ({args.controller})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
