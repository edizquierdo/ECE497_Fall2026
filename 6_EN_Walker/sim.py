"""
Simulation runner for evolved walker controllers -- feedforward network or
CTRNN, selected with `--controller`. Runs episodes, can print scores,
visualize the body's x-position trajectory, and write a per-tick trace
file compatible with visualize_walker.py's gait-diagram, output-trace, and
animation tools.

Trace-file compatibility note: visualize_walker.load_trace() expects
columns [outputs (n_neurons) | per-leg (foot_x, foot_y, foot_state,
joint_x, joint_y) x 6 | cx, cy, vx], and infers n_neurons from the column
count rather than assuming it -- so it works unmodified whether "outputs"
here is a feedforward controller's 18 raw actuator commands or a CTRNN's
full set of neuron outputs (30 for the default modular topology, or
18 + --interneurons for fully_connected).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from walker import WalkerEnv, N_LEGS, OBS_SIZE, ACTION_SIZE
from neural_controller import WalkerController, WalkerCTRNNController, SM_DEFAULT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate an evolved controller for the six-legged walker."
    )
    parser.add_argument("--controller", choices=["feedforward", "ctrnn"], default="feedforward",
                         help="Controller architecture to run (default: feedforward). Must "
                              "match what --genome was evolved with.")
    parser.add_argument("--genome", type=str, default=None,
                         help="Path to numpy file with the evolved genome "
                              "(optional; omit for a randomly initialized controller)")

    # -- feedforward-only --
    parser.add_argument("--hidden", type=int, nargs="+", default=[64],
                         help="[feedforward only] must match what you evolved with")
    parser.add_argument("--activation", choices=["tanh", "relu", "sigmoid"], default="tanh",
                         help="[feedforward only] must match what you evolved with")

    # -- ctrnn-only --
    parser.add_argument("--topology", choices=["modular", "fully_connected"], default="modular",
                         help="[ctrnn only] must match what you evolved with")
    parser.add_argument("--module_size", type=int, default=SM_DEFAULT,
                         help="[ctrnn only] must match what you evolved with")
    parser.add_argument("--interneurons", type=int, default=0,
                         help="[ctrnn only] must match what you evolved with")
    parser.add_argument("--mode", choices=["cpg", "rpg", "mpg"], default="rpg",
                         help="[ctrnn only] sensor condition to run under: cpg = no sensory "
                              "input, rpg = live leg-angle feedback, mpg = alternate by rep")

    parser.add_argument("--duration", type=float, default=220.0,
                         help="Simulated seconds per episode (default: 220, matching "
                              "evolve.py's --duration default)")
    parser.add_argument("--reps", type=int, default=5,
                         help="Number of independent repetitions/episodes to run")
    parser.add_argument("--viztraces", action="store_true",
                         help="Visualize body x/y position over time")
    parser.add_argument("--scores", action="store_true",
                         help="Print average forward speed over all repetitions")
    parser.add_argument("--trace", type=str, default=None,
                         help="Write a per-tick trace file (single episode) to this path, "
                              "compatible with visualize_walker.py")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def _build_controller(controller, hidden_sizes, topology, module_size, n_interneurons, genome_path, activation="tanh"):
    """Construct and load a controller of the requested type. Returns the
    model, ready for act() calls."""
    if controller == "feedforward":
        model = WalkerController(hidden_sizes=hidden_sizes, obs_size=OBS_SIZE, activation=activation)
        if genome_path is not None:
            genome_data = np.load(genome_path)
            genome = torch.tensor(genome_data, dtype=torch.float32)
            expected_size = WalkerController.genome_size(hidden_sizes=hidden_sizes, obs_size=OBS_SIZE)
            if genome.numel() != expected_size:
                raise ValueError(
                    f"Genome at '{genome_path}' has {genome.numel()} parameters, but "
                    f"--hidden {hidden_sizes} expects {expected_size}. Make sure --hidden "
                    f"(and --activation, though it doesn't change parameter count) matches "
                    f"the settings used when this genome was evolved."
                )
            torch.nn.utils.vector_to_parameters(genome, model.parameters())
            print(f"Loaded genome from: {genome_path}")
        else:
            print("Using randomly initialized controller.")
        model.eval()
        return model

    model = WalkerCTRNNController(
        topology=topology, module_size=module_size, n_interneurons=n_interneurons
    )
    if genome_path is not None:
        genome_data = np.load(genome_path)
        expected_size = WalkerCTRNNController.genome_size(
            topology=topology, module_size=module_size, n_interneurons=n_interneurons
        )
        if genome_data.size != expected_size:
            raise ValueError(
                f"Genome at '{genome_path}' has {genome_data.size} parameters, but "
                f"--topology {topology} (module_size={module_size}, interneurons={n_interneurons}) "
                f"expects {expected_size}. Make sure these settings match what this genome "
                f"was evolved with."
            )
        model.set_genome(genome_data)
        print(f"Loaded genome from: {genome_path}")
    else:
        model.set_genome(np.random.default_rng().random(
            WalkerCTRNNController.genome_size(topology, module_size, n_interneurons)
        ))
        print("Using randomly initialized controller.")
    return model


def _act(controller, model, obs, sensors_on):
    if controller == "feedforward":
        return model.act(obs)
    return model.act(obs, sensors_on=sensors_on)


def run_simulation(
    controller="feedforward",
    genome_path=None,
    hidden_sizes=64,
    activation="tanh",
    topology="modular",
    module_size=SM_DEFAULT,
    n_interneurons=0,
    mode="rpg",
    duration=220.0,
    reps=5,
    viztraces=False,
    seed=None,
):
    """Run the walker simulation with an evolved (or random) controller.

    Returns:
        final_avg: average forward speed (cx / duration) across reps.
        episode_speeds: per-rep forward speed.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    model = _build_controller(controller, hidden_sizes, topology, module_size, n_interneurons, genome_path, activation=activation)

    if viztraces:
        n_ticks = int(round(duration / 0.1))
        xpos = np.zeros((reps, n_ticks))

    speeds = np.zeros(reps)

    for r in range(reps):
        env = WalkerEnv(duration=duration)
        rep_seed = None if seed is None else seed + r
        obs, _ = env.reset(seed=rep_seed)

        if controller == "ctrnn":
            rng = np.random.default_rng(rep_seed)
            model.reset_state(rng)
        sensors_on = mode != "cpg" if mode != "mpg" else (r % 2 == 1)

        t = 0
        while True:
            action = _act(controller, model, obs, sensors_on)
            obs, reward, terminated, truncated, info = env.step(action)
            if viztraces:
                xpos[r, t] = info["cx"]
            t += 1
            if terminated or truncated:
                break

        speeds[r] = info["cx"] / duration  # average forward speed, matches evolve.py's fitness signal
        env.close()

    final_avg = float(np.mean(speeds))

    if viztraces:
        plt.figure(figsize=(10, 6))
        for r in range(reps):
            plt.plot(xpos[r], alpha=0.7, label=f"Rep {r+1}")
        plt.xlabel("Time step")
        plt.ylabel("Body X Position (cx)")
        plt.title(f"Six-Legged Walker Position Over Time ({controller} controller)")
        plt.legend(loc="upper left", fontsize="small")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return final_avg, speeds


def write_trace(
    trace_path,
    controller="feedforward",
    genome_path=None,
    hidden_sizes=64,
    activation="tanh",
    topology="modular",
    module_size=SM_DEFAULT,
    n_interneurons=0,
    mode="rpg",
    duration=1000.0,
    seed=None,
):
    """Run one episode and write a per-tick trace file compatible with
    visualize_walker.py's load_trace(): columns are
    [n_neurons outputs | per-leg (foot_x, foot_y, foot_state, joint_x,
    joint_y) x 6 | cx, cy, vx]. For a feedforward controller, "outputs" is
    the 18 raw actuator commands (it has no internal neurons to report).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    model = _build_controller(controller, hidden_sizes, topology, module_size, n_interneurons, genome_path, activation=activation)

    env = WalkerEnv(duration=duration)
    obs, _ = env.reset(seed=seed)

    if controller == "ctrnn":
        rng = np.random.default_rng(seed)
        model.reset_state(rng)
    sensors_on = mode != "cpg"

    rows = []
    n_ticks = int(round(duration / 0.1))
    for _ in range(n_ticks):
        action = _act(controller, model, obs, sensors_on)  # 18 actuator commands in [0,1]
        neuron_outputs = action if controller == "feedforward" else model.Outputs
        obs, reward, terminated, truncated, info = env.step(action)

        row = list(neuron_outputs)
        for i in range(N_LEGS):
            row.extend([
                env.foot_x[i], env.foot_y_real[i], env.foot_state[i],
                env.joint_x[i], env.joint_y_offset[i],
            ])
        row.extend([env.cx, 0.0, env.vx])  # cy is always 0
        rows.append(row)

        if terminated or truncated:
            break

    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w") as fp:
        for row in rows:
            fp.write("\t".join(f"{v:f}" for v in row))
            fp.write("\n")

    return env.cx / duration


def main():
    args = parse_args()
    hidden_sizes = args.hidden[0] if len(args.hidden) == 1 else tuple(args.hidden)

    common_kwargs = dict(
        controller=args.controller,
        genome_path=args.genome,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        topology=args.topology,
        module_size=args.module_size,
        n_interneurons=args.interneurons,
        mode=args.mode,
        seed=args.seed,
    )

    if args.trace:
        avg_speed = write_trace(
            args.trace, duration=max(args.duration, 1000.0), **common_kwargs,
        )
        print(f"Trace written to: {args.trace} (avg forward speed: {avg_speed:.4f})")
        return

    final_avg, speeds = run_simulation(
        duration=args.duration,
        reps=args.reps,
        viztraces=args.viztraces,
        **common_kwargs,
    )

    if args.scores:
        print(f"\nAverage forward speed over {args.reps} episodes: {final_avg:.4f}")
        print(f"Per-rep speeds: {np.round(speeds, 4)}")

    return final_avg


if __name__ == "__main__":
    main()
