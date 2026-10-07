"""
Replay an evolved soft robot: score it, plot its body and behavior, save a
GIF, or watch it live.

A genome saved by `evolve.py --output best.npy` comes with a `best.json`
holding every setting used to evolve it (mode, body, task, controller...),
and sim.py reads that automatically -- no need to re-type matching flags.
`--env` and `--duration` can still be overridden, to test a robot on a
task or episode length it wasn't evolved for.

Run without --genome to try a preset or hand-designed body with a
random, unevolved brain (a quick way to check your setup, or to see what a
body does before evolution touches it).
"""

import json
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from soft_robot import (
    DEFAULT_ENV, DEFAULT_DURATION, PRESET_BODIES, load_body, is_valid_body,
    make_env, run_episode, body_to_string, plot_body,
)
from neural_controller import CONTROLLERS, DEFAULT_HIDDEN, DEFAULT_PERIOD
from evolve import config_path_for, decode_body_from_genome, decode_controller


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate an evolved EvoGym soft robot.")
    parser.add_argument("--genome", type=str, default=None,
                        help="Genome saved by evolve.py --output (its .json settings file must sit "
                             "next to it). Omit to run --body with a random, unevolved brain.")
    parser.add_argument("--body", type=str, default="biped",
                        help=f"[no --genome only] Preset ({', '.join(PRESET_BODIES)}) or .txt/.npy file")
    parser.add_argument("--controller", choices=list(CONTROLLERS), default="local",
                        help="[no --genome only] Kind of random brain to use (default: local)")
    parser.add_argument("--loop", choices=["open", "closed"], default="closed",
                        help="[no --genome only] open- or closed-loop random brain (default: closed)")
    parser.add_argument("--env", type=str, default=None,
                        help="Override the task (default: whatever the genome was evolved on)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Override the episode length in steps")
    parser.add_argument("--showbody", action="store_true", help="Plot the robot's body")
    parser.add_argument("--viztraces", action="store_true",
                        help="Plot center-of-mass trajectory and actuator commands over time")
    parser.add_argument("--gif", type=str, default=None, help="Save an animation to this .gif file")
    parser.add_argument("--render", action="store_true", help="Watch the episode live in a window")
    parser.add_argument("--seed", type=int, default=None,
                        help="[no --genome only] Seed for the random brain's weights")
    return parser.parse_args()


def build(args):
    """Return (body, cfg, genome) for the robot to simulate."""
    if args.genome is None:
        body = load_body(args.body)
        cfg = {"mode": "control", "controller": args.controller, "loop": args.loop,
               "body": body.tolist(), "env": DEFAULT_ENV, "duration": DEFAULT_DURATION,
               "hidden": DEFAULT_HIDDEN[args.controller], "activation": "tanh",
               "period": DEFAULT_PERIOD}
        n_genes = CONTROLLERS[args.controller].genome_size(body.shape, args.loop == "closed", cfg["hidden"])
        genome = np.random.default_rng(args.seed).uniform(-1, 1, n_genes)
        print(f"No --genome given: running body '{args.body}' with a random {args.loop}-loop "
              f"{args.controller} brain.")
    else:
        cfg_path = config_path_for(args.genome)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Settings file {cfg_path} not found -- it's written by "
                                    f"evolve.py --output next to the genome.")
        with open(cfg_path) as f:
            cfg = json.load(f)
        genome = np.load(args.genome)
        body = decode_body_from_genome(genome, cfg)
        brain = "fixed traveling-wave" if cfg["mode"] == "morphology" else \
            f"{cfg['loop']}-loop {cfg['controller']}"
        print(f"Loaded {args.genome} ({cfg['mode']} mode, {brain} brain, evolved on {cfg['env']})")
    if args.env is not None:
        cfg["env"] = args.env
    if args.duration is not None:
        cfg["duration"] = args.duration
    return body, cfg, genome


def main():
    args = parse_args()
    body, cfg, genome = build(args)
    print(body_to_string(body))
    if not is_valid_body(body):
        raise SystemExit("This body can't be built (disconnected, or no actuators).")

    if args.showbody:
        plot_body(body, title=f"Body ({cfg['mode']})")
        plt.tight_layout()

    render_mode = "human" if args.render else ("rgb_array" if args.gif else None)
    env = make_env(body, cfg["env"], cfg["duration"], render_mode=render_mode)
    controller = decode_controller(genome, cfg, body, env)
    reward, trace = run_episode(env, controller, record=True, frame_every=4 if args.gif else 0)
    env.close()

    com = trace["com"]
    print(f"\nTask: {cfg['env']} | steps: {len(com)} | total reward (fitness): {reward:.3f}")
    print(f"Center of mass moved {com[-1, 0] - com[0, 0]:+.3f} in x, {com[-1, 1] - com[0, 1]:+.3f} in y")

    if args.gif:
        import imageio
        imageio.mimsave(args.gif, trace["frames"], duration=0.05, loop=0)
        print(f"Animation saved to: {args.gif}")

    if args.viztraces:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(com[:, 0] - com[0, 0], color="green")
        axes[0].set_ylabel("COM x (from start)")
        axes[0].set_title("Center-of-mass trajectory")
        axes[1].plot(com[:, 1], color="purple")
        axes[1].set_ylabel("COM y (height)")
        im = axes[2].imshow(trace["actions"].T, aspect="auto", cmap="coolwarm",
                            vmin=0.6, vmax=1.6, interpolation="nearest")
        axes[2].set_ylabel("Actuator #\n(row-major)")
        axes[2].set_xlabel("Simulation step")
        axes[2].set_title("Actuator commands (blue = contract, red = expand)")
        fig.colorbar(im, ax=axes[2], orientation="horizontal", fraction=0.08, pad=0.25)
        for ax in axes[:2]:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()

    if args.showbody or args.viztraces:
        plt.show()
    return reward


if __name__ == "__main__":
    main()
