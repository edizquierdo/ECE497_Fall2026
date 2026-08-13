"""
Render an offscreen MP4/GIF of the real MuJoCo hexapod being driven by a
walker.py-evolved genome through hexapod_bridge.py. Useful on headless
machines (no display) where `hexapod_bridge.py --render` (the live
interactive viewer) won't work.
"""

import argparse
import os
import sys

# Offscreen rendering needs a GL backend; on a headless Linux machine with
# no display (e.g. a remote GPU server) MuJoCo/Gymnasium won't pick one
# automatically. 'egl' works on machines with an NVIDIA driver (like the
# lab's GPU server) without needing a display -- but it's a Linux-only
# backend: MuJoCo doesn't support it on macOS/Windows, and setting it there
# raises "invalid value for environment variable MUJOCO_GL: egl" (macOS has
# its own working default -- 'cgl' -- that only kicks in if MUJOCO_GL isn't
# set at all, so this only forces 'egl' on Linux and leaves other platforms
# alone). If you're on a headless Linux machine with no GPU at all, set
# MUJOCO_GL=osmesa yourself before running this script instead (and
# `pip install`/apt-install a software OSMesa implementation if you don't
# already have one). Only sets a default -- won't override an environment
# variable you've already set.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import torch

from hexapod_env import HexapodEnv
from hexapod_bridge import (
    hexapod_obs_to_walker_obs, walker_action_to_hexapod_action, LIFT_GAIN_DEFAULT,
)
from neural_controller import WalkerController, WalkerCTRNNController, SM_DEFAULT
from walker import OBS_SIZE, TS


def parse_args():
    parser = argparse.ArgumentParser(description="Render a video of the hexapod driven by an evolved genome.")
    parser.add_argument("--controller", choices=["feedforward", "ctrnn"], default="feedforward",
                         help="Same meaning as evolve.py/sim.py -- must match how --genome was evolved.")
    parser.add_argument("--genome", type=str, default=None,
                         help="Genome evolved by evolve.py against walker.py. Omit for a random controller.")
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
                         help="Scale on the mapped sweep torque -- see hexapod_bridge.py's "
                              "--torque_scale for why this is worth sweeping.")
    parser.add_argument("--timescale", type=float, default=4.0,
                         help="[ctrnn only] Same meaning as hexapod_bridge.py's --timescale -- "
                              "scales the CTRNN's internal clock relative to the real hexapod's "
                              "physical clock.")
    parser.add_argument("--lift_gain", type=float, default=LIFT_GAIN_DEFAULT,
                         help="Same meaning as hexapod_bridge.py's --lift_gain -- multiplier on "
                              "the PD lift controller's gains, independent of --torque_scale.")
    parser.add_argument("--duration", type=float, default=15.0,
                         help="Simulated seconds to render, same convention as walker.py/sim.py/"
                              "evolve.py/hexapod_bridge.py's --duration. Steps rendered = "
                              "duration / env.dt (0.05s).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="hexapod.mp4",
                         help="Output video path (.mp4 recommended; .gif also works)")
    parser.add_argument("--width", type=int, default=640, help="Frame width in pixels")
    parser.add_argument("--height", type=int, default=480, help="Frame height in pixels")
    parser.add_argument("--fps", type=int, default=20, help="Playback frame rate of the output video")
    return parser.parse_args()


def main():
    args = parse_args()
    hidden_sizes = args.hidden[0] if len(args.hidden) == 1 else tuple(args.hidden)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.controller == "feedforward":
        model = WalkerController(hidden_sizes=hidden_sizes, obs_size=OBS_SIZE)
        if args.genome is not None:
            genome = torch.tensor(np.load(args.genome), dtype=torch.float32)
            torch.nn.utils.vector_to_parameters(genome, model.parameters())
        model.eval()
    else:
        model = WalkerCTRNNController(topology=args.topology, module_size=args.module_size, n_interneurons=args.interneurons)
        if args.genome is not None:
            model.set_genome(np.load(args.genome))
        else:
            model.set_genome(np.random.default_rng(args.seed).random(
                WalkerCTRNNController.genome_size(args.topology, args.module_size, args.interneurons)
            ))

    env = HexapodEnv(render_mode="rgb_array", width=args.width, height=args.height, camera_name="track")
    obs, _ = env.reset(seed=args.seed)
    x0, y0 = env.data.qpos[0], env.data.qpos[1]
    steps = int(round(args.duration / env.dt))

    if args.controller == "ctrnn":
        rng = np.random.default_rng(args.seed)
        model.reset_state(rng)
    sensors_on = args.mode != "cpg"

    frames = [env.render()]
    for t in range(steps):
        walker_obs = hexapod_obs_to_walker_obs(obs)
        if args.controller == "feedforward":
            action18 = model.act(walker_obs)
        else:
            action18 = model.act(walker_obs, sensors_on=sensors_on, dt=TS * args.timescale)

        hexapod_action = walker_action_to_hexapod_action(
            action18, obs, torque_scale=args.torque_scale, lift_gain=args.lift_gain
        )
        obs, reward, terminated, truncated, info = env.step(hexapod_action)
        frames.append(env.render())
        if terminated:
            print(f"Fell at step {t}.")
            break

    env.close()

    displacement = np.hypot(info["x_position"] - x0, info["y_position"] - y0)
    if args.out.lower().endswith(".gif"):
        # imageio's Pillow-based GIF writer takes per-frame duration in ms,
        # not fps (unlike the ffmpeg-based writer used for .mp4/etc.).
        imageio.mimsave(args.out, frames, duration=1000.0 / args.fps)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Wrote {len(frames)} frames to {args.out} (displacement from start: {displacement:.4f})")


if __name__ == "__main__":
    main()
