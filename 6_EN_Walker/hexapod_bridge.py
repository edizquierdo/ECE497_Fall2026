"""
Sim-to-(more-)real transfer test, Phase C: take a controller (feedforward
or CTRNN) evolved against walker.py's idealized physics, and run it on
hexapod.xml's real MuJoCo rigid-body simulation instead -- same evolved
weights, no re-training.

hexapod.xml's leg design (a single pivot per leg with 2 co-located
joints, sweep + lift) was specifically chosen to match walker.py's own
per-leg abstraction as closely as possible, so the two interfaces line
up far more directly than a hip+knee kinematic chain ever could:

  walker.py (idealized):
    observation = 6 leg angles (one per leg)
    action      = 18 values: (fwd, bwd, foot) x 6 legs, each in [0, 1]

  hexapod_env.py (real MuJoCo):
    observation = 35 values: torso pose/velocity + 12 joint angles/velocities
                  (sweep, lift) x 6 legs
    action      = 12 torque commands in [-1, 1]: (sweep, lift) x 6 legs

Each leg's SWEEP angle corresponds directly to walker.py's single
per-leg angle sensor -- no discarded state, unlike a hip+knee design
where a knee angle would have nowhere to go. And the SWEEP torque maps
directly too: `sweep_torque = fwd - bwd`, exactly what the genome was
evolved to produce.

The LIFT torque is the one piece that has to be invented: `foot` in
walker.py is a binary-ish stance/swing signal that walker.py's own
physics interprets by zeroing body velocity when the tripod is
unsupported -- it never commands an actual leg-lift torque. hexapod.xml's
lift joint needs an actual torque command every tick, so `foot` has to be
turned into one somehow. `walker_action_to_hexapod_action()` below does
this with a **closed-loop PD controller**: it reads the lift joint's
*actual* current angle and velocity (from the hexapod's own sensors) and
drives it toward a target angle -- down while `foot > 0.5` (planted), up
while swinging -- tapering the torque off as the joint gets close, rather
than pushing at a constant rate regardless of where the joint already is.
This is the same idea as a servo holding a position, not shoving forever.

Nothing here guarantees the evolved gait still works. A genome that
walks fine in walker.py's idealized physics may drag a leg, slip, or
fail to support its own weight here, because real contact dynamics,
friction, and inertia are all things evolution never saw.
"""

import argparse

import numpy as np
import torch

from hexapod_env import HexapodEnv, LEG_NAMES
from neural_controller import WalkerController, WalkerCTRNNController, SM_DEFAULT
from walker import N_LEGS, OBS_SIZE, TS


def hexapod_obs_to_walker_obs(hexapod_obs: np.ndarray) -> np.ndarray:
    """Extract walker.py-shaped observation (6 leg angles) from the real
    hexapod's 35-value observation.

    hexapod_obs layout: [z(1), quat(4), 12 joint angles (sweep,lift)x6,
    torso vel(6), 12 joint velocities]. Each leg's SWEEP angle (indices
    5, 7, 9, 11, 13, 15) IS walker.py's per-leg angle sensor, directly --
    unlike a hip+knee design, nothing has to be discarded here, since
    hexapod.xml's leg design only has the 2 DOF walker.py's action space
    already accounts for (sweep <-> fwd/bwd, lift <-> foot).
    """
    joint_angles = hexapod_obs[5:17]  # (sweep, lift) x 6 legs
    sweep_angles = joint_angles[0::2]  # every other value, starting at sweep
    return sweep_angles


LIFT_DOWN_DEG = -10.0    # target lift angle while a leg is "planted" (foot > 0.5); joint range is [-15, 60]
LIFT_UP_DEG = 40.0       # target lift angle while a leg is "swinging" (foot <= 0.5)
LIFT_KP = 0.08           # base proportional gain: torque (in [-1,1]) per degree of angle error
LIFT_KD = 0.02           # base derivative gain: torque per (degree/second) of joint velocity
LIFT_GAIN_DEFAULT = 2.0  # default multiplier on LIFT_KP/LIFT_KD -- see --lift_gain


def hexapod_obs_to_lift_state(hexapod_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract each leg's current lift-joint angle and angular velocity (in
    degrees / degrees-per-second) from the real hexapod's 35-value
    observation, in LEG_NAMES order. Layout: [z(1), quat(4), 12 joint
    angles (sweep,lift)x6 at [5:17], torso vel(6) at [17:23], 12 joint
    velocities (sweep,lift)x6 at [23:35]] -- see hexapod_env.py's _get_obs()."""
    joint_angles = hexapod_obs[5:17]
    joint_vels = hexapod_obs[23:35]
    lift_angles_deg = np.degrees(joint_angles[1::2])
    lift_vels_deg = np.degrees(joint_vels[1::2])
    return lift_angles_deg, lift_vels_deg


def walker_action_to_hexapod_action(
    action18: np.ndarray, hexapod_obs: np.ndarray, torque_scale: float = 1.0,
    lift_gain: float = LIFT_GAIN_DEFAULT,
) -> np.ndarray:
    """18 walker.py values -> 12 hexapod torques, in LEG_NAMES order
    (matches walker.py's own leg ordering).

    Sweep is a direct correspondence: `sweep_torque = fwd - bwd`, scaled by
    `torque_scale`. Lift is a closed-loop PD controller (see module
    docstring): it reads the lift joint's actual angle/velocity from
    `hexapod_obs` and drives it toward LIFT_DOWN_DEG (planted) or
    LIFT_UP_DEG (swinging), tapering off as it arrives rather than pushing
    at a constant rate. `lift_gain` scales the PD gains (LIFT_KP/LIFT_KD)
    and is deliberately independent of `torque_scale`, which only affects
    sweep -- see `hexapod_lift_sweep.py` for exploring both together."""
    action18 = np.clip(np.asarray(action18, dtype=float), 0.0, 1.0).reshape(N_LEGS, 3)
    fwd, bwd, foot = action18[:, 0], action18[:, 1], action18[:, 2]

    sweep_torque = torque_scale * (fwd - bwd)  # direct correspondence, in [-1, 1]

    lift_angles_deg, lift_vels_deg = hexapod_obs_to_lift_state(hexapod_obs)
    target_deg = np.where(foot > 0.5, LIFT_DOWN_DEG, LIFT_UP_DEG)
    kp, kd = LIFT_KP * lift_gain, LIFT_KD * lift_gain
    lift_torque = kp * (target_deg - lift_angles_deg) - kd * lift_vels_deg

    hexapod_action = np.empty(12)
    hexapod_action[0::2] = sweep_torque
    hexapod_action[1::2] = lift_torque
    return np.clip(hexapod_action, -1.0, 1.0)


def run_transfer(
    controller="feedforward",
    genome_path=None,
    hidden_sizes=64,
    topology="modular",
    module_size=SM_DEFAULT,
    n_interneurons=0,
    mode="rpg",
    torque_scale=1.0,
    timescale=4.0,
    lift_gain=LIFT_GAIN_DEFAULT,
    duration=50.0,
    seed=None,
    render=False,
):
    """Run an evolved (walker.py-trained) genome on the real hexapod.
    Returns total displacement (direction-agnostic straight-line distance
    from the start position) and whether the hexapod stayed upright the
    whole time.

    Displacement is deliberately NOT raw x-axis progress: unlike walker.py's
    idealized physics (1D by construction, so x is the only axis that
    exists), the real MuJoCo body can drift sideways or yaw off-heading,
    and walker.py's genomes were never evolved with any notion of steering.
    Scoring only x would mark a genome that's visibly walking -- just not
    in +x -- the same as one that never moved at all. sqrt(dx^2 + dy^2)
    from the reset position counts any movement as progress; it doesn't
    care if the walk is straight, curved, or sideways.

    `duration` is in simulated seconds, same convention as walker.py /
    sim.py / evolve.py's `--duration` -- NOT the same thing as a raw step
    count, since hexapod_env.py's per-step dt (0.05s, from MuJoCo's
    timestep=0.01 x frame_skip=5) differs from walker.py's TS (0.1s). The
    number of steps actually run is `duration / env.dt`, computed below
    once the env exists.

    `timescale` (--controller ctrnn only) scales the CTRNN's own internal
    clock relative to the real hexapod's physical clock. The genome was
    evolved with the CTRNN's internal clock and walker.py's body advancing
    in lockstep, one act() per walker.py tick of TS=0.1s -- but each
    hexapod_env.py step here is only 0.05s of real physical time, so the
    two clocks don't automatically line up. `timescale` multiplies the dt
    passed to `act()` (`dt = TS * timescale`): higher values make the
    CPG's commanded leg motion play out faster relative to the body's
    physical response, lower values slow it down. `lift_gain` scales the
    PD lift controller's own gains -- see `hexapod_lift_sweep.py` for
    exploring `timescale` and `lift_gain` together."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if controller == "feedforward":
        model = WalkerController(hidden_sizes=hidden_sizes, obs_size=OBS_SIZE)
        if genome_path is not None:
            genome = torch.tensor(np.load(genome_path), dtype=torch.float32)
            torch.nn.utils.vector_to_parameters(genome, model.parameters())
        model.eval()
    else:
        model = WalkerCTRNNController(topology=topology, module_size=module_size, n_interneurons=n_interneurons)
        if genome_path is not None:
            model.set_genome(np.load(genome_path))
        else:
            model.set_genome(np.random.default_rng(seed).random(
                WalkerCTRNNController.genome_size(topology, module_size, n_interneurons)
            ))

    env = HexapodEnv(render_mode="human" if render else None)
    obs, _ = env.reset(seed=seed)
    x0, y0 = env.data.qpos[0], env.data.qpos[1]
    steps = int(round(duration / env.dt))

    if controller == "ctrnn":
        rng = np.random.default_rng(seed)
        model.reset_state(rng)
    sensors_on = mode != "cpg"

    fell = False
    for t in range(steps):
        walker_obs = hexapod_obs_to_walker_obs(obs)
        if controller == "feedforward":
            action18 = model.act(walker_obs)
        else:
            action18 = model.act(walker_obs, sensors_on=sensors_on, dt=TS * timescale)

        hexapod_action = walker_action_to_hexapod_action(
            action18, obs, torque_scale=torque_scale, lift_gain=lift_gain
        )
        obs, reward, terminated, truncated, info = env.step(hexapod_action)

        if terminated:
            fell = True
            break

    displacement = float(np.hypot(info["x_position"] - x0, info["y_position"] - y0))
    env.close()
    return displacement, fell, t + 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a walker.py-evolved genome on the real MuJoCo hexapod (sim-to-real transfer test)."
    )
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
                         help="Scale on the mapped sweep torque -- walker.py's action magnitudes "
                              "were never tuned against real actuator gains, so this is worth "
                              "sweeping.")
    parser.add_argument("--timescale", type=float, default=4.0,
                         help="[ctrnn only] Scales the CTRNN's internal clock relative to the "
                              "real hexapod's physical clock (dt passed to act() = TS * "
                              "timescale). The genome evolved with its internal clock and "
                              "walker.py's body in lockstep at TS=0.1s/tick, but hexapod_env.py's "
                              "real physical step is only 0.05s -- worth exploring together with "
                              "--lift_gain (see hexapod_lift_sweep.py) rather than assuming the "
                              "default is right.")
    parser.add_argument("--lift_gain", type=float, default=LIFT_GAIN_DEFAULT,
                         help="Multiplier on the PD lift controller's gains (LIFT_KP/LIFT_KD), "
                              "independent of --torque_scale (which scales sweep only) -- worth "
                              "exploring together with --timescale, see hexapod_lift_sweep.py.")
    parser.add_argument("--duration", type=float, default=50.0,
                         help="Simulated seconds to run, same convention as walker.py/sim.py/"
                              "evolve.py's --duration. NOT the same as a raw MuJoCo step count: "
                              "hexapod_env.py's per-step dt (0.05s) differs from walker.py's TS "
                              "(0.1s), so steps run = duration / 0.05.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--render", action="store_true",
                         help="Open an interactive MuJoCo viewer window (requires a display; "
                              "not for headless servers -- see hexapod_render.py instead)")
    return parser.parse_args()


def main():
    args = parse_args()
    hidden_sizes = args.hidden[0] if len(args.hidden) == 1 else tuple(args.hidden)

    displacement, fell, n_steps = run_transfer(
        controller=args.controller,
        genome_path=args.genome,
        hidden_sizes=hidden_sizes,
        topology=args.topology,
        module_size=args.module_size,
        n_interneurons=args.interneurons,
        mode=args.mode,
        torque_scale=args.torque_scale,
        timescale=args.timescale,
        lift_gain=args.lift_gain,
        duration=args.duration,
        seed=args.seed,
        render=args.render,
    )

    status = "fell / became unhealthy" if fell else "stayed upright"
    print(f"Ran {n_steps} steps ({n_steps * 0.05:.1f}s) on the real hexapod ({status}).")
    print(f"Total displacement from start: {displacement:.4f}")


if __name__ == "__main__":
    main()
