"""
Soft-robot bodies and environments, built on Evolution Gym (EvoGym).

An EvoGym robot is a small 2D grid of voxels. Each voxel holds one of five
materials:

    0  empty                 (no voxel)
    1  rigid                 (stiff, passive)
    2  soft                  (squishy, passive)
    3  horizontal actuator   (expands/contracts left-right)
    4  vertical actuator     (expands/contracts up-down)

Row 0 of the grid is the TOP of the robot; the last row touches the ground.
Every actuator voxel receives one number per simulation step, its target
length as a fraction of its rest length, in [0.6, 1.6] (0.6 = squeeze,
1.0 = rest, 1.6 = stretch). The action vector lists actuators in row-major
order (top-left to bottom-right, skipping non-actuator voxels) -- this is
what lets per-voxel genes be mapped onto actuators in `actuator_indices()`.

This module handles everything about the *body* and the *world* that
evolve.py and sim.py share: preset bodies, loading a body from a file,
decoding a body from genes, checking a body is buildable, creating the
Gymnasium environment, and running one episode.
"""

import warnings

import numpy as np

# EvoGym 2.0 imports the deprecated `pkg_resources` module and triggers a
# gymnasium deprecation warning on every import; neither is about anything
# in this project, so they're silenced to keep the output readable.
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

N_MATERIALS = 5
MATERIAL_NAMES = ["empty", "rigid", "soft", "h_act", "v_act"]
ACTUATOR_MATERIALS = (3, 4)

# Actuator command range used by EvoGym: target length as a fraction of rest length.
ACTION_LOW, ACTION_HIGH = 0.6, 1.6
ACTION_MID = (ACTION_LOW + ACTION_HIGH) / 2      # 1.1
ACTION_AMP = (ACTION_HIGH - ACTION_LOW) / 2      # 0.5

# Fitness given to a genome whose body can't be built (disconnected pieces,
# or no actuators at all). Must be below anything a real robot scores.
INVALID_FITNESS = -1.0

DEFAULT_ENV = "Walker-v0"

# Hand-designed bodies, all 5x5. Row 0 is the top of the robot.
PRESET_BODIES = {
    # Two legs joined by a torso; every voxel a horizontal actuator.
    "biped": [
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3],
    ],
    # A low, flat body: vertical actuators on top of horizontal ones.
    "worm": [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3],
    ],
    # A solid block: soft body, rigid shell on top, actuators along the bottom.
    "block": [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [4, 2, 4, 2, 4],
        [3, 3, 3, 3, 3],
    ],
}


def load_body(spec):
    """Return a body grid (2D int array) from a preset name or a file path.

    Files can be `.npy` (as saved by np.save) or plain text, one row per
    line with whitespace-separated material codes -- the easiest way to
    hand-design your own body is a small text file like:

        3 3 3 3 3
        3 3 3 3 3
        3 3 0 3 3
        3 3 0 3 3
        3 3 0 3 3
    """
    if spec in PRESET_BODIES:
        body = np.array(PRESET_BODIES[spec], dtype=int)
    elif str(spec).endswith(".npy"):
        body = np.load(spec).astype(int)
    else:
        body = np.loadtxt(spec, dtype=int, ndmin=2)
    if body.ndim != 2 or body.min() < 0 or body.max() >= N_MATERIALS:
        raise ValueError(f"Body '{spec}' must be a 2D grid of material codes 0-{N_MATERIALS - 1}.")
    return body


def is_valid_body(body):
    """True if EvoGym can build this body: one connected piece, at least one actuator."""
    from evogym import is_connected, has_actuator
    return bool(is_connected(body) and has_actuator(body))


def actuator_indices(body):
    """Flat (row-major) voxel indices of the actuators, in action-vector order."""
    return np.flatnonzero(np.isin(np.asarray(body).flatten(), ACTUATOR_MATERIALS))


def decode_body(genes, grid):
    """Decode `grid*grid*5` genes into a body: each voxel takes the material
    whose gene (one of its five) is largest.

    This 'argmax over logits' encoding means every material is always
    reachable by mutation from every other, and small mutations usually
    leave the body unchanged -- only when two of a voxel's genes swap order
    does its material flip.
    """
    logits = np.asarray(genes, dtype=float).reshape(grid, grid, N_MATERIALS)
    return logits.argmax(axis=-1)


def make_env(body, env_id=DEFAULT_ENV, duration=None, render_mode=None):
    """Create an EvoGym environment for `body`.

    `import evogym.envs` is what registers EvoGym's tasks with Gymnasium.
    It's done here, inside the function, rather than only at the top of the
    file, because evolve.py evaluates genomes in separate Ray worker
    processes, and each worker needs its own registration.

    Args:
        body: 2D int array of material codes.
        env_id: EvoGym task name (default 'Walker-v0').
        duration: Episode length in simulation steps; None = the task's own default.
        render_mode: None (headless), 'rgb_array' (frames for GIFs), or 'human' (window).
    """
    import gymnasium as gym
    import evogym.envs  # noqa: F401  (registers the tasks)
    kwargs = {} if duration is None else {"max_episode_steps": int(duration)}
    return gym.make(env_id, body=np.asarray(body), render_mode=render_mode, **kwargs)


def center_of_mass(env):
    """(x, y) of the robot's center of mass right now, averaged over its point masses."""
    sim = env.unwrapped
    return sim.object_pos_at_time(sim.get_time(), "robot").mean(axis=1)


def run_episode(env, controller, record=False, frame_every=0):
    """Run one full episode of `controller` in `env` and return its total reward.

    With record=True, also return a dict of per-step traces:
        'com'     (T, 2) center of mass (x, y)
        'actions' (T, n_actuators) commands sent to each actuator
        'rewards' (T,)   per-step reward
        'frames'  list of RGB images (only if frame_every > 0 and the env
                  was made with render_mode='rgb_array')
    """
    obs, _ = env.reset(seed=0)
    controller.reset()
    total = 0.0
    trace = {"com": [], "actions": [], "rewards": [], "frames": []}
    t = 0
    while True:
        action = controller.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if record:
            trace["com"].append(center_of_mass(env))
            trace["actions"].append(np.array(action))
            trace["rewards"].append(float(reward))
            if frame_every and t % frame_every == 0:
                frame = env.render()
                if frame is not None:
                    trace["frames"].append(frame)
        t += 1
        if terminated or truncated:
            break
    if not record:
        return total
    for key in ("com", "actions", "rewards"):
        trace[key] = np.array(trace[key])
    return total, trace


def body_to_string(body):
    """Compact text picture of a body, one character per voxel."""
    chars = {0: ".", 1: "R", 2: "S", 3: "H", 4: "V"}
    return "\n".join(" ".join(chars[int(v)] for v in row) for row in np.asarray(body))


# EvoGym's own rendering colors, so body plots match the GIFs.
MATERIAL_COLORS = ["#ffffff", "#262626", "#bfbfbf", "#fd8e3e", "#6dafd6"]


def plot_body(body, ax=None, title=None):
    """Draw a body grid with one colored square per voxel (matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    body = np.asarray(body)
    ax.imshow(body, cmap=ListedColormap(MATERIAL_COLORS), vmin=-0.5, vmax=N_MATERIALS - 0.5)
    ax.set_xticks(np.arange(-0.5, body.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, body.shape[0]), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="#888888", label=n)
                 for c, n in zip(MATERIAL_COLORS[1:], MATERIAL_NAMES[1:])],
        loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small", frameon=False,
    )
    if title:
        ax.set_title(title)
    return ax
