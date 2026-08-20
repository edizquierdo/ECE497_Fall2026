"""
Visualization tools for inspecting an evolved walker's behavior from a trace
file written by `sim.py`'s `write_trace()` (i.e. what `sim.py --trace`
produces).

Three visualizations, all built from the same trace file:

  - `plot_gait_diagram()`   -- classic insect-gait stance/swing diagram
                               (black = foot down, white = foot up), one row
                               per leg, with body speed plotted underneath.
  - `plot_neuron_outputs()` -- all 30 neuron outputs (5 per leg x 6 legs) as
                               a compact grid of small line plots, so you can
                               see the rhythms the CTRNN actually produced.
  - `animate_walker()`      -- a top-down animation of the six legs swinging
                               and the body moving forward, saved as a GIF.

Run this file directly to visualize a trace:

    python visualize_walker.py rpg.dat --out_prefix rpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

from walker import N_LEGS, TS, OFFSET_X, OFFSET_Y

# neural_controller.py's N_COMMAND (3 command channels x 6 legs) and
# SM_DEFAULT (CTRNN module size) are CTRNN-specific and not needed for
# Project 6's feedforward traces, but are used below purely as display
# parameters (how many "output" columns to expect, and a default module
# size for the modular topology view). Importing them from
# neural_controller.py would pull in torch (this script has no other need
# for it) -- so they're given directly here instead. Values match
# neural_controller.py's constants of the same name.
N_COMMAND = 18  # 3 (fwd, bwd, foot) x 6 legs
SM_DEFAULT = 3

# Leg index -> conventional insect-gait label (front/mid/hind, left/right).
# Leg pairs 0<->5, 1<->4, 2<->3 are the CBW (contra-lateral) pairs, i.e.
# directly across the body from each other -- see the body-plan diagram in
# neural_controller.py. That makes 0,1,2 the left-side legs (front to hind)
# and 5,4,3 the right-side legs (front to hind).
LEG_NAMES = {0: "L1", 1: "L2", 2: "L3", 3: "R3", 4: "R2", 5: "R1"}
# Display order top-to-bottom, matching the classic gait-diagram convention.
DISPLAY_ORDER = [3, 4, 5, 2, 1, 0]  # R3, R2, R1, L3, L2, L1


def _truncate_trace(trace: dict[str, np.ndarray], max_time: float | None) -> dict[str, np.ndarray]:
    """Return a copy of `trace` with every array cut to the first
    `max_time` simulated seconds (all arrays are indexed by tick, so this
    is just a shared row-slice). `max_time=None` returns `trace`
    unchanged. Used to make `--max_time` apply consistently to the gait
    diagram and neuron-output plots too, not just the animation -- a
    1000-simulated-second trace (`sim.py --trace`'s minimum duration)
    otherwise always plots all 10,000 ticks on every figure regardless of
    what's actually useful to look at."""
    if max_time is None:
        return trace
    n_ticks_total = len(trace["cx"])
    n_ticks = min(n_ticks_total, int(round(max_time / TS)))
    return {
        key: (value[:n_ticks] if isinstance(value, np.ndarray) else value)
        for key, value in trace.items()
    }


def load_trace(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a trace file written by `write_trace()` into named arrays.
    Works for a trace from either topology -- the neuron count isn't
    assumed, it's inferred from the file's column count.

    Column layout per row (see `write_trace`):
        outputs (n_neurons) | per-leg (foot_x, foot_y, foot_state, joint_x, joint_y) x 6 | cx, cy, vx
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    n_ticks = data.shape[0]
    n_neurons = data.shape[1] - N_LEGS * 5 - 3

    outputs = data[:, :n_neurons]  # (n_ticks, n_neurons) -- flat; see plot_neuron_outputs for topology-aware grouping

    leg_block = data[:, n_neurons : n_neurons + N_LEGS * 5].reshape(n_ticks, N_LEGS, 5)
    foot_x = leg_block[:, :, 0]
    foot_y = leg_block[:, :, 1]
    foot_state = leg_block[:, :, 2]  # >0.5 == stance (foot down)
    joint_x = leg_block[:, :, 3]
    joint_y = leg_block[:, :, 4]

    cx, cy, vx = data[:, -3], data[:, -2], data[:, -1]

    return dict(
        outputs=outputs, n_neurons=n_neurons,
        foot_x=foot_x, foot_y=foot_y, foot_state=foot_state,
        joint_x=joint_x, joint_y=joint_y, cx=cx, cy=cy, vx=vx,
    )


def plot_gait_diagram(
    trace: dict[str, np.ndarray],
    save_path: str | Path,
    title: str = "Gait diagram",
):
    """Stance/swing diagram: one horizontal row per leg, black where the
    foot is down (stance) and white where it's up (swing) -- the standard
    way gaits are visualized in the evolutionary-robotics literature (e.g.
    Beer & Gallagher). Body speed is plotted underneath so you can relate
    gait pattern to forward progress.
    """
    foot_state = trace["foot_state"]
    n_ticks = foot_state.shape[0]
    t = np.arange(n_ticks) * TS

    fig, (ax_gait, ax_speed) = plt.subplots(
        2, 1, figsize=(10, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    stance = (foot_state > 0.5).astype(float)  # (n_ticks, N_LEGS), 1 = stance
    # Rearrange rows into display order (R3,R2,R1,L3,L2,L1 top to bottom).
    image = stance[:, DISPLAY_ORDER].T  # (N_LEGS, n_ticks)
    ax_gait.imshow(
        image, aspect="auto", cmap="gray_r", interpolation="nearest",
        extent=[t[0], t[-1], N_LEGS - 0.5, -0.5], origin="upper",
        vmin=0, vmax=1,  # fixed 0/1 scale -- without this, imshow auto-normalizes from
                         # the data's own min/max, so a trace where every leg is planted
                         # (or every leg swings) the whole window -- a constant array --
                         # collapses to normalized 0 and renders white regardless of
                         # which state it actually was.
    )
    ax_gait.set_yticks(range(N_LEGS))
    ax_gait.set_yticklabels([LEG_NAMES[i] for i in DISPLAY_ORDER])
    ax_gait.set_title(title)
    ax_gait.set_ylabel("Leg")

    ax_speed.plot(t, trace["vx"], color="black", linewidth=0.8)
    ax_speed.set_ylabel("Body speed")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.axhline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_neuron_outputs(
    trace: dict[str, np.ndarray],
    save_path: str | Path,
    title: str = "Neuron outputs",
    topology: str = "fully_connected",
    module_size: int = 0,
):
    """Compact grid of neuron output traces, laid out according to the
    circuit's topology:

      - **modular**: one row per leg (in display order), one column per
        neuron slot within the leg's module (FWD, BWD, FOOT, then any
        free interneurons -- see `neural_controller.py` for the
        neuron-role constants). `module_size` must match what the trace
        was generated with.
      - **fully_connected**: the 18 command neurons shown the same way
        (6 leg-rows x 3 role-columns), plus a second block underneath for
        any shared interneurons (which aren't tied to a particular leg,
        so they're just laid out in a wrapped grid rather than by leg).

    Every subplot shares the same y-axis range [0, 1], since outputs are
    sigmoids.
    """
    outputs = trace["outputs"]  # (n_ticks, n_neurons), flat
    n_ticks = outputs.shape[0]
    t = np.arange(n_ticks) * TS

    if topology == "modular":
        sm = module_size
        outputs_by_leg = outputs.reshape(n_ticks, N_LEGS, sm)
        neuron_labels = ["FWD", "BWD", "FOOT"] + [f"N{i}" for i in range(sm - 3)]

        fig, axes = plt.subplots(
            N_LEGS, sm, figsize=(2.0 * sm, 1.3 * N_LEGS), sharex=True, sharey=True,
        )
        if sm == 1:
            axes = axes.reshape(N_LEGS, 1)
        for row, leg in enumerate(DISPLAY_ORDER):
            for col in range(sm):
                ax = axes[row, col]
                ax.plot(t, outputs_by_leg[:, leg, col], color="steelblue", linewidth=0.8)
                ax.set_ylim(-0.05, 1.05)
                if col == 0:
                    ax.set_ylabel(LEG_NAMES[leg], rotation=0, ha="right", va="center")
                if row == 0:
                    ax.set_title(neuron_labels[col], fontsize=9)
                if row == N_LEGS - 1:
                    ax.set_xlabel("Time (s)", fontsize=8)
                ax.tick_params(labelsize=7)
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    elif topology == "fully_connected":
        n_interneurons = outputs.shape[1] - N_COMMAND
        command = outputs[:, :N_COMMAND].reshape(n_ticks, N_LEGS, 3)
        role_labels = ["FWD", "BWD", "FOOT"]

        inter_rows = -(-n_interneurons // 6) if n_interneurons > 0 else 0  # ceil(n/6), wrapped 6-wide
        total_rows = N_LEGS + inter_rows
        fig, axes = plt.subplots(
            total_rows, 6, figsize=(12, 1.3 * total_rows), sharex=True, sharey=True,
        )
        axes = np.atleast_2d(axes)

        for row, leg in enumerate(DISPLAY_ORDER):
            for col in range(6):
                ax = axes[row, col]
                if col < 3:
                    ax.plot(t, command[:, leg, col], color="steelblue", linewidth=0.8)
                    if row == 0:
                        ax.set_title(role_labels[col], fontsize=9)
                else:
                    ax.axis("off")
                if col == 0:
                    ax.set_ylabel(LEG_NAMES[leg], rotation=0, ha="right", va="center")
                ax.set_ylim(-0.05, 1.05)
                ax.tick_params(labelsize=7)

        for k in range(inter_rows * 6):
            ax = axes[N_LEGS + k // 6, k % 6]
            if k < n_interneurons:
                ax.plot(t, outputs[:, N_COMMAND + k], color="darkorange", linewidth=0.8)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"I{k}", fontsize=8)
                if k // 6 == inter_rows - 1:
                    ax.set_xlabel("Time (s)", fontsize=8)
                ax.tick_params(labelsize=7)
            else:
                ax.axis("off")
        if inter_rows > 0:
            axes[N_LEGS, 0].set_ylabel("interneurons", rotation=0, ha="right", va="center", fontsize=8)

        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    else:
        raise ValueError(f"Unknown topology: {topology!r} (expected 'modular' or 'fully_connected')")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def animate_walker(
    trace: dict[str, np.ndarray],
    save_path: str | Path,
    stride: int = 10,
    fps: int = 20,
    max_time: float | None = 30.0,
    trail: bool = True,
):
    """Top-down animation of the walker: a plain rectangular body with six
    legs rigidly anchored to it (fixed body-relative joints -- joints
    never move relative to the body or to each other), each leg drawn as
    a line from its joint to its foot, and a filled disk marking the foot
    only while it's planted (stance) -- the disk disappears entirely while
    the leg is swinging, so foot-down timing is visually unambiguous.

    `joint_x`/`foot_x` in the trace are already in world coordinates --
    walker.py's step() computes `self.joint_x[i] = self.cx +
    self.joint_x_offset[i]` every tick (and `foot_x` is derived from that
    same world-space `joint_x` while swinging, or held fixed in world
    space while planted, which is correct: a planted foot doesn't move
    under the body). So they're used directly here, with no extra offset
    added.

    `stride` subsamples ticks (e.g. stride=10 with TS=0.1 gives one
    animation frame per 1.0s of simulated time). `max_time` (in simulated
    seconds) caps how much of the trace gets animated -- traces can cover
    many hundreds of simulated seconds (e.g. `--trace` uses a 1000s
    duration), which is far more frames than needed to see the gait and can
    exhaust memory if animated in full. Pass `max_time=None` to animate the
    whole trace.
    """
    cx, cy = trace["cx"], trace["cy"]
    joint_x, joint_y = trace["joint_x"], trace["joint_y"]
    foot_x, foot_y = trace["foot_x"], trace["foot_y"]
    foot_state = trace["foot_state"]

    n_ticks_total = len(cx)
    n_ticks = n_ticks_total if max_time is None else min(n_ticks_total, int(max_time / TS))

    frame_idx = np.arange(0, n_ticks, stride)

    # Already world coordinates -- see docstring. Do NOT add cx/cy again.
    world_joint_x, world_joint_y = joint_x[frame_idx], joint_y[frame_idx]
    world_foot_x, world_foot_y = foot_x[frame_idx], foot_y[frame_idx]
    stance = foot_state[frame_idx] > 0.5

    body_x, body_y = cx[frame_idx], cy[frame_idx]

    # Plain rectangular body, sized with a small margin around the leg
    # attachment points (OFFSET_X/OFFSET_Y), in body-relative coordinates.
    half_x, half_y = 1.15 * OFFSET_X, 1.15 * OFFSET_Y
    body_rect = np.array([
        [-half_x, -half_y], [-half_x, half_y], [half_x, half_y], [half_x, -half_y],
    ])

    x_min = min(world_joint_x.min(), world_foot_x.min(), (body_x - half_x).min()) - 0.5 * OFFSET_X
    x_max = max(world_joint_x.max(), world_foot_x.max(), (body_x + half_x).max()) + 0.5 * OFFSET_X
    y_min = min(world_joint_y.min(), world_foot_y.min(), -half_y) - 0.5 * OFFSET_Y
    y_max = max(world_joint_y.max(), world_foot_y.max(), half_y) + 0.5 * OFFSET_Y

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title("Walker animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    trail_line, = ax.plot([], [], "-", color="lightgray", linewidth=1.0, zorder=0)
    body_patch = Polygon(body_rect, closed=True, facecolor="#CFD8DC",
                          edgecolor="#37474F", linewidth=1.5, zorder=2)
    ax.add_patch(body_patch)
    leg_lines = [ax.plot([], [], "-", color="black", linewidth=2.0, zorder=3)[0] for _ in range(N_LEGS)]
    foot_dots = [ax.plot([], [], "o", markersize=9, color="crimson", zorder=4)[0] for _ in range(N_LEGS)]

    def init():
        trail_line.set_data([], [])
        body_patch.set_xy(body_rect)
        for line, dot in zip(leg_lines, foot_dots):
            line.set_data([], [])
            dot.set_data([], [])
        return [trail_line, body_patch, *leg_lines, *foot_dots]

    def update(frame_i):
        bx, by = body_x[frame_i], body_y[frame_i]
        body_patch.set_xy(body_rect + np.array([bx, by]))
        if trail:
            trail_line.set_data(body_x[: frame_i + 1], body_y[: frame_i + 1])
        for leg in range(N_LEGS):
            jx, jy = world_joint_x[frame_i, leg], world_joint_y[frame_i, leg]
            fx, fy = world_foot_x[frame_i, leg], world_foot_y[frame_i, leg]
            leg_lines[leg].set_data([jx, fx], [jy, fy])
            if stance[frame_i, leg]:
                foot_dots[leg].set_data([fx], [fy])
            else:
                foot_dots[leg].set_data([], [])  # swinging: foot disk disappears
        return [trail_line, body_patch, *leg_lines, *foot_dots]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_idx), init_func=init, blit=True,
    )
    anim.save(save_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def visualize_trace(
    trace_path: str | Path,
    out_prefix: str | Path,
    max_time: float | None = 30.0,
    topology: str = "fully_connected",
    module_size: int = 0,
):
    """Convenience wrapper: load a trace file once and produce all three
    visualizations, named `<out_prefix>_gait.png`, `<out_prefix>_neurons.png`,
    and `<out_prefix>_animation.gif`. `max_time` (in simulated seconds)
    windows all three outputs to the same leading slice of the trace, not
    just the animation -- pass `max_time=None` to use the whole trace.
    `topology`/`module_size` must match whatever the trace was generated
    with (only `plot_neuron_outputs` needs them -- the gait diagram and
    animation are topology-agnostic)."""
    trace = load_trace(trace_path)
    trace = _truncate_trace(trace, max_time)
    out_prefix = str(out_prefix)
    plot_gait_diagram(trace, f"{out_prefix}_gait.png")
    plot_neuron_outputs(trace, f"{out_prefix}_neurons.png", topology=topology, module_size=module_size)
    animate_walker(trace, f"{out_prefix}_animation.gif", max_time=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a walker trace file (gait diagram, neuron outputs, animation)."
    )
    parser.add_argument("trace_path", type=Path, help="Trace file written by --trace (e.g. rpg.dat)")
    parser.add_argument("--out_prefix", type=str, default=None, help="Output filename prefix (default: trace filename stem)")
    parser.add_argument(
        "--topology", choices=["fully_connected", "modular"], default="fully_connected",
        help="Must match the topology the trace was generated with (default: fully_connected, which is what sim.py's --trace writes)",
    )
    parser.add_argument(
        "--module_size", type=int, default=0,
        help="Must match the module size the trace was generated with. Only used with --topology modular (default: 0, since sim.py's --trace has no CTRNN modules)",
    )
    parser.add_argument("--stride", type=int, default=10, help="Animation frame subsampling (default: 10)")
    parser.add_argument("--fps", type=int, default=20, help="Animation frames per second (default: 20)")
    parser.add_argument(
        "--max_time", type=float, default=30.0,
        help="Max simulated seconds to include in ALL THREE outputs (gait diagram, neuron "
             "plots, and animation) -- not just the animation. 0 means the whole trace "
             "(default: 30). This is also the parameter to change if the gait/neuron figures' "
             "time axis is showing more time than you want to look at -- sim.py's --trace "
             "writes at least 1000 simulated seconds by design, and every plot windows to "
             "this value, not just the animation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_prefix = args.out_prefix or args.trace_path.stem
    trace = load_trace(args.trace_path)
    max_time = None if args.max_time == 0 else args.max_time
    trace = _truncate_trace(trace, max_time)
    plot_gait_diagram(trace, f"{out_prefix}_gait.png", title=f"Gait diagram: {args.trace_path.name}")
    plot_neuron_outputs(
        trace, f"{out_prefix}_neurons.png", title=f"Neuron outputs: {args.trace_path.name}",
        topology=args.topology, module_size=args.module_size,
    )
    animate_walker(trace, f"{out_prefix}_animation.gif", stride=args.stride, fps=args.fps, max_time=None)
    print(f"Wrote {out_prefix}_gait.png, {out_prefix}_neurons.png, {out_prefix}_animation.gif")


if __name__ == "__main__":
    main()
