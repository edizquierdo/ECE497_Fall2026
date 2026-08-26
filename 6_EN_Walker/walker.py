"""
Gym-style wrapper around the six-legged walker's body physics: a per-tick
leg-force -> velocity -> balance-check simulation, exposed through a
reset()/step() interface any controller can drive -- feedforward network
or CTRNN, they see the exact same environment.

Observation (6,): each leg's current angle, in
    [BACKWARD_ANGLE_LIMIT, FORWARD_ANGLE_LIMIT] radians.
Action (18,): [fwd_0, bwd_0, foot_0, fwd_1, bwd_1, foot_1, ...], one
    (fwd, bwd, foot) triple per leg, each in [0, 1]. `foot > 0.5` means
    that leg's foot is planted (stance); otherwise it's swinging.

Optional early termination (`patience`/`min_progress`): most genomes early
in evolution never get a stable gait going and just sit near cx=0 for the
whole episode, wasting the rest of `duration` on a foregone conclusion. If
`patience` is set, the episode terminates as soon as the walker's forward
position hasn't advanced by at least `min_progress` over the trailing
`patience` seconds -- a sliding-window check, not just a single check at
the start, so it also catches a gait that walks for a while and then stalls
partway through. This is off by default so it doesn't change the behavior
of anything that already depends on always running the full `duration`
(e.g. `sim.py`'s scoring and trace output).
"""


from collections import deque

import numpy as np

N_LEGS = 6
OFFSET_X = 12.5
OFFSET_Y = 5.0
GLOBAL_LEG_LENGTH = 15.0
TS = 0.1
FORWARD_ANGLE_LIMIT = np.pi / 6.0
BACKWARD_ANGLE_LIMIT = -np.pi / 6.0
MAX_LEG_FORCE = 0.05
MAX_VELOCITY = 1.0
MAX_TORQUE = 0.5
MAX_OMEGA = 1.0

VELOCITY_DECAY = 0.015       # per-tick friction on body velocity
OVEREXTENSION_LIMIT = 25.0   # planted-foot lag (joint_x - foot_x) that kills stability
SUPPORT_RAY_LENGTH = 150.0   # length of the front/back rays cast from body center

OBS_SIZE = N_LEGS          # 6: one angle sensor per leg
ACTION_SIZE = 3 * N_LEGS   # 18: (fwd, bwd, foot) per leg


class WalkerEnv:
    """Body-only six-legged walker with a reset()/step() interface."""

    def __init__(self, duration=220.0, seed=None, patience=None, min_progress=0.5):
        """
        Args:
            duration: Max simulated seconds per episode.
            seed: Optional RNG seed.
            patience: If set, simulated seconds of trailing history to check
                for forward progress. None (default) disables early
                termination -- every episode always runs the full duration.
            min_progress: Minimum forward-position advance (in cx units)
                required over the trailing `patience`-second window to keep
                the episode going. Only used if `patience` is set. Default
                0.5 is well above the near-zero drift a truly-stuck walker
                accumulates from noise, and well below the multi-unit
                advance a genuinely walking gait makes over 10s -- but
                hasn't been tuned against real evolutionary runs, so treat
                it as a starting point to validate, not a settled constant.
        """
        self.duration = duration
        self.n_ticks = int(round(duration / TS))
        self.rng = np.random.default_rng(seed)

        self.patience = patience
        self.min_progress = min_progress
        # patience<=0 disables early termination the same way patience=None
        # does (a zero-length window would otherwise count as "full" on
        # the very first tick).
        self._patience_ticks = (
            None if (patience is None or patience <= 0) else int(round(patience / TS))
        )

        self.joint_x_offset = np.array(
            [OFFSET_X, 0.0, -OFFSET_X, -OFFSET_X, 0.0, OFFSET_X], dtype=float
        )
        self.joint_y_offset = np.array(
            [OFFSET_Y, OFFSET_Y, OFFSET_Y, -OFFSET_Y, -OFFSET_Y, -OFFSET_Y], dtype=float
        )
        self.leg_length = np.full(N_LEGS, GLOBAL_LEG_LENGTH)

        self.observation = None

    def reset(self, seed=None):
        """Randomize starting leg angles and reset body position/velocity."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.angle = self.rng.uniform(
            BACKWARD_ANGLE_LIMIT, FORWARD_ANGLE_LIMIT, N_LEGS
        )
        self.omega = np.zeros(N_LEGS)
        self.foot_state = np.zeros(N_LEGS)
        self.cx = 0.0
        self.vx = 0.0
        self.tick = 0

        self.joint_x = self.joint_x_offset.copy()
        self.foot_x = self.joint_x + self.leg_length * np.sin(self.angle)
        self.foot_y = self.joint_y_offset + self.leg_length * np.cos(self.angle)
        self.foot_y_real = self._mirror_foot_y(self.foot_y, self.joint_y_offset)

        self.observation = self.angle.copy()
        self._progress_window = (
            None if self._patience_ticks is None
            else deque(maxlen=self._patience_ticks)
        )
        return self.observation, {}

    def step(self, action):
        """One Euler tick of body physics, given 18 actuator commands."""
        action = np.clip(np.asarray(action, dtype=float), 0.0, 1.0).reshape(N_LEGS, 3)
        fwd, bwd, foot = action[:, 0], action[:, 1], action[:, 2]

        forward_force = fwd * MAX_LEG_FORCE
        backward_force = bwd * MAX_LEG_FORCE

        # ---- part 1: leg forces -> body velocity ----
        body_force = 0.0
        for i in range(N_LEGS):
            if foot[i] > 0.5:
                self.foot_state[i] = 1.0
                self.omega[i] = 0.0
            else:
                self.foot_state[i] = 0.0

            leg_force = forward_force[i] - backward_force[i]
            if self.foot_state[i] == 1.0:
                # A planted leg only contributes force while its angle is
                # strictly within the joint limits -- once it reaches a
                # limit, it stops pushing until it swings back inside.
                if BACKWARD_ANGLE_LIMIT < self.angle[i] < FORWARD_ANGLE_LIMIT:
                    body_force += leg_force

        # Friction/decay every tick, then clip out any residual *backward*
        # velocity before integrating this tick's force -- so momentum
        # only ever persists (and decays) in the forward direction.
        self.vx -= self.vx * VELOCITY_DECAY
        self.vx = float(np.clip(self.vx, 0.0, MAX_VELOCITY))
        self.vx = float(np.clip(self.vx + TS * body_force, -MAX_VELOCITY, MAX_VELOCITY))

        # ---- part 2: leg angles / foot positions ----
        for i in range(N_LEGS):
            self.joint_x[i] = self.cx + self.joint_x_offset[i]
            if self.foot_state[i] == 1.0:
                new_angle = np.arctan2(
                    self.foot_x[i] - self.joint_x[i],
                    self.foot_y[i] - self.joint_y_offset[i],
                )
                self.omega[i] = (new_angle - self.angle[i]) / TS
                self.angle[i] = new_angle
            else:
                self.omega[i] += TS * MAX_TORQUE * (backward_force[i] - forward_force[i])
                self.omega[i] = float(np.clip(self.omega[i], -MAX_OMEGA, MAX_OMEGA))
                self.angle[i] += TS * self.omega[i]
                # Clip the angle at the joint limits but leave omega alone
                # -- a leg that hits a limit while still being driven into
                # it keeps its saturated angular velocity, so reversing
                # direction takes a beat to spin down through zero rather
                # than snapping back instantly.
                self.angle[i] = float(
                    np.clip(self.angle[i], BACKWARD_ANGLE_LIMIT, FORWARD_ANGLE_LIMIT)
                )
                self.foot_x[i] = self.joint_x[i] + self.leg_length[i] * np.sin(self.angle[i])
                self.foot_y[i] = self.joint_y_offset[i] + self.leg_length[i] * np.cos(self.angle[i])
            self.foot_y_real[i] = (
                self.foot_y[i] if i <= 2
                else self.joint_y_offset[i] - (self.foot_y[i] - self.joint_y_offset[i])
            )

        # ---- check_balance: is the body supported by its planted feet? ----
        if not self._stable():
            self.vx = 0.0
        self.cx += TS * self.vx

        self.tick += 1
        self.observation = self.angle.copy()
        reward = self.vx  # per-tick forward speed; average reward ~= final cx/duration

        # ---- optional early termination: has cx advanced enough over the
        # trailing `patience`-second window? Only active once the window is
        # full (i.e. after `patience` seconds have actually elapsed).
        terminated = False
        if self._progress_window is not None:
            self._progress_window.append(self.cx)
            if len(self._progress_window) == self._progress_window.maxlen:
                progress = self.cx - self._progress_window[0]
                terminated = progress < self.min_progress

        truncated = self.tick >= self.n_ticks
        return self.observation, reward, terminated, truncated, {"cx": self.cx}

    @staticmethod
    def _mirror_foot_y(foot_y, joint_y_offset):
        """Legs 3-5 (the right side) have their y-coordinate mirrored for
        the balance check -- without this, the support-polygon test is
        wrong for the right-side legs."""
        out = foot_y.copy()
        for i in range(3, N_LEGS):
            out[i] = joint_y_offset[i] - (foot_y[i] - joint_y_offset[i])
        return out

    def _stable(self):
        """Is the body's center within the support polygon formed by the
        currently-planted feet? Legs 0-2 are "one side" and legs 3-5 are
        "the other side" (same grouping used by `_mirror_foot_y`). Casts a
        ray forward and a ray backward from the body center along the
        travel axis and checks whether each one crosses the edge of the
        polygon connecting the two sides -- the body counts as stable only
        if both do.
        """
        down = [i for i in range(N_LEGS) if self.foot_state[i] > 0.5]

        num_right = sum(1 for i in down if i <= 2)
        num_left = sum(1 for i in down if i >= 3)
        if len(down) <= 2:
            return False
        if num_left == 0 or num_right == 0:
            return False

        pts = []  # (travel-axis pos, lateral-axis pos, side) per planted foot
        for i in down:
            # Overextension bail-out: if this planted foot has dragged more
            # than OVEREXTENSION_LIMIT behind its joint along the travel
            # axis, the whole stability check fails immediately -- this
            # runs on *every* currently-planted leg, not just the ones
            # that end up forming the support polygon.
            if self.joint_x[i] - self.foot_x[i] > OVEREXTENSION_LIMIT:
                return False
            side = 0 if i <= 2 else 1
            pts.append((self.foot_x[i], self.foot_y_real[i], side))

        # Sort most-forward-first (largest travel coordinate first).
        pts.sort(key=lambda p: -p[0])

        def edge_test(direction):
            n = len(pts)
            if direction > 0:
                i = 0
                while i < n - 1 and pts[i][2] == pts[i + 1][2]:
                    i += 1
                c_u, c_v, _ = pts[0]
                d_u, d_v, _ = pts[i + 1]
            else:
                i = n - 1
                while i > 0 and pts[i][2] == pts[i - 1][2]:
                    i -= 1
                c_u, c_v, _ = pts[n - 1]
                d_u, d_v, _ = pts[i - 1]

            # Body-center ray: fixed at lateral position 0, running
            # `direction` * SUPPORT_RAY_LENGTH along the travel axis.
            a_u, a_v = self.cx, 0.0
            b_u, b_v = self.cx + direction * SUPPORT_RAY_LENGTH, 0.0

            denom = (b_v - a_v) * (d_u - c_u) - (b_u - a_u) * (d_v - c_v)
            if denom == 0.0:
                return False
            r = ((a_u - c_u) * (d_v - c_v) - (a_v - c_v) * (d_u - c_u)) / denom
            s = ((a_u - c_u) * (b_v - a_v) - (a_v - c_v) * (b_u - a_u)) / denom
            return 0.0 <= r <= 1.0 and 0.0 <= s <= 1.0

        return edge_test(+1) and edge_test(-1)

    def close(self):
        pass
