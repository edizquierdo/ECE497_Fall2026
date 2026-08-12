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
                if (
                    BACKWARD_ANGLE_LIMIT <= self.angle[i] <= FORWARD_ANGLE_LIMIT
                    or (self.angle[i] < BACKWARD_ANGLE_LIMIT and leg_force < 0.0)
                    or (self.angle[i] > FORWARD_ANGLE_LIMIT and leg_force > 0.0)
                ):
                    body_force += leg_force

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
                if self.angle[i] < BACKWARD_ANGLE_LIMIT:
                    self.angle[i] = BACKWARD_ANGLE_LIMIT
                    self.omega[i] = 0.0
                if self.angle[i] > FORWARD_ANGLE_LIMIT:
                    self.angle[i] = FORWARD_ANGLE_LIMIT
                    self.omega[i] = 0.0
                self.foot_x[i] = self.joint_x[i] + self.leg_length[i] * np.sin(self.angle[i])
                self.foot_y[i] = self.joint_y_offset[i] + self.leg_length[i] * np.cos(self.angle[i])
            self.foot_y_real[i] = (
                self.foot_y[i] if i <= 2
                else self.joint_y_offset[i] - (self.foot_y[i] - self.joint_y_offset[i])
            )

        # ---- check_balance: tripod support-polygon test ----
        supports = [
            i for i in range(N_LEGS)
            if self.foot_state[i] > 0.5 and (self.joint_x[i] - self.foot_x[i] < 20.0)
        ]
        if not self._supported(supports):
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

    def _supported(self, supports):
        """Point-in-polygon test: is the body center inside the polygon
        formed by the currently-supporting feet? Uses foot_y_real
        (mirrored y for the right-side legs), not raw foot_y."""
        if len(supports) < 3:
            return False
        cy = 0.0
        supported = 0
        j = len(supports) - 1
        for i in range(len(supports)):
            yi = self.foot_y_real[supports[i]]
            yj = self.foot_y_real[supports[j]]
            xi = self.foot_x[supports[i]]
            xj = self.foot_x[supports[j]]
            if (yi < cy) != (yj < cy):
                x_cross = xi + (cy - yi) / (yj - yi) * (xj - xi)
                if x_cross < self.cx:
                    supported ^= 1
            j = i
        return bool(supported)

    def close(self):
        pass
