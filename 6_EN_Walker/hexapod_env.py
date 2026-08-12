"""
Gymnasium environment for the hexapod.xml MuJoCo model (Phase C /
sim-to-real transfer test). Structured the same way Gymnasium's own
Ant-v5 is: a thin MujocoEnv subclass exposing qpos/qvel-based
observations and a Box(-1, 1, (12,)) action space (one torque per sweep/lift
joint, 6 legs x 2 joints).

This is a *real* rigid-body simulation with actual contact dynamics,
friction, and inertia -- unlike walker.py's idealized force/angle model.
Nothing here is guaranteed to behave like walker.py's physics, which is
exactly the point: a controller evolved on the idealized environment has
to survive an interface it wasn't evolved against. See hexapod_bridge.py
for how a walker.py-trained genome's outputs get translated into this
environment's 12-torque action space.
"""

import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

XML_PATH = os.path.join(os.path.dirname(__file__), "hexapod.xml")

# Joint order matches walker.py's leg ordering (front_left, mid_left,
# back_left, back_right, mid_right, front_right), 2 co-located hinge joints (sweep, lift) per leg, both pivoting at the same point -- no separate hip/knee segments.
LEG_NAMES = ["front_left", "mid_left", "back_left", "back_right", "mid_right", "front_right"]

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.2,
}


class HexapodEnv(MujocoEnv, utils.EzPickle):
    """Six-legged MuJoCo hexapod, Gymnasium MujocoEnv interface.

    Observation (35,): torso z-height (1) + torso orientation quaternion
    (4) + all 12 sweep/lift joint angles + torso linear/angular velocity
    (6) + all 12 sweep/lift joint velocities.

    Action (12,): torque command in [-1, 1] for each sweep/lift joint, in
    LEG_NAMES x (sweep, lift) order -- NOT the same action space as
    walker.py's 18-value (fwd, bwd, foot) x 6 legs. See hexapod_bridge.py.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,  # 1 / (timestep=0.01 * frame_skip=5)
    }

    def __init__(self, xml_file=XML_PATH, forward_reward_weight=1.0,
                 ctrl_cost_weight=0.01, healthy_z_range=(0.06, 0.30),
                 reset_noise_scale=0.01, **kwargs):
        utils.EzPickle.__init__(
            self, xml_file, forward_reward_weight, ctrl_cost_weight,
            healthy_z_range, reset_noise_scale, **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        # torso z (1) + torso quat (4) + 12 joint angles + torso lin/ang vel (6) + 12 joint velocities = 35
        observation_space = Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64)

        MujocoEnv.__init__(
            self, xml_file, frame_skip=5, observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG, **kwargs,
        )

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        return min_z <= self.data.qpos[2] <= max_z

    def _get_obs(self):
        # qpos: [x, y, z, quat(4), 12 joint angles] = 19
        # qvel: [vx, vy, vz, wx, wy, wz, 12 joint velocities] = 18
        torso_z = self.data.qpos[2:3]
        torso_quat = self.data.qpos[3:7]
        joint_angles = self.data.qpos[7:19]
        torso_vel = self.data.qvel[0:6]
        joint_vels = self.data.qvel[6:18]
        return np.concatenate([torso_z, torso_quat, joint_angles, torso_vel, joint_vels]).astype(np.float64)

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        forward_reward = self._forward_reward_weight * (x_after - x_before) / self.dt
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        reward = forward_reward - ctrl_cost

        terminated = not self.is_healthy  # torso fell or flipped
        observation = self._get_obs()
        info = {
            "x_position": x_after,
            "y_position": self.data.qpos[1],
            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
