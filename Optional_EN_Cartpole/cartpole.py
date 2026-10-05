"""
CartPole environment wrapper for evolutionary neuroevolution.

This module provides a simplified CartPole interface that matches the
structure of the Braitenberg vehicle, making it easier to adapt evolution
code from phototaxis projects.
"""

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None


class CartPole:
    """
    A wrapper around Gymnasium's CartPole-v1 that provides a similar interface
    to the Braitenberg Vehicle for consistent code structure.

    The cartpole environment has 4 observations (cart position, cart velocity,
    pole angle, pole angular velocity) and 2 discrete actions (left/right).

    Args:
        render_mode: None, "human", or other gymnasium render modes
        max_episode_steps: Maximum steps before an episode is truncated
            (default 500, matching CartPole-v1's built-in limit). Lowering
            this shortens both training episodes and simulated episodes;
            it cannot be raised above what the underlying physics allows
            the pole to balance for, since termination on falling still
            applies regardless of this cap.
    """

    def __init__(self, render_mode=None, max_episode_steps=500):
        if gym is None:
            raise ImportError("Gymnasium not installed. Run: pip install gymnasium")
        self.env = gym.make(
            "CartPole-v1",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Reset the environment
        self.observation, _ = self.env.reset()

    def reset(self):
        """Reset the environment and return initial observation."""
        result = self.env.reset()
        # Gymnasium API compatibility: newer versions return (obs, info);
        # older versions returned just obs. Handle both so this wrapper
        # doesn't break if the installed Gymnasium version changes.
        if isinstance(result, tuple):
            self.observation = result[0]
            return self.observation, {}
        else:
            self.observation = result
            return result, {}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: 0 (left) or 1 (right)

        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.observation = obs
        return obs, float(reward), terminated, truncated, info

    def close(self):
        """Close the environment."""
        self.env.close()
