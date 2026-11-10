import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, old_reward, terminated, truncated, info = self.env.step(action)

        cos_theta = obs[0]
        sin_theta = obs[1]
        angular_velocity = obs[2]

        theta = np.arctan2(sin_theta, cos_theta)
        torque = float(np.asarray(action).item())

        new_reward = -(
            2.0 * theta**2
            + 0.2 * angular_velocity**2
            + 0.5 * torque**2
        )

        return obs, float(new_reward), terminated, truncated, info