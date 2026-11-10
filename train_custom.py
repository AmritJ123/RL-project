import gymnasium as gym
from stable_baselines3 import PPO
from custom_reward import CustomRewardWrapper
import os

os.makedirs("models", exist_ok=True)

env = gym.make("Pendulum-v1")
env = CustomRewardWrapper(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42
)

model.learn(total_timesteps=20000)

model.save("models/ppo_custom")

env.close()

print("Custom reward training completed!")