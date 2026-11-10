import gymnasium as gym
from stable_baselines3 import PPO
import os

os.makedirs("models", exist_ok=True)

env = gym.make("Pendulum-v1")

model = PPO("MlpPolicy", env, verbose=1, seed=42)

model.learn(total_timesteps=20000)

model.save("models/ppo_baseline")

env.close()

print("Training completed! Model saved.")