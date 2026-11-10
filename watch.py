import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Pendulum-v1", render_mode="human")

model = PPO.load("models/ppo_baseline")

obs, info = env.reset()

for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()