import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

baseline = PPO.load("models/ppo_baseline")
custom = PPO.load("models/ppo_custom")


def evaluate(model, seeds=range(10)):
    rewards = []
    torque_costs = []
    upright_percentages = []

    for seed in seeds:
        env = gym.make("Pendulum-v1")
        obs, info = env.reset(seed=seed)

        total_reward = 0
        total_torque = 0
        upright_steps = 0
        steps = 0

        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            theta = np.arctan2(obs[1], obs[0])
            angular_velocity = obs[2]

            total_reward += reward
            total_torque += float(action[0] ** 2)

            if abs(theta) < 0.2 and abs(angular_velocity) < 1.0:
                upright_steps += 1

            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        torque_costs.append(total_torque / steps)
        upright_percentages.append(100 * upright_steps / steps)

        env.close()

    return (
        np.mean(rewards),
        np.mean(torque_costs),
        np.mean(upright_percentages)
    )


baseline_results = evaluate(baseline)
custom_results = evaluate(custom)

print("\nBASELINE PPO")
print(f"Average reward: {baseline_results[0]:.2f}")
print(f"Mean squared torque: {baseline_results[1]:.4f}")
print(f"Upright percentage: {baseline_results[2]:.2f}%")

print("\nCUSTOM REWARD PPO")
print(f"Average reward: {custom_results[0]:.2f}")
print(f"Mean squared torque: {custom_results[1]:.4f}")
print(f"Upright percentage: {custom_results[2]:.2f}%")