import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from rl_engine.training.train_agent import SmartGPUEnv

model_path = os.path.join(PROJECT_ROOT, "rl_engine", "training", "ppo_smartgpu")
model = PPO.load(model_path)
env = SmartGPUEnv()
obs, info = env.reset()

print("Sanity Check: Running 20 steps with trained model...")
for i in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(int(action))
    print(f"Step {i:2d} | GPU selected: {action} | reward: {reward:.2f}")
    if done:
        obs, info = env.reset()
