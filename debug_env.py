import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from rl_engine.training.train_agent import SmartGPUEnv

env = SmartGPUEnv()
obs, info = env.reset()

print("Running 10 steps...")
for step in range(10):
    action = 0  # pick GPU-0
    obs, reward, done, trunc, info = env.step(action)
    if done:
        obs, info = env.reset()
