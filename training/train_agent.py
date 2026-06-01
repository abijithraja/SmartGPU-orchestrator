"""
SmartGPU - PPO Training Script
Run: python training/train_agent.py
Trains for 50,000 steps (~5 min on CPU).
Saves model to training/ppo_smartgpu.zip
"""
import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Allow imports from simulator/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.gpu_simulator import GPUCluster

N_GPUS = 4
OBS_DIM = 6 * N_GPUS   # 6 features per GPU
JOBS_PER_EPISODE = 200
SAVE_PATH = os.path.join(os.path.dirname(__file__), "ppo_smartgpu")


class SmartGPUEnv(gym.Env):
    """Custom Gymnasium environment for GPU scheduling."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.cluster = GPUCluster(n_gpus=N_GPUS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_GPUS)
        self._steps = 0
        self._job_memory = 4
        self._job_intensity = 0.5

    def _get_obs(self) -> np.ndarray:
        obs = []
        for gpu in self.cluster.gpus:
            obs.extend(gpu.to_state_vector(self._job_memory, self._job_intensity))
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cluster = GPUCluster(n_gpus=N_GPUS)
        self._steps = 0
        self._new_job()
        return self._get_obs(), {}

    def _new_job(self):
        self._job_memory = random.randint(2, 20)
        self._job_intensity = random.uniform(0.1, 1.0)

    def step(self, action: int):
        self._steps += 1
        gpu = self.cluster.gpus[action]

        # Compute reward
        oom = self._job_memory > gpu.free_memory
        thermal = gpu.temperature > 85
        saturated = gpu.utilization > 95

        if oom:
            reward = -2.0
        elif thermal or saturated:
            reward = -0.5
        else:
            # Positive reward proportional to how good this GPU is
            utilization_score = (100 - gpu.utilization) / 100
            memory_score = gpu.free_memory / gpu.total_memory
            reward = 0.5 * utilization_score + 0.5 * memory_score

        # Simulate job effect
        if not oom:
            gpu.assign_job(self._job_memory, self._job_intensity)

        # Advance simulation
        self.cluster.step_all()
        self._new_job()

        terminated = self._steps >= JOBS_PER_EPISODE
        return self._get_obs(), reward, terminated, False, {}


def train():
    print("Checking environment...")
    env = SmartGPUEnv()
    check_env(env, warn=True)

    print("Training PPO agent for 50,000 steps...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=50_000)
    model.save(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}.zip")


if __name__ == "__main__":
    train()
