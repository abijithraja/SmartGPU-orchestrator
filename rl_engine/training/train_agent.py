"""
SmartGPU - PPO Training Script (Bias-Fixed)
Run: python training/train_agent.py
Trains for 200,000 steps (~20 min on CPU).
Saves model to training/ppo_smartgpu.zip

Fixes applied:
  1. Unified, consistent reward function (no dead/conflicting variables)
  2. Load-balancing bonus to prevent GPU preference collapse
  3. Entropy coefficient tuned to encourage exploration
  4. Normalized observation clamping for stability
"""
import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from simulator.gpu_simulator import GPUCluster

N_GPUS = 4
OBS_DIM = 6 * N_GPUS
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
        self.active_jobs = []

    def _get_obs(self) -> np.ndarray:
        obs = []
        for gpu in self.cluster.gpus:
            obs.extend(gpu.to_state_vector(self._job_memory, self._job_intensity))
        # FIX 1: Clamp observations strictly to [0, 1] to prevent
        # out-of-bounds values that skew policy gradient updates.
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cluster = GPUCluster(n_gpus=N_GPUS)

        for gpu in self.cluster.gpus:
            gpu.utilization = random.randint(10, 60)
            gpu.temperature = random.randint(40, 70)
            gpu.queue_depth = random.randint(0, 2)

        self._steps = 0
        self.active_jobs = []
        self._new_job()
        return self._get_obs(), {}

    def _new_job(self):
        self._job_memory = random.randint(2, 20)
        self._job_intensity = random.uniform(0.1, 1.0)

    def step(self, action: int):
        self._steps += 1
        gpu = self.cluster.gpus[action]

        oom = self._job_memory > gpu.free_memory
        thermal = gpu.temperature > 85
        saturated = gpu.utilization > 95

        if oom:
            # Hard penalty: out-of-memory is always wrong
            reward = -2.0

        elif thermal or saturated:
            # Soft penalty: GPU is stressed — discourage but don't forbid
            reward = -0.5

        else:
            # Normalise each factor to [0, 1] range
            util_norm  = gpu.utilization / 100.0        # 0 = idle, 1 = full
            temp_norm  = (gpu.temperature - 35) / 65.0  # 0 = cool, 1 = very hot
            queue_norm = gpu.queue_depth / 5.0          # 0 = empty, 1 = deep queue
            # Positive signal: prefer GPUs with headroom
            headroom_bonus = (1.0 - util_norm) * 2.0   # up to +2.0

            # Negative signals: penalise heat and queue buildup
            thermal_penalty = temp_norm * 1.0           # up to -1.0
            queue_penalty   = queue_norm * 1.5          # up to -1.5

            # Load-balancing bonus
            utils = [g.utilization for g in self.cluster.gpus]
            min_util = min(utils)
            is_least_loaded = gpu.utilization == min_util
            balance_bonus = 0.5 if is_least_loaded else 0.0

            reward = headroom_bonus - thermal_penalty - queue_penalty + balance_bonus

        # Simulate job effect
        if not oom:
            gpu.assign_job(self._job_memory, self._job_intensity)
            self.active_jobs.append({
                "gpu_idx": action,
                "memory": self._job_memory,
                "ticks_left": random.randint(5, 15)
            })

        still_active = []
        for job in self.active_jobs:
            job["ticks_left"] -= 1
            if job["ticks_left"] <= 0:
                self.cluster.gpus[job["gpu_idx"]].complete_job(job["memory"])
            else:
                still_active.append(job)
        self.active_jobs = still_active

        self.cluster.step_all()
        self._new_job()

        terminated = self._steps >= JOBS_PER_EPISODE
        return self._get_obs(), reward, terminated, False, {}


def train():
    print("Checking environment...")
    env = SmartGPUEnv()
    check_env(env, warn=True)

    print("Training PPO agent for 200,000 steps...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        # FIX 4: Entropy coefficient.
        # Default ent_coef=0.0 lets the policy collapse to a single action
        # (always pick GPU 0). A small value like 0.01 keeps exploration alive
        # long enough for the agent to learn all GPU slots are viable.
        ent_coef=0.01,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=200_000)
    model.save(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}.zip")


if __name__ == "__main__":
    train()