"""
train_agent.py — PPO training agent with all bias corrections applied.

Bias fixes included:
  FIX 1 (L55-57)  : Clamp observations to [0, 1]
  FIX 2 (L93-120) : Unified reward expression (no dead variables)
  FIX 3 (L111-118): Load-balance bonus for least-loaded GPU
  FIX 4 (L148-152): Entropy coefficient to prevent policy collapse
"""

import numpy as np
from stable_baselines3 import PPO


class GPUSchedulerEnv:
    """
    GPU scheduling environment for the PPO agent.
    All four training-level bias corrections are applied here.
    """

    def __init__(self, cluster):
        self.cluster = cluster

    # ------------------------------------------------------------------
    # FIX 1: Observation clamping
    # ------------------------------------------------------------------
    def _get_obs(self):
        """
        Build the observation vector and clamp every value to [0, 1].

        Without clamping, transient spikes (e.g. utilisation briefly > 100
        under driver bugs, or temperature readings during sensor glitches)
        produce out-of-bounds inputs that skew policy-gradient updates and
        destabilise training.
        """
        obs = []
        for gpu in self.cluster.gpus:
            obs.extend([
                gpu.utilization / 100.0,
                (gpu.temperature - 35) / 65.0,
                gpu.queue_depth / 5.0,
                gpu.free_memory / gpu.total_memory,
            ])

        # FIX 1: Clamp observations strictly to [0, 1] to prevent
        # out-of-bounds values that skew policy gradient updates.
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    # ------------------------------------------------------------------
    # FIX 2 + FIX 3: Unified reward with load-balance bonus
    # ------------------------------------------------------------------
    def _compute_reward(self, gpu):
        """
        Single, unified reward expression.

        FIX 2 — Previously `queue_depth_penalty`, `temperature_penalty`,
        and `utilization_penalty` were computed but NEVER referenced in the
        final reward expression, so the agent effectively trained on a
        constant signal and learned to ignore queue/thermal state entirely.
        All factors are now combined into one expression.

        FIX 3 — Without a load-balance incentive the agent had no reason to
        spread work across GPUs and collapsed to always preferring GPU-0
        (the first index seen during rollout).  A +0.5 bonus for choosing
        the least-loaded GPU breaks this symmetry.
        """

        # Normalise each factor to [0, 1]
        util_norm  = gpu.utilization / 100.0        # 0 = idle,  1 = full
        temp_norm  = (gpu.temperature - 35) / 65.0  # 0 = cool,  1 = very hot
        queue_norm = gpu.queue_depth / 5.0           # 0 = empty, 1 = deep queue

        # Positive signal: prefer GPUs with headroom
        headroom_bonus = (1.0 - util_norm) * 2.0    # up to +2.0

        # Negative signals: penalise heat and queue buildup
        thermal_penalty = temp_norm * 1.0            # up to -1.0
        queue_penalty   = queue_norm * 1.5           # up to -1.5

        # FIX 3: Load-balancing bonus.
        # Reward the agent for choosing the GPU with the LOWEST utilisation
        # across the cluster. Without this, the agent had no incentive to
        # spread load and collapsed to always preferring GPU 0.
        utils = [g.utilization for g in self.cluster.gpus]
        min_util = min(utils)
        is_least_loaded = gpu.utilization == min_util
        balance_bonus = 0.5 if is_least_loaded else 0.0

        # FIX 2: All factors combined into one unified reward expression.
        reward = headroom_bonus - thermal_penalty - queue_penalty + balance_bonus
        return reward

    # ------------------------------------------------------------------
    # FIX 4: Entropy coefficient
    # ------------------------------------------------------------------
    def build_model(self, env):
        """
        Construct the PPO model with an explicit entropy coefficient.

        FIX 4 — The default ent_coef=0.0 allows the policy to collapse to a
        single deterministic action (always pick GPU 0) as soon as it finds
        any positive reward there.  Setting ent_coef=0.01 applies a small
        entropy bonus that keeps exploration alive long enough for the agent
        to discover that all GPU slots are viable, preventing preference
        collapse without meaningfully slowing convergence.
        """
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            # FIX 4: Entropy coefficient.
            # Default ent_coef=0.0 lets the policy collapse to a single action
            # (always pick GPU 0). A small value like 0.01 keeps exploration
            # alive long enough for the agent to learn all GPU slots are viable.
            ent_coef=0.01,
        )
        return model
