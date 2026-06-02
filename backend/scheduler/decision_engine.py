"""
Decision Engine - the core AI scheduling brain.
Uses PPO agent from Stable-Baselines3, with Rule Guard safety filter
and cold-start round-robin fallback.
"""
import os
import sys
import random
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Lazy-load the model (avoid import errors if SB3 not installed at boot)
_model = None
_model_loaded = False


def _load_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model
        
    from config import MODEL_PATH
    model_path = os.path.abspath(MODEL_PATH)
    
    try:
        from stable_baselines3 import PPO
        
        # Let SB3 handle the .zip extension and file loading
        _model = PPO.load(model_path)
        logger.warning(f"PPO MODEL LOADED: {model_path}")
        
    except Exception as e:
        logger.warning(f"PPO load failed at {model_path}: {e}. Using round-robin fallback.")
        _model = None
        
    _model_loaded = True
    return _model


def _build_obs(gpu_states: List[dict], job_memory: int, job_intensity: float) -> np.ndarray:
    """Build the observation vector the PPO agent expects."""
    obs = []
    total_mem = max(g.get("total_memory", 24) for g in gpu_states)
    for g in gpu_states:
        obs.extend([
            g["utilization"] / 100.0,
            g["free_memory"] / max(g.get("total_memory", 24), 1),
            g["temperature"] / 100.0,
            g.get("queue_depth", 0) / 10.0,
            job_memory / 80.0,
            job_intensity,
        ])
    # Pad or truncate to exactly 4 GPUs x 6 features = 24
    target_len = 4 * 6
    obs = obs[:target_len]
    while len(obs) < target_len:
        obs.extend([0.0] * 6)
    return np.array(obs, dtype=np.float32)


def _round_robin_select(gpu_states: List[dict], job_memory: int) -> Optional[dict]:
    """Fallback: pick the GPU with the most free memory (safe round-robin)."""
    valid = [g for g in gpu_states if g["free_memory"] >= job_memory]
    if not valid:
        return None
    return max(valid, key=lambda g: g["free_memory"])


def _get_rl_scores(
    gpu_states: List[dict],
    job_memory: int,
    job_intensity: float,
) -> List[float]:
    """
    Get a score per GPU from the PPO agent.
    Falls back to a heuristic score if model unavailable.
    """
    model = _load_model()
    if model is None:
        # Heuristic fallback scores
        return [
            (g["free_memory"] / max(g.get("total_memory", 24), 1)) * 0.6
            + (1 - g["utilization"] / 100) * 0.4
            for g in gpu_states
        ]

    obs = _build_obs(gpu_states, job_memory, job_intensity)
    # Get raw action probabilities from the policy network
    try:
        import torch
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.squeeze().numpy()
        return list(probs)
    except Exception:
        # If torch call fails, use predict
        action, _ = model.predict(obs, deterministic=True)
        scores = [0.1] * len(gpu_states)
        scores[int(action)] = 0.9
        return scores


def schedule_job(
    gpu_states: List[dict],
    job_memory: int,
    job_intensity: float,
    experience_count: int,
    cold_start_threshold: int,
) -> Tuple[Optional[dict], List[float], bool]:
    """
    Main scheduling entry point.
    Returns (selected_gpu, rl_scores, used_rl_agent).
    Returns (None, [], False) if no safe GPU available.
    """
    from scheduler.rule_guard import apply_rules

    # Cold start: fall back to round-robin until enough data
    if experience_count < cold_start_threshold:
        gpu = _round_robin_select(gpu_states, job_memory)
        scores = [0.0] * len(gpu_states)
        return gpu, scores, False

    # Get RL scores
    rl_scores = _get_rl_scores(gpu_states, job_memory, job_intensity)

    logger.warning(f"RL SCORES: {rl_scores}")

    # Apply rule guard (safety filter)
    result = apply_rules(job_memory, gpu_states, rl_scores)
    if result is None:
        return None, rl_scores, True   # Hold in queue

    selected_gpu, _ = result

    logger.warning("USED RL: True")
    logger.warning(f"SELECTED GPU: {selected_gpu['id']}")

    return selected_gpu, rl_scores, True


def get_confidence(rl_scores: List[float]) -> float:
    """Returns softmax confidence of the top action."""
    if not rl_scores or max(rl_scores) == 0:
        return 0.5
    scores = np.array(rl_scores)
    exp_scores = np.exp(scores - np.max(scores))
    softmax = exp_scores / exp_scores.sum()
    return round(float(np.max(softmax)), 3)