"""
Rule Guard - Hard safety filter between PPO scores and final decision.
The agent is probabilistic; the Rule Guard provides correctness guarantees.
"""
from typing import List, Optional, Tuple


def apply_rules(
    job_memory: int,
    gpu_states: List[dict],
    rl_scores: List[float],
) -> Optional[Tuple[dict, float]]:
    """
    Filter GPU candidates that violate physical constraints.
    Returns (best_gpu, score) or None if all GPUs are unsafe (hold job).
    """
    valid = []
    for gpu, score in zip(gpu_states, rl_scores):
        reasons = []

        if job_memory > gpu["free_memory"]:
            reasons.append(f"OOM risk ({job_memory}GB req vs {gpu['free_memory']}GB free)")
        if gpu["temperature"] > 85:
            reasons.append(f"thermal risk ({gpu['temperature']}C > 85C)")
        if gpu["utilization"] > 95:
            reasons.append(f"saturated ({gpu['utilization']}% util)")

        if not reasons:
            valid.append((gpu, score))

    if not valid:
        return None  # All GPUs unsafe - caller must hold job and retry

    # Return the GPU with the highest RL score among valid candidates
    return max(valid, key=lambda x: x[1])


def get_rejection_reasons(job_memory: int, gpu_states: List[dict]) -> dict:
    """Returns per-GPU rejection reasons for the explainability engine."""
    reasons = {}
    for gpu in gpu_states:
        r = []
        if job_memory > gpu["free_memory"]:
            r.append(f"OOM risk: only {gpu['free_memory']:.1f}GB free vs {job_memory}GB required")
        if gpu["temperature"] > 85:
            r.append(f"thermal risk: {gpu['temperature']:.1f}C")
        if gpu["utilization"] > 95:
            r.append(f"saturated: {gpu['utilization']:.1f}% utilization")
        if r:
            reasons[gpu["id"]] = r
    return reasons
