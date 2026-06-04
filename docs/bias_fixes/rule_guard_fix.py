"""
rule_guard.py — Hard constraint filter acting as the final bias safety net.

Bias fix included:
  FIX 8 (continued): Rule-based guard that eliminates any GPU that violates
                     hard constraints regardless of its RL score, then picks
                     the highest-scoring valid GPU with an extra queue
                     tie-breaker.
"""


def select_valid_gpu(gpu_states: list, rl_scores: list, job_memory: float):
    """
    Filter GPU candidates by hard constraints and return the best one.

    Even after all training and score-level corrections, the agent might
    assign a job to a GPU that is thermally unsafe, out of memory, or
    queue-saturated.  This function is the last line of defence: it rejects
    any GPU that violates a hard rule and then picks the highest adjusted
    score among the survivors.

    Parameters
    ----------
    gpu_states : list[dict]  — current GPU metrics snapshot.
    rl_scores  : list[float] — post-adjustment scores from decision_engine.
    job_memory : float       — GB of memory the incoming job requires.

    Returns
    -------
    dict — the selected GPU state dict, or raises RuntimeError if no GPU
           passes the constraints.
    """
    valid = []

    for gpu, score in zip(gpu_states, rl_scores):
        reasons = []

        # Hard constraint checks — each appends a human-readable reason
        # so that failures are easy to diagnose in logs.
        if gpu.get("failed"):
            reasons.append("GPU offline")

        if job_memory > gpu["free_memory"]:
            reasons.append(
                f"OOM risk ({job_memory}GB required vs {gpu['free_memory']}GB free)"
            )

        if gpu["temperature"] > 85:
            reasons.append(
                f"thermal risk ({gpu['temperature']}°C > 85°C threshold)"
            )

        if gpu.get("queue_depth", 0) >= 2:
            reasons.append("GPU queue full (queue_depth ≥ 2)")

        if gpu["utilization"] > 95:
            reasons.append(
                f"GPU saturated ({gpu['utilization']}% utilisation)"
            )

        if reasons:
            # Log the rejection so operators can see which GPUs were
            # excluded and why — critical for debugging bias regressions.
            print(
                f"[RULE GUARD] GPU {gpu['id']} excluded: "
                + ", ".join(reasons)
            )
            continue

        # Apply an extra queue penalty at the score level so that among
        # equally valid GPUs the one with the shortest queue wins.
        adjusted_score = score - (gpu.get("queue_depth", 0) * 0.25)
        valid.append((gpu, adjusted_score))

    if not valid:
        raise RuntimeError(
            "[RULE GUARD] No valid GPU found for job — all candidates "
            "failed constraint checks. Check cluster health."
        )

    # Return the GPU with the highest adjusted RL score among valid candidates.
    return max(valid, key=lambda x: x[1])[0]
