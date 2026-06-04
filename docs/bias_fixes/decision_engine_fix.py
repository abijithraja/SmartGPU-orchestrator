"""
decision_engine.py — RL score post-processing with bias corrections applied.

Bias fix included:
  FIX 8 (L167-186): Queue / utilisation / temperature penalties subtracted
                    from raw RL scores before GPU selection.
"""


def adjust_rl_scores(gpu_states, rl_scores, real_queue: dict) -> list:
    """
    Apply score-level bias corrections to raw RL scores.

    The PPO model outputs a raw preference score per GPU, but those scores
    can still reflect residual training bias (e.g. a slight preference for
    GPU-0 or for low-index GPUs).  Subtracting explicit penalties for queue
    depth, utilisation, and temperature aligns the final ranking with actual
    cluster health, regardless of whatever bias the model internalized.

    Parameters
    ----------
    gpu_states : list[dict]  — current GPU metrics snapshot.
    rl_scores  : list[float] — raw scores produced by the RL agent,
                               one per GPU (same order as gpu_states).
    real_queue : dict        — {gpu_id: queue_depth} from the database,
                               used to break ties when the in-memory
                               queue_depth is stale or unavailable.

    Returns
    -------
    list[float] — adjusted scores, same length and order as rl_scores.
    """
    adjusted = list(rl_scores)  # work on a copy; don't mutate the input

    # ------------------------------------------------------------------
    # FIX 8: Score-level bias correction
    # ------------------------------------------------------------------
    # Subtract queue, utilisation, and temperature penalties from the raw
    # RL score so that even a biased agent ends up selecting the healthiest
    # GPU after post-processing.
    for i, gpu in enumerate(gpu_states):
        # Use the real DB queue depth when available to stay consistent
        # across workers and to break ties more accurately.
        real_q_depth = real_queue.get(gpu["id"], 0)

        queue_penalty = (
            max(gpu.get("queue_depth", 0), real_q_depth) * 0.20
        )
        util_penalty = (
            gpu["utilization"] / 100
        ) * 0.80
        temp_penalty = (
            gpu["temperature"] / 100
        ) * 0.40

        adjusted[i] -= (queue_penalty + util_penalty + temp_penalty)

    return adjusted
