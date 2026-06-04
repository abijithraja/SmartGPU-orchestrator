"""
worker.py — Job dispatch worker with runtime bias corrections applied.

Bias fixes included:
  FIX 5 (L80-88)   : Hybrid override — force least-loaded GPU when agent picks >80% util
  FIX 6 (L222-225) : Repeated-assignment penalty (-5 × recent_count)
  FIX 7 (L227-230) : Cluster load-imbalance penalty (-std_dev of utilizations)
"""

import numpy as np
from backend.database.db import SessionLocal
from backend.database.models import Job


def dispatch_job(job, gpu_states, rl_agent):
    """
    Select a GPU for the given job using the RL agent, then apply
    runtime bias corrections before committing the assignment.

    Parameters
    ----------
    job        : Job ORM object with .memory_required attribute.
    gpu_states : list[dict] — current snapshot of each GPU's metrics.
    rl_agent   : trained PPO agent with a .predict() method.
    """

    # --- RL agent selection ---
    obs = _build_observation(gpu_states)
    action, _ = rl_agent.predict(obs, deterministic=True)
    selected_gpu = gpu_states[action]
    used_rl = True

    # ------------------------------------------------------------------
    # FIX 5: Hybrid Scheduler Override
    # ------------------------------------------------------------------
    # If the RL agent stubbornly picks a heavily loaded GPU, force
    # selection of the least loaded valid GPU to balance the cluster.
    # This acts as a hard safety net for the bias that survives training:
    # the agent may still prefer a familiar GPU even when it is saturated.
    if selected_gpu["utilization"] > 80:
        available_gpus = [
            g for g in gpu_states
            if g["free_memory"] >= job.memory_required
            and g["utilization"] < 80
        ]
        if available_gpus:
            selected_gpu = min(available_gpus, key=lambda g: g["utilization"])
            used_rl = False
            print(
                f"[WORKER] Hybrid Override: Forced least loaded GPU "
                f"{selected_gpu['id']} (agent picked overloaded GPU)"
            )

    # ------------------------------------------------------------------
    # Reward shaping — applied AFTER selection, fed back to training loop
    # ------------------------------------------------------------------
    reward = _base_reward(selected_gpu)

    # FIX 6: Repeated Assignment Penalty
    # If the agent keeps routing jobs to the same GPU, penalise proportionally
    # to how many of the last 10 jobs landed there. This breaks the
    # "favourite GPU" habit that persists even after training fixes.
    db = SessionLocal()
    try:
        recent_jobs = (
            db.query(Job)
            .filter(Job.assigned_gpu_id.isnot(None))
            .order_by(Job.created_at.desc())
            .limit(10)
            .all()
        )
    finally:
        db.close()
    recent_assignment_count = sum(
        1 for j in recent_jobs if j.assigned_gpu_id == selected_gpu["id"]
    )
    reward -= min(recent_assignment_count * 0.1, 0.5)   # max -0.5 instead of -50

    # FIX 7: Cluster Load-Imbalance Penalty
    # Penalise by the standard deviation of GPU utilizations across the
    # cluster. A perfectly balanced cluster has std_dev ≈ 0, so this term
    # is near zero when load is spread evenly and grows as skew increases,
    # giving the training loop a signal to seek balance.
    std_dev = np.std([g["utilization"] for g in gpu_states])
    reward -= (float(std_dev) / 100.0)

    return selected_gpu, reward, used_rl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_observation(gpu_states):
    """Flatten GPU metrics into a normalised observation vector."""
    obs = []
    for g in gpu_states:
        obs.extend([
            g["utilization"] / 100.0,
            (g["temperature"] - 35) / 65.0,
            g.get("queue_depth", 0) / 5.0,
            g["free_memory"] / max(g.get("total_memory", 1), 1),
        ])
    return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)


def _base_reward(gpu):
    """Minimal base reward before the bias-correction penalties are applied."""
    util_norm = gpu["utilization"] / 100.0
    return (1.0 - util_norm) * 2.0   # headroom bonus only; other factors in train_agent
