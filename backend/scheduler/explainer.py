"""
Explainability Engine - generates human-readable explanations for every
GPU scheduling decision. This is SmartGPU's key differentiator.
"""
from typing import List
from scheduler.rule_guard import get_rejection_reasons


def explain_decision(
    selected_gpu: dict,
    all_gpus: List[dict],
    job_memory: int,
    rl_scores: List[float],
    confidence: float,
    baseline_speedup_pct: float,
) -> str:
    """
    Returns a human-readable explanation of why a GPU was selected.
    Example output:
      GPU-2 selected - Free memory: 18.4 GB (highest available),
      Utilisation: 22% (lowest active node), Temperature: 41C (safe),
      Predicted completion 26% faster than round-robin.
      Confidence: 0.87. Rejected: GPU-0 (OOM risk: only 6.1 GB free vs 8 GB required).
    """
    gpu_id = selected_gpu["id"]
    free_mem = selected_gpu["free_memory"]
    util = selected_gpu["utilization"]
    temp = selected_gpu["temperature"]

    # Memory comparison
    all_free = [g["free_memory"] for g in all_gpus]
    mem_rank = "highest available" if free_mem == max(all_free) else f"{round(free_mem, 1)} GB free"

    # Utilization comparison
    all_util = [g["utilization"] for g in all_gpus]
    util_rank = "lowest active node" if util == min(all_util) else f"{round(util, 1)}%"

    # Temperature assessment
    if temp < 50:
        temp_note = "well within safe range"
    elif temp < 70:
        temp_note = "acceptable"
    else:
        temp_note = "warm but within limits"

    # Build primary reason string
    lines = [
        f"{gpu_id} selected",
        f"  - Free memory: {free_mem:.1f} GB ({mem_rank})",
        f"  - Utilisation: {util:.1f}% ({util_rank})",
        f"  - Temperature: {temp:.1f}C ({temp_note})",
        f"  - Predicted completion: {baseline_speedup_pct:.0f}% faster than round-robin baseline",
        f"  - Confidence score: {confidence:.2f}",
    ]

    # Rejection reasons for other GPUs
    rejections = get_rejection_reasons(job_memory, all_gpus)
    for other_gpu in all_gpus:
        gid = other_gpu["id"]
        if gid == gpu_id:
            continue
        if gid in rejections:
            reasons_str = "; ".join(rejections[gid])
            lines.append(f"  - Rejected {gid}: {reasons_str}")

    return "\n".join(lines)
