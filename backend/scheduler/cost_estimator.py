"""
Cost Estimator - Azure GPU pricing x duration.
Compares AI decision cost vs round-robin baseline.
"""

# Azure pay-as-you-go rates (USD/hour) as of 2025
GPU_RATES_USD_PER_HOUR = {
    "Standard_NC6": 0.90,
    "Standard_NC12": 1.80,
    "Standard_NC24": 3.60,
    "Standard_NC6s_v3": 3.06,
    "Standard_NC12s_v3": 6.12,
}

DEFAULT_RATE = 1.00


def estimate_cost(sku: str, duration_seconds: float) -> float:
    """Estimate cost in USD for a job running on a given GPU SKU."""
    rate = GPU_RATES_USD_PER_HOUR.get(sku, DEFAULT_RATE)
    return round((duration_seconds / 3600.0) * rate, 4)


def estimate_baseline_duration(memory_required: int, compute_intensity: float) -> float:
    """
    Estimate baseline (round-robin, worst-case) duration in seconds.
    Used as comparison point to measure AI speedup.
    """
    base = 120.0 + memory_required * 8.0 + compute_intensity * 180.0
    return round(base, 1)


def estimate_ai_duration(
    memory_required: int,
    compute_intensity: float,
    gpu_utilization: float,
) -> float:
    """
    Estimate AI-scheduled duration in seconds.
    Better GPU placement = faster completion.
    """
    baseline = estimate_baseline_duration(memory_required, compute_intensity)
    utilization_penalty = 1.0 + (gpu_utilization / 200.0)  # busy GPU = slower
    return round(baseline * utilization_penalty * 0.78, 1)  # ~22% avg speedup


def compute_savings(
    sku: str,
    ai_duration_s: float,
    baseline_duration_s: float,
) -> dict:
    ai_cost = estimate_cost(sku, ai_duration_s)
    baseline_cost = estimate_cost(sku, baseline_duration_s)
    savings = round(baseline_cost - ai_cost, 4)
    pct = round((savings / baseline_cost) * 100, 1) if baseline_cost > 0 else 0
    return {
        "ai_cost_usd": ai_cost,
        "baseline_cost_usd": baseline_cost,
        "savings_usd": savings,
        "savings_pct": pct,
    }
