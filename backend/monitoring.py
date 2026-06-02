from prometheus_client import Counter, Gauge

# =========================
# COUNTERS
# =========================

ai_decisions_total = Counter(
    "smartgpu_ai_decisions_total",
    "Total AI decisions made"
)

jobs_processed_total = Counter(
    "smartgpu_jobs_processed_total",
    "Total jobs processed"
)

# =========================
# JOB METRICS
# =========================

jobs_running_total = Gauge(
    "smartgpu_jobs_running_total",
    "Total running jobs"
)

jobs_queued_total = Gauge(
    "smartgpu_jobs_queued_total",
    "Total queued jobs"
)

job_confidence = Gauge(
    "smartgpu_job_confidence",
    "Latest AI confidence"
)

# =========================
# GPU METRICS
# =========================

gpu_utilization = Gauge(
    "smartgpu_gpu_utilization",
    "Current GPU Utilization",
    ["gpu"]
)

gpu_temperature = Gauge(
    "smartgpu_gpu_temperature",
    "Current GPU Temperature",
    ["gpu"]
)

gpu_free_memory = Gauge(
    "smartgpu_gpu_free_memory",
    "Current GPU Free Memory",
    ["gpu"]
)

gpu_queue_depth = Gauge(
    "smartgpu_gpu_queue_depth",
    "Current GPU Queue Depth",
    ["gpu"]
)

# =========================
# COST METRICS
# =========================

cost_savings_percent = Gauge(
    "smartgpu_cost_savings_percent",
    "Cost savings percent"
)

job_rl_cost = Gauge(
    "smartgpu_job_rl_cost_usd",
    "RL job cost"
)

job_baseline_cost = Gauge(
    "smartgpu_job_baseline_cost_usd",
    "Baseline cost"
)

cost_savings_usd = Gauge(
    "smartgpu_cost_savings_usd",
    "Savings in USD"
)

# =========================
# COUNTER HELPERS
# =========================

def record_ai_decision() -> None:
    ai_decisions_total.inc()
    print("AI DECISION RECORDED")


def record_job_processed() -> None:
    jobs_processed_total.inc()
    print("JOB PROCESSED")


# =========================
# JOB STATUS METRICS
# =========================

def update_jobs_metrics(
    processed: int,
    running: int,
    queued: int,
) -> None:

    jobs_running_total.set(running)
    jobs_queued_total.set(queued)


# =========================
# GPU METRICS UPDATE
# =========================

def update_gpu_metrics(gpu_states: list) -> None:

    for gpu in gpu_states:

        gid = gpu.get("id", "unknown")

        gpu_utilization.labels(
            gpu=gid
        ).set(
            gpu.get("utilization", 0)
        )

        gpu_temperature.labels(
            gpu=gid
        ).set(
            gpu.get("temperature", 0)
        )

        gpu_free_memory.labels(
            gpu=gid
        ).set(
            gpu.get("free_memory", 0)
        )

        gpu_queue_depth.labels(
            gpu=gid
        ).set(
            gpu.get("queue_depth", 0)
        )


# =========================
# COST SAVINGS & AI HELPERS
# =========================

def update_cost_savings(percent: float) -> None:
    cost_savings_percent.set(percent)


def update_job_confidence(confidence: float) -> None:
    job_confidence.set(confidence)


def update_job_costs(
    rl_cost: float,
    baseline_cost: float,
    savings_usd: float,
) -> None:
    job_rl_cost.set(rl_cost)
    job_baseline_cost.set(baseline_cost)
    cost_savings_usd.set(savings_usd)