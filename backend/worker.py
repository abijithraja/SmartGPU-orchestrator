import time
import uuid
from datetime import datetime

from celery_app import celery

from config import COLD_START_THRESHOLD

from database.db import SessionLocal
from database.models import Job, RLExperience

from services.gpu_service import (
    get_gpu_status,
    assign_job_to_gpu,
    complete_job_on_gpu,
)

from scheduler.decision_engine import (
    schedule_job,
    get_confidence,
)

from scheduler.cost_estimator import (
    estimate_ai_duration,
    estimate_baseline_duration,
    compute_savings,
)

from monitoring import (
    record_job_processed,
    update_job_confidence,
    update_job_costs,
)


@celery.task
def process_job(job_id: str):

    db = SessionLocal()

    try:

        parsed_job_id = uuid.UUID(job_id)

        job = (
            db.query(Job)
            .filter(Job.id == parsed_job_id)
            .first()
        )

        if not job:
            return

        job.status = "running"
        job.started_at = datetime.utcnow()

        db.commit()

        gpu_states = get_gpu_status()

        experience_count = (
            db.query(RLExperience)
            .count()
        )

        selected_gpu, scores, used_rl = schedule_job(
            gpu_states=gpu_states,
            job_memory=job.memory_required,
            job_intensity=job.compute_intensity,
            experience_count=experience_count,
            cold_start_threshold=COLD_START_THRESHOLD,
        )

        if selected_gpu is None:

            job.status = "queued"

            db.commit()

            return

        confidence = get_confidence(scores)

        assign_job_to_gpu(
            selected_gpu["id"],
            job.memory_required,
            job.compute_intensity,
        )

        ai_duration = estimate_ai_duration(
            job.memory_required,
            job.compute_intensity,
            selected_gpu["utilization"],
        )

        baseline_duration = estimate_baseline_duration(
            job.memory_required,
            job.compute_intensity,
        )

        cost_data = compute_savings(
            selected_gpu["sku"],
            ai_duration,
            baseline_duration,
        )

        job.assigned_gpu_id = selected_gpu["id"]
        job.assigned_gpu_sku = selected_gpu["sku"]

        job.ai_confidence = confidence

        job.actual_duration_s = ai_duration
        job.baseline_duration_s = baseline_duration

        job.actual_cost = cost_data["ai_cost_usd"]
        job.baseline_cost = cost_data["baseline_cost_usd"]

        job.predicted_cost = cost_data["ai_cost_usd"]

        job.ai_explanation = (
            f"Selected {selected_gpu['id']} "
            f"using {'PPO' if used_rl else 'Round Robin'} "
            f"(confidence {confidence})"
        )

        db.commit()

        experience = RLExperience(
            state_json={
                "gpu_states": gpu_states
            },
            action_gpu_id=selected_gpu["id"],
            action_score=confidence,
            completion_s=ai_duration,
            baseline_s=baseline_duration,
            retrain_used=used_rl,
            job_id=job.id,
        )

        db.add(experience)

        db.commit()

        update_job_confidence(
            confidence
        )

        update_job_costs(
            cost_data["ai_cost_usd"],
            cost_data["baseline_cost_usd"],
            cost_data["savings_usd"],
        )

        print(
            f"[WORKER] Assigned "
            f"{job.id} -> {selected_gpu['id']}"
        )

        time.sleep(5)

        complete_job_on_gpu(
            selected_gpu["id"],
            job.memory_required,
        )

        job.status = "completed"

        job.completed_at = datetime.utcnow()

        db.commit()

        record_job_processed()

        print(
            f"[WORKER] Completed {job.id}"
        )

    finally:
        db.close()