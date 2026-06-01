"""
Job Service - orchestrates the full scheduling pipeline.
"""
import uuid
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from database.models import Job, RLExperience
from schemas.job_schema import JobCreate, JobResponse
from services.gpu_service import get_gpu_status
from scheduler.decision_engine import schedule_job, get_confidence
from scheduler.explainer import explain_decision
from scheduler.cost_estimator import (
    estimate_baseline_duration,
    estimate_ai_duration,
    compute_savings,
)
from config import COLD_START_THRESHOLD

logger = logging.getLogger(__name__)


def get_experience_count(db: Session) -> int:
    return db.query(RLExperience).count()


def create_job(db: Session, job: JobCreate) -> JobResponse:
    # 1. Fetch live GPU metrics
    gpu_states = get_gpu_status()

    # 2. Check cold-start status
    exp_count = get_experience_count(db)

    # 3. Run scheduling pipeline
    selected_gpu, rl_scores, used_rl = schedule_job(
        gpu_states=gpu_states,
        job_memory=job.memory_required,
        job_intensity=job.compute_intensity,
        experience_count=exp_count,
        cold_start_threshold=COLD_START_THRESHOLD,
    )

    if selected_gpu is None:
        # No safe GPU available - queue the job but do not assign yet
        db_job = Job(
            status="queued",
            model_name=job.model_name,
            memory_required=job.memory_required,
            compute_intensity=job.compute_intensity,
            priority=job.priority,
            ai_explanation="No safe GPU available. Job queued - will retry in 60s.",
        )
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return JobResponse(
            job_id=str(db_job.id),
            status="queued",
            explanation=db_job.ai_explanation,
        )

    # 4. Calculate costs and explanation
    confidence = get_confidence(rl_scores) if used_rl else 0.5
    baseline_duration = estimate_baseline_duration(job.memory_required, job.compute_intensity)
    ai_duration = estimate_ai_duration(
        job.memory_required, job.compute_intensity, selected_gpu["utilization"]
    )
    costs = compute_savings(selected_gpu.get("sku", "Standard_NC6"), ai_duration, baseline_duration)
    speedup_pct = costs["savings_pct"]

    explanation = explain_decision(
        selected_gpu=selected_gpu,
        all_gpus=gpu_states,
        job_memory=job.memory_required,
        rl_scores=rl_scores,
        confidence=confidence,
        baseline_speedup_pct=speedup_pct,
    )

    # 5. Save job to DB
    db_job = Job(
        status="running",
        model_name=job.model_name,
        memory_required=job.memory_required,
        compute_intensity=job.compute_intensity,
        priority=job.priority,
        assigned_gpu_id=selected_gpu["id"],
        assigned_gpu_sku=selected_gpu.get("sku", "Standard_NC6"),
        ai_explanation=explanation,
        ai_confidence=confidence,
        predicted_cost=costs["ai_cost_usd"],
        baseline_cost=costs["baseline_cost_usd"],
        baseline_duration_s=baseline_duration,
        started_at=datetime.utcnow(),
    )
    db.add(db_job)
    db.flush()

    # 6. Log experience to RL table (reward filled on completion)
    exp = RLExperience(
        state_json={
            "gpu_states": gpu_states,
            "job_memory": job.memory_required,
            "job_intensity": job.compute_intensity,
        },
        action_gpu_id=selected_gpu["id"],
        action_score=confidence,
        baseline_s=baseline_duration,
        job_id=db_job.id,
    )
    db.add(exp)
    db.commit()
    db.refresh(db_job)

    # 7. Dispatch Celery task
    from worker import process_job
    process_job.delay(str(db_job.id))

    return JobResponse(
        job_id=str(db_job.id),
        status=db_job.status,
        assigned_gpu=selected_gpu["id"],
        explanation=explanation,
        confidence=confidence,
        predicted_cost_usd=costs["ai_cost_usd"],
        baseline_cost_usd=costs["baseline_cost_usd"],
    )


def get_job(db: Session, job_id: str) -> dict:
    try:
        parsed = uuid.UUID(job_id)
    except ValueError:
        return {"error": "Invalid job ID"}

    job = db.query(Job).filter(Job.id == parsed).first()
    if not job:
        return {"error": "Job not found"}

    return {
        "job_id": str(job.id),
        "status": job.status,
        "model_name": job.model_name,
        "assigned_gpu": job.assigned_gpu_id,
        "ai_explanation": job.ai_explanation,
        "confidence": job.ai_confidence,
        "costs": {
            "predicted_usd": job.predicted_cost,
            "baseline_usd": job.baseline_cost,
            "actual_usd": job.actual_cost,
            "savings_usd": (
                round(job.baseline_cost - job.actual_cost, 4)
                if job.baseline_cost and job.actual_cost
                else None
            ),
        },
        "timing": {
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration_s": job.actual_duration_s,
        },
        "retry_count": job.retry_count,
        "oom_occurred": job.oom_occurred,
    }


def get_all_jobs(db: Session, limit: int = 50) -> list:
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
    return [
        {
            "job_id": str(j.id),
            "status": j.status,
            "model_name": j.model_name,
            "assigned_gpu": j.assigned_gpu_id,
            "confidence": j.ai_confidence,
            "predicted_cost_usd": j.predicted_cost,
            "baseline_cost_usd": j.baseline_cost,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]
