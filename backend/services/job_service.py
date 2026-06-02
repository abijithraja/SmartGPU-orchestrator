import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from database.models import Job
from schemas.job_schema import JobCreate
from worker import process_job
from monitoring import record_ai_decision


def create_job(db: Session, job: JobCreate):
    """
    Create a new job and queue it for processing.
    """

    db_job = Job(
        status="queued",
        model_name=job.model_name,
        memory_required=job.memory_required,
        compute_intensity=job.compute_intensity,
        priority=job.priority,
    )

    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    # Record AI scheduling metric
    record_ai_decision()

    # Send job to Celery worker
    process_job.delay(str(db_job.id))

    return {
        "job_id": str(db_job.id),
        "status": db_job.status,
    }


def get_job(db: Session, job_id: str):
    """
    Get details of a specific job.
    """

    try:
        parsed_job_id = uuid.UUID(job_id)
    except ValueError:
        return {"error": "Invalid job id"}

    db_job = db.query(Job).filter(Job.id == parsed_job_id).first()

    if db_job is None:
        return {"error": "Job not found"}

    return {
        "job_id": str(db_job.id),
        "status": db_job.status,
        "assigned_gpu": db_job.assigned_gpu_id,
        "confidence": db_job.ai_confidence,
        "predicted_cost_usd": db_job.predicted_cost,
        "actual_cost_usd": db_job.actual_cost,
        "baseline_cost_usd": db_job.baseline_cost,
        "explanation": db_job.ai_explanation,
        "details": {
            "model_name": db_job.model_name,
            "memory_required": db_job.memory_required,
            "compute_intensity": db_job.compute_intensity,
            "priority": db_job.priority,
        },
        "created_at": (
            db_job.created_at.isoformat()
            if db_job.created_at
            else None
        ),
    }


def get_all_jobs(db: Session, limit: int = 50):
    """
    Return recent jobs.
    """

    jobs = (
        db.query(Job)
        .order_by(Job.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "job_id": str(job.id),
            "status": job.status,
            "model_name": job.model_name,
            "assigned_gpu": job.assigned_gpu_id,
            "confidence": job.ai_confidence,
            "actual_cost": job.actual_cost,
            "baseline_cost": job.baseline_cost,
            "cost_saved": (
                job.baseline_cost - job.actual_cost
                if job.baseline_cost and job.actual_cost
                else 0
            ),
            "explanation": job.ai_explanation,
            "created_at": (
                job.created_at.isoformat()
                if job.created_at
                else None
            ),
            "completed_at": (
                job.completed_at.isoformat()
                if job.completed_at
                else None
            ),
            "retry_count": job.retry_count or 0,
            "gpu_failed": job.gpu_failed or False,
            "failure_reason": job.failure_reason,
        }
        for job in jobs
    ]


def get_running_jobs(db: Session):

    jobs = (
        db.query(Job)
        .filter(Job.status == "running")
        .all()
    )

    return [
        {
            "job_id": str(job.id),
            "model_name": job.model_name,
            "assigned_gpu": job.assigned_gpu_id,
            "status": job.status,
            "progress": job.progress or 0,
        }
        for job in jobs
    ]


def get_queued_jobs(db: Session):

    jobs = (
        db.query(Job)
        .filter(Job.status == "queued")
        .all()
    )

    return [
        {
            "job_id": str(job.id),
            "model_name": job.model_name,
            "priority": job.priority,
            "status": job.status,
            "wait_time": int(
                (
                    datetime.utcnow()
                    - job.created_at
                ).total_seconds()
            ) if job.created_at else 0,
        }
        for job in jobs
    ]