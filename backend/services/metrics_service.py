from sqlalchemy.orm import Session

from database.models import Job
from sqlalchemy import func


def get_jobs_processed(db: Session) -> int:

    return (
        db.query(Job)
        .filter(Job.status == "completed")
        .count()
    )


def get_jobs_running(db: Session) -> int:

    return (
        db.query(Job)
        .filter(Job.status == "running")
        .count()
    )


def get_jobs_queued(db: Session) -> int:

    return (
        db.query(Job)
        .filter(Job.status == "queued")
        .count()
    )


def get_ai_decisions(db: Session) -> int:
    return (
        db.query(Job)
        .filter(Job.assigned_gpu_id.isnot(None))
        .count()
    )


def get_cost_savings(db: Session) -> float:
    result = db.query(func.sum(Job.baseline_cost - Job.predicted_cost)).scalar()
    return float(result) if result else 0.0