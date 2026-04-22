import uuid

from sqlalchemy.orm import Session

from database.models import Job
from schemas.job_schema import JobCreate
from worker import process_job


def create_job(db: Session, job: JobCreate):
    db_job = Job(
        status="queued",
        model_name=job.model_name,
        memory_required=job.memory_required,
        priority=job.priority,
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    process_job.delay(str(db_job.id))

    return {"job_id": str(db_job.id), "status": db_job.status}


def get_job(db: Session, job_id: str):
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
        "details": {
            "model_name": db_job.model_name,
            "memory_required": db_job.memory_required,
            "priority": db_job.priority,
        },
        "created_at": db_job.created_at.isoformat() if db_job.created_at else None,
    }
