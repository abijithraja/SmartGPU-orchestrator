from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.db import get_db
from schemas.job_schema import JobCreate, JobResponse
from services.job_service import (
    create_job,
    get_job,
    get_all_jobs,
    get_running_jobs,
    get_queued_jobs,
)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("/", response_model=JobResponse)
def submit_job(job: JobCreate, db: Session = Depends(get_db)):
    """Submit a new AI training job. Returns scheduling decision with explanation."""
    return create_job(db, job)


@router.get("/")
def list_jobs(limit: int = 50, db: Session = Depends(get_db)):
    """List recent jobs with costs and AI decisions."""
    return get_all_jobs(db, limit)


@router.get("/running")
def running_jobs(db: Session = Depends(get_db)):
    return get_running_jobs(db)


@router.get("/queued")
def queued_jobs(db: Session = Depends(get_db)):
    return get_queued_jobs(db)


@router.get("/{job_id}")
def job_status(job_id: str, db: Session = Depends(get_db)):
    """Get detailed status of a specific job."""
    return get_job(db, job_id)
