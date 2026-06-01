from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.db import get_db
from schemas.job_schema import JobCreate
from services.job_service import create_job, get_job

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("/")
def submit_job(job: JobCreate, db: Session = Depends(get_db)):
    return create_job(db, job)


@router.get("/{job_id}")
def job_status(job_id: str, db: Session = Depends(get_db)):
    return get_job(db, job_id)
