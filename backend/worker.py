import time
import uuid

from celery_app import celery
from database.db import SessionLocal
from database.models import Job
from monitoring import record_job_processed


@celery.task
def process_job(job_id: str):
    db = SessionLocal()
    try:
        try:
            parsed_job_id = uuid.UUID(job_id)
        except ValueError:
            return

        job = db.query(Job).filter(Job.id == parsed_job_id).first()
        if not job:
            return

        job.status = "running"
        db.commit()

        print(f"[WORKER] Running job {job_id}")

        time.sleep(5)

        job.status = "completed"
        db.commit()

        record_job_processed()

        print(f"[WORKER] Completed job {job_id}")
    finally:
        db.close()
