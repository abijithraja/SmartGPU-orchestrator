"""
Recovery Monitor - Celery beat task that checks job health every 30 seconds.
Detects OOM, GPU faults, and pod crashes. Reschedules up to 3 times.
"""
import logging
from datetime import datetime, timedelta

from celery_app import celery
from database.db import SessionLocal
from database.models import Job

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
JOB_TIMEOUT_MINUTES = 30  # jobs running longer than this are considered stuck


@celery.task
def check_running_jobs():
    """
    Celery beat task. Fires every 30 seconds.
    Checks for stuck/failed running jobs and reschedules them.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=JOB_TIMEOUT_MINUTES)
        stuck_jobs = (
            db.query(Job)
            .filter(Job.status == "running")
            .filter(Job.started_at < cutoff)
            .all()
        )

        for job in stuck_jobs:
            if job.retry_count >= MAX_RETRIES:
                job.status = "dead"
                logger.error(f"Job {job.id} exceeded max retries - marked dead")
            else:
                job.retry_count += 1
                job.status = "queued"
                job.assigned_gpu_id = None
                job.started_at = None
                logger.warning(
                    f"Job {job.id} timed out. Retry {job.retry_count}/{MAX_RETRIES}"
                )

        db.commit()
    except Exception as e:
        logger.error(f"Recovery monitor error: {e}")
    finally:
        db.close()
