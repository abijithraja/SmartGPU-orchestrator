"""
Celery worker - simulates job execution and computes reward on completion.
"""
import time
import uuid
import logging
import random
from datetime import datetime

from celery_app import celery
from database.db import SessionLocal
from database.models import Job, RLExperience

logger = logging.getLogger(__name__)


@celery.task
def process_job(job_id: str):
    db = SessionLocal()
    try:
        parsed_job_id = uuid.UUID(job_id)
        job = db.query(Job).filter(Job.id == parsed_job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        logger.info(f"[WORKER] Processing job {job_id} on {job.assigned_gpu_id}")

        # Simulate job execution time based on memory + intensity
        base_duration = 10 + job.memory_required * 2 + job.compute_intensity * 30
        jitter = random.gauss(0, 2)
        duration = max(5, base_duration + jitter)
        time.sleep(min(duration, 60))   # cap at 60s for demo

        # Determine if OOM occurred (rare, <2% with good scheduling)
        oom = random.random() < 0.015

        # Update job record
        job.status = "failed" if oom else "completed"
        job.completed_at = datetime.utcnow()
        job.actual_duration_s = duration
        job.oom_occurred = oom

        # Compute actual cost
        from scheduler.cost_estimator import estimate_cost
        sku = job.assigned_gpu_sku or "Standard_NC6"
        job.actual_cost = estimate_cost(sku, duration)

        db.commit()

        # Compute and record reward in RLExperience
        exp = (
            db.query(RLExperience)
            .filter(RLExperience.job_id == parsed_job_id)
            .first()
        )
        if exp:
            baseline_s = exp.baseline_s or duration * 1.3
            speedup = (baseline_s - duration) / baseline_s if baseline_s > 0 else 0
            reward = speedup - (2.0 if oom else 0.0)
            exp.reward = round(reward, 4)
            exp.completion_s = duration
            db.commit()

        logger.info(
            f"[WORKER] Job {job_id} {'FAILED (OOM)' if oom else 'completed'} "
            f"in {duration:.1f}s"
        )

    except Exception as e:
        logger.error(f"[WORKER] Error processing job {job_id}: {e}")
        if db:
            try:
                job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
                if job:
                    job.status = "failed"
                    db.commit()
            except Exception:
                pass
    finally:
        db.close()
