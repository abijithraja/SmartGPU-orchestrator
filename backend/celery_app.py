from celery import Celery
from celery.schedules import timedelta

from config import REDIS_URL

celery = Celery(
    "smartgpu",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker", "scheduler.recovery_monitor"],
)

celery.conf.beat_schedule = {
    "check-running-jobs-every-30s": {
        "task": "scheduler.recovery_monitor.check_running_jobs",
        "schedule": timedelta(seconds=30),
    },
}

celery.conf.timezone = "UTC"
