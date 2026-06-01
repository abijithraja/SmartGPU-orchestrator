from celery import Celery

from config import REDIS_URL

celery = Celery(
    "smartgpu",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker"],
)
