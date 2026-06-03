from celery import Celery

from config import REDIS_URL

celery = Celery(
    "smartgpu",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker"],
)

celery.conf.update(
    broker_connection_retry_on_startup=True
)
