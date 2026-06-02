import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from celery_app import celery
from backend.training.retrain import retrain


@celery.task
def retrain_rl_model():
    """Celery task to retrain PPO model in the background."""
    retrain()
