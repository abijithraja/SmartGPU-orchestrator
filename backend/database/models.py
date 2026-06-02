import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Boolean, Column, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from database.db import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String, default="queued")   # queued|running|completed|failed|dead

    model_name = Column(String, nullable=False)
    memory_required = Column(Integer, nullable=False)   # GB
    compute_intensity = Column(Float, default=0.5)      # 0-1
    priority = Column(String, default="normal")         # low|normal|high

    assigned_gpu_id = Column(String, nullable=True)
    assigned_gpu_sku = Column(String, nullable=True)

    # AI decision details
    ai_explanation = Column(Text, nullable=True)
    ai_confidence = Column(Float, nullable=True)

    # Cost tracking
    predicted_cost = Column(Float, nullable=True)       # USD
    baseline_cost = Column(Float, nullable=True)        # round-robin USD
    actual_cost = Column(Float, nullable=True)

    # Timing
    started_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)
    actual_duration_s = Column(Float, nullable=True)
    baseline_duration_s = Column(Float, nullable=True)

    # Recovery
    retry_count = Column(Integer, default=0)
    oom_occurred = Column(Boolean, default=False)

    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class RLExperience(Base):
    __tablename__ = "rl_experience"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    state_json = Column(JSONB)                  # GPU metrics + job requirements
    action_gpu_id = Column(String)             # Which GPU was selected
    action_score = Column(Float)               # Confidence from model
    reward = Column(Float, nullable=True)      # Filled on job completion
    completion_s = Column(Float, nullable=True)
    baseline_s = Column(Float, nullable=True)
    oom_occurred = Column(Boolean, default=False)
    retrain_used = Column(Boolean, default=False)
    job_id = Column(UUID(as_uuid=True), nullable=True)