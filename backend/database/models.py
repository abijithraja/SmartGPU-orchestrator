import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Column, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from database.db import Base


class Job(Base):
	__tablename__ = "jobs"

	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	status = Column(String, default="queued")

	model_name = Column(String)
	memory_required = Column(Integer)
	priority = Column(String)

	created_at = Column(TIMESTAMP, default=datetime.utcnow)
