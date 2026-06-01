from typing import Optional
from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    memory_required: int = Field(..., ge=1, le=80, description="GPU memory in GB")
    compute_intensity: float = Field(0.5, ge=0.0, le=1.0)
    priority: str = Field("normal", pattern="^(low|normal|high)$")


class JobResponse(BaseModel):
    job_id: str
    status: str
    assigned_gpu: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    predicted_cost_usd: Optional[float] = None
    baseline_cost_usd: Optional[float] = None

    class Config:
        from_attributes = True
