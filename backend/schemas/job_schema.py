from pydantic import BaseModel


class JobCreate(BaseModel):
    model_name: str
    memory_required: int
    priority: str
