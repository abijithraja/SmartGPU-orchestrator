from fastapi import APIRouter
from services.gpu_service import get_gpu_status, get_gpu_by_id

router = APIRouter(prefix="/gpus", tags=["GPUs"])


@router.get("/")
def gpu_status():
    """Returns live GPU metrics for all nodes."""
    return get_gpu_status()
 
@router.get("/{gpu_id}")
def single_gpu(gpu_id: str):
    """Returns live metrics for a specific GPU node."""
    gpu = get_gpu_by_id(gpu_id)
    if not gpu:
        return {"error": "GPU not found"}
    return gpu
