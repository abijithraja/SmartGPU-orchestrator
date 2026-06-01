from fastapi import APIRouter

from services.gpu_service import get_gpu_status

router = APIRouter(prefix="/gpus", tags=["GPUs"])


@router.get("/")
def gpu_status():
    return get_gpu_status()
