from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.db import get_db
from database.models import Job

router = APIRouter(tags=["Cluster"])

@router.get("/metrics")
def cluster_metrics(db: Session = Depends(get_db)):
    # Get all active jobs (running or queued) and completed/failed stats
    active_jobs = db.query(Job).filter(Job.status == "running").all()
    queued_jobs = db.query(Job).filter(Job.status == "queued").all()
    
    completed_jobs = db.query(Job).filter(Job.status == "completed").count()
    failed_jobs = db.query(Job).filter(Job.status.in_(["failed", "dead"])).count()
    
    total_queued = len(queued_jobs)
    
    # Calculate Cluster Health
    total_jobs_finished = completed_jobs + failed_jobs
    success_rate = completed_jobs / max(1, total_jobs_finished)
    queue_score = max(0.0, 1.0 - (total_queued / 50.0))
    gpu_availability = 1.0 # Assuming all 4 GPUs are available
    
    cluster_health = int((gpu_availability * 0.4 + success_rate * 0.4 + queue_score * 0.2) * 100)
    
    # Group running jobs by GPU
    from collections import defaultdict
    gpu_to_jobs = defaultdict(list)
    for job in active_jobs:
        if job.assigned_gpu_id:
            gpu_to_jobs[job.assigned_gpu_id].append(job)
            
    # Compute GPU stats
    gpus = []
    gpu_definitions = [
        {"id": "gpu-0", "total_memory": 12},
        {"id": "gpu-1", "total_memory": 24},
        {"id": "gpu-2", "total_memory": 48},
        {"id": "gpu-3", "total_memory": 12},
    ]
    
    for i, gpu_def in enumerate(gpu_definitions):
        gpu_id = gpu_def["id"]
        total_memory = gpu_def["total_memory"]
        jobs_on_gpu = gpu_to_jobs.get(gpu_id, [])
        
        utilization = min(100.0, sum(job.compute_intensity * 100 for job in jobs_on_gpu))
        memory_used = min(float(total_memory), sum(job.memory_required for job in jobs_on_gpu))
        temperature = min(95.0, 35.0 + (utilization * 0.55))
        
        # Distribute queue length evenly
        queue_length = total_queued // 4 + (1 if i < total_queued % 4 else 0)
        
        gpus.append({
            "id": gpu_id,
            "utilization": round(utilization, 1),
            "memory_used": round(memory_used, 1),
            "memory_total": total_memory,
            "temperature": round(temperature, 1),
            "queue_length": queue_length,
            "failed": False
        })
        
    return {
        "cluster_health": cluster_health,
        "gpus": gpus
    }
