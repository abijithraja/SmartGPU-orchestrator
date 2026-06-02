"""
Migration Engine - Detects when a running job should be migrated
to a different GPU due to thermal or saturation issues.
"""


def should_migrate_job(
    gpu_util: float,
    gpu_temp: float,
) -> bool:
    """
    Returns True if the GPU is overloaded and the job
    should be migrated to another GPU.
    """
    return (
        gpu_util > 95
        or gpu_temp > 85
    )
