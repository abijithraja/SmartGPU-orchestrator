"""
GPU Service - returns live GPU state.
In development: uses the GPU simulator.
In production: queries Prometheus NVIDIA exporter.
"""
import os
from typing import List

# Import simulator
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from simulator.gpu_simulator import GPUCluster

# Singleton cluster for development/demo
_cluster = GPUCluster(n_gpus=4)


def get_gpu_status() -> List[dict]:
    """Returns current GPU state for all nodes."""
    _cluster.step_all()  # Advance simulation tick
    return _cluster.get_states()


def get_gpu_by_id(gpu_id: str) -> dict:
    """Returns state of a specific GPU."""
    all_gpus = get_gpu_status()
    for gpu in all_gpus:
        if gpu["id"] == gpu_id:
            return gpu
    return {}
