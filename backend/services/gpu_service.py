"""
GPU Service - returns live GPU state.
In development: uses the GPU simulator.
In production: queries Prometheus NVIDIA exporter.
"""

import os
import sys
from typing import List

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simulator.gpu_simulator import GPUCluster
from monitoring import update_gpu_metrics

# Singleton cluster for development/demo
_cluster = GPUCluster(n_gpus=4)


def get_gpu_status() -> List[dict]:
    """Returns current GPU state for all nodes."""
    _cluster.step_all()
    states = _cluster.get_states()
    update_gpu_metrics(states)
    return states


def get_gpu_by_id(gpu_id: str) -> dict:
    """Returns state of a specific GPU."""
    all_gpus = get_gpu_status()

    for gpu in all_gpus:
        if gpu["id"] == gpu_id:
            return gpu

    return {}