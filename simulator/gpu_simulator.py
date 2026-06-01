"""
SmartGPU - Realistic GPU Simulator
Generates lifelike GPU metrics for RL training and offline demo.
No Azure credits required.
"""
import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulatedGPU:
    id: str
    name: str
    total_memory: int         # GB
    sku: str = "Standard_NC6"  # Azure SKU for cost estimation

    # Live state
    utilization: float = 0.0       # 0-100%
    free_memory: float = 0.0       # GB
    temperature: float = 35.0      # C
    queue_depth: int = 0           # waiting jobs
    active_job_load: float = 0.0   # extra util from running job

    # Internals
    _temp_lag: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.free_memory = float(self.total_memory)
        self._temp_lag = [35.0, 35.0]

    def assign_job(self, memory_required: int, compute_intensity: float):
        """Called when a job is dispatched to this GPU."""
        self.free_memory = max(0, self.free_memory - memory_required)
        self.active_job_load = compute_intensity * 80.0   # 0-80 util points
        self.queue_depth += 1

    def complete_job(self, memory_required: int):
        """Called when a job finishes."""
        self.free_memory = min(self.total_memory, self.free_memory + memory_required)
        self.active_job_load = max(0, self.active_job_load - 80.0)
        self.queue_depth = max(0, self.queue_depth - 1)

    def step(self):
        """Advance one 15-second simulation tick."""
        # Base utilization: random idle + active job load
        base_idle = random.gauss(15, 5)
        self.utilization = min(100, max(0, base_idle + self.active_job_load + random.gauss(0, 3)))

        # Temperature lags utilization by 2 ticks (thermal inertia)
        self._temp_lag.append(self.utilization)
        lagged_util = self._temp_lag.pop(0)
        target_temp = 35 + (lagged_util / 100) * 55   # 35-90C range
        self.temperature = 0.85 * self.temperature + 0.15 * target_temp + random.gauss(0, 0.5)
        self.temperature = max(30, min(95, self.temperature))

    def to_state_vector(self, job_memory: int = 0, job_intensity: float = 0.5) -> List[float]:
        """Returns normalized feature vector for the RL agent."""
        return [
            self.utilization / 100.0,
            self.free_memory / self.total_memory,
            self.temperature / 100.0,
            self.queue_depth / 10.0,
            job_memory / 32.0,
            job_intensity,
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "sku": self.sku,
            "utilization": round(self.utilization, 1),
            "free_memory": round(self.free_memory, 1),
            "total_memory": self.total_memory,
            "temperature": round(self.temperature, 1),
            "queue_depth": self.queue_depth,
        }


class GPUCluster:
    """A simulated cluster of N GPUs."""

    def __init__(self, n_gpus: int = 4):
        skus = ["Standard_NC6", "Standard_NC12", "Standard_NC24", "Standard_NC6"]
        memories = [12, 24, 48, 12]
        self.gpus: List[SimulatedGPU] = [
            SimulatedGPU(
                id=f"gpu-{i}",
                name=f"GPU Node {i}",
                total_memory=memories[i % len(memories)],
                sku=skus[i % len(skus)],
            )
            for i in range(n_gpus)
        ]

    def step_all(self):
        for gpu in self.gpus:
            gpu.step()

    def get_states(self) -> List[dict]:
        return [g.to_dict() for g in self.gpus]
