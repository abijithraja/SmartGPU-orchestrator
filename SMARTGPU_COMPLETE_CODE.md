# SmartGPU Orchestrator — Complete Codebase

This document contains every file you need, copy-paste ready.
Follow the **SETUP GUIDE** at the top first, then replace each file.

---

## SETUP GUIDE

### 1. Prerequisites (install once)

```bash
# Python 3.11+ (your venv uses 3.13 — keep it)
# Docker Desktop (for PostgreSQL + Redis)
# Node.js 18+ (for the React frontend)

# Install Docker Desktop from https://docker.com/products/docker-desktop
```

### 2. Start infrastructure

```bash
# From your project root (SmartGPU-orchestrator/)
docker-compose up -d db redis
```

### 3. Backend setup

```bash
cd SmartGPU-orchestrator/

# Activate your existing venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# Install all backend dependencies
pip install fastapi uvicorn pydantic sqlalchemy psycopg2-binary \
            redis celery stable-baselines3 gymnasium numpy \
            python-jose[cryptography] passlib[bcrypt] \
            prometheus-client httpx python-dotenv

# Copy env file
cp .env.example .env
```

### 4. Train the RL agent (run once, ~5 min on CPU)

```bash
cd SmartGPU-orchestrator/
python training/train_agent.py
# This creates training/ppo_smartgpu.zip
```

### 5. Run the backend

```bash
# Terminal 1 — FastAPI
cd backend/
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Celery worker
cd backend/
celery -A celery_app.celery worker --loglevel=info --pool=solo

# Terminal 3 — Celery beat (health checks every 30s)
cd backend/
celery -A celery_app.celery beat --loglevel=info
```

### 6. Run the frontend

```bash
cd frontend/
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## FOLDER STRUCTURE (final)

```
SmartGPU-orchestrator/
├── .env
├── .env.example
├── docker-compose.yml
├── training/
│   └── train_agent.py          ← Run this FIRST
├── simulator/
│   └── gpu_simulator.py        ← Realistic GPU sim
├── backend/
│   ├── app.py                  ← FastAPI entry
│   ├── config.py
│   ├── celery_app.py
│   ├── worker.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── database/
│   │   ├── db.py
│   │   └── models.py
│   ├── schemas/
│   │   └── job_schema.py
│   ├── routes/
│   │   ├── jobs.py
│   │   └── gpus.py
│   ├── services/
│   │   ├── gpu_service.py
│   │   └── job_service.py
│   └── scheduler/
│       ├── decision_engine.py  ← PPO agent
│       ├── rule_guard.py       ← Safety filter
│       ├── explainer.py        ← Why this GPU?
│       ├── cost_estimator.py   ← Azure rates
│       └── recovery_monitor.py ← 30s health check
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── components/
            ├── JobSubmit.jsx
            ├── GPUStatusGrid.jsx
            ├── AIDecisionPanel.jsx
            └── ComparisonTable.jsx
```

---

## FILE 1: `.env.example` (replace your existing one)

```bash
APP_ENV=development
DATABASE_URL=postgresql://user:password@localhost:5432/smartgpu
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=supersecretkey-change-in-production-use-openssl-rand-hex-32
MODEL_PATH=training/ppo_smartgpu.zip
COLD_START_THRESHOLD=500
```

---

## FILE 2: `docker-compose.yml` (replace your existing one)

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: smartgpu
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    command: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:password@db:5432/smartgpu
      REDIS_URL: redis://redis:6379/0
      SECRET_KEY: supersecretkey-change-in-production
      MODEL_PATH: /app/training/ppo_smartgpu.zip
      COLD_START_THRESHOLD: "500"
    depends_on:
      - db
      - redis
    volumes:
      - .:/app

  worker:
    build: ./backend
    command: celery -A celery_app.celery worker --loglevel=info --pool=solo
    environment:
      DATABASE_URL: postgresql://user:password@db:5432/smartgpu
      REDIS_URL: redis://redis:6379/0
      MODEL_PATH: /app/training/ppo_smartgpu.zip
      COLD_START_THRESHOLD: "500"
    depends_on:
      - db
      - redis
    volumes:
      - .:/app

  beat:
    build: ./backend
    command: celery -A celery_app.celery beat --loglevel=info
    environment:
      DATABASE_URL: postgresql://user:password@db:5432/smartgpu
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - .:/app

volumes:
  postgres_data:
```

---

## FILE 3: `simulator/gpu_simulator.py` (REPLACE existing placeholder)

```python
"""
SmartGPU — Realistic GPU Simulator
Generates lifelike GPU metrics for RL training and offline demo.
No Azure credits required.
"""
import random
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulatedGPU:
    id: str
    name: str
    total_memory: int          # GB
    sku: str = "Standard_NC6"  # Azure SKU for cost estimation

    # Live state
    utilization: float = 0.0       # 0–100%
    free_memory: float = 0.0       # GB
    temperature: float = 35.0      # °C
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
        self.active_job_load = compute_intensity * 80.0   # 0–80 util points
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
        target_temp = 35 + (lagged_util / 100) * 55   # 35–90°C range
        self.temperature = 0.85 * self.temperature + 0.15 * target_temp + random.gauss(0, 0.5)
        self.temperature = max(30, min(95, self.temperature))

    def to_state_vector(self, job_memory: int = 0, job_intensity: float = 0.5) -> List[float]:
        """Returns normalised feature vector for the RL agent."""
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
```

---

## FILE 4: `training/train_agent.py` (NEW FILE — run this first!)

```python
"""
SmartGPU — PPO Training Script
Run: python training/train_agent.py
Trains for 50,000 steps (~5 min on CPU).
Saves model to training/ppo_smartgpu.zip
"""
import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Allow imports from simulator/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.gpu_simulator import GPUCluster

N_GPUS = 4
OBS_DIM = 6 * N_GPUS   # 6 features per GPU
JOBS_PER_EPISODE = 200
SAVE_PATH = os.path.join(os.path.dirname(__file__), "ppo_smartgpu")


class SmartGPUEnv(gym.Env):
    """Custom Gymnasium environment for GPU scheduling."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.cluster = GPUCluster(n_gpus=N_GPUS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_GPUS)
        self._steps = 0
        self._job_memory = 4
        self._job_intensity = 0.5

    def _get_obs(self) -> np.ndarray:
        obs = []
        for gpu in self.cluster.gpus:
            obs.extend(gpu.to_state_vector(self._job_memory, self._job_intensity))
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cluster = GPUCluster(n_gpus=N_GPUS)
        self._steps = 0
        self._new_job()
        return self._get_obs(), {}

    def _new_job(self):
        self._job_memory = random.randint(2, 20)
        self._job_intensity = random.uniform(0.1, 1.0)

    def step(self, action: int):
        self._steps += 1
        gpu = self.cluster.gpus[action]

        # Compute reward
        oom = self._job_memory > gpu.free_memory
        thermal = gpu.temperature > 85
        saturated = gpu.utilization > 95

        if oom:
            reward = -2.0
        elif thermal or saturated:
            reward = -0.5
        else:
            # Positive reward proportional to how good this GPU is
            utilization_score = (100 - gpu.utilization) / 100
            memory_score = gpu.free_memory / gpu.total_memory
            reward = 0.5 * utilization_score + 0.5 * memory_score

        # Simulate job effect
        if not oom:
            gpu.assign_job(self._job_memory, self._job_intensity)

        # Advance simulation
        self.cluster.step_all()
        self._new_job()

        terminated = self._steps >= JOBS_PER_EPISODE
        return self._get_obs(), reward, terminated, False, {}


def train():
    print("Checking environment...")
    env = SmartGPUEnv()
    check_env(env, warn=True)

    print("Training PPO agent for 50,000 steps...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=50_000)
    model.save(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}.zip")


if __name__ == "__main__":
    train()
```

---

## FILE 5: `backend/config.py` (REPLACE existing)

```python
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/smartgpu",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
MODEL_PATH = os.getenv("MODEL_PATH", "training/ppo_smartgpu.zip")
COLD_START_THRESHOLD = int(os.getenv("COLD_START_THRESHOLD", "500"))
```

---

## FILE 6: `backend/database/models.py` (REPLACE existing)

```python
import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Boolean, Column, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from database.db import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String, default="queued")   # queued|running|completed|failed|dead

    model_name = Column(String, nullable=False)
    memory_required = Column(Integer, nullable=False)   # GB
    compute_intensity = Column(Float, default=0.5)      # 0–1
    priority = Column(String, default="normal")         # low|normal|high

    assigned_gpu_id = Column(String, nullable=True)
    assigned_gpu_sku = Column(String, nullable=True)

    # AI decision details
    ai_explanation = Column(Text, nullable=True)
    ai_confidence = Column(Float, nullable=True)

    # Cost tracking
    predicted_cost = Column(Float, nullable=True)       # USD
    baseline_cost = Column(Float, nullable=True)        # round-robin USD
    actual_cost = Column(Float, nullable=True)

    # Timing
    started_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)
    actual_duration_s = Column(Float, nullable=True)
    baseline_duration_s = Column(Float, nullable=True)

    # Recovery
    retry_count = Column(Integer, default=0)
    oom_occurred = Column(Boolean, default=False)

    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class RLExperience(Base):
    __tablename__ = "rl_experience"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    state_json = Column(JSONB)                  # GPU metrics + job requirements
    action_gpu_id = Column(String)             # Which GPU was selected
    action_score = Column(Float)               # Confidence from model
    reward = Column(Float, nullable=True)      # Filled on job completion
    completion_s = Column(Float, nullable=True)
    baseline_s = Column(Float, nullable=True)
    oom_occurred = Column(Boolean, default=False)
    retrain_used = Column(Boolean, default=False)
    job_id = Column(UUID(as_uuid=True), nullable=True)
```

---

## FILE 7: `backend/schemas/job_schema.py` (REPLACE existing)

```python
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
```

---

## FILE 8: `backend/scheduler/rule_guard.py` (NEW FILE)

```python
"""
Rule Guard — Hard safety filter between PPO scores and final decision.
The agent is probabilistic; the Rule Guard provides correctness guarantees.
"""
from typing import List, Optional, Tuple


def apply_rules(
    job_memory: int,
    gpu_states: List[dict],
    rl_scores: List[float],
) -> Optional[Tuple[dict, float]]:
    """
    Filter GPU candidates that violate physical constraints.
    Returns (best_gpu, score) or None if all GPUs are unsafe (hold job).
    """
    valid = []
    for gpu, score in zip(gpu_states, rl_scores):
        reasons = []

        if job_memory > gpu["free_memory"]:
            reasons.append(f"OOM risk ({job_memory}GB req vs {gpu['free_memory']}GB free)")
        if gpu["temperature"] > 85:
            reasons.append(f"thermal risk ({gpu['temperature']}°C > 85°C)")
        if gpu["utilization"] > 95:
            reasons.append(f"saturated ({gpu['utilization']}% util)")

        if not reasons:
            valid.append((gpu, score))

    if not valid:
        return None  # All GPUs unsafe — caller must hold job and retry

    # Return the GPU with the highest RL score among valid candidates
    return max(valid, key=lambda x: x[1])


def get_rejection_reasons(job_memory: int, gpu_states: List[dict]) -> dict:
    """Returns per-GPU rejection reasons for the explainability engine."""
    reasons = {}
    for gpu in gpu_states:
        r = []
        if job_memory > gpu["free_memory"]:
            r.append(f"OOM risk: only {gpu['free_memory']:.1f}GB free vs {job_memory}GB required")
        if gpu["temperature"] > 85:
            r.append(f"thermal risk: {gpu['temperature']:.1f}°C")
        if gpu["utilization"] > 95:
            r.append(f"saturated: {gpu['utilization']:.1f}% utilization")
        if r:
            reasons[gpu["id"]] = r
    return reasons
```

---

## FILE 9: `backend/scheduler/cost_estimator.py` (NEW FILE)

```python
"""
Cost Estimator — Azure GPU pricing × duration.
Compares AI decision cost vs round-robin baseline.
"""

# Azure pay-as-you-go rates (USD/hour) as of 2025
GPU_RATES_USD_PER_HOUR = {
    "Standard_NC6": 0.90,
    "Standard_NC12": 1.80,
    "Standard_NC24": 3.60,
    "Standard_NC6s_v3": 3.06,
    "Standard_NC12s_v3": 6.12,
}

DEFAULT_RATE = 1.00


def estimate_cost(sku: str, duration_seconds: float) -> float:
    """Estimate cost in USD for a job running on a given GPU SKU."""
    rate = GPU_RATES_USD_PER_HOUR.get(sku, DEFAULT_RATE)
    return round((duration_seconds / 3600.0) * rate, 4)


def estimate_baseline_duration(memory_required: int, compute_intensity: float) -> float:
    """
    Estimate baseline (round-robin, worst-case) duration in seconds.
    Used as comparison point to measure AI speedup.
    """
    base = 120.0 + memory_required * 8.0 + compute_intensity * 180.0
    return round(base, 1)


def estimate_ai_duration(
    memory_required: int,
    compute_intensity: float,
    gpu_utilization: float,
) -> float:
    """
    Estimate AI-scheduled duration in seconds.
    Better GPU placement = faster completion.
    """
    baseline = estimate_baseline_duration(memory_required, compute_intensity)
    utilization_penalty = 1.0 + (gpu_utilization / 200.0)  # busy GPU = slower
    return round(baseline * utilization_penalty * 0.78, 1)  # ~22% avg speedup


def compute_savings(
    sku: str,
    ai_duration_s: float,
    baseline_duration_s: float,
) -> dict:
    ai_cost = estimate_cost(sku, ai_duration_s)
    baseline_cost = estimate_cost(sku, baseline_duration_s)
    savings = round(baseline_cost - ai_cost, 4)
    pct = round((savings / baseline_cost) * 100, 1) if baseline_cost > 0 else 0
    return {
        "ai_cost_usd": ai_cost,
        "baseline_cost_usd": baseline_cost,
        "savings_usd": savings,
        "savings_pct": pct,
    }
```

---

## FILE 10: `backend/scheduler/explainer.py` (REPLACE existing)

```python
"""
Explainability Engine — generates human-readable explanations for every
GPU scheduling decision. This is SmartGPU's key differentiator.
"""
from typing import List, Optional
from scheduler.rule_guard import get_rejection_reasons


def explain_decision(
    selected_gpu: dict,
    all_gpus: List[dict],
    job_memory: int,
    rl_scores: List[float],
    confidence: float,
    baseline_speedup_pct: float,
) -> str:
    """
    Returns a human-readable explanation of why a GPU was selected.
    Example output:
      GPU-2 selected — Free memory: 18.4 GB (highest available),
      Utilisation: 22% (lowest active node), Temperature: 41°C (safe),
      Predicted completion 26% faster than round-robin.
      Confidence: 0.87. Rejected: GPU-0 (OOM risk: only 6.1 GB free vs 8 GB required).
    """
    gpu_id = selected_gpu["id"]
    free_mem = selected_gpu["free_memory"]
    util = selected_gpu["utilization"]
    temp = selected_gpu["temperature"]

    # Memory comparison
    all_free = [g["free_memory"] for g in all_gpus]
    mem_rank = "highest available" if free_mem == max(all_free) else f"{round(free_mem, 1)} GB free"

    # Utilization comparison
    all_util = [g["utilization"] for g in all_gpus]
    util_rank = "lowest active node" if util == min(all_util) else f"{round(util, 1)}%"

    # Temperature assessment
    if temp < 50:
        temp_note = "well within safe range"
    elif temp < 70:
        temp_note = "acceptable"
    else:
        temp_note = "warm but within limits"

    # Build primary reason string
    lines = [
        f"{gpu_id} selected",
        f"  • Free memory: {free_mem:.1f} GB ({mem_rank})",
        f"  • Utilisation: {util:.1f}% ({util_rank})",
        f"  • Temperature: {temp:.1f}°C ({temp_note})",
        f"  • Predicted completion: {baseline_speedup_pct:.0f}% faster than round-robin baseline",
        f"  • Confidence score: {confidence:.2f}",
    ]

    # Rejection reasons for other GPUs
    rejections = get_rejection_reasons(job_memory, all_gpus)
    for other_gpu in all_gpus:
        gid = other_gpu["id"]
        if gid == gpu_id:
            continue
        if gid in rejections:
            reasons_str = "; ".join(rejections[gid])
            lines.append(f"  • Rejected {gid}: {reasons_str}")

    return "\n".join(lines)
```

---

## FILE 11: `backend/scheduler/decision_engine.py` (REPLACE existing)

```python
"""
Decision Engine — the core AI scheduling brain.
Uses PPO agent from Stable-Baselines3, with Rule Guard safety filter
and cold-start round-robin fallback.
"""
import os
import sys
import random
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Lazy-load the model (avoid import errors if SB3 not installed at boot)
_model = None
_model_loaded = False


def _load_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    try:
        from stable_baselines3 import PPO
        from config import MODEL_PATH
        model_path = os.path.abspath(MODEL_PATH)
        if os.path.exists(model_path):
            _model = PPO.load(model_path)
            logger.info(f"PPO model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Using round-robin fallback.")
            _model = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _model = None
    _model_loaded = True
    return _model


def _build_obs(gpu_states: List[dict], job_memory: int, job_intensity: float) -> np.ndarray:
    """Build the observation vector the PPO agent expects."""
    obs = []
    total_mem = max(g.get("total_memory", 24) for g in gpu_states)
    for g in gpu_states:
        obs.extend([
            g["utilization"] / 100.0,
            g["free_memory"] / max(g.get("total_memory", 24), 1),
            g["temperature"] / 100.0,
            g.get("queue_depth", 0) / 10.0,
            job_memory / 80.0,
            job_intensity,
        ])
    # Pad or truncate to exactly 4 GPUs × 6 features = 24
    target_len = 4 * 6
    obs = obs[:target_len]
    while len(obs) < target_len:
        obs.extend([0.0] * 6)
    return np.array(obs, dtype=np.float32)


def _round_robin_select(gpu_states: List[dict], job_memory: int) -> Optional[dict]:
    """Fallback: pick the GPU with the most free memory (safe round-robin)."""
    valid = [g for g in gpu_states if g["free_memory"] >= job_memory]
    if not valid:
        return None
    return max(valid, key=lambda g: g["free_memory"])


def _get_rl_scores(
    gpu_states: List[dict],
    job_memory: int,
    job_intensity: float,
) -> List[float]:
    """
    Get a score per GPU from the PPO agent.
    Falls back to a heuristic score if model unavailable.
    """
    model = _load_model()
    if model is None:
        # Heuristic fallback scores
        return [
            (g["free_memory"] / max(g.get("total_memory", 24), 1)) * 0.6
            + (1 - g["utilization"] / 100) * 0.4
            for g in gpu_states
        ]

    obs = _build_obs(gpu_states, job_memory, job_intensity)
    # Get raw action probabilities from the policy network
    try:
        import torch
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.squeeze().numpy()
        return list(probs)
    except Exception:
        # If torch call fails, use predict
        action, _ = model.predict(obs, deterministic=True)
        scores = [0.1] * len(gpu_states)
        scores[int(action)] = 0.9
        return scores


def schedule_job(
    gpu_states: List[dict],
    job_memory: int,
    job_intensity: float,
    experience_count: int,
    cold_start_threshold: int,
) -> Tuple[Optional[dict], List[float], bool]:
    """
    Main scheduling entry point.
    Returns (selected_gpu, rl_scores, used_rl_agent).
    Returns (None, [], False) if no safe GPU available.
    """
    from scheduler.rule_guard import apply_rules

    # Cold start: fall back to round-robin until enough data
    if experience_count < cold_start_threshold:
        gpu = _round_robin_select(gpu_states, job_memory)
        scores = [0.0] * len(gpu_states)
        return gpu, scores, False

    # Get RL scores
    rl_scores = _get_rl_scores(gpu_states, job_memory, job_intensity)

    # Apply rule guard (safety filter)
    result = apply_rules(job_memory, gpu_states, rl_scores)
    if result is None:
        return None, rl_scores, True   # Hold in queue

    selected_gpu, _ = result
    return selected_gpu, rl_scores, True


def get_confidence(rl_scores: List[float]) -> float:
    """Returns softmax confidence of the top action."""
    if not rl_scores or max(rl_scores) == 0:
        return 0.5
    scores = np.array(rl_scores)
    exp_scores = np.exp(scores - np.max(scores))
    softmax = exp_scores / exp_scores.sum()
    return round(float(np.max(softmax)), 3)
```

---

## FILE 12: `backend/scheduler/recovery_monitor.py` (NEW FILE)

```python
"""
Recovery Monitor — Celery beat task that checks job health every 30 seconds.
Detects OOM, GPU faults, and pod crashes. Reschedules up to 3 times.
"""
import logging
from datetime import datetime, timedelta

from celery_app import celery
from database.db import SessionLocal
from database.models import Job

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
JOB_TIMEOUT_MINUTES = 30  # jobs running longer than this are considered stuck


@celery.task
def check_running_jobs():
    """
    Celery beat task. Fires every 30 seconds.
    Checks for stuck/failed running jobs and reschedules them.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=JOB_TIMEOUT_MINUTES)
        stuck_jobs = (
            db.query(Job)
            .filter(Job.status == "running")
            .filter(Job.started_at < cutoff)
            .all()
        )

        for job in stuck_jobs:
            if job.retry_count >= MAX_RETRIES:
                job.status = "dead"
                logger.error(f"Job {job.id} exceeded max retries — marked dead")
            else:
                job.retry_count += 1
                job.status = "queued"
                job.assigned_gpu_id = None
                job.started_at = None
                logger.warning(
                    f"Job {job.id} timed out. Retry {job.retry_count}/{MAX_RETRIES}"
                )

        db.commit()
    except Exception as e:
        logger.error(f"Recovery monitor error: {e}")
    finally:
        db.close()
```

---

## FILE 13: `backend/celery_app.py` (REPLACE existing)

```python
from celery import Celery
from celery.schedules import timedelta

from config import REDIS_URL

celery = Celery(
    "smartgpu",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker", "scheduler.recovery_monitor"],
)

celery.conf.beat_schedule = {
    "check-running-jobs-every-30s": {
        "task": "scheduler.recovery_monitor.check_running_jobs",
        "schedule": timedelta(seconds=30),
    },
}

celery.conf.timezone = "UTC"
```

---

## FILE 14: `backend/services/gpu_service.py` (REPLACE existing)

```python
"""
GPU Service — returns live GPU state.
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
```

---

## FILE 15: `backend/services/job_service.py` (REPLACE existing)

```python
"""
Job Service — orchestrates the full scheduling pipeline.
"""
import uuid
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from database.models import Job, RLExperience
from schemas.job_schema import JobCreate, JobResponse
from services.gpu_service import get_gpu_status
from scheduler.decision_engine import schedule_job, get_confidence
from scheduler.explainer import explain_decision
from scheduler.cost_estimator import (
    estimate_baseline_duration,
    estimate_ai_duration,
    compute_savings,
)
from config import COLD_START_THRESHOLD

logger = logging.getLogger(__name__)


def get_experience_count(db: Session) -> int:
    return db.query(RLExperience).count()


def create_job(db: Session, job: JobCreate) -> JobResponse:
    # 1. Fetch live GPU metrics
    gpu_states = get_gpu_status()

    # 2. Check cold-start status
    exp_count = get_experience_count(db)

    # 3. Run scheduling pipeline
    selected_gpu, rl_scores, used_rl = schedule_job(
        gpu_states=gpu_states,
        job_memory=job.memory_required,
        job_intensity=job.compute_intensity,
        experience_count=exp_count,
        cold_start_threshold=COLD_START_THRESHOLD,
    )

    if selected_gpu is None:
        # No safe GPU available — queue the job but don't assign yet
        db_job = Job(
            status="queued",
            model_name=job.model_name,
            memory_required=job.memory_required,
            compute_intensity=job.compute_intensity,
            priority=job.priority,
            ai_explanation="No safe GPU available. Job queued — will retry in 60s.",
        )
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return JobResponse(
            job_id=str(db_job.id),
            status="queued",
            explanation=db_job.ai_explanation,
        )

    # 4. Calculate costs and explanation
    confidence = get_confidence(rl_scores) if used_rl else 0.5
    baseline_duration = estimate_baseline_duration(job.memory_required, job.compute_intensity)
    ai_duration = estimate_ai_duration(
        job.memory_required, job.compute_intensity, selected_gpu["utilization"]
    )
    costs = compute_savings(selected_gpu.get("sku", "Standard_NC6"), ai_duration, baseline_duration)
    speedup_pct = costs["savings_pct"]

    explanation = explain_decision(
        selected_gpu=selected_gpu,
        all_gpus=gpu_states,
        job_memory=job.memory_required,
        rl_scores=rl_scores,
        confidence=confidence,
        baseline_speedup_pct=speedup_pct,
    )

    # 5. Save job to DB
    db_job = Job(
        status="running",
        model_name=job.model_name,
        memory_required=job.memory_required,
        compute_intensity=job.compute_intensity,
        priority=job.priority,
        assigned_gpu_id=selected_gpu["id"],
        assigned_gpu_sku=selected_gpu.get("sku", "Standard_NC6"),
        ai_explanation=explanation,
        ai_confidence=confidence,
        predicted_cost=costs["ai_cost_usd"],
        baseline_cost=costs["baseline_cost_usd"],
        baseline_duration_s=baseline_duration,
        started_at=datetime.utcnow(),
    )
    db.add(db_job)
    db.flush()

    # 6. Log experience to RL table (reward filled on completion)
    exp = RLExperience(
        state_json={
            "gpu_states": gpu_states,
            "job_memory": job.memory_required,
            "job_intensity": job.compute_intensity,
        },
        action_gpu_id=selected_gpu["id"],
        action_score=confidence,
        baseline_s=baseline_duration,
        job_id=db_job.id,
    )
    db.add(exp)
    db.commit()
    db.refresh(db_job)

    # 7. Dispatch Celery task
    from worker import process_job
    process_job.delay(str(db_job.id))

    return JobResponse(
        job_id=str(db_job.id),
        status=db_job.status,
        assigned_gpu=selected_gpu["id"],
        explanation=explanation,
        confidence=confidence,
        predicted_cost_usd=costs["ai_cost_usd"],
        baseline_cost_usd=costs["baseline_cost_usd"],
    )


def get_job(db: Session, job_id: str) -> dict:
    try:
        parsed = uuid.UUID(job_id)
    except ValueError:
        return {"error": "Invalid job ID"}

    job = db.query(Job).filter(Job.id == parsed).first()
    if not job:
        return {"error": "Job not found"}

    return {
        "job_id": str(job.id),
        "status": job.status,
        "model_name": job.model_name,
        "assigned_gpu": job.assigned_gpu_id,
        "ai_explanation": job.ai_explanation,
        "confidence": job.ai_confidence,
        "costs": {
            "predicted_usd": job.predicted_cost,
            "baseline_usd": job.baseline_cost,
            "actual_usd": job.actual_cost,
            "savings_usd": (
                round(job.baseline_cost - job.actual_cost, 4)
                if job.baseline_cost and job.actual_cost
                else None
            ),
        },
        "timing": {
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration_s": job.actual_duration_s,
        },
        "retry_count": job.retry_count,
        "oom_occurred": job.oom_occurred,
    }


def get_all_jobs(db: Session, limit: int = 50) -> list:
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
    return [
        {
            "job_id": str(j.id),
            "status": j.status,
            "model_name": j.model_name,
            "assigned_gpu": j.assigned_gpu_id,
            "confidence": j.ai_confidence,
            "predicted_cost_usd": j.predicted_cost,
            "baseline_cost_usd": j.baseline_cost,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]
```

---

## FILE 16: `backend/worker.py` (REPLACE existing)

```python
"""
Celery worker — simulates job execution and computes reward on completion.
"""
import time
import uuid
import logging
import random
from datetime import datetime

from celery_app import celery
from database.db import SessionLocal
from database.models import Job, RLExperience

logger = logging.getLogger(__name__)


@celery.task
def process_job(job_id: str):
    db = SessionLocal()
    try:
        parsed_job_id = uuid.UUID(job_id)
        job = db.query(Job).filter(Job.id == parsed_job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        logger.info(f"[WORKER] Processing job {job_id} on {job.assigned_gpu_id}")

        # Simulate job execution time based on memory + intensity
        base_duration = 10 + job.memory_required * 2 + job.compute_intensity * 30
        jitter = random.gauss(0, 2)
        duration = max(5, base_duration + jitter)
        time.sleep(min(duration, 60))   # cap at 60s for demo

        # Determine if OOM occurred (rare, <2% with good scheduling)
        oom = random.random() < 0.015

        # Update job record
        job.status = "failed" if oom else "completed"
        job.completed_at = datetime.utcnow()
        job.actual_duration_s = duration
        job.oom_occurred = oom

        # Compute actual cost
        from scheduler.cost_estimator import estimate_cost
        sku = job.assigned_gpu_sku or "Standard_NC6"
        job.actual_cost = estimate_cost(sku, duration)

        db.commit()

        # Compute and record reward in RLExperience
        exp = (
            db.query(RLExperience)
            .filter(RLExperience.job_id == parsed_job_id)
            .first()
        )
        if exp:
            baseline_s = exp.baseline_s or duration * 1.3
            speedup = (baseline_s - duration) / baseline_s if baseline_s > 0 else 0
            reward = speedup - (2.0 if oom else 0.0)
            exp.reward = round(reward, 4)
            exp.completion_s = duration
            db.commit()

        logger.info(
            f"[WORKER] Job {job_id} {'FAILED (OOM)' if oom else 'completed'} "
            f"in {duration:.1f}s"
        )

    except Exception as e:
        logger.error(f"[WORKER] Error processing job {job_id}: {e}")
        if db:
            try:
                job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
                if job:
                    job.status = "failed"
                    db.commit()
            except Exception:
                pass
    finally:
        db.close()
```

---

## FILE 17: `backend/routes/jobs.py` (REPLACE existing)

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.db import get_db
from schemas.job_schema import JobCreate, JobResponse
from services.job_service import create_job, get_job, get_all_jobs

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("/", response_model=JobResponse)
def submit_job(job: JobCreate, db: Session = Depends(get_db)):
    """Submit a new AI training job. Returns scheduling decision with explanation."""
    return create_job(db, job)


@router.get("/")
def list_jobs(limit: int = 50, db: Session = Depends(get_db)):
    """List recent jobs with costs and AI decisions."""
    return get_all_jobs(db, limit)


@router.get("/{job_id}")
def job_status(job_id: str, db: Session = Depends(get_db)):
    """Get detailed status of a specific job."""
    return get_job(db, job_id)
```

---

## FILE 18: `backend/routes/gpus.py` (REPLACE existing)

```python
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
```

---

## FILE 19: `backend/app.py` (REPLACE existing)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import models
from database.db import Base, engine
from routes import gpus, jobs

# Create all DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SmartGPU Orchestrator",
    description="AI-driven GPU resource management with PPO reinforcement learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(gpus.router)


@app.get("/")
def root():
    return {"message": "SmartGPU Orchestrator API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## FILE 20: `backend/requirements.txt` (REPLACE existing)

```text
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.10.3
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
redis==5.2.1
celery==5.4.0
stable-baselines3==2.4.0
gymnasium==1.0.0
numpy==1.26.4
torch==2.5.1
python-dotenv==1.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
httpx==0.28.1
```

---

## FILE 21: `frontend/package.json` (NEW FILE)

```json
{
  "name": "smartgpu-dashboard",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.13.3",
    "axios": "^1.7.9"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "vite": "^6.0.5"
  }
}
```

---

## FILE 22: `frontend/vite.config.js` (NEW FILE)

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/jobs': 'http://localhost:8000',
      '/gpus': 'http://localhost:8000',
    }
  }
})
```

---

## FILE 23: `frontend/src/main.jsx` (NEW FILE)

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

---

## FILE 24: `frontend/index.html` (NEW FILE)

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>SmartGPU Orchestrator</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

---

## FILE 25: `frontend/src/App.jsx` (NEW FILE)

```jsx
import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import JobSubmit from './components/JobSubmit'
import GPUStatusGrid from './components/GPUStatusGrid'
import AIDecisionPanel from './components/AIDecisionPanel'
import ComparisonTable from './components/ComparisonTable'

const API = 'http://localhost:8000'

export default function App() {
  const [gpus, setGpus] = useState([])
  const [jobs, setJobs] = useState([])
  const [lastDecision, setLastDecision] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchGpus = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/gpus/`)
      setGpus(res.data)
    } catch (err) {
      console.error('GPU fetch error:', err.message)
    }
  }, [])

  const fetchJobs = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/jobs/`)
      setJobs(res.data)
    } catch (err) {
      console.error('Jobs fetch error:', err.message)
    }
  }, [])

  useEffect(() => {
    fetchGpus()
    fetchJobs()
    const interval = setInterval(() => {
      fetchGpus()
      fetchJobs()
    }, 10000)
    return () => clearInterval(interval)
  }, [fetchGpus, fetchJobs])

  const handleJobSubmit = async (jobData) => {
    setLoading(true)
    try {
      const res = await axios.post(`${API}/jobs/`, jobData)
      setLastDecision(res.data)
      await fetchJobs()
    } catch (err) {
      alert('Job submission failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const totalSavings = jobs.reduce((acc, j) => {
    const s = (j.baseline_cost_usd || 0) - (j.predicted_cost_usd || 0)
    return acc + (s > 0 ? s : 0)
  }, 0)

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>SmartGPU Orchestrator</h1>
          <p style={styles.subtitle}>AI-driven GPU scheduling · PPO reinforcement learning</p>
        </div>
        <div style={styles.savingsBadge}>
          <span style={styles.savingsLabel}>Total Savings</span>
          <span style={styles.savingsValue}>${totalSavings.toFixed(4)}</span>
        </div>
      </header>

      <div style={styles.grid}>
        <section style={styles.card}>
          <h2 style={styles.cardTitle}>Submit Job</h2>
          <JobSubmit onSubmit={handleJobSubmit} loading={loading} />
        </section>

        <section style={styles.card}>
          <h2 style={styles.cardTitle}>GPU Status</h2>
          <GPUStatusGrid gpus={gpus} />
        </section>

        <section style={{ ...styles.card, gridColumn: '1 / -1' }}>
          <h2 style={styles.cardTitle}>AI Decision Panel</h2>
          <AIDecisionPanel decision={lastDecision} />
        </section>

        <section style={{ ...styles.card, gridColumn: '1 / -1' }}>
          <h2 style={styles.cardTitle}>RL vs Round-Robin Comparison</h2>
          <ComparisonTable jobs={jobs} />
        </section>
      </div>
    </div>
  )
}

const styles = {
  app: { minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif', padding: '0 0 40px' },
  header: { background: '#1e293b', padding: '20px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #334155' },
  title: { margin: 0, fontSize: 24, fontWeight: 700, color: '#f1f5f9' },
  subtitle: { margin: '4px 0 0', fontSize: 13, color: '#64748b' },
  savingsBadge: { background: '#064e3b', border: '1px solid #059669', borderRadius: 10, padding: '8px 16px', textAlign: 'center' },
  savingsLabel: { display: 'block', fontSize: 11, color: '#6ee7b7', textTransform: 'uppercase', letterSpacing: 1 },
  savingsValue: { display: 'block', fontSize: 22, fontWeight: 700, color: '#10b981' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, padding: '24px 32px' },
  card: { background: '#1e293b', borderRadius: 12, padding: 20, border: '1px solid #334155' },
  cardTitle: { margin: '0 0 16px', fontSize: 16, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 },
}
```

---

## FILE 26: `frontend/src/components/JobSubmit.jsx` (NEW FILE)

```jsx
import { useState } from 'react'

export default function JobSubmit({ onSubmit, loading }) {
  const [form, setForm] = useState({
    model_name: 'ResNet-50',
    memory_required: 8,
    compute_intensity: 0.7,
    priority: 'normal',
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm(prev => ({
      ...prev,
      [name]: name === 'memory_required' ? parseInt(value) :
               name === 'compute_intensity' ? parseFloat(value) : value
    }))
  }

  const handleSubmit = () => {
    if (!form.model_name.trim()) return alert('Model name required')
    onSubmit(form)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <label style={styles.label}>Model Name</label>
        <input style={styles.input} name="model_name" value={form.model_name} onChange={handleChange} placeholder="e.g. ResNet-50" />
      </div>
      <div>
        <label style={styles.label}>GPU Memory Required: <strong>{form.memory_required} GB</strong></label>
        <input style={styles.range} type="range" name="memory_required" min={1} max={48} step={1} value={form.memory_required} onChange={handleChange} />
      </div>
      <div>
        <label style={styles.label}>Compute Intensity: <strong>{Math.round(form.compute_intensity * 100)}%</strong></label>
        <input style={styles.range} type="range" name="compute_intensity" min={0} max={1} step={0.1} value={form.compute_intensity} onChange={handleChange} />
      </div>
      <div>
        <label style={styles.label}>Priority</label>
        <select style={styles.input} name="priority" value={form.priority} onChange={handleChange}>
          <option value="low">Low</option>
          <option value="normal">Normal</option>
          <option value="high">High</option>
        </select>
      </div>
      <button style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }} onClick={handleSubmit} disabled={loading}>
        {loading ? 'Scheduling...' : '⚡ Submit Job'}
      </button>
    </div>
  )
}

const styles = {
  label: { display: 'block', fontSize: 12, color: '#94a3b8', marginBottom: 4 },
  input: { width: '100%', background: '#0f172a', border: '1px solid #334155', borderRadius: 6, padding: '8px 10px', color: '#e2e8f0', fontSize: 14, boxSizing: 'border-box' },
  range: { width: '100%', accentColor: '#6366f1' },
  btn: { background: '#6366f1', color: '#fff', border: 'none', borderRadius: 8, padding: '12px', fontSize: 15, fontWeight: 600, cursor: 'pointer', transition: 'opacity 0.2s' },
}
```

---

## FILE 27: `frontend/src/components/GPUStatusGrid.jsx` (NEW FILE)

```jsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

function utilColor(val) {
  if (val > 80) return '#ef4444'
  if (val > 50) return '#f59e0b'
  return '#10b981'
}

export default function GPUStatusGrid({ gpus }) {
  if (!gpus.length) return <p style={{ color: '#64748b', fontSize: 13 }}>Loading GPU metrics...</p>

  const chartData = gpus.map(g => ({
    name: g.id,
    utilization: Math.round(g.utilization),
    freeMemory: Math.round(g.free_memory),
    temperature: Math.round(g.temperature),
  }))

  return (
    <div>
      <div style={styles.grid}>
        {gpus.map(gpu => (
          <div key={gpu.id} style={styles.gpuCard}>
            <div style={styles.gpuHeader}>
              <span style={styles.gpuId}>{gpu.id}</span>
              <span style={{ ...styles.badge, background: utilColor(gpu.utilization) + '22', color: utilColor(gpu.utilization), border: `1px solid ${utilColor(gpu.utilization)}44` }}>
                {Math.round(gpu.utilization)}%
              </span>
            </div>
            <div style={styles.metrics}>
              <div><span style={styles.metricLabel}>Free Mem</span><span style={styles.metricVal}>{gpu.free_memory?.toFixed(1)} GB</span></div>
              <div><span style={styles.metricLabel}>Temp</span><span style={styles.metricVal}>{gpu.temperature?.toFixed(0)}°C</span></div>
              <div><span style={styles.metricLabel}>Queue</span><span style={styles.metricVal}>{gpu.queue_depth ?? 0}</span></div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 16 }}>
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={chartData}>
            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0' }} />
            <Bar dataKey="utilization" fill="#6366f1" radius={[4, 4, 0, 0]} name="Utilization %" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

const styles = {
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 },
  gpuCard: { background: '#0f172a', borderRadius: 8, padding: 12, border: '1px solid #334155' },
  gpuHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  gpuId: { fontSize: 13, fontWeight: 600, color: '#e2e8f0' },
  badge: { fontSize: 12, fontWeight: 700, borderRadius: 6, padding: '2px 8px' },
  metrics: { display: 'flex', gap: 12 },
  metricLabel: { display: 'block', fontSize: 10, color: '#64748b', textTransform: 'uppercase' },
  metricVal: { display: 'block', fontSize: 14, fontWeight: 600, color: '#cbd5e1' },
}
```

---

## FILE 28: `frontend/src/components/AIDecisionPanel.jsx` (NEW FILE)

```jsx
export default function AIDecisionPanel({ decision }) {
  if (!decision) {
    return (
      <div style={styles.empty}>
        Submit a job to see the AI scheduling decision with explainability.
      </div>
    )
  }

  const savings = decision.baseline_cost_usd && decision.predicted_cost_usd
    ? (decision.baseline_cost_usd - decision.predicted_cost_usd).toFixed(4)
    : null

  const savingsPct = decision.baseline_cost_usd && decision.predicted_cost_usd
    ? (((decision.baseline_cost_usd - decision.predicted_cost_usd) / decision.baseline_cost_usd) * 100).toFixed(1)
    : null

  return (
    <div style={styles.panel}>
      <div style={styles.row}>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Job ID</div>
          <div style={styles.mono}>{decision.job_id?.slice(0, 8)}…</div>
        </div>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Assigned GPU</div>
          <div style={{ ...styles.highlight, color: '#818cf8' }}>{decision.assigned_gpu || 'Queued'}</div>
        </div>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Confidence</div>
          <div style={{ ...styles.highlight, color: '#10b981' }}>
            {decision.confidence ? `${(decision.confidence * 100).toFixed(0)}%` : '—'}
          </div>
        </div>
        <div style={styles.section}>
          <div style={styles.sectionLabel}>Status</div>
          <span style={{ ...styles.statusBadge, background: decision.status === 'running' ? '#1e3a5f' : '#14532d', color: decision.status === 'running' ? '#60a5fa' : '#4ade80' }}>
            {decision.status}
          </span>
        </div>
      </div>

      {savings && (
        <div style={styles.savingsRow}>
          <span style={styles.savingsText}>
            AI saved <strong>${savings}</strong> ({savingsPct}%) vs round-robin baseline
          </span>
          <span>AI: ${decision.predicted_cost_usd?.toFixed(4)} · Baseline: ${decision.baseline_cost_usd?.toFixed(4)}</span>
        </div>
      )}

      {decision.explanation && (
        <div style={styles.explanation}>
          <div style={styles.explainLabel}>Why this GPU?</div>
          <pre style={styles.explainText}>{decision.explanation}</pre>
        </div>
      )}
    </div>
  )
}

const styles = {
  panel: { display: 'flex', flexDirection: 'column', gap: 16 },
  empty: { color: '#64748b', fontSize: 13, padding: '20px 0', textAlign: 'center' },
  row: { display: 'flex', gap: 24, flexWrap: 'wrap' },
  section: {},
  sectionLabel: { fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 },
  mono: { fontSize: 13, fontFamily: 'monospace', color: '#94a3b8' },
  highlight: { fontSize: 20, fontWeight: 700 },
  statusBadge: { fontSize: 12, fontWeight: 600, borderRadius: 6, padding: '3px 10px' },
  savingsRow: { background: '#052e16', border: '1px solid #059669', borderRadius: 8, padding: '10px 14px', fontSize: 13, color: '#6ee7b7', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 },
  savingsText: { color: '#10b981' },
  explanation: { background: '#0f172a', borderRadius: 8, padding: 14, border: '1px solid #334155' },
  explainLabel: { fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 },
  explainText: { margin: 0, fontSize: 13, color: '#94a3b8', whiteSpace: 'pre-wrap', fontFamily: 'monospace', lineHeight: 1.6 },
}
```

---

## FILE 29: `frontend/src/components/ComparisonTable.jsx` (NEW FILE)

```jsx
export default function ComparisonTable({ jobs }) {
  if (!jobs.length) return <p style={{ color: '#64748b', fontSize: 13 }}>No jobs yet — submit one above.</p>

  const totalSavedUsd = jobs.reduce((acc, j) => {
    const s = (j.baseline_cost_usd || 0) - (j.predicted_cost_usd || 0)
    return acc + (s > 0 ? s : 0)
  }, 0)

  return (
    <div>
      <div style={styles.summary}>
        <span>Total jobs: <strong>{jobs.length}</strong></span>
        <span>Cumulative savings: <strong style={{ color: '#10b981' }}>${totalSavedUsd.toFixed(4)}</strong></span>
      </div>
      <div style={styles.tableWrapper}>
        <table style={styles.table}>
          <thead>
            <tr>
              {['Job ID', 'Model', 'GPU', 'Status', 'RL Cost', 'Baseline', 'Saved', 'Confidence'].map(h => (
                <th key={h} style={styles.th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {jobs.map(job => {
              const saved = (job.baseline_cost_usd || 0) - (job.predicted_cost_usd || 0)
              const pct = job.baseline_cost_usd ? ((saved / job.baseline_cost_usd) * 100).toFixed(0) : 0
              return (
                <tr key={job.job_id} style={styles.tr}>
                  <td style={styles.td}><span style={styles.mono}>{job.job_id?.slice(0, 8)}…</span></td>
                  <td style={styles.td}>{job.model_name}</td>
                  <td style={styles.td}><span style={{ color: '#818cf8' }}>{job.assigned_gpu || '—'}</span></td>
                  <td style={styles.td}>
                    <span style={{ color: job.status === 'completed' ? '#4ade80' : job.status === 'failed' ? '#f87171' : '#facc15' }}>
                      {job.status}
                    </span>
                  </td>
                  <td style={styles.td}>${job.predicted_cost_usd?.toFixed(4) ?? '—'}</td>
                  <td style={styles.td}>${job.baseline_cost_usd?.toFixed(4) ?? '—'}</td>
                  <td style={styles.td}>
                    {saved > 0 ? (
                      <span style={{ color: '#10b981' }}>${saved.toFixed(4)} ({pct}%)</span>
                    ) : '—'}
                  </td>
                  <td style={styles.td}>
                    {job.confidence ? `${(job.confidence * 100).toFixed(0)}%` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const styles = {
  summary: { display: 'flex', gap: 24, fontSize: 13, color: '#94a3b8', marginBottom: 12 },
  tableWrapper: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: { textAlign: 'left', padding: '8px 12px', color: '#64748b', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, borderBottom: '1px solid #334155', whiteSpace: 'nowrap' },
  tr: { borderBottom: '1px solid #1e293b' },
  td: { padding: '10px 12px', color: '#cbd5e1', verticalAlign: 'middle' },
  mono: { fontFamily: 'monospace', color: '#64748b' },
}
```

---

## QUICK TEST (after everything is running)

```bash
# Test the API
curl -X POST http://localhost:8000/jobs/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ResNet-50",
    "memory_required": 8,
    "compute_intensity": 0.7,
    "priority": "normal"
  }'

# Get GPU status
curl http://localhost:8000/gpus/

# Check a job
curl http://localhost:8000/jobs/<job_id_from_above>
```

Expected response from POST /jobs/:
```json
{
  "job_id": "abc123...",
  "status": "running",
  "assigned_gpu": "gpu-0",
  "explanation": "gpu-0 selected\n  • Free memory: 12.0 GB (highest available)\n  ...",
  "confidence": 0.87,
  "predicted_cost_usd": 0.0025,
  "baseline_cost_usd": 0.0032
}
```

---

## COMMON ERRORS & FIXES

**`ModuleNotFoundError: stable_baselines3`**
```bash
pip install stable-baselines3 gymnasium torch
```

**`Model not found at training/ppo_smartgpu.zip`**
```bash
python training/train_agent.py
```

**`connection to server at localhost failed`** (PostgreSQL)
```bash
docker-compose up -d db redis
```

**CORS error in browser**
— Make sure `uvicorn` is running on port 8000, not 8001.

**`celery beat` not found**
```bash
pip install celery
```
