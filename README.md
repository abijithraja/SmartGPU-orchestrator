# SmartGPU Orchestrator

AI-driven GPU resource orchestration platform for intelligent, explainable, and cost-aware scheduling of AI workloads.

## Current Status (Important)

This project is currently in the **research and progressive build stage**.

- We are validating architecture, scheduling strategy, and learning loops.
- We are building the platform **phase by phase**, not as a single final release.
- Several core modules are intentionally placeholders while the simulator, RL policy, and deployment layers are developed incrementally.

In short: this repository reflects an active research build, with production-grade goals and progressive implementation.

## Project Vision

SmartGPU Orchestrator aims to replace static GPU scheduling heuristics with a reinforcement-learning-driven decision engine that can:

- Improve GPU utilization under dynamic workloads.
- Reduce OOM and thermal-risk scheduling failures.
- Explain every scheduling decision in human-readable form.
- Track cost impact per decision against a baseline strategy.
- Continuously improve through periodic retraining from live experience data.

## Problem We Are Solving

Traditional schedulers (FIFO/round-robin/fixed priority) are rigid and often blind to real-time GPU conditions such as memory pressure, thermal state, and queue behavior. In expensive GPU environments, that leads to avoidable waste and unstable execution.

SmartGPU explores an adaptive alternative:

- Observe live metrics
- Score candidate GPUs
- Apply safety guardrails
- Pick the best placement
- Explain and log the outcome
- Retrain over time

## Research Goals and Target Outcomes

Target outcomes from the project report (not yet fully achieved in code):

- GPU utilization: 75-90% target range
- OOM-caused failures: below 2%
- Cost waste vs optimal: below 10%
- Explainability: full per-decision rationale
- Learning loop: retraining after experience accumulation

## What Is Implemented Today

Current implementation in this repository:

- FastAPI backend skeleton with core routes:
	- `POST /jobs/`
	- `GET /jobs/{job_id}`
	- `GET /gpus/`
- PostgreSQL-backed `jobs` table via SQLAlchemy.
- Celery + Redis async worker flow for simulated job execution lifecycle (`queued -> running -> completed`).
- Containerized backend worker image and local dependencies via Docker Compose.
- Initial scheduler/explainer/simulator placeholders for progressive replacement with RL-based logic.

## What Is Planned Next (Progressive Build)

Planned capabilities from the roadmap:

- Realistic GPU simulator and synthetic training environment.
- PPO-based scheduler (Stable-Baselines3) with safety rule guard.
- Explainability panel and side-by-side baseline comparison mode.
- Cost estimator and experience logging for retraining loop.
- Prometheus + Grafana integration for live cluster telemetry.
- Kubernetes and Azure AKS deployment with GPU-aware scheduling.

## Architecture (Planned End State)

- Frontend: React + Vite dashboard for job submission, GPU status, and AI decision insights.
- Backend: FastAPI control plane with validation, routing, and orchestration APIs.
- Queue/Workers: Redis + Celery for asynchronous execution and retries.
- Data: PostgreSQL for jobs, scheduling outcomes, and training experiences.
- AI Layer: PPO policy + explainability + rule guard + simulator.
- Infra: Docker, Kubernetes (AKS), Prometheus, Grafana.

## Repository Structure

```text
SmartGPU-orchestrator/
├── backend/                # FastAPI, routes, services, scheduler placeholders, worker
├── simulator/              # GPU simulator placeholder
├── rl_engine/              # RL agents/models/training scaffolding
├── monitoring/             # Prometheus/Grafana scaffolding
├── infrastructure/         # Azure and Docker infra scaffolding
├── frontend_dashboard/     # Planned frontend app (currently empty)
├── docs/                   # Planned project docs (currently empty)
├── docker-compose.yml
├── Full report.md
└── README.md
```

## Local Development

### Prerequisites

- Python 3.11+
- Docker + Docker Compose

### 1) Start infrastructure services

From the project root:

```bash
docker compose up -d db redis
```

### 2) Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3) Run FastAPI app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4) Run Celery worker

In a second terminal from `backend`:

```bash
celery -A celery_app.celery worker --loglevel=info
```

API root: `http://localhost:8000`

## Basic API Example

Submit a job:

```http
POST /jobs/
Content-Type: application/json

{
	"model_name": "resnet50",
	"memory_required": 8,
	"priority": "high"
}
```

Check status:

```http
GET /jobs/{job_id}
```

Get mock GPU status:

```http
GET /gpus/
```

## Progressive Roadmap

This project is intentionally being built in phases:

1. Foundation: backend skeleton, data flow, queueing.
2. Simulator-first development for safe/offline AI training.
3. RL decision engine and explainability.
4. Comparison mode and measurable value demonstration.
5. Cloud-native deployment and observability.
6. Dashboard and UX completion.

## Notes for Reviewers

- The report describes the target architecture and research direction.
- The codebase currently reflects an in-progress implementation of that design.
- Placeholders are expected and deliberate at this stage.

## License

License to be finalized.
