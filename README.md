# SmartGPU Orchestrator

SmartGPU Orchestrator is a cloud-based AI infrastructure platform designed to intelligently allocate GPU resources to AI training workloads. By leveraging a trained Reinforcement Learning (RL) agent, the system optimizes scheduling decisions to maximize resource utilization and minimize costs, making every allocation explainable and self-improving.

**Note: This is an ongoing research project exploring the application of reinforcement learning in cloud infrastructure orchestration.**

## Architecture

The system operates across a five-stage pipeline:
1. User submission via the dashboard.
2. FastAPI validates and queues the job in Redis.
3. The PPO Reinforcement Learning Agent scores all available GPUs based on live telemetry (utilization, free memory, temperature).
4. The Explainability Engine generates human-readable reasoning for the assignment, estimating cost savings.
5. Kubernetes schedules the Docker container, while Prometheus and Grafana track the live metrics.

```mermaid
flowchart TD
    User([User]) --> |Submits Job| API[FastAPI Backend]
    API --> |Queues Job| Redis[(Redis Queue)]
    Redis --> Worker[Celery Worker]
    
    subgraph AI Engine
        Worker --> PPO[PPO RL Agent]
        PPO --> Guard[Rule Guard]
        Guard --> Explainer[Explainability Engine]
    end
    
    subgraph Infrastructure
        Explainer --> K8s[Kubernetes / AKS]
        K8s --> GPU0[GPU Node 0]
        K8s --> GPU1[GPU Node 1]
    end
    
    subgraph Monitoring
        GPU0 --> Prom[Prometheus Exporter]
        GPU1 --> Prom
        Prom --> Grafana[Grafana Dashboard]
        Prom --> PPO
    end
```

## Key Features

- **AI-Driven Scheduling**: Replaces static heuristics with a Proximal Policy Optimization (PPO) reinforcement learning agent.
- **Explainability**: Provides human-readable reasoning per scheduling decision, ensuring the AI model is transparent.
- **Self-Improving**: Automatically logs experiences and retrains the model continuously.
- **Cost Tracking**: Computes predicted cost versus a round-robin baseline to measure actual financial impact.
- **Hybrid Safety Layer**: Includes a hard rule guard to prevent Out-Of-Memory (OOM) errors and thermal throttling, overriding the AI when necessary.

## Technology Stack

- **Frontend**: React, Vite, Recharts
- **Backend**: FastAPI, Python, Redis, Celery, PostgreSQL
- **AI Engine**: Stable-Baselines3 (PPO)
- **Infrastructure**: Docker, Kubernetes (Azure AKS)
- **Monitoring**: Prometheus, Grafana

## References

For detailed methodology, architecture analysis, and competitive landscape comparisons, please refer to the primary project documentation included in this repository:
- `Full report.md`
- `SmartGPU_Report.docx`
- `SMARTGPU_COMPLETE_CODE.md`

This research demonstrates that intelligent scheduling directly reduces GPU waste, prevents OOM-caused job failures, and significantly lowers computational costs compared to naive static schedulers.
