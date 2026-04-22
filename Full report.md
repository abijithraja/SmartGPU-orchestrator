SmartGPU Orchestrator — Project Report
AI-Driven GPU Resource Management
Cloud-Based AI Infrastructure System | Full Project Report — Architecture, Implementation & Competitive Analysis
AI Engine: PPO RL Agent + Explainability | Infrastructure: Docker + Kubernetes + Azure AKS | Monitoring: Prometheus + Grafana + Real-Time

1. Executive Summary
One-Line Pitch: SmartGPU Orchestrator is a cloud-based AI infrastructure platform that uses a trained Reinforcement Learning agent to intelligently allocate GPU resources to AI training jobs — making every scheduling decision explainable, cost-tracked, and self-improving.
SmartGPU Orchestrator is a production-grade platform that solves a real and expensive problem: GPU scheduling inefficiency. Companies waste millions of dollars annually on idle or poorly utilised GPU clusters because existing schedulers use static rules — round-robin, FIFO, or fixed priority queues — that cannot adapt to dynamic workload patterns.
SmartGPU replaces static heuristics with a Proximal Policy Optimisation (PPO) reinforcement learning agent that observes live GPU metrics, scores all available nodes, selects the optimal assignment, explains why it made that choice, estimates the cost impact, and retrains itself every 500 jobs. It is the only system in its class that gets smarter with every scheduling decision it makes.
AspectRatingNotesProblem Relevance★★★★★ StrongGPU waste is a real, costly problem at scaleTech Stack★★★★★ StrongDocker + K8s + Prometheus + Grafana + Azure — industry standardAI Engine★★★★★ DefinedPPO agent (Stable-Baselines3), trained on GPU simulatorExplainability★★★★★ UniqueHuman-readable reason per scheduling decisionSelf-Improvement★★★★★ UniqueRetrains every 500 jobs from live experience dataScope Realism★★★★☆ ManagedCold start + hard rule guard + honest limitations statedDemo-ability★★★★☆ SolidGPU simulator enables offline demo without cloud cost

2. Problem Statement
2.1 Current State of GPU Management
Modern AI infrastructure teams manage GPU clusters through one of three approaches: static priority queues (FIFO), manual resource allocation by engineers, or vendor black-box schedulers embedded in cloud platforms like AWS SageMaker or Google Vertex AI. Each of these fails at a different level.
Three Root Failures:

Static schedulers cannot adapt to workload patterns. A round-robin scheduler that assigns the next job to GPU-0 has no knowledge that GPU-0 is at 89% utilisation and GPU-3 is idle. It assigns anyway, causing job slowdowns or OOM crashes.
Cloud platform schedulers are opaque. When SageMaker picks a compute node, you receive no explanation and have no ability to override or improve the logic. Every scheduling decision is a black box.
GPU waste is expensive and measurable. An A100 GPU on Azure costs ~$3.00 per hour. A cluster of 8 running at 50% utilisation wastes $12/hour — $105,000 per year on a single cluster. Intelligent scheduling directly reduces this figure.

2.2 Business Impact
MetricCurrent State (Naive Scheduling)SmartGPU TargetGPU Utilisation45–60% average75–90% averageOOM-caused Job Failures8–15% of jobs< 2% of jobsCost vs Optimal30–40% waste< 10% wasteScheduling TransparencyNone — black boxFull — reason per decisionSystem LearningNever improvesRetrains every 500 jobs

3. Solution Overview
SmartGPU Orchestrator is a full-stack AI scheduling platform built on a five-stage pipeline. Every stage feeds the next, and the entire loop closes back on itself through the training data logging and retraining system.
StepAction1User submits job via React dashboard, CLI, or REST API2FastAPI validates, assigns UUID, queues in Redis3AI Engine reads live GPU metrics from Prometheus4PPO agent scores all GPUs; Rule Guard filters unsafe candidates5Explainability engine generates human-readable reason for selection6Cost estimator computes predicted cost vs. round-robin baseline7Experience row logged to PostgreSQL (state, action, reward=NULL)8Kubernetes schedules Docker container on chosen GPU node9Health check monitors every 30s; Recovery branch handles failures10On completion: actual cost + duration recorded, reward computed11Dashboard refreshes: shows savings, AI decision, GPU utilisation12Every 500 rows: PPO agent retrains; updated model hot-reloaded

4. System Architecture
4.1 Architecture Layers
Frontend Layer — React + Vite Dashboard: React + Vite frontend with four primary views: Job Submission Form, GPU Status Grid (live utilisation, memory, temperature per node), AI Decision Panel (which GPU was selected and why), and Comparison Table (RL scheduler vs. round-robin side-by-side). Recharts provides live updating visualisations. The dashboard polls the FastAPI backend every 10 seconds.
Backend Layer — FastAPI Central Coordinator: FastAPI handles job submission validation (JWT auth, parameter bounds checking, memory feasibility), writes job records to PostgreSQL, publishes job IDs to the Redis queue, and exposes GET endpoints for GPU status, job status, and the AI decision log. A FastAPI background task triggers the retrain check after every completed job.
AI Intelligence Layer — PPO Agent + Simulator: Three sub-components: the GPU Simulator (generates realistic synthetic GPU metrics), the PPO Agent (Stable-Baselines3, trained on the simulator for 50,000 steps), and the Explainability Engine (generates human-readable decision breakdown with confidence scores).
Infrastructure Layer — Kubernetes + Azure AKS: Each AI training job runs in a Docker container with explicit resource limits. The AI scheduler's decision is encoded as a Kubernetes node affinity rule. Prometheus with the NVIDIA GPU Exporter scrapes metrics every 15 seconds. Grafana provides dashboards for utilisation curves, memory consumption, and temperature trends.
4.2 Folder Structure
smartgpu/
├── frontend/               # React + Vite dashboard
│   └── src/components/
│       ├── JobSubmit.jsx
│       ├── GPUStatusGrid.jsx
│       ├── AIDecisionPanel.jsx    ← THE differentiator
│       └── ComparisonTable.jsx
├── backend/                # FastAPI application
│   ├── api/               # Route handlers
│   ├── scheduler/
│   │   ├── rl_agent.py        ← PPO model (Stable-Baselines3)
│   │   ├── simulator.py       ← GPU simulator for training
│   │   ├── explainer.py       ← Why this GPU? engine
│   │   ├── rule_guard.py      ← Hard constraint filter
│   │   ├── cost_estimator.py  ← Azure rate × duration
│   │   └── recovery_monitor.py← Celery beat, 30s health poll
│   ├── models/
│   │   └── rl_experience.py   ← SQLAlchemy model for log table
│   └── monitoring/
├── k8s/                    # Kubernetes manifests
├── docker/                 # Dockerfiles
├── training/               # RL training scripts
│   └── train_agent.py     ← 50,000 step training on simulator
└── docs/
    └── architecture.md

5. The AI Engine (Core Differentiator)
5.1 GPU Simulator
The simulator solves three problems simultaneously: provides training data for the PPO agent without live GPU hardware; enables realistic demo without burning Azure credits; and generates thousands of historical GPU/job pairs for training.
pythonclass SimulatedGPU:
    id, name, total_memory   # Node identity
    current_utilisation      # Spikes when job starts, decays when done
    free_memory              # Drops by job_mem_req when job begins
    temperature              # Follows load with 2-step lag
    queue_depth              # Jobs waiting on this node

    def step(self):          # Called every 15s simulation tick
        self.utilisation = smooth(self.base + active_job_load)
        self.temperature = lag_filter(self.utilisation)
5.2 PPO Agent (Proximal Policy Optimisation)
ComponentDefinitionState vector per GPU[utilisation, free_memory, temperature, queue_depth]Job vector[required_memory, compute_intensity, priority_flag]ActionGPU index (0 to N-1) — which GPU to assignReward ++1 proportional to completion speedup vs. round-robinReward −−2 if OOM crash occurs; −0.5 for thermal throttlingTraining50,000 steps on simulator; ~30 min on CPU
5.3 Rule Guard (Hybrid Safety Layer)
pythondef apply_rules(job, gpu_states, rl_scores):
    valid = []
    for gpu, score in zip(gpu_states, rl_scores):
        if job.required_memory > gpu.free_memory: continue   # OOM guaranteed
        if gpu.temperature > 85:               continue   # Thermal risk
        if gpu.utilisation > 95:               continue   # Saturated
        valid.append((gpu, score))
    if not valid:
        return None   # Hold job in queue, retry in 60s
    return max(valid, key=lambda x: x[1]).gpu
This design is constrained reinforcement learning: the agent provides intelligent preference ordering; the rules provide safety guarantees.
5.4 Explainability Engine
Example Explainability Output:

GPU-2 selected — Free memory: 18.4 GB (highest available), Utilisation: 22% (lowest active node), Temperature: 41°C (well within safe range), Predicted completion: 26% faster than round-robin baseline. Confidence score: 0.87. Rejected: GPU-0 (OOM risk: only 6.1 GB free vs. 8.0 GB required), GPU-1 (thermal risk: 87°C).


6. Three Key Improvements
6.1 Cost Optimisation Metric
pythonGPU_RATES = {
    'Standard_NC6':   0.90,   # USD/hour on Azure
    'Standard_NC12':  1.80,
    'Standard_NC24':  3.60,
}

def estimate_cost(gpu_sku: str, duration_seconds: float) -> float:
    rate = GPU_RATES.get(gpu_sku, 1.0)
    return round((duration_seconds / 3600) * rate, 4)
6.2 Failure Recovery (Self-Healing Infrastructure)
A Celery beat task fires every 30 seconds. If a failure is detected — OOM error, GPU hardware fault, pod crash — the recovery branch activates: the failed GPU is masked (forced to score 0), the AI agent reschedules to the next best GPU, and the job retries up to 3 times. After 3 failures, the job goes to dead-letter status and the user is notified.
6.3 Training Data Logging (Self-Improving System)
sql-- PostgreSQL: rl_experience table
id            UUID PRIMARY KEY
timestamp     TIMESTAMPTZ
state_json    JSONB   -- GPU metrics + job requirements at decision time
action_gpu_id TEXT    -- Which GPU was selected
action_score  FLOAT   -- Confidence from the model
reward        FLOAT   -- Computed after job completes (initially NULL)
completion_s  FLOAT   -- Actual job duration
baseline_s    FLOAT   -- What round-robin would have taken
oom_occurred  BOOLEAN
job_id        UUID REFERENCES jobs(id)
retrain_used  BOOLEAN -- Flipped after this row used in retraining
Reward formula: reward = (baseline_s - completion_s) / baseline_s - (2.0 if oom_occurred else 0.0)
Once 500 unused rows accumulate, the PPO agent retrains, saves new weights, and hot-reloads into the scheduler. The system literally gets smarter as it runs.

7. Safety, Cold Start & Engineering Maturity
7.1 Cold Start Strategy
pythondef schedule(job, gpu_states):
    if experience_count() < COLD_START_THRESHOLD:   # Default: 500 rows
        return round_robin_fallback(gpu_states)     # Safe, predictable
    return rl_agent.predict(build_state_vector(job, gpu_states))
Why 500 rows? A typical cluster processes 10–50 jobs per day. 500 rows represents 10–50 days of live experience — enough to observe a meaningful distribution of workloads. Every round-robin decision during cold start is still logged, actively building the dataset the PPO agent will train on.
7.2 Scaling Limitations (Engineering Honesty)

Fixed state vector size. The PPO agent is trained with a fixed number of GPU slots. Adding a new GPU node requires retraining. V2 path: a Graph Neural Network (GNN) policy that handles variable cluster sizes.
Single-cluster scope. SmartGPU operates within one Azure AKS cluster. No cross-region or cross-cloud spillover. V2: hierarchical agent architecture (cluster router + per-cluster schedulers).
Exclusive GPU assignment. One job per GPU at a time. NVIDIA MIG fractional GPU partitioning is out of scope for v1 — a deliberate simplicity trade-off.


8. Competitive Landscape & Differentiation
8.1 Existing Products

NVIDIA Base Command Platform — Static priority queues and fairshare scheduling. No learning from workload history. No explainability. Requires NVIDIA hardware lock-in. Enterprise pricing: tens of thousands per year.
Run:ai — Kubernetes-native GPU scheduler. Supports fractional GPU sharing. Scheduling is quota and priority-based, not AI-driven. No self-healing, no per-job cost tracking.
AWS SageMaker — Compute node selection is entirely AWS-controlled and opaque. No visibility into why a machine was chosen. No custom scheduling logic. Locked to AWS infrastructure.
Google Vertex AI — Managed, opaque, GCP-locked. Auto-scaling at VM level, not GPU-assignment level. No explainability or custom intelligence.
Slurm (HPC Scheduler) — Industry standard for HPC. Rule-based, queue-based, static configuration. No learning, no explainability, no cost tracking.
Volcano (Kubernetes Batch) — CNCF open-source batch scheduler. Gang scheduling support. Policy-based, not AI-driven. No cost tracking, no failure recovery, no explainability.
Kubeflow Pipelines — Orchestrates ML workflow pipelines. Delegates actual compute scheduling to the default Kubernetes scheduler. No GPU-aware intelligence beyond resource request matching.

8.2 Capability Comparison Matrix
CapabilityNVIDIA BCRun:aiSageMakerSlurmVolcanoSmartGPUAI-Driven SchedulingNoNoNoNoNoYes (PPO)Learns from HistoryNoNoNoNoNoYes (500-row retrain)ExplainabilityNoNoNoNoNoYes (per-decision)Per-Job Cost TrackingLimitedNoCloud billingNoNoYes + vs. baselineSelf-Healing RecoveryNoNoPartialNoNoYes (auto-reschedule)Cold Start SafetyN/AN/AN/AN/AN/AYes (round-robin fallback)Hard Rule GuardN/AN/AN/AYesN/AYes (hybrid RL + rules)Open & CustomisableNoNoNoYesYesYesIndividual-Dev ViableNoNoNoNoNoYes (Azure credits)
Core Differentiator: Every other system uses static rules or opaque cloud logic. None have an agent that observes outcomes, computes a reward, and retrains itself. None tell you why a GPU was chosen. SmartGPU is the only system that gets smarter over time, explains every decision, prevents memory failures, recovers automatically from hardware faults, and shows exactly how much money the intelligent decision saved.

9. Build Phases — Start to Finish
PhaseDurationGoalDeliverablePhase 1Week 1–2Foundation: skeleton running locallyFastAPI + PostgreSQL + Redis + React (mocked data) in Docker ComposePhase 2Week 3GPU Simulator (most critical phase)Python SimulatedGPU class generating realistic metric patterns; 50k-step training datasetPhase 3Week 4–5AI Engine: PPO agent + explainabilityTrained PPO model; explainability output; comparison mode vs. round-robinPhase 4Week 6Comparison Mode (killer demo feature)Side-by-side dashboard: RL vs. round-robin, % improvement per jobPhase 5Week 7–8Kubernetes + Azure deploymentAKS cluster; Prometheus NVIDIA exporter; Grafana dashboards; real GPU jobsPhase 6Week 9Dashboard polish + AI Decision PanelComplete UI with explainability panel; savings counter; confidence scores
Phase 2 Priority Note: The GPU Simulator is the most strategically important phase. Without it, you will burn Azure credits just testing. Build the simulator first, train the PPO agent entirely on simulated data, and use it for demos throughout development. Only in Phase 5 do you connect to real Azure GPU hardware.

10. Complete Technology Stack
LayerTechnologyPurposeFrontendReact + Vite + RechartsDashboard, job submission form, AI decision panel, comparison tableBackendFastAPI (Python)Same language as ML layer; clean async API; JWT auth; job validationAI EngineStable-Baselines3 (PPO)Genuine RL agent; trainable offline on simulator; explainableJob QueueRedis + CeleryPriority job queue with retry logic; Celery beat for health checksDatabasePostgreSQLJob records, rl_experience log, training data, cost historyContainersDockerReproducible job execution environments; dependency isolationOrchestrationKubernetes (Azure AKS)Container scheduling; node affinity rules; resource limitsCloudMicrosoft AzureAKS cluster; Standard_NC GPU VM nodes; free tier for testingMonitoringPrometheus + NVIDIA Exporter15-second GPU metric scraping; alerting; RL state constructionVisualisationGrafanaReal-time GPU dashboards; utilisation curves; temperature trendsGPU SimulatorCustom PythonOffline training; realistic metric patterns; demo without cloud costRL TrainingStable-Baselines3 PPO50,000 training steps; ~30 min on CPU; hot-reloadable weights

11. What Makes This an A-Grade Project
A Concrete, Defensible AI Model: PPO (Proximal Policy Optimisation) via Stable-Baselines3 — not a heuristic, not a classifier with an 'AI' label. You can describe the state vector, the action space, the reward function, and the training process clearly.
A Measurable Result (Comparison Mode): The comparison table showing RL scheduler vs. round-robin side-by-side — with actual job names, GPU assignments, completion times, and % improvement — gives you a concrete, data-backed claim.
An Explainability Panel: The AI Decision Panel is the feature that no other GPU scheduler — open source or commercial — has. It makes the AI transparent, trustworthy, and uniquely yours.
SmartGPU in One Sentence: A user submits an AI training job → FastAPI validates it → Redis queues it → the PPO agent reads live GPU metrics and selects the optimal GPU → explains the decision in plain English → estimates cost savings → logs the experience → Kubernetes runs the job in Docker on Azure → health checks protect it → Prometheus monitors it → on completion the reward is computed → the dashboard shows savings → every 500 jobs the model retrains itself and gets measurably smarter.
