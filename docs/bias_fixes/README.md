# SmartGPU — Model Bias Detection & Correction Reference

This directory contains **standalone, well-documented reference implementations** of all 8 bias fixes applied to the SmartGPU PPO scheduling model.

These files are **not** direct drop-in replacements for the production code; they extract and document the bias-correction logic in isolation for clarity, auditing, and reporting purposes.

---

## Fix Index

| Fix # | File | Bias Problem | Correction Mechanism |
|-------|------|-------------|---------------------|
| 1 | `train_agent.py` (L55-57) | Out-of-bounds observations destabilize gradient updates | `np.clip(obs, 0.0, 1.0)` — clamp all observations to `[0, 1]` |
| 2 | `train_agent.py` (L93-120) | Reward variables (`queue_penalty`, `temperature_penalty`) computed but **never used** | Unified all signals into a single reward expression |
| 3 | `train_agent.py` (L111-118) | Agent collapses to always picking GPU-0 (preference collapse) | Load-balance bonus: `+0.5` for choosing the least-loaded GPU |
| 4 | `train_agent.py` (L148-152) | Policy entropy collapses to deterministic single action | Set `ent_coef=0.01` (default was `0.0`) |
| 5 | [`worker_fix.py`](worker_fix.py) (L38-52) | Agent stubbornly picks a heavily loaded GPU at runtime | Hybrid override: force least-loaded GPU when util > 80% |
| 6 | [`worker_fix.py`](worker_fix.py) (L59-71) | Agent repeatedly assigns jobs to the same GPU | Reward penalty: `-5 × recent_assignment_count` (last 10 jobs) |
| 7 | [`worker_fix.py`](worker_fix.py) (L77-82) | Cluster-wide load imbalance goes uncorrected | Reward penalty: `-std_dev(all GPU utilizations)` |
| 8 | [`decision_engine_fix.py`](decision_engine_fix.py) + [`rule_guard_fix.py`](rule_guard_fix.py) | Raw RL scores still reflect residual training bias | Score-level queue/util/temp penalties + hard constraint filter |

---

## Where Fixes Live in Production

| Reference File | Production File | Key Function | Fixes |
|---------------|----------------|-------------|-------|
| [`train_agent_fix.py`](train_agent_fix.py) | `rl_engine/training/train_agent.py` | `SmartGPUEnv.step()`, `train()` | 1, 2, 3, 4 |
| [`worker_fix.py`](worker_fix.py) | `backend/worker.py` | `process_job()` | 5, 6, 7 |
| [`decision_engine_fix.py`](decision_engine_fix.py) | `backend/scheduler/decision_engine.py` | `schedule_job()` | 8 (score adjustment) |
| [`rule_guard_fix.py`](rule_guard_fix.py) | `backend/scheduler/rule_guard.py` | `apply_rules()` | 8 (hard constraints) |

---

## How to Verify Bias Corrections are Working

1. **GPU Distribution** — After 20+ jobs, check the frontend "Jobs Assigned Distribution" pie chart. All 4 GPUs should have roughly equal shares.
2. **Hybrid Override Logs** — Search worker logs for `[WORKER] Hybrid Override:`. Frequent overrides indicate the agent still has residual bias.
3. **RL Confidence Trend** — The confidence chart should show values between 0.3–0.8. Sustained 0.95+ indicates the agent is over-confident (possible collapse).
4. **Rule Guard Rejections** — Search logs for `[RULE GUARD] GPU ... excluded:`. These confirm the safety net is active.
