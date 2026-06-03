import time
import uuid
from datetime import datetime

from celery_app import celery

from config import COLD_START_THRESHOLD

from database.db import SessionLocal
from database.models import Job, RLExperience

from services.gpu_service import (
    get_gpu_status,
    assign_job_to_gpu,
    complete_job_on_gpu,
)

from scheduler.decision_engine import (
    schedule_job,
    get_confidence,
)

from scheduler.cost_estimator import (
    estimate_ai_duration,
    estimate_baseline_duration,
    compute_savings,
)

from monitoring import (
    record_job_processed,
    update_job_confidence,
    update_job_costs,
    update_cluster_alerts,
)


@celery.task
def process_job(job_id: str):

    db = SessionLocal()

    try:

        parsed_job_id = uuid.UUID(job_id)

        job = (
            db.query(Job)
            .filter(Job.id == parsed_job_id)
            .first()
        )

        if not job:
            return

        job.status = "running"
        job.started_at = datetime.utcnow()

        db.commit()

        gpu_states = get_gpu_status()

        experience_count = (
            db.query(RLExperience)
            .count()
        )

        selected_gpu, scores, used_rl = schedule_job(
            gpu_states=gpu_states,
            job_memory=job.memory_required,
            job_intensity=job.compute_intensity,
            experience_count=experience_count,
            cold_start_threshold=COLD_START_THRESHOLD,
        )

        if selected_gpu is None:
            job.status = "queued"
            db.commit()
            return

        # --- Hybrid Scheduler Override ---
        # If the RL agent stubbornly picks a heavily loaded GPU, 
        # force selection of the least loaded valid GPU to balance the cluster.
        if selected_gpu["utilization"] > 80:
            available_gpus = [g for g in gpu_states if g["free_memory"] >= job.memory_required and g["utilization"] < 80]
            if available_gpus:
                selected_gpu = min(available_gpus, key=lambda g: g["utilization"])
                used_rl = False
                print(f"[WORKER] Hybrid Override: Forced least loaded GPU {selected_gpu['id']}")

        # --- OOM Detection ---
        if job.memory_required > selected_gpu["free_memory"]:

            job.oom_occurred = True

            print(
                f"[WORKER] OOM detected for job {job.id} "
                f"({job.memory_required}GB > "
                f"{selected_gpu['free_memory']}GB free)"
            )

            # OOM Recovery: reduce memory by 20% and retry
            job.memory_required = int(
                job.memory_required * 0.8
            )

            job.status = "queued"

            db.commit()

            process_job.delay(str(job.id))

            return

        # --- Migration Check ---
        from scheduler.migration_engine import (
            should_migrate_job
        )

        if should_migrate_job(
            selected_gpu["utilization"],
            selected_gpu["temperature"],
        ):

            print(
                f"[WORKER] MIGRATING JOB {job.id} "
                f"(util={selected_gpu['utilization']}%, "
                f"temp={selected_gpu['temperature']}C)"
            )

            job.status = "queued"

            db.commit()

            process_job.delay(str(job.id))

            return

        # --- Smart Alerts ---
        if selected_gpu["temperature"] > 80:
            update_cluster_alerts(1)
        else:
            update_cluster_alerts(0)

        confidence = get_confidence(scores)

        assign_job_to_gpu(
            selected_gpu["id"],
            job.memory_required,
            job.compute_intensity,
        )

        ai_duration = estimate_ai_duration(
            job.memory_required,
            job.compute_intensity,
            selected_gpu["utilization"],
        )

        baseline_duration = estimate_baseline_duration(
            job.memory_required,
            job.compute_intensity,
        )

        cost_data = compute_savings(
            selected_gpu["sku"],
            ai_duration,
            baseline_duration,
        )

        job.assigned_gpu_id = selected_gpu["id"]
        job.assigned_gpu_sku = selected_gpu["sku"]

        job.ai_confidence = confidence

        job.actual_duration_s = ai_duration
        job.baseline_duration_s = baseline_duration

        job.actual_cost = cost_data["ai_cost_usd"]
        job.baseline_cost = cost_data["baseline_cost_usd"]

        job.predicted_cost = cost_data["ai_cost_usd"]

        job.ai_explanation = (
            f"GPU {selected_gpu['id']} selected. "
            f"Free Memory: {selected_gpu['free_memory']}GB, "
            f"Utilization: {selected_gpu['utilization']}%, "
            f"Temperature: {selected_gpu['temperature']}C, "
            f"Queue Depth: {selected_gpu['queue_depth']}. "
            f"Decision source: "
            f"{'PPO Reinforcement Learning' if used_rl else 'Round Robin'}."
        )

        db.commit()

        experience = RLExperience(
            state_json={
                "gpu_states": gpu_states
            },
            action_gpu_id=selected_gpu["id"],
            action_score=confidence,
            completion_s=ai_duration,
            baseline_s=baseline_duration,
            retrain_used=used_rl,
            job_id=job.id,
        )

        # Multi-objective reward for PPO learning
        reward = 0

        # 1. Base Savings Bonus
        reward += (
            cost_data["savings_usd"] * 100
        )

        # 2. Resource/Utilization Penalty (Fix #1)
        reward += (selected_gpu["free_memory"] * 0.4)
        reward -= (selected_gpu["utilization"] * 0.3)

        # 3. Queue & Temperature Penalties
        reward -= (selected_gpu["queue_depth"] * 2)
        reward -= (selected_gpu["temperature"] * 0.05)

        # 4. Repeated Assignment Penalty (Fix #2)
        recent_jobs = db.query(Job).filter(Job.assigned_gpu_id.isnot(None)).order_by(Job.created_at.desc()).limit(10).all()
        recent_assignment_count = sum(1 for j in recent_jobs if j.assigned_gpu_id == selected_gpu["id"])
        reward -= (recent_assignment_count * 5)

        # 5. Load Balance Bonus (Fix #4)
        import numpy as np
        std_dev = np.std([g["utilization"] for g in gpu_states])
        reward -= float(std_dev)

        experience.reward = round(
            reward,
            3
        )

        db.add(experience)

        db.commit()

        # Auto-retrain PPO every 50 experiences
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from backend.tasks.retrain_task import (
            retrain_rl_model
        )

        experience_count = (
            db.query(RLExperience)
            .count()
        )

        if experience_count % 50 == 0:
            retrain_rl_model.delay()

        update_job_confidence(
            confidence
        )

        update_job_costs(
            cost_data["ai_cost_usd"],
            cost_data["baseline_cost_usd"],
            cost_data["savings_usd"],
        )

        run_time = max(
            30,
            int((job.memory_required * job.compute_intensity) / 10)
        )

        print(
            f"[WORKER] Running job for {run_time}s"
        )

        # Resume from checkpoint if retrying
        start_progress = (
            job.checkpoint_progress or 0
        )

        start_tick = int(
            (start_progress / 100) * run_time
        )

        # Update progress each second for live progress bar
        for i in range(start_tick, run_time):
            job.progress = int(
                ((i + 1) / run_time) * 100
            )

            # Checkpoint every tick
            job.checkpoint_progress = (
                job.progress
            )

            db.commit()
            time.sleep(1)

        complete_job_on_gpu(
            selected_gpu["id"],
            job.memory_required,
        )

        job.status = "completed"

        job.completed_at = datetime.utcnow()

        db.commit()

        record_job_processed()

        print(
            f"[WORKER] Completed {job.id}"
        )

    except Exception as e:

        print(
            f"[WORKER] Job {job_id} failed: {e}"
        )

        # Re-fetch job in case of stale state
        job = (
            db.query(Job)
            .filter(Job.id == uuid.UUID(job_id))
            .first()
        )

        if job:

            job.gpu_failed = True
            job.failure_reason = str(e)

            # Automatic retry (up to 3 attempts)
            if job.retry_count < 3:

                job.retry_count += 1

                job.status = "queued"

                db.commit()

                print(
                    f"[WORKER] Retrying job {job.id} "
                    f"(attempt {job.retry_count}/3)"
                )

                process_job.delay(str(job.id))

            else:

                job.status = "dead"

                db.commit()

                print(
                    f"[WORKER] Job {job.id} marked DEAD "
                    f"after 3 retries"
                )

    finally:
        db.close()