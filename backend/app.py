from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

import time
from sqlalchemy.exc import OperationalError

from database import models
from database.db import Base, engine, SessionLocal

from routes import gpus, jobs

from monitoring import update_jobs_metrics

from services.metrics_service import (
    get_jobs_processed,
    get_jobs_running,
    get_jobs_queued,
)

# Wait for PostgreSQL before creating tables
for attempt in range(30):
    try:
        Base.metadata.create_all(bind=engine)
        print("Database connected successfully")
        break
    except OperationalError:
        print(f"Waiting for PostgreSQL... ({attempt + 1}/30)")
        time.sleep(2)
else:
    raise RuntimeError("Could not connect to PostgreSQL")

import threading

from services.metrics_updater import (
    start_metrics_loop
)

threading.Thread(
    target=start_metrics_loop,
    daemon=True
).start()

app = FastAPI(
    title="SmartGPU Orchestrator",
    description="AI-driven GPU resource management with PPO reinforcement learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(gpus.router)


@app.get("/")
def root():
    return {
        "message": "SmartGPU Orchestrator API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.get("/metrics")
def metrics():

    db = SessionLocal()

    try:
        processed = get_jobs_processed(db)

        running = get_jobs_running(db)

        queued = get_jobs_queued(db)

        update_jobs_metrics(
            processed=processed,
            running=running,
            queued=queued,
        )

    finally:
        db.close()

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )