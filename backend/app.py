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
