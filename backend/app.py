from fastapi import FastAPI

from database import models
from database.db import Base, engine
from routes import gpus, jobs

app = FastAPI(title="SmartGPU Orchestrator")

Base.metadata.create_all(bind=engine)

app.include_router(jobs.router)
app.include_router(gpus.router)


@app.get("/")
def root():
    return {"message": "SmartGPU API running"}
