import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/smartgpu",
)

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/rl_engine/training/ppo_smartgpu",
)

COLD_START_THRESHOLD = int(
    os.getenv(
        "COLD_START_THRESHOLD",
        "0"
    )
)