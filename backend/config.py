import os

DATABASE_URL = os.getenv(
	"DATABASE_URL",
	"postgresql://user:password@localhost:5432/smartgpu",
)

REDIS_URL = os.getenv(
	"REDIS_URL",
	"redis://localhost:6379/0",
)
