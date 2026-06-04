import requests
import time

print("Submitting 1st job (will use round-robin fallback)...")
response = requests.post("http://localhost:8000/jobs/", json={"model_name": "test-1", "memory_required": 4, "compute_intensity": 0.5})
print("Response:", response.status_code, response.text)

time.sleep(1)

print("Submitting 2nd job (will trigger the RL model load!)...")
response = requests.post("http://localhost:8000/jobs/", json={"model_name": "test-2", "memory_required": 6, "compute_intensity": 0.5})
print("Response:", response.status_code, response.text)

print("Jobs submitted! The model should now be loaded in the worker logs.")
