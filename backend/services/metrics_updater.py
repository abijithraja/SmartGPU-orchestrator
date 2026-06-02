import time

from monitoring import (
    update_gpu_metrics
)

from services.gpu_service import (
    get_gpu_status
)

def start_metrics_loop():

    while True:

        gpu_states = get_gpu_status()

        update_gpu_metrics(
            gpu_states
        )

        time.sleep(5)
