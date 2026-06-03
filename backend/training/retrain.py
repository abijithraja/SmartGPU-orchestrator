import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
        )
    ),
)

from stable_baselines3 import PPO
from rl_engine.training.train_agent import SmartGPUEnv

MODEL_PATH = "/app/rl_engine/models/ppo_smartgpu"


def retrain():
    try:
        print("[RETRAIN] Loading environment...")

        env = SmartGPUEnv()

        print("[RETRAIN] Loading PPO model...")

        model = PPO.load(
            MODEL_PATH,
            env=env,
        )

        print("[RETRAIN] Retraining PPO for 5,000 steps...")

        model.learn(
            total_timesteps=100000,
            reset_num_timesteps=False,
        )

        print("[RETRAIN] Saving model...")

        model.save(MODEL_PATH)

        print("[RETRAIN] Retraining complete.")

    except Exception as e:
        import traceback

        print(f"[RETRAIN] Failed: {repr(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    retrain()
