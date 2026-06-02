import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "rl_engine",
        "training",
        "ppo_smartgpu",
    )
)


def retrain():
    """
    Incrementally retrain the PPO model for 5,000 steps
    without resetting the learning counter.
    """
    try:
        from stable_baselines3 import PPO

        model = PPO.load(MODEL_PATH)

        print("[RETRAIN] Retraining PPO for 5,000 steps...")

        model.learn(
            total_timesteps=5000,
            reset_num_timesteps=False,
        )

        model.save(MODEL_PATH)

        print("[RETRAIN] Retraining complete.")

    except Exception as e:
        print(f"[RETRAIN] Failed: {e}")
