import os
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.robotic_arm_env import CustomEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

urdf      = "/home/arjan/arm_rl/urdf/robotic_arm/urdf/robotic_arm.urdf"
path      = "models/ppo_robotic_arm"
opt_path  = "models/ppo_robotic_arm_opt"
n_envs    = 4
params_file = "tuning/best_params_latest.json"

def make_env():
    def _init():
        return Monitor(CustomEnv(urdf, render=False))
    return _init

if __name__ == "__main__":

    #Load best params
    if not os.path.exists(params_file):
        print(f"No params file found at {params_file}")
        print("Run tune.py first!")
        exit()

    with open(params_file, "r") as f:
        best = json.load(f)

    print(f"Loaded params from {params_file}")
    print(f"Best reward from tuning : {best.pop('best_reward'):.2f}")
    print(f"Params : {best}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = f"./logs/train_{timestamp}/"

    #Load or create model
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    if os.path.exists(path + ".zip"):
        print(f"\nLoading existing model from {path}...")
        model = PPO.load(path, env=env, tensorboard_log=log_path,
                         custom_objects={
                             "learning_rate": best["learning_rate"],
                             "clip_range":    best["clip_range"],
                         })
        model.gamma         = best["gamma"]
        model.gae_lambda    = best["gae_lambda"]
        model.ent_coef      = best["ent_coef"]
        model.vf_coef       = best["vf_coef"]
        model.max_grad_norm = best["max_grad_norm"]
    else:
        print("\nNo existing model, starting fresh...")
        model = PPO("MlpPolicy", env, verbose=1, device="cuda",
                    tensorboard_log=log_path,
                    learning_rate=best["learning_rate"],
                    gamma=best["gamma"],
                    gae_lambda=best["gae_lambda"],
                    clip_range=best["clip_range"],
                    ent_coef=best["ent_coef"],
                    vf_coef=best["vf_coef"],
                    batch_size=best["batch_size"],
                    n_steps=best["n_steps"],
                    max_grad_norm=best["max_grad_norm"])

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="models/",
        name_prefix="ppo_robotic_arm_opt",
    )

    print("\nStarting training (200k steps)...")
    model.learn(total_timesteps=200_000, reset_num_timesteps=False, callback=checkpoint_callback)
    model.save(opt_path)
    print(f"\nModel saved → {opt_path}.zip")
    print(f"Original untouched → {path}.zip")

    #Evaluation
    env.close()
    eval_env = DummyVecEnv([lambda: Monitor(CustomEnv(urdf, render=False))])

    print("\nStarting evaluation (Ctrl+C to stop)...")
    obs = eval_env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = eval_env.step(action)
            if dones[0]:
                obs = eval_env.reset()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        model.save(opt_path)
        print(f"Model saved → {opt_path}.zip")
        eval_env.close()