import gymnasium as gym
from stable_baselines3 import PPO
from envs.robotic_arm_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime
import json
import optuna
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("tuning", exist_ok=True)

urdf = "/home/arjan/arm_rl/urdf/robotic_arm/urdf/robotic_arm.urdf"
n_envs=4

def make_env():
    def _init():
        return Monitor(CustomEnv(urdf,render=False))
    return _init

# Optuna Tuning 
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma         = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda    = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range    = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef      = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    vf_coef       = trial.suggest_float("vf_coef", 0.3, 0.9)
    batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps       = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    # env = DummyVecEnv([lambda: Monitor(CustomEnv(urdf))])
    env=SubprocVecEnv([make_env() for _ in range(n_envs)])
    model = PPO("MlpPolicy", 
                env, 
                verbose=0,
                learning_rate=learning_rate, 
                gamma=gamma, 
                gae_lambda=gae_lambda,
                clip_range=clip_range, 
                ent_coef=ent_coef, 
                vf_coef=vf_coef,
                batch_size=batch_size, 
                n_steps=n_steps, 
                max_grad_norm=max_grad_norm,
                device="cuda")

    model.learn(total_timesteps=50000,reset_num_timesteps=False)    
    mean_reward, _ =evaluate_policy(model, env, n_eval_episodes=5, deterministic=True) #:use evaluate_policy libarary by importing it

    env.close()
    return mean_reward
if __name__=="__main__":
    print("Starting Optuna hyperparameter search...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    best = study.best_trial.params
    best["best_reward"] = study.best_trial.value

    # Save best params to JSON
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_parms= f"tuning/best_params_{timestamp}.json"
    latest_parms= "tuning/best_params_latest.json"

    with open(tuning_parms,"w") as f:
        json.dump(best, f, indent=4)

    with open(latest_parms, "w") as f:
        json.dump(best, f, indent=4)

    print(f"\nBest Reward : {study.best_trial.value:.2f}")
    print(f"Best Params : {best}")
    print(f"\nSaved → {tuning_parms}")
    print(f"Saved → {latest_parms}  (always latest)")
