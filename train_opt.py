import gymnasium as gym
from stable_baselines3 import PPO
from envs.robotic_arm_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime
import optuna
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

path = "models/ppo_robotic_arm"
opt_path="models/ppo_robotic_arm_opt"
urdf = "/home/arjan/arm_rl/urdf/robotic_arm/urdf/robotic_arm.urdf"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"./logs/optuna_{timestamp}/"
n_envs=4

def make_env():
    def _init():
        return Monitor(CustomEnv(urdf))
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

    env = DummyVecEnv([lambda: Monitor(CustomEnv(urdf))])
    # env=SubprocVecEnv([make_env() for _ in range(n_envs)])
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
                device="cpu")

    model.learn(total_timesteps=50000,reset_num_timesteps=False)    
    mean_reward, _ =evaluate_policy(model, env, n_eval_episodes=5, deterministic=True) #:use evaluate_policy libarary by importing it
    # total_reward=0.0
    # for _ in range(1000):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, _ = env.step(action)
    #     total_reward += reward[0]
    #     if done[0]:
    #         obs = env.reset()

    env.close()
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

#Final Trainining 
best = study.best_trial.params
print(f"  Best Reward : {study.best_trial.value:.2f}")
print(f" Best Params : {study.best_trial.params}")

env = DummyVecEnv([lambda: Monitor(CustomEnv(urdf))])
# env=SubprocVecEnv([make_env() for _ in range(n_envs)])


if os.path.exists(path +".zip"):
    model=PPO.load(path,env=env,tensorboard_log=log_path) #continue training from previous one
    model.learning_rate  = best["learning_rate"]
    model.gamma          = best["gamma"]
    model.gae_lambda     = best["gae_lambda"]
    model.clip_range     = best["clip_range"]
    model.ent_coef       = best["ent_coef"]
    model.vf_coef        = best["vf_coef"]
    model.max_grad_norm  = best["max_grad_norm"]

else:
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                learning_rate=best["learning_rate"],
                gamma=best["gamma"],
                gae_lambda=best["gae_lambda"],
                clip_range=best["clip_range"],
                ent_coef=best["ent_coef"],
                vf_coef=best["vf_coef"],
                batch_size=best["batch_size"],
                n_steps=best["n_steps"],
                max_grad_norm=best["max_grad_norm"],
                device="cpu",
                tensorboard_log=log_path)


checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="models/", name_prefix="ppo_robotic_arm_x")

print("Starting final training...")
model.learn(total_timesteps=200000, reset_num_timesteps=False, callback=checkpoint_callback)
model.save(opt_path)
print(f"Model saved to {opt_path}.zip")
# env.close()
# eval_env=SubprocVecEnv([make_env()])  #single env for eval
#Evaluation
print("Starting evaluation...")
obs = env.reset()
try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones[0]:
            obs = env.reset()

except KeyboardInterrupt:
    print("\nInterrupted during evaluation.")

finally:
    model.save(opt_path)
    print(f"Model saved to {opt_path}.zip")
    env.close()