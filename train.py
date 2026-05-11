import gymnasium as gym
from stable_baselines3 import PPO
from envs.robotic_arm_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

urdf="/home/arjan/arm_rl/urdf/robotic_arm/urdf/robotic_arm.urdf"
path="models/ppo_robotic_arm"
# env=CustomEnv(urdf)
# env=Monitor(env)  #Tracks reward for TensorBoard
env=DummyVecEnv([lambda:Monitor(CustomEnv(urdf))])

if os.path.exists(path+".zip"):
    print("Loading existing model...")
    model = PPO.load(path, env=env)
else:
    print("Starting fresh...")
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                batch_size=64,
                n_steps=2048,
                max_grad_norm=0.5,
                device="cpu",
                tensorboard_log="./logs/")

checkpoint_callback=CheckpointCallback(save_freq=5000, save_path="models/",name_prefix="ppo_robotic_arm")

print("Starting Training..")
model.learn(total_timesteps=200000, reset_num_timesteps=False,callback=checkpoint_callback) # reset_timesteps=False :-for continue learning

print("Training completed")
model.save(path)

obs= env.reset()

print("Model evaluation")
#Model Evaluation
try:
    while True:
        action, _states = model.predict(obs, deterministic=True) #Predict the action
        obs, rewards, dones, info = env.step(action) #Steps through the env (dones combine terminated and turncated)

        if dones[0]:
            obs =env.reset()

except KeyboardInterrupt:
    print("\n Saving current model ")

finally:
    env.close()
    print("Saving model...")
    model.save(path)  # Always runs
    print(f"Model saved to {path}.zip")

