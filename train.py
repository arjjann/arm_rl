import gymnasium as gym
from stable_baselines3 import PPO
from envs.robotic_arm_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

urdf="/home/arjan/arm_rl/urdf/robotic_arm/urdf/robotic_arm.urdf"
# env=CustomEnv(urdf)
# env=Monitor(env)  #Tracks reward for TensorBoard
env=DummyVecEnv([lambda:Monitor(CustomEnv(urdf))])

model=PPO("MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,# learning rate (try smaller if unstable)
            gamma=0.99,# discount factor (how far future rewards matter)
            gae_lambda=0.95,# GAE smoothing (stability of advantage estimates)
            clip_range=0.2, # PPO clipping (controls policy update size)
            ent_coef=0.01,  # entropy bonus (exploration strength)
            vf_coef=0.5, # value function loss weight
            batch_size=64, # batch size per update
            n_steps=2048, # how many steps collected before update
            max_grad_norm=0.5, # gradient clipping (stability)
            device="cpu",
            tensorboard_log="./logs/"
                    )
model.learn(total_timesteps=200000, reset_num_timesteps=False) # reset_timesteps=False :-for continue learning
model.save("models/ppo_robotic_arm")
obs= env.reset()

try:
    while True:
        action, _states = model.predict(obs, deterministic=True) #Predict the action
        obs, rewards, dones, info = env.step(action) #Steps through the env (dones combine terminated and turncated)

        if dones[0]:
            obs =env.reset()

except KeyboardInterrupt:
    print("\n Saving current model ")
    model.save("models/ppo_robotic_arm_interrupt")

finally:
    env.close()
    print("Saving model...")
    model.save("models/ppo_robotic_arm")  # Always runs
    print("Model saved to models/ppo_robotic_arm")

