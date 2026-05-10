from stable_baselines3 import PPO
from envs.robotic_arm_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

urdf = "/home/arjan/arm_rl/urdf/robotic_arm/urdf/robotic_arm.urdf"

# Load environment
env = DummyVecEnv([lambda: Monitor(CustomEnv(urdf))])

# Load trained model
model = PPO.load("models/ppo_robotic_arm", env=env)

# Test loop
obs = env.reset()
episode = 0

try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        if dones[0]:
            episode += 1
            print(f"Episode {episode} | reward: {rewards[0]:.3f}")
            obs = env.reset()

except KeyboardInterrupt:
    print("\nTesting stopped")

finally:
    env.close()