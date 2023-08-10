import gym
from stable_baselines3 import PPO
import os

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make("LunarLander-v2")
env.reset()

# Check if a saved model checkpoint exists
checkpoint_path = f"{models_dir}/299000.zip"
if os.path.exists(checkpoint_path):
    model = PPO.load(checkpoint_path, env=env)
else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
EPISODES = 300

for i in range(300, 600):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    
    
env.close()