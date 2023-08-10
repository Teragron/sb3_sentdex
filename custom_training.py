import gymnasium
from stable_baselines3 import PPO
import os
from ping_pong import EggCatcherEnv




models_dir = "models/PPO_GPT"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
env = EggCatcherEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10

for i in range(1, 30000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_GPT")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    
    
env.close()