import gym
from stable_baselines3 import PPO, A2C

# Create the LunarLander environment
env = gym.make("LunarLander-v2", render_mode="human")

# Load the PPO model
models_dir = "models/PPO_2"
models_path = f"{models_dir}/1000.zip"
model = PPO.load(models_path, env=env)

# Initialize variables
episodes_to_run = 2
episode_count = 0

while episode_count < episodes_to_run:
    # Reset the environment and get the initial observation
    observation, info = env.reset(seed=42)#seed=42
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # Take an action using the PPO model
        action, _ = model.predict(observation)

        # Execute the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # If the episode ends or is truncated, reset the environment
        if terminated or truncated:
            observation, info = env.reset()

    episode_count += 1

env.close()


#tensorboard --logdir=logs