# import gymnasium

# from huggingface_sb3 import load_from_hub, package_to_hub
# from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor


import gymnasium as gym
from src.env.pacman_env_new import PacmanEnv
# First, we create our environment called LunarLander-v2
env = PacmanEnv(layout="classic", enable_render=True, state_active=True, player_lives=3)
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action

# env = make_vec_env(lambda: PacmanEnv(layout='classic', enable_render=True, state_active=True, player_lives=3), n_envs=16)

# model = PPO('MlpPolicy', env, verbose=True, n_steps=10, batch_size=10)

# model.learn(1000000, log_interval=1, progress_bar=1)

# # TODO: Specify file name for model and save the model to file
# model_name = "ppo-pacman"
# model.save(model_name)

env.close()