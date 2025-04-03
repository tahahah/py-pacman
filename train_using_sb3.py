# import gymnasium

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
import matplotlib.pyplot as plt
import gymnasium as gym
import torchvision.transforms as T
from collections import deque
from gymnasium.wrappers import FrameStackObservation
import wandb
from wandb.integration.sb3 import WandbCallback
import os
from dotenv import load_dotenv
import pygame as pg
import cv2
import huggingface_hub
# Load environment variables if you have a .env file with WANDB_API_KEY
load_dotenv()
import os
from src.env.pacman_env_new import PacmanEnv
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation


# Initialize wandb
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 20000000,
    "env_name": "PacmanEnv",
    "layout": "classic",
    "n_steps": 256,
    "batch_size": 256,  # Increased from 32 to 256 for better GPU utilization
    "n_epochs": 4,
    "gamma": 0.99,  # Discount factor (default is 0.99)
    "ent_coef": 0.01,  # Entropy coefficient - increased to encourage exploration
    "learning_rate": 3e-4,  # Learning rate
    "clip_range": 0.2,  # PPO clip range
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Maximum gradient norm for gradient clipping
}

# Custom wrapper to add additional info and resize observations
class PacmanInfoWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        
        # Update observation space to match the new shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, *shape), dtype=np.uint8
        )
    
    def observation(self, observation):
        # Resize observation using cv2
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        
        # Convert back to channel-first format (C, H, W)
        return np.transpose(observation, (2, 0, 1))

# Custom callback to log metrics and save model checkpoints
class PacmanMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, save_freq=100000, save_model_freq=None):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.step_count = 0
        self.save_model_freq = save_model_freq
        self.model_save_count = 0
    
    def _on_step(self):
        self.step_count += 1
        
        # Check if it's time to save the model
        if self.save_model_freq is not None and self.step_count % self.save_model_freq == 0:
            self.save_model()
        
        # Regular metrics logging
        try:
            action = self.locals.get("actions", [None])[0]
            done = self.locals["dones"][0]
            info = self.locals["infos"][0]
            if self.step_count % self.save_freq == 0:
                # Get the current observation from the selected environment
                obs = self.locals["new_obs"][0]  # Shape: (3, 84, 84)
                
                # Convert to HWC format for visualization
                obs_hwc = np.transpose(obs, (1, 2, 0))
                
                # Create a figure to show the observation
                plt.figure(figsize=(8, 8))
                plt.imshow(obs_hwc)
                plt.title(f"Observation at Step {self.step_count}")
                plt.axis('off')
                
                # Add some metadata to the image
                plt.figtext(0.5, 0.01, 
                        f"Action: {action}, Reward: {self.locals['rewards'][0]:.2f}, Pellets Left: {info.get('pellets left', 170)}",
                        ha="center", fontsize=10, weight='bold')
                
                # Log the figure to wandb
                wandb.log({
                    "observation/image": wandb.Image(plt.gcf(), caption=f"Step {self.step_count}"),
                })
                
                plt.close()
            
            # If episode is done, reset episode stats
            if done:
                # Log episode stats
                wandb.log({
                    "train/pellets_left": info.get("pellets left", 170)
                })
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return True
    
    def save_model(self):
        """Save the model at the current checkpoint"""
        try:
            self.model_save_count += 1
            checkpoint_path = f"models/{wandb.run.id}/ppo-pacman-checkpoint-{self.step_count}"
            print(f"Saving model checkpoint at step {self.step_count} to {checkpoint_path}")
            
            # Save the model
            self.model.save(checkpoint_path)
            
            # Log to wandb
            wandb.log({
                "checkpoint/step": self.step_count,
                "checkpoint/path": checkpoint_path
            })
            
            # Optionally upload to Hugging Face
            try:
                repo_id = f"Tahahah/PacmanRL"
                huggingface_hub.upload_file(
                    path_or_fileobj=checkpoint_path, 
                    path_in_repo=f"checkpoints/checkpoint-{self.step_count}", 
                    repo_id=repo_id, 
                    repo_type="model"
                )
                print(f"Uploaded checkpoint to Hugging Face: {repo_id}")
            except Exception as e:
                print(f"Error uploading to Hugging Face: {str(e)}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            import traceback
            traceback.print_exc()

# Function to create the environment
def make_env():
    # Create base environment
    env = PacmanEnv(layout="classic", enable_render=True, render_mode="rgb_array", state_active=False, player_lives=3)
    
    # Wrap with Monitor for statistics recording
    env = Monitor(env)
    
    # Wrap with our custom wrapper for pellet tracking and resizing
    
    # Add additional wrappers
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)
    
    return env

# Create vectorized environment
env = make_vec_env(make_env, n_envs=8)

# Calculate model saving frequency (every 1/5th of total timesteps)
save_model_freq = config["total_timesteps"] // 5

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

if device == "cuda":
    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Set PyTorch to use the GPU
    torch.cuda.set_device(0)


run = wandb.init(
    project="PacmanRL",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,       # auto-upload the videos of agents playing the game
    save_code=True,         # optional
)

# Create directories for videos and models
os.makedirs(f"videos/{run.id}", exist_ok=True)
os.makedirs(f"models/{run.id}", exist_ok=True)

# Load a pretrained model if it exists, otherwise create a new one
model_name = f"models/xask3fxp/ppo-pacman-final"
try:
    model = PPO.load(model_name, env=env, device=device)
    print(f"Loaded pretrained model from {model_name}")
except FileNotFoundError:
    model = PPO(
        config["policy_type"], 
        env, 
        verbose=1,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        device=device,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs={"normalize_images": False}
    )
    print("Created new model")

# Create a list of callbacks
callbacks = [
    PacmanMetricsCallback(
        save_freq=100000,  # Observation logging frequency
        save_model_freq=save_model_freq  # Model saving frequency
    ),
    WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
]

# Train the model with our callbacks
model.learn(
    total_timesteps=config["total_timesteps"], 
    log_interval=1, 
    progress_bar=True,
    callback=callbacks
)

# Save the final model
model_name = f"models/{run.id}/ppo-pacman-final"
model.save(model_name)

# Run evaluation and record video
print("Running model evaluation and recording video...")

# Create a separate environment for evaluation
eval_env = PacmanEnv(layout="classic", enable_render=True, render_mode="rgb_array", state_active=False, player_lives=3)
eval_env = Monitor(eval_env)
eval_env = SkipFrame(eval_env, skip=4)
eval_env = GrayScaleObservation(eval_env)
eval_env = ResizeObservation(eval_env, shape=(84, 84))
eval_env = FrameStackObservation(eval_env, stack_size=4)

# Create video directory
video_dir = f"videos/{run.id}"
os.makedirs(video_dir, exist_ok=True)
video_path = f"{video_dir}/final_evaluation.mp4"

# Wrap the environment with VecVideoRecorder
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecVideoRecorder(
    eval_env,
    video_path,
    record_video_trigger=lambda x: True,  # Always record
    video_length=2000,  # Record a longer video for evaluation
    name_prefix="final-evaluation"
)

# Load the trained model
eval_model = PPO.load(model_name, env=eval_env)

# Run evaluation
obs = eval_env.reset()
done = False
total_reward = 0
step_count = 0
max_steps = 2000  # Set a maximum number of steps

print("Starting evaluation...")
while step_count < max_steps:
    action, _ = eval_model.predict(obs, deterministic=True)
    obs, reward, terminated, info = eval_env.step(action)
    total_reward += reward[0]
    step_count += 1
    done = terminated[0]
    if done:
        print(f"Episode finished after {step_count} steps with reward {total_reward}")
        break

# Close the environment to ensure video is saved
eval_env.close()

# Log the video to wandb
if os.path.exists(video_path):
    print(f"Uploading evaluation video to wandb: {video_path}")
    wandb.log({
        "evaluation/video": wandb.Video(video_path, fps=30, format="mp4"),
        "evaluation/total_reward": total_reward,
        "evaluation/episode_length": step_count
    })
else:
    print(f"Warning: Evaluation video not found at {video_path}")

# Save the model to Hugging Face
huggingface_hub.login(token=os.environ['HF_TOKEN'])

repo_id = f"Tahahah/PacmanRL"
try:
    huggingface_hub.upload_file(path_or_fileobj=model_name+".zip", path_in_repo=f"checkpoints/{model_name}", repo_id=repo_id, repo_type="model")
except huggingface_hub.utils.RepositoryNotFoundError:
    huggingface_hub.create_repo(repo_id, repo_type="model")
    huggingface_hub.upload_file(path_or_fileobj=model_name+".zip", path_in_repo=f"checkpoints/{model_name}", repo_id=repo_id, repo_type="model")

# Close the environment
env.close()

# Finish the wandb run
run.finish()