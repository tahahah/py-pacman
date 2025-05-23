# import gymnasium

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
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
                actual_checkpoint_file_path = checkpoint_path + ".zip"
                huggingface_hub.upload_file(
                    path_or_fileobj=actual_checkpoint_file_path, 
                    path_in_repo=f"checkpoints/checkpoint-{self.step_count}.zip", 
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
raw_env = make_vec_env(make_env, n_envs=8)
# Normalize observations and rewards
env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Create a separate, raw environment for evaluation (will be wrapped by EvalCallback's logic if needed)
# For VecNormalize, it's often better to let EvalCallback handle the eval_env's normalization
# or pass a VecNormalize instance that will be synced.
# The key is that the model expects observations normalized by the *training* env's stats.
# We will ensure the EvalCallback saves the *training* env's stats.
eval_env_for_callback = make_vec_env(make_env, n_envs=1)
# We will pass this raw env to EvalCallback. The callback will use the model's normalization stats implicitly during prediction.
# The VecNormalize stats saved will be from the training env.

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
pretrained_model_load_path_prefix = "models/bot3vy5v/ppo-pacman-final" # This is a path prefix
pretrained_model_zip_path = f"{pretrained_model_load_path_prefix}.zip"

try:
    # Check for VecNormalize stats associated with the pretrained model
    pretrained_vec_normalize_stats_path = f"{pretrained_model_load_path_prefix}_vecnormalize.pkl"
    
    current_env_is_vec_normalize = isinstance(env, VecNormalize)
    
    if os.path.exists(pretrained_model_zip_path):
        if os.path.exists(pretrained_vec_normalize_stats_path) and current_env_is_vec_normalize:
            print(f"Loading VecNormalize stats for pretrained model from: {pretrained_vec_normalize_stats_path}")
            # Load stats into the existing 'env' VecNormalize wrapper.
            # 'raw_env' is the original DummyVecEnv that 'env' wraps.
            env = VecNormalize.load(pretrained_vec_normalize_stats_path, raw_env) # raw_env is the underlying non-normalized env stack
            env.training = True # Ensure it's in training mode for subsequent learning
            print(f"Successfully loaded and applied VecNormalize stats to the training environment.")
        elif current_env_is_vec_normalize:
            print(f"VecNormalize stats not found at {pretrained_vec_normalize_stats_path} for pretrained model, but model file exists.")
            print("The existing VecNormalize wrapper on 'env' will be used, which will learn stats from scratch if not already learned.")
        
        print(f"Loading pretrained model from {pretrained_model_zip_path}")
        model = PPO.load(pretrained_model_zip_path, env=env, device=device, custom_objects={'learning_rate': config['learning_rate'], 'ent_coef': config['ent_coef']}) # Pass env which might now have loaded stats
        print(f"Successfully loaded pretrained model.")
    else:
        raise FileNotFoundError # Pretrained model zip not found, proceed to create new

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

# Setup evaluation callback
# This will save the best model according to the evaluation environment
# and log evaluation metrics
eval_callback = EvalCallback(
    eval_env_for_callback, # Pass the raw environment for evaluation 
    best_model_save_path=f"models/{run.id}/best_model/",
    log_path=f"models/{run.id}/eval_logs/", 
    eval_freq=max(config["n_steps"] * 8 // 10, 1), # Evaluate 10 times per training run, or at least once
    n_eval_episodes=5,
    deterministic=True, 
    render=False,
    callback_on_new_best=None, # Could add a custom callback here if needed
    # When a new best model is found, save the VecNormalize stats
    # This is crucial for loading the model later with the correct normalization
    # When a new best model is found by EvalCallback, save the VecNormalize stats of the *training* environment.
    # self.model.get_vec_normalize_env() should give the training VecNormalize instance.
    callback_after_eval=lambda: self.model.get_vec_normalize_env().save(os.path.join(self.best_model_save_path, "vecnormalize.pkl"))
)

callbacks.append(eval_callback)

# Train the model with our callbacks
model.learn(
    total_timesteps=config["total_timesteps"], 
    log_interval=1, 
    progress_bar=True,
    callback=callbacks
)

# Save the final model and VecNormalize stats
final_model_path_prefix = f"models/{run.id}/ppo-pacman-final"
model.save(final_model_path_prefix) # Saves as final_model_path_prefix.zip
# Save the VecNormalize statistics alongside the model zip file
env.save(f"{final_model_path_prefix}_vecnormalize.pkl")

final_model_zip_path = f"{final_model_path_prefix}.zip" # Actual path to the zip file

# Note: When loading the model, you'll need to load the VecNormalize stats as well
# Example:
# model = PPO.load("path_to_model")
# env = VecNormalize.load("path_to_vecnormalize.pkl", DummyVecEnv([make_env]))
# env.training = False # Important for evaluation
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# Run evaluation and record video
print("Running model evaluation and recording video...")

# Create a raw environment for video evaluation
video_eval_raw_env = PacmanEnv(layout="classic", enable_render=True, render_mode="rgb_array", state_active=False, player_lives=3)
video_eval_raw_env = Monitor(video_eval_raw_env)
video_eval_raw_env = SkipFrame(video_eval_raw_env, skip=4)
video_eval_raw_env = GrayScaleObservation(video_eval_raw_env)
video_eval_raw_env = ResizeObservation(video_eval_raw_env, shape=(84, 84))
video_eval_raw_env = FrameStackObservation(video_eval_raw_env, stack_size=4)

# Wrap in DummyVecEnv before VecNormalize.load
video_eval_dummy_env = DummyVecEnv([lambda: video_eval_raw_env])

# Load the VecNormalize stats saved with the final model
vec_normalize_stats_path_for_final_model = f"{final_model_path_prefix}_vecnormalize.pkl"

if os.path.exists(vec_normalize_stats_path_for_final_model):
    print(f"Loading VecNormalize stats for video evaluation from: {vec_normalize_stats_path_for_final_model}")
    video_eval_normalized_env = VecNormalize.load(vec_normalize_stats_path_for_final_model, video_eval_dummy_env)
    video_eval_normalized_env.training = False  # Set to evaluation mode
    video_eval_normalized_env.norm_reward = False # Don't normalize rewards for pure evaluation
else:
    print(f"WARNING: VecNormalize stats not found at {vec_normalize_stats_path_for_final_model}. Video evaluation will use unnormalized observations.")
    video_eval_normalized_env = video_eval_dummy_env # Fallback, likely suboptimal

# Wrap the normalized environment with VecVideoRecorder for the final evaluation
video_output_folder = f"videos/{run.id}/final_eval/"
os.makedirs(video_output_folder, exist_ok=True)
video_eval_vec_env_recorded = VecVideoRecorder(
    video_eval_normalized_env, 
    video_output_folder, 
    record_video_trigger=lambda x: x == 0, # Record the first episode
    video_length=2000, # Max length of video, can be adjusted
    name_prefix=f"final-ppo-pacman-{run.id}"
)

# Evaluate the final model with video recording
print("Evaluating final model and recording video...")
# Load the model with the video-recorded environment
# Note: The model was already saved as 'final_model_zip_path'. We load it here for evaluation with the video recorder.
final_model_for_eval = PPO.load(final_model_zip_path, env=video_eval_vec_env_recorded, device=device)

# The evaluate_policy call will trigger the VideoRecorder to save the video.
mean_reward, std_reward = evaluate_policy(final_model_for_eval, video_eval_vec_env_recorded, n_eval_episodes=1, deterministic=True, render=False, callback=None)
print(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the video recorder to ensure the video file is finalized
video_eval_vec_env_recorded.close() 

# Log the recorded video to WandB
# The VecVideoRecorder saves files in the specified video_output_folder.
# Let's try to find the most recent mp4 file in that folder.
if os.path.exists(video_output_folder):
    video_files = [os.path.join(video_output_folder, f) for f in os.listdir(video_output_folder) if f.endswith(".mp4")]
    if video_files:
        latest_video_file = max(video_files, key=os.path.getctime)
        print(f"Logging video {latest_video_file} to WandB.")
        wandb.log({"evaluation/video": wandb.Video(latest_video_file, fps=15, format="mp4"),
                   "evaluation/mean_reward": mean_reward}) # fps can be adjusted
    else:
        print(f"No video files found in {video_output_folder} to log to WandB.")
else:
    print(f"Video folder {video_output_folder} not found. Cannot log video.")


# Create video directory
video_dir = f"videos/{run.id}"
os.makedirs(video_dir, exist_ok=True)

# Wrap the NORMALIZED environment with VecVideoRecorder
video_eval_vec_env = VecVideoRecorder(
    video_eval_normalized_env, # Use the normalized environment
    video_dir,
    record_video_trigger=lambda x: True,  # Always record
    video_length=2000,  # Record a longer video for evaluation
    name_prefix="final-evaluation"
)
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