# import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FrameStackObservation
from dotenv import load_dotenv
import huggingface_hub
# Load environment variables if you have a .env file with WANDB_API_KEY
load_dotenv()
import os
from src.env.pacman_env_new import PacmanEnv
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

model_name = "./ppo-pacman-final.zip"

print("Creating environment for evaluation...")
# Create a separate environment for evaluation
eval_env = PacmanEnv(layout="classic", enable_render=True, render_mode="rgb_array", state_active=False, player_lives=3)
eval_env = Monitor(eval_env)
eval_env = SkipFrame(eval_env, skip=4)
eval_env = GrayScaleObservation(eval_env)
eval_env = ResizeObservation(eval_env, shape=(84, 84))
eval_env = FrameStackObservation(eval_env, stack_size=4)

# Create video directory
import uuid

print("Creating video directory...")
video_dir = f"videos/{uuid.uuid4()}"
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

print("Loading the trained model...")
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

# Save the model to Hugging Face
huggingface_hub.login(token=os.environ['HF_TOKEN'])

repo_id = f"Tahahah/PacmanRL"
try:
    huggingface_hub.upload_file(path_or_fileobj=model_name, path_in_repo=f"checkpoints/{model_name}", repo_id=repo_id, repo_type="model")
except huggingface_hub.utils.RepositoryNotFoundError:
    huggingface_hub.create_repo(repo_id, repo_type="model")
    huggingface_hub.upload_file(path_or_fileobj=model_name, path_in_repo=f"checkpoints/{model_name}", repo_id=repo_id, repo_type="model")

# Close the environment
eval_env.close()
