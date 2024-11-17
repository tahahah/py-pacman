import argparse

import logging
from typing import Any, List, Optional, Tuple
import json
import math
import os
import pickle
import queue
import sys
import time
from collections import namedtuple
from itertools import count
from ActionEncoder import ActionEncoder
import huggingface_hub
import numpy as np
import redis
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import wandb
import zstandard as zstd
from datasets import Dataset
from dotenv import load_dotenv
from gym.wrappers import FrameStack
from PIL import Image
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from redis.retry import Retry
from replay_buffer import ReplayBuffer
from src.env.pacman_env import PacmanEnv
from wrappers import GrayScaleObservation, ResizeObservation, SkipFrame
from model import DQN
# Load environment variables from .env file
load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'))
wandb.init(project="PacmanDataGen", job_type="pacman", magic=True)

# Get HF_TOKEN from environment variables
HF_TOKEN = os.getenv('HF_TOKEN')

# if gpu is to be used
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
logging.warning(f"CUDA available: {USE_CUDA}")

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))

MAX_MESSAGE_SIZE = 500 * 1024 * 1024  # 500 MB


class PacmanAgent:
    def __init__(self, input_dim, output_dim, model_name="pacman_policy_net_gamengen_1_rainbow_negative_pellet_reward_multienv"):
        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00004, eps=1.5e-4)
        self.steps_done = 0
        self.batch_size = 32

        # Try to load the model from Hugging Face if it exists
        self.pretrained_model = "pacman_policy_net_gamengen_1_rainbow_negative_pellet_reward"
        self.model_name = model_name
        try:
            huggingface_hub.login(token=HF_TOKEN)
            model_path = huggingface_hub.hf_hub_download(repo_id=f"Tahahah/{self.pretrained_model if self.pretrained_model else self.model_name}", filename="checkpoints/pacman.pth", repo_type="model")
            state_dict = torch.load(model_path, map_location=device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            logging.warning(f"Model loaded from Hugging Face: {self.model_name}")
        except Exception as e:
            logging.warning(f"Could not load model from Hugging Face: {e}")

    def select_actions_batch(self, states_batch: torch.Tensor, epsilon: float, n_actions: int) -> torch.Tensor:
        """Select actions for a batch of states"""
        batch_size = states_batch.size(0)
        actions = torch.empty(batch_size, device=device, dtype=torch.long)
        
        # For exploration
        random_mask = torch.rand(batch_size, device=device) < epsilon
        random_actions = torch.randint(0, n_actions, (batch_size,), device=device)
        
        # For exploitation
        with torch.no_grad():
            policy_actions = self.policy_net(states_batch).max(1)[1]
        
        # Combine random and policy actions
        actions[random_mask] = random_actions[random_mask]
        actions[~random_mask] = policy_actions[~random_mask]
        
        return actions

    def optimize_model(self, memory, gamma=0.99):
        if len(memory) < self.batch_size:
            return

        # Sample a batch of transitions using prioritized replay
        state, next_state, action, reward, done, indices, weights = memory.sample(self.batch_size)
        
        # Convert to tensors and move to device
        state = torch.tensor(state, device=device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
        action = torch.tensor(action, device=device, dtype=torch.long)
        reward = torch.tensor(reward, device=device, dtype=torch.float32)
        done = torch.tensor(done, device=device, dtype=torch.float32)
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(-1))

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_net(next_state).max(1)[0]
            next_state_values[done.bool()] = 0.0
            expected_state_action_values = (next_state_values * gamma) + reward

        # Compute TD errors for priority updating
        td_errors = torch.abs(state_action_values.squeeze() - expected_state_action_values).detach()
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss(reduction='none')
        loss = (weights * criterion(state_action_values.squeeze(), expected_state_action_values)).mean()

        # Update priorities in replay buffer
        memory.update_priorities(indices, td_errors.cpu().numpy())
        
        return loss

    def update_target_network(self):
        # Soft update of target network
        tau = 0.005
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filename):
        # Save the model locally
        torch.save(self.policy_net.state_dict(), filename)
        
        # Save the model to Hugging Face with retries
        huggingface_hub.login(token=HF_TOKEN)
        repo_id = f"Tahahah/{self.model_name}"

        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        for attempt in range(max_retries):
            try:
                try:
                    huggingface_hub.upload_file(
                        path_or_fileobj=filename,
                        path_in_repo=f"checkpoints/{filename}",
                        repo_id=repo_id,
                        repo_type="model"
                    )
                except huggingface_hub.utils.RepositoryNotFoundError:
                    logging.warning(f"Repository {repo_id} not found. Creating it...")
                    huggingface_hub.create_repo(repo_id, repo_type="model")
                    huggingface_hub.upload_file(
                        path_or_fileobj=filename,
                        path_in_repo=f"checkpoints/{filename}",
                        repo_id=repo_id,
                        repo_type="model"
                    )
                logging.warning(f"RL Model saved locally as {filename} and uploaded to Hugging Face as {self.model_name}")
                break
            except Exception as e:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                if attempt < max_retries - 1:
                    logging.warning(f"Attempt {attempt + 1} failed to save model to HuggingFace: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"Failed to save model to HuggingFace after {max_retries} attempts: {str(e)}")
                    
    @classmethod
    def load_model(cls, input_dim, output_dim, filename):
        agent = cls(input_dim, output_dim)
        state_dict = torch.load(filename, map_location=device)
        agent.policy_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
        return agent


from pydantic import BaseModel, Field
class DataRecord(BaseModel):
    episode: int
    frames: List[Any]
    actions: List[int]
    batch_id: int
    is_last_batch: bool

class VectorizedPacmanEnv:
    def __init__(self, num_envs: int, layout: str, frames_to_skip: int):
        self.num_envs = num_envs
        self.envs = []
        self.base_envs = []  # Store base envs for rendering
        self.game_states = [None] * num_envs  # Track game states
        for _ in range(num_envs):
            env = PacmanEnv(layout=layout)
            # Store base env before wrapping
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            self.base_envs.append(base_env)
            
            env = SkipFrame(env, skip=frames_to_skip)
            env = GrayScaleObservation(env)
            env = ResizeObservation(env, shape=84)
            env = FrameStack(env, num_stack=4)
            self.envs.append(env)
        
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
        # Initialize game states
        for i, env in enumerate(self.envs):
            self.game_states[i] = {
                'score': env.game.score,
                'total_rewards': env.game.total_rewards,
                'pellets': env.maze.get_number_of_pellets()
            }
        
    def reset(self) -> np.ndarray:
        states = []
        for i, env in enumerate(self.envs):
            state = env.reset(mode='rgb_array')
            # Update initial game state
            self.game_states[i] = {
                'score': env.game.score,
                'total_rewards': env.game.total_rewards,
                'pellets': env.maze.get_number_of_pellets()
            }
            states.append(state)
        return np.array(states)

    def step(self, actions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        next_states, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(action.item())
            
            # Track pellet changes
            current_pellets = env.maze.get_number_of_pellets()
            info['current_pellets'] = current_pellets
            
            if done:
                next_state = env.reset(mode='rgb_array')
                # Reset game state tracking
                self.game_states[i] = {
                    'score': env.game.score,
                    'total_rewards': env.game.total_rewards,
                    'pellets': env.maze.get_number_of_pellets(),
                    'episode_reward': env.game.total_rewards
                }
            else:
                # Update game state tracking
                self.game_states[i] = {
                    'score': env.game.score,
                    'total_rewards': env.game.total_rewards,
                    'pellets': current_pellets,
                    'episode_reward': env.game.total_rewards
                }
            
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            np.array(next_states),
            np.array(rewards),
            np.array(dones),
            infos
        )

    def render(self, mode: str = 'human', env_idx: int = 0) -> Optional[np.ndarray]:
        if mode == 'human':
            self.envs[env_idx].render(mode)
            return None
        elif mode == 'rgb_array':
            return self.base_envs[env_idx].render(mode='rgb_array')

    def get_number_of_pellets(self) -> List[int]:
        return [env.maze.get_number_of_pellets() for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()

class PacmanTrainer:
    def __init__(self, layout, episodes, frames_to_skip, save_locally, enable_rmq, log_video_to_wandb=True, num_envs=8):
        self.layout = layout
        self.episodes = episodes
        self.frames_to_skip = frames_to_skip
        self.num_envs = num_envs
        self.env = VectorizedPacmanEnv(num_envs, layout, frames_to_skip)
        self.agent = None
        self.memory = None
        self.save_queue = queue.Queue()
        self.connection = None
        self.channel = None
        self.save_locally = save_locally | False
        self.enable_rmq = enable_rmq
        self.action_encoder = ActionEncoder()
        self.log_video_to_wandb = log_video_to_wandb
        logging.basicConfig(level=logging.warning, format='%(asctime)s - %(levelname)s - %(message)s')

    def _setup_rabbitmq(self):
        if not self.enable_rmq:
            return

        import pika

        # Set up the connection to RabbitMQ
        credentials = pika.PlainCredentials('pacman', 'pacman_pass')
        parameters = pika.ConnectionParameters(
            'rabbitmq-host',
            5672,
            '/',
            credentials
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Declare the queue
        self.channel.queue_declare(queue='HF_upload_queue')
        
        # Create redis client with retry mechanism
        self.redis_client = redis.StrictRedis(
            host='redis', 
            port=6379, 
            db=0, 
            decode_responses=False, 
            password="pacman", 
            health_check_interval=30, 
            socket_keepalive=True,
            retry=Retry(ExponentialBackoff(cap=10, base=1), 25),
            retry_on_error=[ConnectionError, TimeoutError]
        )
        self.episode_keys_buffer = []

    def _close_rabbitmq(self):
        if self.connection:
            self.connection.close()

    def _save_data_to_redis(self, episode, frames_buffer, actions_buffer):
        logging.warning("_save_data_to_redis invoked")
        key = f"episode_{episode}"
        
        # Serialize data using pickle
        data = {
            'episode': episode,
            'frames': frames_buffer,
            'actions': actions_buffer
        }
        serialized_data = pickle.dumps(data)
        
        # Compress the serialized data
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(serialized_data)
        
        # Log the sizes of the serialized and compressed data
        original_size = sys.getsizeof(serialized_data)
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / original_size
        logging.warning(f"Original size: {original_size} bytes, Compressed size: {compressed_size} bytes, Compression ratio: {compression_ratio:.2f}")
        
        # Clear the original buffers to free memory
        frames_buffer.clear()
        actions_buffer.clear()
        
        # Log the data being saved
        logging.warning(f"Saving compressed data for episode {episode} to Redis with key {key}")
        
        self.redis_client.set(key, compressed_data)
        self.episode_keys_buffer.append(key)

        del data
        del serialized_data
        del compressed_data
        
        # Log the current buffer size
        logging.warning(f"Current episode keys buffer size: {len(self.episode_keys_buffer)}")

        # Publish keys to the queue every 20 episodes
        if len(self.episode_keys_buffer) >= 20:
            logging.warning("Buffer size reached 20, publishing keys to queue")
            self._publish_keys_to_queue()
            self.episode_keys_buffer.clear()
            logging.warning("Episode keys buffer cleared after publishing")

    def _publish_keys_to_queue(self):
        if self.enable_rmq:
            message = json.dumps(self.episode_keys_buffer)
            self.channel.basic_publish(exchange='', routing_key='HF_upload_queue', body=message)
            logging.warning(f"Published keys to RabbitMQ queue 'HF_upload_queue': {self.episode_keys_buffer}")


    def train(self):
        if self.enable_rmq:
            self._setup_rabbitmq()
        
        states = self.env.reset()
        n_actions = self.env.action_space.n

        self.agent = PacmanAgent(states[0].shape, n_actions)
        self.memory = ReplayBuffer(32 * self.num_envs)  # Increased batch size for multiple envs

        frames_buffers = [[] for _ in range(self.num_envs)]
        actions_buffers = [[] for _ in range(self.num_envs)]
        max_batch_size = 500 * 1024 * 1024  # 400 MB
        episode_rewards = [0.0] * self.num_envs
        episode_steps = [0] * self.num_envs
        episode_counts = [0] * self.num_envs  # Track episodes per environment
        total_episodes = 0  # Track total episodes across all environments
        previous_frames = [None] * self.num_envs

        # Enable automatic mixed precision
        scaler = torch.amp.GradScaler('cuda')

        while total_episodes < self.episodes:
            epsilon = self._get_epsilon(total_episodes)
            
            # Log epsilon value
            wandb.log({
                "training/epsilon": epsilon,
                "training/episode": total_episodes
            })
            
            # Convert states to tensor once
            states_tensor = torch.tensor(states, device=device, dtype=torch.float32)
            
            # Get actions for all environments in parallel
            with torch.amp.autocast('cuda'):
                actions = self.agent.select_actions_batch(states_tensor, epsilon, n_actions)
            
            # Step all environments (now returns numpy arrays)
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # Update episode tracking and handle data collection
            episodes_finished = 0
            for i, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
                reward_clipped = max(-1.0, min(float(reward), 1.0))
                episode_rewards[i] += reward_clipped
                episode_steps[i] += 1

                # Collect frames only if needed
                if done or (self.log_video_to_wandb and total_episodes % 2000 == 0):
                    current_frame = self.env.render(mode='rgb_array', env_idx=i)
                    if done:
                        frames_buffers[i].append(current_frame)
                    else:
                        previous_frames[i] = current_frame

                if done:
                    pellets_left = info['current_pellets']
                    episode_counts[i] += 1
                    episodes_finished += 1
                    logging.warning(f"Environment #{i} finished episode #{episode_counts[i]} after {episode_steps[i]} timesteps with total reward: {episode_rewards[i]} and {pellets_left} pellets left.")
                    
                    wandb.log({
                        f"env_{i}/episode": episode_counts[i],
                        f"env_{i}/reward": episode_rewards[i],
                        f"env_{i}/pellets_left": pellets_left,
                        f"env_{i}/epsilon": epsilon
                    })
                    
                    # Reset counters for this environment
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
            
            # Update total episodes only after all environments are processed
            if episodes_finished > 0:
                total_episodes += 1
                
                # Save model periodically based on total episodes
                if total_episodes % 1000 == 0:
                    self.agent.save_model('pacman.pth')
                    logging.warning(f"Saved model at total episode {total_episodes}")

                # Log video periodically based on total episodes
                if self.log_video_to_wandb and total_episodes % 2000 == 0:
                    for i in range(self.num_envs):
                        if len(frames_buffers[i]) > 0:
                            frames = [np.array(frame).astype(np.uint8) for frame in frames_buffers[i]]
                            if frames[0].max() <= 1.0:
                                frames = [frame * 255 for frame in frames]
                            frames = np.stack(frames)
                            frames = np.transpose(frames, (0, 3, 1, 2))
                            video = wandb.Video(frames, fps=10, format="mp4")
                            wandb.log({
                                f"video_env_{i}": video,
                                f"image_env_{i}": wandb.Image(previous_frames[i]) if previous_frames[i] is not None else None,
                            })
                            frames_buffers[i] = []  # Clear buffer after logging

            # Cache experience in replay buffer
            self.memory.cache_batch(
                states,  # Already numpy
                next_states,  # Already numpy
                actions.cpu().numpy(),
                rewards,  # Already numpy
                dones  # Already numpy
            )

            states = next_states

            # Optimize model
            if len(self.memory) >= 32 * self.num_envs:
                # Compute loss with mixed precision
                with torch.amp.autocast('cuda'):
                    loss = self.agent.optimize_model(self.memory)
                
                # Scale loss and backprop
                scaler.scale(loss).backward()
                
                # Unscale before gradient clipping
                scaler.unscale_(self.agent.optimizer)
                torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), max_norm=10.0)
                
                # Step optimizer and update scaler
                scaler.step(self.agent.optimizer)
                scaler.update()

            # Update target network periodically
            if total_episodes > 2 and total_episodes % 100 == 0:
                self.agent.update_target_network()
                logging.warning(f"Updated target network at total episode {total_episodes}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logging.warning('Training Complete')
        self.env.close()
        self.agent.save_model('pacman.pth')
        if self.enable_rmq:
            self._close_rabbitmq()
    
    def _get_buffer_size(self, frames_buffer, actions_buffer):
        # Estimate the size of the buffers in bytes
        buffer_size = sum([frame.nbytes for frame in frames_buffer]) + \
                      sum([sys.getsizeof(action) for action in actions_buffer])
        return buffer_size
    def _get_epsilon(self, frame_idx):
        # Start with a lower initial epsilon and decay faster
        initial_epsilon = 0.95  # Lower initial exploration rate
        min_epsilon = 0.05      # Minimum exploration rate
        decay_rate = 45512       # Faster decay rate

        return min_epsilon + (initial_epsilon - min_epsilon) * math.exp(-1. * frame_idx / decay_rate)
    
    def _save_data(self, data_record: DataRecord):
        self.save_queue.put(data_record)
        if self.enable_rmq:
            self._publish_to_rabbitmq(self.save_queue.get())

    def _save_remaining_data(self, data_record: DataRecord):
        if data_record.frames:
            self._save_data(data_record)

    def _publish_to_rabbitmq(self, data: DataRecord):
        import pickle

        # Serialize the data using pickle
        message = pickle.dumps(data.dict())

        # Publish the message to the queue
        self.channel.basic_publish(exchange='',
                                   routing_key='HF_upload_queue',
                                   body=message)

        logging.warning("Published dataset to RabbitMQ queue 'HF_upload_queue'")

    def _save_frames_locally(self, frames, episode, actions):
        # Create a directory for the episode if it doesn't exist
        episode_dir = f"data/episode_{episode}_frs{self.frames_to_skip}"
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)

        # Save each frame as a PNG file with the episode and action in the filename
        for idx, frame in enumerate(frames):
            action = actions[idx]
            # Check if the frame is completely black
            if not np.any(frame):
                logging.warning(f"Frame {idx} is completely black")
            
            filename = os.path.join(episode_dir, f"{idx:05d}.png")
            Image.fromarray(frame).save(filename)
            # logging.warning(f"Saved frame {idx} of episode {episode} with action {action} to {filename}")

class PacmanRunner:
    def __init__(self, layout):
        self.layout = layout
        self.env = self._create_environment()
        self.agent = None

    def _create_environment(self):
        env = PacmanEnv(self.layout)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        return env

    def run(self):
        screen = self.env.reset(mode='rgb_array')
        n_actions = self.env.action_space.n

        self.agent = PacmanAgent.load_model(screen.shape, n_actions, 'pacman.pth')

        for _ in range(10):
            screen = self.env.reset(mode='rgb_array')
            self.env.render(mode='human')

            for _ in count():
                self.env.render(mode='human')
                action = self.agent.select_action(screen, 0, n_actions)
                screen, _, done, _ = self.env.step(action)

                if done:
                    break

def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the agent that interacts with the sm env')
    parser.add_argument('-lay', '--layout', type=str, nargs=1,
                        help="Name of layout to load in the game")
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the agent')
    parser.add_argument('-e', '--episodes', type=int, nargs=1,
                        help="The number of episode to use during training")
    parser.add_argument('-frs', '--frames_to_skip', type=int, nargs=1,
                        help="The number of frames to skip during training, so the agent doesn't have to take "
                             "an action a every frame")
    parser.add_argument('-r', '--run', action='store_true',
                        help='run the trained agent')
    parser.add_argument('-loc', '--save_locally', action='store_true',
                        help='Save the frames')
    parser.add_argument('-rmq', '--enable_rmq', action='store_true',
                        help='Enable RabbitMQ for saving data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    layout = args.layout[0]
    episodes = args.episodes[0] if args.episodes else 1000

    if args.train:
        frames_to_skip = args.frames_to_skip[0] if args.frames_to_skip is not None else 4
        trainer = PacmanTrainer(layout=layout, episodes=episodes, frames_to_skip=frames_to_skip, save_locally=args.save_locally, enable_rmq=args.enable_rmq)
        trainer.train()

    if args.run:
        runner = PacmanRunner(layout)
        runner.run()
