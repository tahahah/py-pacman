import argparse
import json
import math
import os
import pickle
import queue
import sys
from collections import namedtuple
from itertools import count
from ActionEncoder import ActionEncoder
import huggingface_hub
import numpy as np
import redis
import torch
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

# Load environment variables from .env file
load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'))
wandb.init(project="PacmanDataGen", job_type="pacman", magic=True)

# Get HF_TOKEN from environment variables
HF_TOKEN = os.getenv('HF_TOKEN')

# if gpu is to be used
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

MAX_MESSAGE_SIZE = 500 * 1024 * 1024  # 500 MB

class DQN(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        c, h, w = input_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        c, h, w = input_dim
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        features = self.features(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class PacmanAgent:
    def __init__(self, input_dim, output_dim, model_name="pacman_policy_net_gamengen_1_duelingDQN"):
        self.policy_net = DuelingDQN(input_dim, output_dim).to(device)
        self.target_net = DuelingDQN(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00005, eps=1.5e-4)
        self.steps_done = 0

        # Try to load the model from Hugging Face if it exists
        self.model_name = model_name
        try:
            huggingface_hub.login(token=HF_TOKEN)
            model_path = huggingface_hub.hf_hub_download(repo_id=f"Tahahah/{self.model_name}", filename="checkpoints/pacman.pth", repo_type="model")
            state_dict = torch.load(model_path, map_location=device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            logging.info(f"Model loaded from Hugging Face: {self.model_name}")
        except Exception as e:
            logging.warning(f"Could not load model from Hugging Face: {e}")

    def select_action(self, state, epsilon, n_actions):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state.__array__(), device=device).unsqueeze(0)
                return self.policy_net(state).max(1)[1].item()

    def optimize_model(self, memory, gamma=0.99):
        if self.steps_done < 1e3:
            return
            
        state, next_state, action, reward, done, indices, weights = memory.sample(32)
        
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state).max(1)[1]
            next_state_values = self.target_net(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_state_values[done] = 0.0
            expected_state_action_values = (next_state_values * gamma) + reward
            
        # Calculate TD errors for priority updating
        td_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach()
        
        # Calculate weighted loss
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(state_action_values, 
                expected_state_action_values.unsqueeze(1), reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        
        self.optimizer.step()
        
        # Update priorities in buffer
        memory.update_priorities(indices, td_errors.squeeze().cpu().numpy())
        
    def update_target_network(self):
        # Soft update of target network
        tau = 0.005
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filename):
        # Save the model locally
        torch.save(self.policy_net.state_dict(), filename)
        
        # Save the model to Hugging Face
        huggingface_hub.login(token=HF_TOKEN)

        repo_id = f"Tahahah/{self.model_name}"
        try:
            huggingface_hub.upload_file(path_or_fileobj=filename, path_in_repo=f"checkpoints/{filename}", repo_id=repo_id, repo_type="model")
        except huggingface_hub.utils.RepositoryNotFoundError:
            huggingface_hub.create_repo(repo_id, repo_type="model")
            huggingface_hub.upload_file(path_or_fileobj=filename, path_in_repo=f"checkpoints/{filename}", repo_id=repo_id, repo_type="model")

        logging.info(f"RL Model saved locally as {filename} and uploaded to Hugging Face as {self.model_name}")

    @classmethod
    def load_model(cls, input_dim, output_dim, filename):
        agent = cls(input_dim, output_dim)
        state_dict = torch.load(filename, map_location=device)
        agent.policy_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
        return agent

import logging
from typing import Any, List

from pydantic import BaseModel, Field
class DataRecord(BaseModel):
    episode: int
    frames: List[Any]
    actions: List[int]
    batch_id: int
    is_last_batch: bool

class PacmanTrainer:
    def __init__(self, layout, episodes, frames_to_skip, save_locally, enable_rmq):
        self.layout = layout
        self.episodes = episodes
        self.frames_to_skip = frames_to_skip
        self.env = self._create_environment()
        self.agent = None
        self.memory = None
        self.save_queue = queue.Queue()
        self.connection = None
        self.channel = None
        self.save_locally = save_locally | False
        self.enable_rmq = enable_rmq
        self.action_encoder = ActionEncoder()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _create_environment(self):
        env = PacmanEnv(layout=self.layout)
        env = SkipFrame(env, skip=self.frames_to_skip)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        return env

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
        logging.info("_save_data_to_redis invoked")
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
        logging.info(f"Original size: {original_size} bytes, Compressed size: {compressed_size} bytes, Compression ratio: {compression_ratio:.2f}")
        
        # Clear the original buffers to free memory
        frames_buffer.clear()
        actions_buffer.clear()
        
        # Log the data being saved
        logging.info(f"Saving compressed data for episode {episode} to Redis with key {key}")
        
        self.redis_client.set(key, compressed_data)
        self.episode_keys_buffer.append(key)

        del data
        del serialized_data
        del compressed_data
        
        # Log the current buffer size
        logging.info(f"Current episode keys buffer size: {len(self.episode_keys_buffer)}")

        # Publish keys to the queue every 20 episodes
        if len(self.episode_keys_buffer) >= 20:
            logging.info("Buffer size reached 20, publishing keys to queue")
            self._publish_keys_to_queue()
            self.episode_keys_buffer.clear()
            logging.info("Episode keys buffer cleared after publishing")

    def _publish_keys_to_queue(self):
        if self.enable_rmq:
            message = json.dumps(self.episode_keys_buffer)
            self.channel.basic_publish(exchange='', routing_key='HF_upload_queue', body=message)
            logging.info(f"Published keys to RabbitMQ queue 'HF_upload_queue': {self.episode_keys_buffer}")


    def train(self):
        if self.enable_rmq:
            self._setup_rabbitmq()

        screen = self.env.reset(mode='rgb_array')
        n_actions = self.env.action_space.n

        self.agent = PacmanAgent(screen.shape, n_actions)
        self.memory = ReplayBuffer(32)  # BATCH_SIZE = 32

        frames_buffer, actions_buffer = [], []
        max_batch_size = 500 * 1024 * 1024  # 400 MB

        for i_episode in range(self.episodes):
            state = self.env.reset(mode='rgb_array')
            ep_reward = 0.
            epsilon = self._get_epsilon(i_episode)
            logging.info("-----------------------------------------------------")
            logging.info(f"Starting episode {i_episode} with epsilon {epsilon}")

            for t in count():
                current_frame = self.env.render(mode='rgb_array')
                self.env.render(mode='human')

                action = self.agent.select_action(state, epsilon, n_actions)
                next_state, reward, done, _ = self.env.step(action)
                reward = max(-1.0, min(reward, 1.0))
                ep_reward += reward
                
                if self.enable_rmq or self.save_locally:
                    frames_buffer.append(current_frame)
                    actions_buffer.append(self.action_encoder(action))

                self.memory.cache(state, next_state, action, reward, done)

                state = next_state if not done else None

                self.agent.optimize_model(self.memory, gamma=0.99)
                if done:
                    pellets_left = self.env.maze.get_number_of_pellets()
                    if self.save_locally:
                        self._save_frames_locally(frames=frames_buffer, episode=i_episode, actions=actions_buffer)
                    logging.info(f"Episode #{i_episode} finished after {t + 1} timesteps with total reward: {ep_reward} and {pellets_left} pellets left.")
                    
                    # Log the reward to wandb
                    wandb.log({"episode": i_episode, "reward": ep_reward, "pellets_left": pellets_left})
                    
                    break

                # Check if the batch size limit is reached
            if self.enable_rmq:
                buffer_size = self._get_buffer_size(frames_buffer, actions_buffer)
                logging.info(f"Buffer size: {buffer_size} bytes")
                if buffer_size >= max_batch_size:
                    logging.warning("BUFFER SIZE EXCEEDING 500MB")
                self._save_data_to_redis(i_episode, frames_buffer, actions_buffer)
                frames_buffer, actions_buffer = [], []
                # batch_id += 1

            # Send remaining data at the end of the episode
            if frames_buffer and self.enable_rmq:
                self._save_data_to_redis(i_episode, frames_buffer, actions_buffer)
                frames_buffer, actions_buffer = [], []

            if i_episode > 2: 
                if i_episode % 10 == 0:
                    self.agent.update_target_network()
                    logging.info(f"Updated target network at episode {i_episode}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logging.info(torch.cuda.memory_summary())
                    torch.autograd.set_detect_anomaly(True)

                if i_episode % 1000 == 0:
                    self.agent.save_model('pacman.pth')
                    logging.info(f"Saved model at episode {i_episode}")

        logging.info('Training Complete')
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
        initial_epsilon = 0.5  # Lower initial exploration rate
        min_epsilon = 0.1      # Minimum exploration rate
        decay_rate = 5e6       # Faster decay rate

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

        logging.info("Published dataset to RabbitMQ queue 'HF_upload_queue'")

    def _save_frames_locally(self, frames, episode, actions):
        # Create a directory for the episode if it doesn't exist
        episode_dir = f"episode_{episode}_frs{self.frames_to_skip}"
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)

        # Save each frame as a PNG file with the episode and action in the filename
        for idx, frame in enumerate(frames):
            action = actions[idx]
            filename = os.path.join(episode_dir, f"{idx:05d}.png")
            Image.fromarray(frame).save(filename)
            # logging.info(f"Saved frame {idx} of episode {episode} with action {action} to {filename}")

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
