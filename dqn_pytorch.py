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
import random

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

ATOMS = 51  # default for C51
V_MIN = -10  # adjusted for Pacman environment
V_MAX = 10  # adjusted for Pacman environment
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOMS).to(device)

class RainbowDQN(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, input_dim, output_dim):
        super(RainbowDQN, self).__init__()
        c, h, w = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

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

        # Noisy Linear Layers
        self.noisy_linear1 = NoisyLinear(3136, 512)
        self.noisy_linear2 = NoisyLinear(512, 512)

        # Dueling Network
        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, ATOMS)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, ATOMS * output_dim)
        )

    def forward(self, x):
        features = self.features(x)
        noisy1 = self.noisy_linear1(features)
        noisy2 = self.noisy_linear2(noisy1)

        values = self.value_stream(noisy2).view(-1, 1, ATOMS)
        advantages = self.advantage_stream(noisy2).view(-1, self.output_dim, ATOMS)

        # Compute Q-values
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        self.noisy_linear1.reset_noise()
        self.noisy_linear2.reset_noise()

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize regular weights and biases
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize noisy weights and biases
        self.noisy_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Standard initialization for weights and biases
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x):
        # Ensure input tensor is properly shaped
        if len(x.shape) == 3:
            x = x.view(-1, self.in_features)
        elif len(x.shape) == 4:
            x = x.view(x.size(0), -1)
            
        return F.linear(x, self.weight + self.noisy_weight, self.bias + self.noisy_bias)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.noisy_weight.data = torch.outer(epsilon_out, epsilon_in)
        self.noisy_bias.data = epsilon_out

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
class PacmanAgent:
    def __init__(self, input_dim, output_dim, model_name="pacman_policy_net_gamengen_1_rainbowDQN"):
        self.q_network = RainbowDQN(input_dim, output_dim).to(device)
        self.target_network = RainbowDQN(input_dim, output_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)
        self.model_name = model_name
        self.steps_done = 0

        # Try to load the model from Hugging Face if it exists
        self.model_name = model_name
        try:
            huggingface_hub.login(token=HF_TOKEN)
            model_path = huggingface_hub.hf_hub_download(repo_id=f"Tahahah/{self.model_name}", filename="checkpoints/pacman.pth", repo_type="model")
            state_dict = torch.load(model_path, map_location=device)
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)
            logging.info(f"Model loaded from Hugging Face: {self.model_name}")
        except Exception as e:
            logging.warning(f"Could not load model from Hugging Face: {e}")

    def select_action(self, state, epsilon, n_actions):
        if random.random() < epsilon:
            return random.randrange(n_actions)
        else:
            with torch.no_grad():
                # Check if state is already a tensor
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state.__array__()).to(device)
                
                # Ensure state has correct shape [batch_size, channels, height, width]
                if len(state.shape) == 3:
                    state = state.unsqueeze(0)
                
                dist = self.q_network(state)
                # Calculate expected value for each action
                q_values = (dist * SUPPORT.unsqueeze(0).unsqueeze(0)).sum(2)
                return q_values.max(1)[1].item()

    def optimize_model(self, memory, gamma=0.99, n_step=3):
        if len(memory.buffer) < 1000:  # Wait for enough samples
            return

        # Sample from prioritized replay buffer
        states, next_states, actions, rewards, dones, indices, weights = memory.sample(32)
        
        # Get current Q-value distribution
        current_q_dist = self.q_network(states)  # Shape: [batch_size, n_actions, ATOMS]
        
        # Reshape actions for gathering
        actions = actions.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
        actions = actions.expand(-1, -1, ATOMS)        # Shape: [batch_size, 1, ATOMS]
        current_q_dist = current_q_dist.gather(1, actions)  # Shape: [batch_size, 1, ATOMS]
        
        # Get next Q-value distribution
        with torch.no_grad():
            next_q_dist = self.target_network(next_states)  # Shape: [batch_size, n_actions, ATOMS]
            
            # Select best actions based on expected value
            expected_values = (next_q_dist * SUPPORT.unsqueeze(0).unsqueeze(0)).sum(2)
            next_actions = expected_values.max(1)[1]  # Shape: [batch_size]
            
            # Reshape next_actions for gathering
            next_actions = next_actions.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
            next_actions = next_actions.expand(-1, -1, ATOMS)        # Shape: [batch_size, 1, ATOMS]
            next_q_dist = next_q_dist.gather(1, next_actions)        # Shape: [batch_size, 1, ATOMS]
            
            # Compute projected distribution
            projected_dist = self._project_dist(next_q_dist.squeeze(1), rewards, dones, gamma)
        
        # Compute KL divergence loss
        loss = -(projected_dist * torch.log(current_q_dist.squeeze(1) + 1e-8)).sum(1)
        
        # Apply importance sampling weights
        weights = torch.FloatTensor(weights).to(device)
        weighted_loss = (loss * weights).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        priorities = loss.detach().cpu().numpy()
        memory.update_priorities(indices, priorities)
        
        # Reset noise for both networks
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def _project_dist(self, dist, rewards, dones, gamma):
        batch_size = rewards.shape[0]
        
        # Project next state distribution using Bellman update
        rewards = rewards.unsqueeze(1)  # [batch_size, 1]
        dones = dones.unsqueeze(1)     # [batch_size, 1]
        
        # Compute projected values
        support = SUPPORT.unsqueeze(0)  # [1, ATOMS]
        delta_z = float(V_MAX - V_MIN) / (ATOMS - 1)
        projected_support = rewards + (1 - dones) * gamma * support
        
        # Clamp projected values to support range
        projected_support = projected_support.clamp(V_MIN, V_MAX)
        
        # Compute projection
        b = (projected_support - V_MIN) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()
        
        projected_dist = torch.zeros_like(dist)
        
        # Handle corner cases
        upper_bound_mask = upper < ATOMS
        lower_bound_mask = lower >= 0
        
        # Distribute probability
        for i in range(batch_size):
            for j in range(ATOMS):
                if upper_bound_mask[i, j] and lower_bound_mask[i, j]:
                    projected_dist[i, lower[i, j]] += dist[i, j] * (upper[i, j] - b[i, j])
                    projected_dist[i, upper[i, j]] += dist[i, j] * (b[i, j] - lower[i, j])
        
        return projected_dist
    
    def update_target_network(self):
        tau = 0.005  # Soft update parameter
        for target_param, policy_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filename):
        # Save the model locally
        torch.save(self.q_network.state_dict(), filename)

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
        agent.q_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(state_dict)
        return agent

class ProportionalPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def cache(self, state, next_state, action, reward, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, next_state, action, reward, done))
        else:
            self.buffer[self.pos] = (state, next_state, action, reward, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))

        states = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(batch[1]), dtype=torch.float32).to(device)
        actions = torch.tensor(batch[2], dtype=torch.int64).to(device)
        rewards = torch.tensor(batch[3], dtype=torch.float32).to(device)
        dones = torch.tensor(batch[4], dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        return states, next_states, actions, rewards, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

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
        self.memory = ProportionalPrioritizedReplayBuffer(100000)  # Use the new prioritized replay buffer

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

                self.agent.optimize_model(self.memory, gamma=0.99, n_step=3)
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
        initial_epsilon = 0.8  # Lower initial exploration rate
        min_epsilon = 0.05      # Minimum exploration rate
        decay_rate = 5e2       # Faster decay rate

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
