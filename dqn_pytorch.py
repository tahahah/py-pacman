import argparse
import concurrent.futures
import math
import os
import queue
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from datasets import Dataset
from dotenv import load_dotenv
from gym.wrappers import FrameStack
from PIL import Image

from replay_buffer import ReplayBuffer
from src.env.pacman_env import PacmanEnv
from wrappers import GrayScaleObservation, ResizeObservation, SkipFrame

# Load environment variables from .env file
load_dotenv()

# Get HF_TOKEN from environment variables
HF_TOKEN = os.getenv('HF_TOKEN')

# if gpu is to be used
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

MAX_MESSAGE_SIZE = 500 * 1024 * 1024  # 500 MB

class DQN(nn.Module):
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


class PacmanAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.steps_done = 0

    def select_action(self, state, epsilon, n_actions):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state.__array__(), device=device).unsqueeze(0)
                return self.policy_net(state).max(1)[1].item()

    def optimize_model(self, memory, gamma):
        if self.steps_done < 1e3:
            return

        state, next_state, action, reward, done = memory.sample()
        
        state = state.to(device)
        next_state = next_state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        done = done.to(device)

        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1))

        next_state_values = torch.zeros(memory.batch_size, device=device)
        with torch.no_grad():
            next_state_values[~done] = self.target_net(next_state).max(1)[0].detach()

        expected_state_action_values = (next_state_values * gamma) + reward

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

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
    frame: int
    frames: List[Any]
    actions: List[int]
    next_frames: List[Any]
    dones: List[bool]
    batch_id: int
    is_last_batch: bool

class PacmanTrainer:
    def __init__(self, layout, episodes, frames_to_skip):
        self.layout = layout
        self.episodes = episodes
        self.frames_to_skip = frames_to_skip
        self.env = self._create_environment()
        self.agent = None
        self.memory = None
        self.save_queue = queue.Queue()
        self.connection = None
        self.channel = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _create_environment(self):
        env = PacmanEnv(layout=self.layout)
        env = SkipFrame(env, skip=self.frames_to_skip)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        return env

    def _setup_rabbitmq(self):
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

    def _close_rabbitmq(self):
        if self.connection:
            self.connection.close()

    def train(self):
        self._setup_rabbitmq()

        screen = self.env.reset(mode='rgb_array')
        n_actions = self.env.action_space.n

        self.agent = PacmanAgent(screen.shape, n_actions)
        self.memory = ReplayBuffer(32)  # BATCH_SIZE = 32

        frames_buffer, actions_buffer, next_frames_buffer, dones_buffer = [], [], [], []
        batch_id = 0
        max_batch_size = 500 * 1024 * 1024  # 500 MB

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

                next_frame = self.env.render(mode='rgb_array')

                frames_buffer.append(current_frame)
                actions_buffer.append(action)
                next_frames_buffer.append(next_frame)
                dones_buffer.append(done)

                self.memory.cache(state, next_state, action, reward, done)

                state = next_state if not done else None

                self.agent.optimize_model(self.memory, gamma=0.99)
                if done:
                    logging.info(f"Episode #{i_episode} finished after {t + 1} timesteps with total reward: {ep_reward}")
                    break

                # Check if the batch size limit is reached
                if self._get_buffer_size(frames_buffer, actions_buffer, next_frames_buffer, dones_buffer) >= max_batch_size:
                    data_record = DataRecord(
                        episode=i_episode,
                        frame=len(frames_buffer),
                        frames=frames_buffer.copy(),
                        actions=actions_buffer.copy(),
                        next_frames=next_frames_buffer.copy(),
                        dones=dones_buffer.copy(),
                        batch_id=batch_id,
                        is_last_batch=False
                    )
                    self._save_data(data_record)
                    frames_buffer, actions_buffer, next_frames_buffer, dones_buffer = [], [], [], []
                    batch_id += 1

            # Send remaining data at the end of the episode
            if frames_buffer:
                data_record = DataRecord(
                    episode=i_episode,
                    frame=len(frames_buffer),
                    frames=frames_buffer.copy(),
                    actions=actions_buffer.copy(),
                    next_frames=next_frames_buffer.copy(),
                    dones=dones_buffer.copy(),
                    batch_id=batch_id,
                    is_last_batch=True
                )
                self._save_data(data_record)
                frames_buffer, actions_buffer, next_frames_buffer, dones_buffer = [], [], [], []
                batch_id = 0

            if i_episode % 10 == 0:
                self.agent.update_target_network()
                logging.info(f"Updated target network at episode {i_episode}")

            if i_episode % 1000 == 0:
                self.agent.save_model('pacman.pth')
                logging.info(f"Saved model at episode {i_episode}")

        logging.info('Training Complete')
        self.env.close()
        self.agent.save_model('pacman.pth')
        self._close_rabbitmq()

    def _get_buffer_size(self, frames_buffer, actions_buffer, next_frames_buffer, dones_buffer):
        # Estimate the size of the buffers in bytes
        return sum([frame.nbytes for frame in frames_buffer]) + \
               sum([frame.nbytes for frame in next_frames_buffer]) + \
               len(actions_buffer) * 4 + len(dones_buffer) * 1

    def _get_epsilon(self, frame_idx):
        return 0.1 + (1.0 - 0.1) * math.exp(-1. * frame_idx / 1e7)

    def _save_data(self, data_record: DataRecord):
        self.save_queue.put(data_record)
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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    layout = args.layout[0]
    episodes = args.episodes[0] if args.episodes else 1000

    if args.train:
        frames_to_skip = args.frames_to_skip[0] if args.frames_to_skip is not None else 10
        trainer = PacmanTrainer(layout=layout, episodes=episodes, frames_to_skip=frames_to_skip)
        trainer.train()

    if args.run:
        runner = PacmanRunner(layout)
        runner.run()
