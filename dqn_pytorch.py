import argparse
import math
import random
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from gym.wrappers import FrameStack

from replay_buffer import ReplayBuffer
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

from src.env.pacman_env import PacmanEnv

# if gpu is to be used
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
steps_done = 0


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def select_action(state, epsilon, policy_net, n_actions):
    """
     Given a state, choose an epsilon-greedy action and update value of step.

     Inputs:
     state(LazyFrame): A single observation of the current state, dimension is (state_dim)
     Outputs:
     action_idx (int): An integer representing which action Mario will perform
     """
    global steps_done
    # EXPLORE
    if np.random.rand() < epsilon:
        action_idx = np.random.randint(n_actions)

    # EXPLOIT
    else:
        state = state.__array__()
        if USE_CUDA:
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        action_values = policy_net(state)
        action_idx = torch.argmax(action_values, axis=1).item()

    # increment step
    steps_done += 1
    return action_idx


def optimize_model(memory: ReplayBuffer, policy_net, optimizer, target_net, gamma):
    if steps_done < 1e3:
        return

    state, next_state, action, reward, done = memory.sample()
    state = state.cuda()
    next_state = next_state.cuda()
    action = action.cuda()
    reward = reward.cuda()
    done = done.cuda()
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            next_state)), device=device, dtype=torch.bool)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state).gather(1, action.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(memory.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(next_state).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(input_dim, output_dim, filename):
    model = DQN(input_dim, output_dim)
    state_dict = torch.load(filename, map_location="cuda:0")
    model.load_state_dict(state_dict)
    return model.to(device)


def train_agent(layout: str, episodes: int = 10000, frames_to_skip: int = 4):
    GAMMA = 0.99
    EPSILON = 1.0
    EPS_END = 0.1
    EPS_DECAY = 1e7
    TARGET_UPDATE = 10
    BATCH_SIZE = 64

    epsilon_by_frame = lambda frame_idx: EPS_END + (EPSILON - EPS_END) * math.exp(
        -1. * frame_idx / EPS_DECAY)

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    env = PacmanEnv(layout=layout)
    env = SkipFrame(env, skip=frames_to_skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    screen = env.reset(mode='rgb_array')

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(screen.shape, n_actions).to(device)
    target_net = DQN(screen.shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayBuffer(BATCH_SIZE)

    for i_episode in range(episodes):
        # Initialize the environment and state
        state = env.reset(mode='rgb_array')
        ep_reward = 0.
        EPSILON = epsilon_by_frame(i_episode)

        for t in count():
            # Select and perform an action
            env.render(mode='human')
            action = select_action(state, EPSILON, policy_net, n_actions)
            next_state, reward, done, info = env.step(action)
            reward = max(-1.0, min(reward, 1.0))
            ep_reward += reward

            memory.cache(state, next_state, action, reward, done)

            # Observe new state
            if done:
                next_state = None

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(memory, policy_net, optimizer, target_net, GAMMA)
            if done:
                print("Episode #{}, lasts for {} timestep, total reward: {}".format(i_episode, t + 1, ep_reward))
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 1000 == 0:
            save_model(target_net, 'pacman.pth')

    print('Complete')
    env.render()
    env.close()

    save_model(target_net, 'pacman.pth')


def run_agent(layout: str):
    env = PacmanEnv(layout)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    screen = env.reset(mode='rgb_array')
    n_actions = env.action_space.n

    model = load_model(screen.shape, n_actions, 'pacman.pth')

    for i in range(10):

        env.render(mode='human')
        screen = env.reset(mode='rgb_array')

        for _ in count():
            env.render(mode='human')
            action = select_action(screen, 0, model, n_actions)
            screen, reward, done, info = env.step(action)

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
        train_agent(layout=layout, episodes=episodes, frames_to_skip=frames_to_skip)

    if args.run:
        run_agent(layout)
