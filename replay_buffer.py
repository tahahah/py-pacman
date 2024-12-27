from collections import deque
import torch
import random

USE_CUDA = torch.cuda.is_available()


# class ReplayBuffer(object):
#     def __init__(self, batch_size=32):
#         self.memory = deque(maxlen=10000)
#         self.batch_size = batch_size

#     def cache(self, state, next_state, action, reward, done):
#         """
#         Store the experience to self.memory (replay buffer)

#         Inputs:
#         state (LazyFrame),
#         next_state (LazyFrame),
#         action (int),
#         reward (float),
#         done(bool))
#         """
#         state = state.__array__()
#         next_state = next_state.__array__()

#         state = torch.tensor(state)
#         next_state = torch.tensor(next_state)
#         action = torch.tensor([action])
#         reward = torch.tensor([reward])
#         done = torch.tensor([done])

#         self.memory.append((state, next_state, action, reward, done,))

#     def sample(self):
#         """
#         Retrieve a batch of experiences from memory
#         """
#         batch = random.sample(self.memory, self.batch_size)
#         state, next_state, action, reward, done = map(torch.stack, zip(*batch))
#         return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha

    def cache(self, state, next_state, action, reward, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, next_state, action, reward, done))
        else:
            self.buffer[self.position] = (state, next_state, action, reward, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.array(batch[0])
        next_states = np.array(batch[1])
        actions = np.array(batch[2])
        rewards = np.array(batch[3])
        dones = np.array(batch[4])

        return states, next_states, actions, rewards, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)