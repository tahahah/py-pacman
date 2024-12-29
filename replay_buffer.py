from collections import deque
import torch
import random
import numpy as np
import gc

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        
    def cache(self, state, next_state, action, reward, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        # Convert states to uint8 to save memory
        if isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.uint8)
        if isinstance(next_state, np.ndarray):
            next_state = np.asarray(next_state, dtype=np.uint8)
            
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, next_state, action, reward, done))
        else:
            # Free old tensors explicitly
            old_state, old_next_state, _, _, _ = self.buffer[self.position]
            del old_state
            del old_next_state
            self.buffer[self.position] = (state, next_state, action, reward, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
        # Periodic garbage collection
        if self.position % 1000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        
        # Convert to tensors only when sampling
        states = torch.from_numpy(np.array(batch[0], dtype=np.float32) / 255.0).to(device)
        next_states = torch.from_numpy(np.array(batch[1], dtype=np.float32) / 255.0).to(device)
        actions = torch.from_numpy(np.array(batch[2])).to(device)
        rewards = torch.from_numpy(np.array(batch[3], dtype=np.float32)).to(device)
        dones = torch.from_numpy(np.array(batch[4], dtype=np.float32)).to(device)

        return states, next_states, actions, rewards, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)