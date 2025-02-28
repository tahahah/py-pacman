from collections import deque
import torch
import random
import numpy as np
import gc

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_steps=3, gamma=0.95):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)
        # Add a counter for garbage collection
        self.cache_count = 0
        
    def _get_n_step_info(self):
        """Return the n-step reward, next_state, and done flag."""
        reward, next_state, done = 0.0, None, False
        
        for idx, (_, _, _, r, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if d:
                done = True
                next_state = self.n_step_buffer[-1][1]  # Use the latest next_state
                break
                
        # If not done, use the latest transition's next_state
        if not done and len(self.n_step_buffer) == self.n_steps:
            next_state = self.n_step_buffer[-1][1]
            
        return reward, next_state, done
        
    def cache(self, state, next_state, action, reward, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        # Convert states to uint8 to save memory
        if isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.uint8)
        if isinstance(next_state, np.ndarray):
            next_state = np.asarray(next_state, dtype=np.uint8)
        
        # Add the transition to the n-step buffer
        self.n_step_buffer.append((state, next_state, action, reward, done))
        
        # Only add to the replay buffer if we have enough transitions for an n-step return
        if len(self.n_step_buffer) < self.n_steps and not done:
            return
            
        # Get the n-step information
        n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
        
        # Add the n-step transition to the buffer
        state = self.n_step_buffer[0][0]  # Initial state
        action = self.n_step_buffer[0][2]  # Initial action
        
        # If there's no next_state (because all transitions led to done), use the last one
        if n_step_next_state is None:
            n_step_next_state = self.n_step_buffer[-1][1]
            
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, n_step_next_state, action, n_step_reward, n_step_done))
        else:
            # Free old tensors explicitly
            old_state, old_next_state, _, _, _ = self.buffer[self.position]
            del old_state
            del old_next_state
            self.buffer[self.position] = (state, n_step_next_state, action, n_step_reward, n_step_done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
        # If this was a terminal state, clear the n-step buffer
        if done:
            self.n_step_buffer.clear()
            
        # Periodic garbage collection
        self.cache_count += 1
        if self.cache_count % 1000 == 0:
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
        
        # Convert to tensors only when sampling, but don't convert to device yet
        # This will be done in optimize_model to maintain the same interface
        states = torch.from_numpy(np.array(batch[0])).float()
        next_states = torch.from_numpy(np.array(batch[1])).float()
        actions = torch.from_numpy(np.array(batch[2])).long()
        rewards = torch.from_numpy(np.array(batch[3], dtype=np.float32)).float()
        dones = torch.from_numpy(np.array(batch[4], dtype=np.float32)).float()

        return states, next_states, actions, rewards, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)