# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    assert input.dim() == 2, f"Expected 2D input, got {input.dim()}D"
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
  def __init__(self, input_dim, output_dim, 
                atoms=51,  # Default value commonly used in Rainbow DQN
                architecture='canonical',  # Using simpler architecture as default
                hidden_size=512,  # Common default value
                noisy_std=0.1,  # Common default value
                advantage_groups=4):  # Target number of groups for advantage calculation
    super(DQN, self).__init__()
    self.atoms = atoms
    self.action_space = output_dim
    
    # Handle advantage grouping
    self.advantage_groups = min(advantage_groups, output_dim)
    
    # Calculate actions per group (can be uneven)
    self.group_sizes = []
    base_size = output_dim // self.advantage_groups
    remainder = output_dim % self.advantage_groups
    
    for i in range(self.advantage_groups):
        # Distribute remainder across groups
        size = base_size + (1 if i < remainder else 0)
        self.group_sizes.append(size)
    
    # Calculate cumulative group sizes for indexing
    self.group_cumsum = [0]
    for size in self.group_sizes:
        self.group_cumsum.append(self.group_cumsum[-1] + size)
    
    c, h, w = input_dim
    
    # Define value range for distributional RL
    self.v_min, self.v_max = -60, 80
    self.support = torch.linspace(self.v_min, self.v_max, self.atoms)
    self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
    
    # Feature extraction layers
    if architecture == 'canonical':
        self.convs = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_output_size = self._get_conv_output(input_dim)
    else:
        # Add alternative architectures here if needed
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Common layers
    self.fc_common = NoisyLinear(conv_output_size, hidden_size, std_init=noisy_std)
    
    # Value stream
    self.fc_value = NoisyLinear(hidden_size, atoms, std_init=noisy_std)
    
    # Advantage stream
    self.fc_rewards = NoisyLinear(hidden_size, output_dim * atoms, std_init=noisy_std)
    
  def _get_conv_output(self, shape):
    o = self.convs(torch.zeros(1, *shape))
    return int(np.prod(o.size()))
    
  def reset_noise(self):
    self.fc_common.reset_noise()
    self.fc_value.reset_noise()
    self.fc_rewards.reset_noise()
    
  def features(self, x):
    x = self.convs(x)
    x = x.view(x.size(0), -1)
    return F.relu(self.fc_common(x))
    
  def value_stream(self, x):
    return self.fc_value(x)
    
  def advantage_stream(self, x):
    return self.fc_rewards(x)
    
  def forward(self, x, log=False, return_distribution=False, ref_dist=None):
    batch_size = x.size(0)
    
    # Extract features
    features = self.features(x)
    
    # Value Stream
    value = self.value_stream(features)
    value = value.view(batch_size, 1, self.atoms)
    
    # Advantage Stream
    advantage = self.advantage_stream(features)
    advantage = advantage.view(batch_size, self.action_space, self.atoms)
    
    # Combine streams using dueling architecture
    # Reshape advantage for group-wise normalization
    grouped_advantage = []
    for i in range(self.advantage_groups):
        start_idx = self.group_cumsum[i]
        end_idx = self.group_cumsum[i+1]
        group_adv = advantage[:, start_idx:end_idx, :]
        # Normalize within group
        group_adv = group_adv - group_adv.mean(dim=1, keepdim=True)
        grouped_advantage.append(group_adv)
    
    # Reconstruct the full advantage tensor
    normalized_advantage = torch.cat(grouped_advantage, dim=1)
    
    # Combine value and advantage
    q_dist = value + normalized_advantage
    
    if log:
        q_dist = F.log_softmax(q_dist, dim=2)
    else:
        q_dist = F.softmax(q_dist, dim=2)
    
    if return_distribution:
        return q_dist
    else:
        # Calculate expected Q-values using the support
        q_values = (q_dist * self.support.to(x.device)).sum(2)
        return q_values