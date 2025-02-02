# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F


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
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
  def __init__(self, input_dim, output_dim, 
                atoms=51,  # Default value commonly used in Rainbow DQN
                architecture='canonical',  # Using simpler architecture as default
                hidden_size=512,  # Common default value
                noisy_std=0.1):  # Common default value
    super(DQN, self).__init__()
    self.atoms = atoms
    self.action_space = output_dim
    c, h, w= input_dim
    if architecture == 'canonical':
        self.convs = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
        self.conv_output_size = 3136  # Updated for 128x128 input (64 * 12 * 12)==9216 edit: reverted back for 84x84
    elif architecture == 'data-efficient':
        self.convs = nn.Sequential(
            nn.Conv2d(c, 32, 5, stride=5, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
        self.conv_output_size = 1024  # Updated for 128x128 input (64 * 4 * 4)

    self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
    self.fc1 = NoisyLinear(7*7*64, 512)
    self.fc2 = NoisyLinear(512, output_dim * atoms)

  def forward(self, x, log=False, return_distribution=False):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    v, a = x.view(-1, 1, self.atoms), x.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
        q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
        q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    
    if return_distribution:
        return q
    else:
        # Calculate the expected Q-values
        support = torch.linspace(-10, 10, self.atoms, device=x.device)  # Adjust support values as needed
        q_values = (q * support).sum(2)
        return q_values

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()