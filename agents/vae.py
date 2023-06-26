# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from agents.helpers import (extract,
                            Losses)
from utils.utils import Progress, Silent

class Decoder(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Decoder, self).__init__()
        input_dim = state_dim + action_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                nn.Mish(),
                                nn.Linear(256, 256),
                                nn.Mish(),
                                nn.Linear(256, 256),
                                nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    # ------------------------------------------ sampling ------------------------------------------#

    def forward(self, s):
        batch_size = s.shape[0]
        a = torch.randn(batch_size, self.action_dim).to(s.device) 
        a = self.final_layer(self.mid_layer(torch.cat([s, a], dim=1)))
        return a.clamp_(-self.max_action, self.max_action)
    
class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Encoder, self).__init__()
        input_dim = state_dim + action_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                nn.Mish(),
                                nn.Linear(256, 256),
                                nn.Mish(),
                                nn.Linear(256, 256),
                                nn.Mish())
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_var_layer = nn.Linear(256, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    # ------------------------------------------ sampling ------------------------------------------#

    def forward(self, a, s):
        h = self.mid_layer(torch.cat([s, a], dim=1))
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        return mean, log_var 

class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(VAE, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = max_action
        self.encoder = Encoder(state_dim, action_dim, max_action)
        self.decoder = Decoder(state_dim, action_dim, max_action)
    
    def reparam(self, mean, log_var):
        epsilon = torch.randn_like(mean).to(mean.device)
        z = mean + epsilon * torch.exp(0.5 * log_var)
        return z
    
    def forward(self, a, s):    
        mean, log_var = self.encoder(a, s)
        z = self.reparam(mean, log_var)
        recon_a = self.decoder(z)
        return recon_a, z, mean, log_var
    
    def KL_loss(self, mean, log_var):
        return -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    
    def reconstruction_loss(self, recon_a, a):
        return F.mse_loss(recon_a, a)
    
    def sample(self, s):
        batch_size = s.shape[0]
        z = torch.randn(batch_size, self.action_dim).to(s.device) 
        a = self.decoder(z)
        return a.clamp_(-self.max_action, self.max_action)
   