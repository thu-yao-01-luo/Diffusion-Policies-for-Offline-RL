# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 activation=nn.ReLU,
                ):
        super(MLP, self).__init__()
        self.device = device

        input_dim = state_dim 
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       activation(),
                                       nn.Linear(256, 256),
                                       activation(),
                                       nn.Linear(256, 256),
                                       activation(),
                                       )
        self.final_layer = nn.Linear(256, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = state
        x = self.mid_layer(x)
        return torch.tanh(self.final_layer(x))