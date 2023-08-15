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

# @DeprecationWarning
class MLP1(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16,
                 activation=nn.Mish,
                ):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            # nn.Mish(),
            activation(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                    #    nn.Mish(),
                                       activation(),
                                       nn.Linear(256, 256),
                                    #    nn.Mish(),
                                       activation(),
                                       nn.Linear(256, 256),
                                    #    nn.Mish()
                                       activation(),
                                       )
        self.final_layer = nn.Linear(256, action_dim)
        # self.layer_norm = nn.LayerNorm(action_dim)
        # torch.nn.init.normal_(self.final_layer.weight, mean=0.0, std=0.5) # output layer init  
        # torch.nn.init.normal_(self.final_layer.bias, mean=0.0, std=0.0)
        # pass
        self.apply(weights_init_)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        # return torch.tanh(self.layer_norm(self.final_layer(x)))
        return torch.tanh(self.final_layer(x))

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

        input_dim = state_dim + action_dim 
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       activation(),
                                       nn.Linear(256, 256),
                                       activation(),
                                       nn.Linear(256, 256),
                                       activation(),
                                       )
        self.final_layer = nn.Linear(256, action_dim)
        self.apply(weights_init_)

    def forward(self, x, state):
        x = torch.cat([x, state], dim=1)
        x = self.mid_layer(x)
        return torch.tanh(self.final_layer(x))