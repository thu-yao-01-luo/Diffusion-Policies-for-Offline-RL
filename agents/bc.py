# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from agents.diffusion_ import Diffusion
from agents.model_ import MLP
import time
from agents.helpers import EMA
from config import Config

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Diffusion_BC(object):
    def __init__(self, state_dim, action_dim, max_action, device, args: Config):
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=args.beta_schedule, n_timesteps=args.T, scale=args.scale,
                               predict_epsilon=args.predict_epsilon).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)

        self.scale = args.scale
        self.lr_decay = args.lr_decay
        self.grad_norm = args.grad_norm
        self.MSBE_coef = args.MSBE_coef

        self.step = 0
        self.step_start_ema = args.step_start_ema
        self.ema = EMA(args.ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = args.update_ema_every
        if args.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=args.lr_maxt, eta_min=0.)
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = args.discount
        self.discount2 = args.discount2
        self.tau = args.tau
        self.eta = args.eta
        self.device = device
        self.max_q_backup = args.max_q_backup
        self.compute_consistency = args.compute_consistency
        self.iql_style = args.iql_style
        self.expectile = args.expectile
        self.quantile = args.quantile
        self.temperature = args.temperature
        self.bc_weight = args.bc_weight
        self.tune_bc_weight = args.tune_bc_weight
        self.std_threshold = args.std_threshold
        self.bc_lower_bound = args.bc_lower_bound
        self.bc_decay = args.bc_decay
        self.value_threshold = args.value_threshold
        self.bc_upper_bound = args.bc_upper_bound
        self.consistency = args.consistency
        self.scale = args.scale
        self.debug = args.debug
        self.g_mdp = args.g_mdp
        self.policy_freq = args.policy_delay
        self.norm_q = args.norm_q
        self.consistency_coef = args.consistency_coef
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.add_noise = args.add_noise
        self.resample = args.resample
        self.bc_sequence = args.bc_sequence

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'v_loss': [], "forward_time": [], "backward_time": [],
                  'critic_loss': [], 'consistency_loss': [], 'MSBE_loss': [], "bc_weight": [], "target_q": [], 
                  "max_next_ac": [], "td_error": [], "consistency_error": [], "actor_q": [], "true_bc_loss": [], 
                  "action_norm": [], "new_action_max": [], "new_action_mean": [], "critic_norm": [], "actor_norm": [],}
        for ind in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)
            """ Q Training """
            # bc_loss = self.actor.p_losses(action, state, t).mean()
            # bc_loss = self.actor.loss(action, state).mean()
            starting_time = time.time()
            if self.bc_sequence:
                bc_loss = F.mse_loss(self.actor.sample(state=state), action)
            else:
                bc_loss = self.actor.loss(action, state).mean()
            ending_time = time.time()
            actor_loss = bc_loss
            self.actor_optimizer.zero_grad()
            back_starting_time = time.time()
            actor_loss.backward()
            back_ending_time = time.time()
            self.actor_optimizer.step() 
            if log_writer is not None:
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            metric['bc_loss'].append(bc_loss.item())
            metric["forward_time"].append(ending_time - starting_time)
            metric["backward_time"].append(back_ending_time - back_starting_time)
            if self.lr_decay:
                self.actor_lr_scheduler.step()
        return metric

    def sample_action(self, state, noise_scale=0.0):
        # # if state.ndim==1:
        # #     state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        # if state.ndim==1 and torch.is_tensor(state)==False:
        #     state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        # elif state.ndim==1 and torch.is_tensor(state)==True:
        #     state = state.float().unsqueeze(0)
        # elif state.ndim==2 and torch.is_tensor(state)==False:
        #     state = torch.tensor(state, dtype=torch.float)
        # elif state.ndim==2 and torch.is_tensor(state)==True:    
        #     state = state.float()
        # else:
        #     raise NotImplementedError
        # # state = torch.tensor(state, dtype=torch.float).to(self.device)
        # state = state.to(self.device)
        # action = self.actor.sample(state)
        # action += noise_scale * torch.randn_like(action)
        # action = action.clamp(-self.max_action, self.max_action)
        # return action.cpu().data.numpy().flatten()
        assert type(state) is np.ndarray
        state = torch.tensor(state, dtype=torch.float)
        if state.ndim==1:
            state = state.unsqueeze(0)
        assert state.ndim==2 and torch.is_tensor(state)==True # (b, o)
        state = state.to(self.device)
        action = self.actor.sample(state)
        action += noise_scale * torch.randn_like(action)
        action = action.clamp(-self.max_action, self.max_action)
        # return action.cpu().data.numpy().flatten()
        return action.cpu().data.numpy().squeeze()


    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
