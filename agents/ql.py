# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
import utils.logger_zhiao as logger_zhiao

from agents.diffusion_ import Diffusion
from agents.model_ import MLP
import time
from agents.helpers import EMA, SinusoidalPosEmb

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def expectile_loss(q, target_q, expectile=0.7):
    diff = q - target_q
    return torch.mean(torch.where(diff > 0, expectile * diff ** 2, (1 - expectile) * diff ** 2))

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, t_dim=16, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim + t_dim

        self.q_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.apply(weights_init_)

    def forward(self, state, action, t=0):
        t = torch.tensor([t] * state.shape[0], dtype=torch.float32, device=state.device)
        t = self.time_mlp(t)
        x = torch.cat([state, action, t], dim=1)
        return self.q_network(x)

class TestCritic(nn.Module):
    def __init__(self, state_dim, action_dim, t_dim=16, hidden_dim=256):
        super(TestCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.t_dim = t_dim

        self.q_network1 = QNetwork(state_dim, action_dim, t_dim, hidden_dim)
        self.q_network2 = QNetwork(state_dim, action_dim, t_dim, hidden_dim)

    def q(self, state, action, t=0):
        return self.q_network1(state, action, t), self.q_network2(state, action, t)
    
    def v(self, state):
        action = torch.randn((state.shape[0], self.action_dim), device=state.device)
        return torch.min(self.q_network1(state, action, 1), self.q_network2(state, action, 1))
    
    def q1(self, state, action, t=0):
        return self.q_network1(state, action)
    
    def qmin(self, state, action, t=0):
        return torch.min(self.q_network1(state, action), self.q_network2(state, action))

class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='vp',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 MSBE_coef=0.05,
                 discount2=0.99,
                 compute_consistency=True,
                 iql_style="discount",
                 expectile=0.7,
                 quantile=0.6,
                 temperature=1.0,
                 bc_weight=1.0,
                 log_every=10,
                 tune_bc_weight=False,
                 std_threshold=1e-4,
                 bc_lower_bound=1e-2,
                 bc_decay=0.995,
                 value_threshold=2.5e-4,
                 bc_upper_bound=1e2,
                 consistency=True,
                 scale=1.0,
                 predict_epsilon=False,
                 debug=False,
                 g_mdp=True, 
                 policy_freq=2,
                 norm_q=True,
                 consistency_coef=1.0,
                 target_noise=0.2, 
                 noise_clip=0.5,
                 add_noise=False,
                 test_critic=False,
                ):
        self.model = MLP(state_dim=state_dim,
                         action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps, scale=scale, predict_epsilon=predict_epsilon).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.scale = scale
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.MSBE_coef = MSBE_coef

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        self.test_critic = test_critic
        self.critic = TestCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.discount2 = discount2
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup
        self.compute_consistency = compute_consistency
        self.iql_style = iql_style
        self.expectile = expectile
        self.quantile = quantile
        self.temperature = temperature
        self.bc_weight = bc_weight
        self.log_every = log_every
        self.tune_bc_weight = tune_bc_weight
        self.std_threshold = std_threshold
        self.bc_lower_bound = bc_lower_bound
        self.bc_decay = bc_decay
        self.value_threshold = value_threshold
        self.bc_upper_bound = bc_upper_bound
        self.consistency = consistency
        self.scale = scale
        self.debug = debug
        self.g_mdp = g_mdp
        self.policy_freq = policy_freq
        self.norm_q = norm_q
        self.consistency_coef = consistency_coef
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.add_noise = add_noise

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'v_loss': [],
                'critic_loss': [], 'consistency_loss': [], 'MSBE_loss': [], "bc_weight": [], "target_q": [], 
                "max_next_ac": [], "td_error": [], "consistency_error": [], "actor_q": [], "true_bc_loss": [], 
                "action_norm": [], "new_action_max": [], "new_action_mean": [], "critic_norm": [], "actor_norm": [],
                "critic_forward": [], "critic_backward": [], "actor_forward": [], "actor_backward": [], "MSBE_time": [], 
                "consistency_time": [], "q_time": [], "bc_time": []}
        for ind in range(iterations):
            # Sample replay buffer / batch
            # add time computation for critic loss, actor loss, q loss, bc loss, and append to the metric
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)
            """ Q Training """
            reward = reward.reshape(-1, 1)
            not_done = not_done.reshape(-1, 1)
            start_time = time.time()
            with torch.no_grad():
                target_v = self.critic_target.v(next_state) # (b, 1)
                target_q = (reward + not_done * self.discount * target_v).detach() # (b,)
            q1, q2 = self.critic.q(state, action) # (b, 1)
            MSBE_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) # (b,)->(1,)
            msbe_time = time.time() - start_time
            start_time = time.time()
            with torch.no_grad():
                denoised_noisy_action = self.ema_model.sample(state)
                # target_v = self.critic.qmin(state, denoised_noisy_action, 0).detach() # (b, 1)->(b,)
                # target_v = self.critic.qmin(state, action, 0).detach() # (b, 1)->(b,)
                # target_v = self.critic.qmin(state, denoised_noisy_action, t_scalar).detach() # (b, 1)->(b,)
                target_v = self.critic_target.qmin(state, denoised_noisy_action).detach() # (b, 1)->(b,)
            v_loss = F.mse_loss(self.critic.v(state), target_v)
            consistency_time = time.time() - start_time
            # current_v = self.critic.qmin(state, noisy_action, t_scalar+1)
            # v_loss = expectile_loss(current_v, target_v, self.expectile)
            critic_loss = MSBE_loss + self.MSBE_coef * v_loss
            self.critic_optimizer.zero_grad()
            start_time = time.time()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_( # type: ignore
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2) 
                if log_writer is not None:
                    log_writer.add_scalar(
                            'Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                metric['critic_norm'].append(critic_grad_norms.max().item()) 
            self.critic_optimizer.step()
            critic_time = time.time() - start_time
            metric['MSBE_time'].append(msbe_time)
            metric['consistency_time'].append(consistency_time)
            metric['critic_forward'].append(msbe_time + consistency_time)
            metric['critic_backward'].append(critic_time)
            """ Actor Training """
            if self.step % self.policy_freq == 0:
                start_time = time.time()
                denoised_noisy_action = self.actor.sample(state)
                q_value = self.critic.qmin(state, denoised_noisy_action)
                q_loss = - q_value.mean() / q_value.abs().mean() if self.norm_q else - q_value.mean()
                q_time = time.time() - start_time
                start_time = time.time()
                # bc_loss = self.actor.p_losses(action, state, t).mean()
                bc_loss = self.actor.loss(action, state).mean()
                bc_time = time.time() - start_time
                actor_loss = q_loss + self.bc_weight * bc_loss
                self.actor_optimizer.zero_grad()
                start_time = time.time()
                actor_loss.backward()
                if self.grad_norm > 0:
                    actor_grad_norms = nn.utils.clip_grad_norm_( # type: ignore
                        self.actor.parameters(), max_norm=self.grad_norm, norm_type=2) 
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                self.actor_optimizer.step()
                actor_time = time.time() - start_time
                if log_writer is not None:
                    log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                    log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                metric['ql_loss'].append(q_value.mean().item())
                metric['bc_loss'].append(bc_loss.item())
                metric['q_time'].append(q_time)
                metric['bc_time'].append(bc_time)
                metric['actor_forward'].append(q_time + bc_time)
                metric['actor_backward'].append(actor_time)
            
            """ Step Target network """
            # self.step_ema()
            if self.step % 2 == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
            self.step += 1
            """ Log """
            if log_writer is not None:
                log_writer.add_scalar('MSBE Loss', MSBE_loss.item(), self.step)
                log_writer.add_scalar('V Loss', v_loss.item(), self.step)   
            metric['MSBE_loss'].append(MSBE_loss.item())
            metric['v_loss'].append(v_loss.item())
            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
        return metric

    def sample_action(self, state, noise_scale=0.0):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if state.ndim==1:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # action = self.actor.model(state, torch.randn_like(state, device=state.device) * noise_scale)
        # action = self.actor.model(state, torch.randn([state.shape[0], self.action_dim], device=state.device))
        # action = self.actor.model(state)
        # action = self.actor.sample(state=state)
        action = self.actor.sample(state)
        action += noise_scale * torch.randn_like(action)
        action = action.clamp(-self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
