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
# from dreamfuser.logger import logger as logger_zhiao
import utils.logger_zhiao as logger_zhiao

from agents.diffusion import Diffusion
from agents.model import MLP
import time
from agents.helpers import EMA, SinusoidalPosEmb


class Q_function(nn.Module):
    """
    MLP Model
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 t_dim=16):

        super(Q_function, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state, noisy_action, time):

        t = self.time_mlp(time)
        x = torch.cat([state, noisy_action, t], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class NoisyCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, t_dim=16):
        super(NoisyCritic, self).__init__()
        self.q1_model = Q_function(state_dim, action_dim, hidden_dim, t_dim)
        self.q2_model = Q_function(state_dim, action_dim, hidden_dim, t_dim)

    def forward(self, state, noisy_action, t):
        return self.q1_model(state, noisy_action, t), self.q2_model(state, noisy_action, t)

    def q1(self, state, noisy_action, t):
        return self.q1_model(state, noisy_action, t)

    def q_min(self, state, noisy_action, t):
        q1, q2 = self.forward(state, noisy_action, t)
        return torch.min(q1, q2)

# iql_style
def expectile_loss(q, target_q, expectile=0.7):
    diff = q - target_q
    return torch.mean(torch.where(diff > 0, expectile * diff ** 2, (1 - expectile) * diff ** 2))

def quantile_loss(q, target_q, tau=0.6):
    diff = q - target_q
    return torch.mean(torch.where(diff > 0, tau * diff, (tau - 1) * diff))

def exponential_loss(q, target_q, eta=1.0):
    diff = q - target_q
    return torch.mean(torch.exp(eta * diff) - eta * diff)

class Diffusion_AC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
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
                 ):

        self.model = MLP(state_dim=state_dim,
                         action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.MSBE_coef = MSBE_coef

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = NoisyCritic(state_dim, action_dim).to(device)
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
        
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)
    
    # ---------------------------- #

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [],
                  'critic_loss': [], 'consistency_loss': [], 'MSBE_loss': [], "bc_weight": []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            # begin_time = time.time()
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)
            # print('sample time: ', time.time() - begin_time)
            total_t = torch.tensor(
                self.actor.n_timesteps, dtype=torch.long, device=self.device)

            """ 
            noisy action 
            tricky part: t = 0 does not mean that the action is noise free! 
            it is the first noised action actually! so we need to shift the time by 1.
            so Q(s, a^t, t+1) actually means $Q(s, a^t, t)$ and Q(s, a, 0) is the noise free action Q function.
            Q(s, a^0, 1) and Q(s, a, 0) are different! or a = a^{-1}, below we use a^{-1} to denote the noise free action.
            and use $Q(s, a^t, t+1)$ pattern
            """
            # begin_time = time.time()
            t = torch.randint(0, self.actor.n_timesteps,
                              (batch_size,), device=self.device).long()
            noise = torch.randn_like(action)
            noisy_action = self.actor.q_sample(action, t, noise)
            # print('add noise sample time: ', time.time() - begin_time)
            """ Q Training """
            # begin_time = time.time()
            # consistency loss
            if self.compute_consistency:
                # Q_1(s, a^t, t+1), Q_2(s, a^t, t+1)
                current_q1, current_q2 = self.critic(state, noisy_action, t+1)
                denoised_noisy_action = self.ema_model.p_sample(
                    noisy_action, t, state)  # a^{t-1}, a = a^{-1}
                # Q'_1(s, a^{t-1}, t), Q'_2(s, a^{t-1}, t)
                target_q1, target_q2 = self.critic_target(
                    state, denoised_noisy_action, t)
                # \hat Q = min(Q'_1(s', a^{t-1}, t), Q'_2(s', a^{t-1}, t))
                target_q = self.discount2 * torch.min(target_q1, target_q2).detach()
                if self.iql_style == "discount":
                    consistency_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                elif self.iql_style == "expectile":
                    consistency_loss = expectile_loss(current_q1, target_q, self.expectile) + expectile_loss(current_q2, target_q, self.expectile)
                elif self.iql_style == "quantile":
                    consistency_loss = quantile_loss(current_q1, target_q, self.quantile) + quantile_loss(current_q2, target_q, self.quantile)
                elif self.iql_style == "exponential":
                    consistency_loss = exponential_loss(current_q1, target_q, self.temperature) + exponential_loss(current_q2, target_q, self.temperature)
                else:
                    raise NotImplementedError
            else:
                # Q_1(s, a^t, t+1), Q_2(s, a^t, t+1)
                current_q1, current_q2 = self.critic(state, noisy_action, t+1)
                target_q1, target_q2 = self.critic_target(
                    state, action, t)  # Q'_1(s, a, t), Q'_2(s, a, t)
                # \hat Q = min(Q'_1(s', a, t), Q'_2(s', a, t))
                target_q = torch.min(target_q1, target_q2).detach()
                consistency_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            # MSBE loss
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(
                    next_state, repeats=10, dim=0)
                # next_action_rpt = self.ema_model(next_state_rpt)
                next_action_rpt = torch.randn(
                    next_state_rpt.shape[0], self.action_dim, device=self.device)  # random noise
                target_q1, target_q2 = self.critic_target(
                    next_state_rpt, next_action_rpt, total_t.expand(next_state_rpt.shape[0]))
                target_q1 = target_q1.view(
                    batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(
                    batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                # next_action = self.ema_model(next_state)
                next_action = torch.randn_like(action)  # random noise
                target_q1, target_q2 = self.critic_target(
                    next_state, next_action, total_t.expand(next_state.shape[0]))
                target_q = torch.min(target_q1, target_q2)
            target_q = (reward + not_done * self.discount * target_q).detach()
            current_q1, current_q2 = self.critic(
                state, action, torch.zeros_like(t))
            MSBE_loss = F.mse_loss(current_q1, target_q) + \
                F.mse_loss(current_q2, target_q)

            critic_loss = consistency_loss + self.MSBE_coef * MSBE_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.p_losses(action, state, t)
            new_action = self.actor.p_sample(noisy_action, t, state)

            q1_new_action, q2_new_action = self.critic(state, new_action, t)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = self.bc_weight * bc_loss + self.eta * q_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            self.step += 1
            """ Log """
            if log_writer is not None: 
                if self.grad_norm > 0:
                    log_writer.add_scalar(
                        'Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar(
                        'Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar(
                    'Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar(
                    'Target_Q Mean', target_q.mean().item(), self.step)
                log_writer.add_scalar('Consistency Loss',
                                      consistency_loss.item(), self.step)
                log_writer.add_scalar('MSBE Loss', MSBE_loss.item(), self.step)
            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            metric['consistency_loss'].append(consistency_loss.item())
            metric['MSBE_loss'].append(MSBE_loss.item())
            metric['bc_weight'].append(self.bc_weight)

            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
            
        # if self.tune_bc_weight and np.std(metric['bc_loss']) < self.std_threshold:
        #     self.bc_weight = max(self.bc_lower_bound, self.bc_weight * self.bc_decay)
        if self.tune_bc_weight and np.mean(metric['bc_loss']) < self.value_threshold:
            self.bc_weight = max(self.bc_lower_bound, self.bc_weight * self.bc_decay)
        else:
            self.bc_weight = min(self.bc_upper_bound, self.bc_weight / self.bc_decay)

        return metric

    # ---------------------------- #

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action, torch.zeros(
                action.shape[0], device=self.device).long()).flatten()
            idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)
        return action[idx].cpu().data.numpy().flatten()

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
