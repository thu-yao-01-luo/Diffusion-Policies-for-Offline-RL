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

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class SQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q_network(x)

class QNetwork2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork2, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q1 = SQNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = SQNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)  

class VNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(VNetwork, self).__init__()
        self.state_dim = state_dim

        self.v_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weights_init_)

    def forward(self, state):
        return self.v_network(state)

class TestCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TestCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network1 = SQNetwork(state_dim, action_dim, hidden_dim)
        self.q_network2 = SQNetwork(state_dim, action_dim, hidden_dim)
        self.v_network = VNetwork(state_dim, hidden_dim)

    def forward(self, state, noisy_action, t):
        # if torch.all(t == 0): 
        #     return self.q_network1(state, noisy_action), self.q_network2(state, noisy_action)
        # elif torch.all(t == 1):
        #     return self.v_network1(state), self.v_network2(state)
        # else:
        #     raise ValueError("t must be either 0 or 1")
        raise NotImplementedError 

    def q1(self, state, action):
        return self.q_network1(state, action).reshape(-1)
        
    def q(self, state, action):
        q1 = self.q_network1(state, action)
        q2 = self.q_network2(state, action)
        return q1.reshape(-1), q2.reshape(-1)

    def qmin(self, state, action):
        return torch.min(self.q_network1(state, action), self.q_network2(state, action)).reshape(-1)
    
    def v(self, state):
        return self.v_network(state).reshape(-1)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        # noise = torch.randn([state.shape[0], self.num_actions]).to(device=state.device)
        # x = F.relu(self.linear1(torch.concat([state, noise], dim=1)))
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        noise = noise.to(device=mean.device)
        action = mean + noise
        return action, torch.tensor(0.), mean
        # return action

class T1DAC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                #  beta_schedule='linear',
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
                         action_dim=action_dim, device=device).to(device)

        self.model = DeterministicPolicy(num_inputs=state_dim, num_actions=action_dim, hidden_dim=256).to(device)
        self.actor = self.model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.scale = scale
        print("scale: ", self.scale)
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.MSBE_coef = MSBE_coef

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        # self.test_critic = test_critic
        # self.critic = TestCritic(state_dim, action_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(
        #     self.critic.parameters(), lr=3e-4)

        # if lr_decay:
        #     self.actor_lr_scheduler = CosineAnnealingLR(
        #         self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
        #     self.critic_lr_scheduler = CosineAnnealingLR(
        #         self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

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

        # self.gamma = args.gamma
        self.gamma = self.discount
        self.alpha = 0.2
        # self.action_range = [action_space.low, action_space.high]
        self.action_range = [-1, 1]

        # self.target_update_interval = args.target_update_interval
        self.target_update_interval = update_ema_every
        # self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.automatic_entropy_tuning = False

        num_inputs = state_dim

        # self.critic = QNetwork(num_inputs, action_dim, 256).to(device=self.device)

        self.critic = QNetwork2(num_inputs, action_dim, 256).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # self.value = ValueNetwork(num_inputs, 256).to(device=self.device)
        # self.value_target = ValueNetwork(num_inputs, 256).to(self.device)
        self.value = VNetwork(num_inputs, 256).to(device=self.device)
        self.value_target = VNetwork(num_inputs, 256).to(self.device)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=lr)
        hard_update(self.value_target, self.value)

        self.policy = DeterministicPolicy(num_inputs, action_dim, 256).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        # o, a, o2, r, d
        # state, action, next_state, reward, not_done
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            vf_next_target = self.value_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # pi, log_pi, mean, log_std = self.policy.sample(state_batch)
        pi, log_pi, mean = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # Regularization Loss
        # reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        reg_loss = 0.001 * (mean.pow(2).mean())
        policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        vf = self.value(state_batch)
        
        with torch.no_grad():
            # vf_target = min_qf_pi - (self.alpha * log_pi)
            vf_target = min_qf_pi.detach() 

        vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

        self.value_optim.zero_grad()
        vf_loss.backward()
        self.value_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.value_target, self.value, self.tau)

        return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    def new_train(self, replay_buffer, iterations, batch_size=100, log_writer=None, t=0):
        metric = {"vf_loss": [], "qf1_loss": [], "qf2_loss": [], "policy_loss": []}
        for ind in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch =\
                state, action, reward, next_state, not_done
            reward_batch = reward_batch.reshape(-1, 1)
            mask_batch = mask_batch.reshape(-1, 1)
            with torch.no_grad():
                vf_next_target = self.value_target(next_state_batch)
                next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            # pi, log_pi, mean, log_std = self.policy.sample(state_batch)
            pi, log_pi, mean = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            # Regularization Loss
            # reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
            reg_loss = 0.001 * (mean.pow(2).mean())
            policy_loss += reg_loss

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            vf = self.value(state_batch)
            
            with torch.no_grad():
                # vf_target = min_qf_pi - (self.alpha * log_pi)
                vf_target = min_qf_pi.detach() 

            vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

            self.value_optim.zero_grad()
            vf_loss.backward()
            self.value_optim.step()

            if t % self.target_update_interval == 0:
                soft_update(self.value_target, self.value, self.tau)
            """ Log """
            vf_loss = vf_loss.item()
            qf1_loss = qf1_loss.item()
            qf2_loss = qf2_loss.item()
            policy_loss = policy_loss.item()
            metric['vf_loss'].append(vf_loss)
            metric['qf1_loss'].append(qf1_loss)
            metric['qf2_loss'].append(qf2_loss)
            metric['policy_loss'].append(policy_loss)
            if log_writer is not None:
                log_writer.add_scalar('vf_loss', vf_loss, t)
                log_writer.add_scalar('qf1_loss', qf1_loss, t)
                log_writer.add_scalar('qf2_loss', qf2_loss, t)
                log_writer.add_scalar('policy_loss', policy_loss, t)
        return metric
    
    def train2(self, replay_buffer, iterations, batch_size=100, log_writer=None, t=0):
        metric = {"vf_loss": [], "qf1_loss": [], "qf2_loss": [], "policy_loss": []}
        for ind in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch =\
                state, action, reward, next_state, not_done
            reward_batch = reward_batch.reshape(-1, 1)
            mask_batch = mask_batch.reshape(-1, 1)
            with torch.no_grad():
                vf_next_target = self.value_target(next_state_batch)
                next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)
            qf1, qf2 = self.critic.q(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1, qf2 = qf1.reshape(-1, 1), qf2.reshape(-1, 1)
            qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            # pi, log_pi, mean, log_std = self.policy.sample(state_batch)
            pi, log_pi, mean = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic.q(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            # Regularization Loss
            # reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
            reg_loss = 0.001 * (mean.pow(2).mean())
            policy_loss += reg_loss

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # vf = self.value(state_batch)
            vf = self.critic.v(state_batch)
            
            with torch.no_grad():
                # vf_target = min_qf_pi - (self.alpha * log_pi)
                vf_target = min_qf_pi.detach() 

            vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

            self.critic_optim.zero_grad()
            # self.value_optim.zero_grad()
            vf_loss.backward()
            # self.value_optim.step()
            self.critic_optim.step()

            if t % self.target_update_interval == 0:
                soft_update(self.value_target, self.value, self.tau)
            """ Log """
            vf_loss = vf_loss.item()
            qf1_loss = qf1_loss.item()
            qf2_loss = qf2_loss.item()
            policy_loss = policy_loss.item()
            metric['vf_loss'].append(vf_loss)
            metric['qf1_loss'].append(qf1_loss)
            metric['qf2_loss'].append(qf2_loss)
            metric['policy_loss'].append(policy_loss)
            if log_writer is not None:
                log_writer.add_scalar('vf_loss', vf_loss, t)
                log_writer.add_scalar('qf1_loss', qf1_loss, t)
                log_writer.add_scalar('qf2_loss', qf2_loss, t)
                log_writer.add_scalar('policy_loss', policy_loss, t)
        return metric
    
    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None, t=0):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [],
                  'critic_loss': [], 'consistency_loss': [], 'MSBE_loss': [], "bc_weight": [], "target_q": [], 
                  "max_next_ac": [], "td_error": [], "consistency_error": [], "actor_q": [], "true_bc_loss": [], 
                  "action_norm": [], "new_action_max": [], "new_action_mean": []}
        for ind in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)
            # noisy_action = torch.randn_like(action) * self.scale
            """ Q Training """
            with torch.no_grad():
                # target_v = self.critic.v(next_state) # (b,)
                # target_v = self.critic_target.v(next_state) # (b,)
                target_v = self.value_target(next_state)
                target_q = (reward.reshape(-1) + not_done.reshape(-1) * self.discount * target_v.reshape(-1)).detach().reshape(-1) # (b,)
            # q1, q2 = self.critic.q(state, noisy_action) # (b,)->(1,)
            # q1, q2 = self.critic(state, noisy_action) # (b,)->(1,)
            q1, q2 = self.critic(state, action) # (b,)->(1,)
            q1, q2 = q1.reshape(-1), q2.reshape(-1)
            MSBE_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) # (b,)->(1,)

            self.critic_optim.zero_grad()
            MSBE_loss.backward()
            self.critic_optim.step()

            # denoised_noisy_action=self.actor.model(noisy_action, t, state) # (b, a)
            # denoised_noisy_action = self.model(noisy_action, state) # (b, a)
            denoised_noisy_action = self.actor(state) # (b, a)
            # q_loss = - self.critic.qmin(state, denoised_noisy_action).mean() # (b,)->(1,)
            q_loss = - torch.min(*self.critic(state, denoised_noisy_action)).mean() # (b,)->(1,)
            reg_loss = 0.001 * (denoised_noisy_action.pow(2).mean())
            policy_loss = q_loss + reg_loss

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            with torch.no_grad():
                # target_v = self.critic.qmin(state, denoised_noisy_action).detach().reshape(-1) # (b,)->(1,)
                target_v = torch.min(*self.critic(state, denoised_noisy_action)).detach().reshape(-1) # (b,)->(1,)
            # v_loss = F.mse_loss(self.critic.v_network(state).reshape(-1), target_v) # (b,)->(1,)
            v_loss = F.mse_loss(self.value(state).reshape(-1), target_v) # (b,)->(1,)
            # self.critic_optimizer.zero_grad()
            self.value_optim.zero_grad()
            v_loss.backward()
            # self.critic_optimizer.step()
            self.value_optim.step()

            """ Step Target network """
            # if self.step % 2 == 0:
            if t % self.target_update_interval == 0:
            #     for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #         target_param.data.copy_(
            #             self.tau * param.data + (1 - self.tau) * target_param.data)
            # if ind % self.target_update_interval == 0:
                for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):    
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
            self.step += 1
            """ Log """
            if log_writer is not None:
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('MSBE Loss', MSBE_loss.item(), self.step)
                log_writer.add_scalar('V Loss', v_loss.item(), self.step)
            metric['ql_loss'].append(q_loss.item())
            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
        return metric

    def old_train(self, replay_buffer, iterations, batch_size=100, log_writer=None, t=0):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [],
                  'critic_loss': [], 'consistency_loss': [], 'MSBE_loss': [], "bc_weight": [], "target_q": [], 
                  "max_next_ac": [], "td_error": [], "consistency_error": [], "actor_q": [], "true_bc_loss": [], 
                  "action_norm": [], "new_action_max": [], "new_action_mean": []}
        for ind in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)
            noisy_action = torch.randn_like(action) * self.scale
            """ Q Training """
            with torch.no_grad():
                # target_v = self.critic.v(next_state) # (b,)
                target_v = self.critic_target.v(next_state) # (b,)
                target_q = (reward.reshape(-1) + not_done.reshape(-1) * self.discount * target_v).detach().reshape(-1) # (b,)
            q1, q2 = self.critic.q(state, noisy_action) # (b,)->(1,)
            MSBE_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) # (b,)->(1,)

            self.critic_optimizer.zero_grad()
            MSBE_loss.backward()
            self.critic_optimizer.step()

            # denoised_noisy_action=self.actor.model(noisy_action, t, state) # (b, a)
            # denoised_noisy_action = self.model(noisy_action, state) # (b, a)
            denoised_noisy_action = self.actor(state) # (b, a)
            q_loss = - self.critic.qmin(state, denoised_noisy_action).mean() # (b,)->(1,)
            reg_loss = 0.01 * (denoised_noisy_action.pow(2).mean())
            policy_loss = q_loss + reg_loss

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            with torch.no_grad():
                target_v = self.critic.qmin(state, denoised_noisy_action).detach().reshape(-1) # (b,)->(1,)
            v_loss = F.mse_loss(self.critic.v_network(state).reshape(-1), target_v) # (b,)->(1,)
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            self.critic_optimizer.step()

            """ Step Target network """
            # if self.step % 2 == 0:
            if t % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
            self.step += 1
            """ Log """
            if log_writer is not None:
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('MSBE Loss', MSBE_loss.item(), self.step)
                log_writer.add_scalar('V Loss', v_loss.item(), self.step)
            metric['ql_loss'].append(q_loss.item())
            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
        return metric

    def sample_action(self, state, noise_scale=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # action = self.model(torch.randn_like(state, device=state.device) * noise_scale, state)
        # action = self.actor(state)
        action = self.policy.sample(state)[0]
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
