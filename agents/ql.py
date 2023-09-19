# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
# from agents.diffusion_ import Diffusion_prime as Diffusion
# from agents.model_ import MLP_wo_tanh as MLP
from agents.nb_diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA, SinusoidalPosEmb
from config import Config

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def expectile_loss(q, target_q, expectile=0.7):
    diff = q - target_q
    return torch.mean(torch.where(diff > 0, expectile * diff ** 2, (1 - expectile) * diff ** 2))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, activation=nn.Mish):
        super(Critic, self).__init__()
        self.action_dim = action_dim
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                    #   nn.Mish(),
                                      activation(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.Mish(),
                                      activation(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.Mish(),
                                      activation(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                    #   nn.Mish(),
                                      activation(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.Mish(),
                                      activation(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.Mish(),
                                      activation(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q(self, state, action):
        return self.forward(state, action)

    def v(self, state):
        action = torch.randn((state.shape[0], self.action_dim), device=state.device)
        return self.qmin(state, action)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q2(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q2_model(x)
    
    def qmin(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, t_dim=16, hidden_dim=256):
#         super(QNetwork, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.input_dim = state_dim + action_dim + t_dim

#         self.q_network = nn.Sequential(
#             nn.Linear(self.input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
        
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(t_dim),
#             nn.Linear(t_dim, t_dim * 2),
#             nn.ReLU(),
#             nn.Linear(t_dim * 2, t_dim),
#         )
#         self.apply(weights_init_)

#     def forward(self, state, action, t=0):
#         t = torch.tensor([t] * state.shape[0], dtype=torch.float32, device=state.device)
#         t = self.time_mlp(t)
#         x = torch.cat([state, action, t], dim=1)
#         return self.q_network(x)

# class TestCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, t_dim=16, hidden_dim=256, max_time_step=1):
#         super(TestCritic, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.t_dim = t_dim
#         self.max_t = max_time_step

#         self.q_network1 = QNetwork(state_dim, action_dim, t_dim, hidden_dim)
#         self.q_network2 = QNetwork(state_dim, action_dim, t_dim, hidden_dim)

#     def q(self, state, action, t=0):
#         return self.q_network1(state, action, t), self.q_network2(state, action, t)
    
#     def v(self, state):
#         action = torch.randn((state.shape[0], self.action_dim), device=state.device)
#         return torch.min(self.q_network1(state, action, self.max_t), self.q_network2(state, action, self.max_t))
    
#     def q1(self, state, action, t=0):
#         return self.q_network1(state, action, t)
    
#     def qmin(self, state, action, t=0):
#         return torch.min(self.q_network1(state, action, t), self.q_network2(state, action, t))

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, t_dim=16, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim + t_dim

        self.q_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            # nn.ReLU(),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            # nn.ReLU(),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.apply(weights_init_)

    def forward(self, state, action, t):
        t = self.time_mlp(t)
        x = torch.cat([state, action, t], dim=1)
        return self.q_network(x)

class TestCritic(nn.Module):
    def __init__(self, state_dim, action_dim, t_dim=16, hidden_dim=256, max_time_step=1):
        super(TestCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.t_dim = t_dim
        self.max_t = max_time_step

        self.q_network1 = QNetwork(state_dim, action_dim, t_dim, hidden_dim)
        self.q_network2 = QNetwork(state_dim, action_dim, t_dim, hidden_dim)

    def q(self, state, action, t=None):
        if t == None:
            t = torch.zeros((state.shape[0],), device=state.device)
        return self.q_network1(state, action, t), self.q_network2(state, action, t)
    
    def v(self, state):
        action = torch.randn((state.shape[0], self.action_dim), device=state.device)
        t = torch.tensor([self.max_t] * state.shape[0], dtype=torch.float32, device=state.device)
        return torch.min(self.q_network1(state, action, t), self.q_network2(state, action, t))
    
    def q1(self, state, action, t=None):
        if t == None: 
            t = torch.zeros((state.shape[0],), device=state.device)
        return self.q_network1(state, action, t)
    
    def q2(self, state, action, t=None):
        if t == None: 
            t = torch.zeros((state.shape[0],), device=state.device)
        return self.q_network2(state, action, t)

    def qmin(self, state, action, t=None):
        if t == None:
            t = torch.zeros((state.shape[0],), device=state.device)
        return torch.min(self.q_network1(state, action, t), self.q_network2(state, action, t))

class Diffusion_QL(object):
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
        self.resample = args.resample

        self.step = 0
        self.step_start_ema = args.step_start_ema
        self.ema = EMA(args.ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = args.update_ema_every
        self.test_critic = args.test_critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.ablation = args.ablation
        if self.test_critic and self.ablation:
            print("Using Test Critic")
            self.critic = TestCritic(state_dim, action_dim, max_time_step=args.T).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        if args.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=args.lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=args.lr_maxt, eta_min=0.)

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
        self.critic_ema = args.critic_ema

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
            q1, q2 = self.critic.q(state, action) # (b, 1)
            if self.ablation: 
                with torch.no_grad():
                    target_v = self.critic_target.v(next_state) # (b, 1)
                    target_q = (reward + not_done * self.discount * target_v) # (b,)
                MSBE_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) # (b,)->(1,)
                with torch.no_grad():
                    denoised_noisy_action = self.ema_model.sample(state)
                    target_v = self.critic_target.qmin(state, denoised_noisy_action) # (b, 1)->(b,)
                v_loss = F.mse_loss(self.critic.v(state), target_v)
                critic_loss = self.MSBE_coef * MSBE_loss + v_loss
            else:
                with torch.no_grad():
                    target_q = self.critic_target.qmin(next_state, self.ema_model.sample(next_state)) # (b, 1)->(b,)
                target_q = reward + not_done * self.discount * target_q # (b,)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) # (b,)->(1,)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_( # type: ignore
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2) 
                if log_writer is not None:
                    log_writer.add_scalar(
                            'Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                metric['critic_norm'].append(critic_grad_norms.max().item()) 
            self.critic_optimizer.step()
            """ Actor Training """
            if self.step % self.policy_freq == 0:
                denoised_noisy_action = self.actor.sample(state)
                q1, q2 = self.critic.q(state, denoised_noisy_action)
                if np.random.uniform() < 0.5:
                    q_loss = - q1.mean() / q1.abs().mean().detach() if self.norm_q else - q1.mean()
                else:
                    q_loss = - q2.mean() / q2.abs().mean().detach() if self.norm_q else - q2.mean()
                bc_loss = self.actor.loss(action, state)
                actor_loss = q_loss + self.bc_weight * bc_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.grad_norm > 0:
                    actor_grad_norms = nn.utils.clip_grad_norm_( # type: ignore
                        self.actor.parameters(), max_norm=self.grad_norm, norm_type=2) 
                    if log_writer is not None:
                        log_writer.add_scalar(
                            'Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                self.actor_optimizer.step()
                if log_writer is not None:
                    log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                    log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                metric['ql_loss'].append(q1.mean().item())
                metric['bc_loss'].append(bc_loss.item())
            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.critic_ema == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
            self.step += 1
            """ Log """
            if log_writer is not None:
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            metric['critic_loss'].append(critic_loss.item())
            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
        return metric

    def sample_action(self, state, noise_scale=0.0):
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
        # state = state.to(self.device)
        # if self.resample:
        #     state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        #     with torch.no_grad():
        #         action = self.actor.sample(state_rpt)
        #         q_value = self.critic_target.qmin(state_rpt, action).flatten()
        #         idx = torch.multinomial(F.softmax(q_value), 1)
        #     return action[idx].cpu().data.numpy().flatten()
        # else:
        #     action = self.actor.sample(state)
        #     action += noise_scale * torch.randn_like(action)
        #     action = action.clamp(-self.max_action, self.max_action)
        #     return action.cpu().data.numpy().flatten()
        assert type(state) is np.ndarray
        state = torch.tensor(state, dtype=torch.float)
        if state.ndim==1:
            state = state.unsqueeze(0)
        assert state.ndim==2 and torch.is_tensor(state)==True # (b, o)
        state = state.to(self.device)
        if self.resample:
            state_rpt = torch.repeat_interleave(state, repeats=50, dim=0) # (50b, o)
            with torch.no_grad():
                action = self.actor.sample(state_rpt) # (50b, a)
                action += noise_scale * torch.randn_like(action)
                action = action.clamp(-self.max_action, self.max_action)
                # q_value = self.critic_target.qmin(state_rpt, action).flatten() # (50b,)
                q_value = self.critic_target.qmin(state_rpt, action).reshape(-1, 50) # (50b,)
                idx = torch.multinomial(F.softmax(q_value, dim=1), 1) # (b, 1)
                idx = idx.flatten() + torch.arange(idx.shape[0]) * 50 #(b,)
            # return action[idx].cpu().data.numpy().flatten()
            return action[idx].cpu().data.numpy().squeeze()
        else:
            action = self.actor.sample(state)
            action += noise_scale * torch.randn_like(action)
            action = action.clamp(-self.max_action, self.max_action)
            # return action.cpu().data.numpy().flatten()
            return action.cpu().data.numpy().squeeze()

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

    # def sample_action(self, state, noise_scale=0.0):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
    #     with torch.no_grad():
    #         action = self.actor.sample(state_rpt)
    #         q_value = self.critic_target.q_min(state_rpt, action).flatten()
    #         idx = torch.multinomial(F.softmax(q_value), 1)
    #     return action[idx].cpu().data.numpy().flatten()