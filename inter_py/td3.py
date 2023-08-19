import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from copy import deepcopy
import d4rl
import os
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.utils_zhiao import load_config
import utils.logger_zhiao as logger_zhiao
from utils.logger import logger
from dataclasses import dataclass, field
from torch import nn
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
from demo_env import CustomEnvironment, compute_gaussian_density
from helpers import mlp

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

class td3:
    def __init__(self, env_fn, actor_critic, args) -> None:
            # Set up optimizers for policy and q-function
            actor_critic = MLPActorCritic
            env = env_fn()
            self.ac = actor_critic(env.observation_space, env.action_space, hidden_sizes=[args.hid]*args.l,)
            self.pi_lr = args.lr
            self.q_lr = args.lr
            self.target_noise = args.target_noise
            self.noise_clip = args.noise_clip
            self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
            self.q_optimizer = Adam(self.q_params, lr=self.q_lr) 
            self.ac_targ = deepcopy(self.ac)
            self.act_limit = env.action_space.high[0]
            self.gamma = args.discount
            self.act_dim = env.action_space.shape[0]
            self.act_noise = args.act_noise
            self.polyak = 1 - args.tau
            self.policy_delay = args.policy_delay
            for p in self.ac_targ.parameters():
                p.requires_grad = False
        
    # Set up function for computing TD3 Q-losses
    def compute_loss_q(self, data):
        ac = self.ac
        ac_targ = self.ac_targ
        target_noise = self.target_noise
        noise_clip = self.noise_clip
        act_limit = self.act_limit
        gamma = self.gamma

        # o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, o2, r, d = data
        o = o.to(torch.float32)
        a = a.to(torch.float32)
        o2 = o2.to(torch.float32)
        r = r.to(torch.float32)
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                         Q2Vals=q2.cpu().detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        ac = self.ac
        o = data[0]
        o = o.to(torch.float32)
        na = ac.pi(o)
        q1_pi = ac.q1(o, na)
        return -q1_pi.mean()

    def update(self, data, timer):
        policy_delay = self.policy_delay
        ac = self.ac
        ac_targ = self.ac_targ
        polyak = self.polyak
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        logger_zhiao.logkv_mean('LossQ', loss_q.item())

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            logger_zhiao.logkv_mean('LossPi', loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(self, o, noise_scale):
        ac = self.ac
        act_dim = self.act_dim
        act_limit = self.act_limit
        
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)  

    def sample_action(self, o, noise_scale=0.0):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        o = torch.tensor(o, dtype=torch.float32).to(device)
        return self.get_action(o, self.act_noise)

    def train(self, update_every, replay_buffer, batch_size):
        for i in range(update_every):
            batch = replay_buffer.sample(batch_size)
            self.update(batch, i)

