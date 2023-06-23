# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gym
import numpy as np
import d4rl
import torch


if __name__ == "__main__":
    file_path = "/home/kairong/Diffusion-Policies-for-Offline-RL/actor_7200.pth"
    env = gym.make('hopper-medium-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device='cuda',
        discount=0.999,
        tau=0.005,
        max_q_backup=False,
        beta_schedule='vp',
        n_timesteps=1,
        eta=1.0,
        lr=3e-4,
        lr_decay=False,
        lr_maxt=8000,
        grad_norm=9.0,
        MSBE_coef=1.0,
        discount2=0.999,
        compute_consistency=True,
        iql_style="discount",
        expectile=0.7,
        quantile=0.6,
        temperature=1.0
    )
    from agents.ac_diffusion import Diffusion_AC as Agent
    agent = Agent(
                **kwargs
                )
    dataset = d4rl.qlearning_dataset(env)
    states = dataset['observations'][:10000]
    actions = dataset['actions'][:10000]
    dir = "/home/kairong/Diffusion-Policies-for-Offline-RL/test/checkpoints"
    agent.load_model(dir, id=7200)
    my_actions = []
    rewards = []
    for i in range(10000):
        state = states[i]
        agent_action = agent.sample_action(state)
        t = torch.tensor(0, dtype=torch.float32).unsqueeze(0).cuda()
        torch_state = torch.FloatTensor(state.reshape(1, -1)).to(agent.device)
        torch_action = torch.FloatTensor(agent_action.reshape(1, -1)).to(agent.device)
        q_value = agent.critic.q1(torch_state, torch_action, t).cpu().detach().numpy()
        my_actions.append(agent_action)
        rewards.append(q_value.item())
    my_actions = np.array(my_actions)
    rewards = np.array(rewards)
    np.save("actions.npy", my_actions)
    np.save("rewards.npy", rewards)