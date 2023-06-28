# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
import os
import argparse
import gym
import numpy as np
import d4rl
import matplotlib.pyplot as plt
import torch

def bug_mix_reward_q():
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
    
def infer():
    # env = gym.make('hopper-medium-v2')
    env= gym.make('halfcheetah-medium-v2')
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
        bc_weight=7.5,
        tune_bc_weight=False,
        bc_lower_bound=1e-2,
        bc_decay=0.995,
        value_threshold=2.5e-4,
        bc_upper_bound=1e2,
        scale=1.0,
        predict_epsilon=True,
        # MSBE_coef=1.0,
        # discount2=0.999,
        # compute_consistency=True,
        # iql_style="discount",
        # expectile=0.7,
        # quantile=0.6,
        # temperature=1.0
    )
    # from agents.ac_diffusion import Diffusion_AC as Agent
    # agent = Agent(
    #             **kwargs
    #             )
    from agents.ql_diffusion import Diffusion_QL as Agent
    agent = Agent(
        **kwargs
    )   
    dataset = env.get_dataset()
    # dataset = d4rl.get_dataset(env_id)

    print(dataset.keys())
    qpos = dataset["infos/qpos"]
    qvel = dataset["infos/qvel"]
    states = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    # dir = "/home/kairong/Diffusion-Policies-for-Offline-RL/test/checkpoints"
    dir = "/home/kairong/Diffusion-Policies-for-Offline-RL/test/checkpoints/ql/"

    # agent.load_model(dir, id=7550)
    agent.load_model(dir, id=9200)
    my_actions = []
    rewards = []
    q_values = []
    env.reset()
    for i in range(2000):
        env.set_state(qpos[i], qvel[i])
        state = states[i]
        agent_action = agent.sample_action(state)
        _, reward, _, _ = env.step(agent_action)
        t = torch.tensor(0, dtype=torch.float32).unsqueeze(0).cuda()
        torch_state = torch.FloatTensor(state.reshape(1, -1)).to(agent.device)
        torch_action = torch.FloatTensor(agent_action.reshape(1, -1)).to(agent.device)
        q_value = agent.critic.q1(torch_state, torch_action, t).cpu().detach().numpy()
        my_actions.append(agent_action)
        # rewards.append(q_value.item())
        rewards.append(reward)
        q_values.append(q_value.item())
    my_actions = np.array(my_actions)
    rewards = np.array(rewards)
    q_values = np.array(q_values)
    action_path = os.path.join(dir, "actions.npy")
    reward_path = os.path.join(dir, "rewards.npy")
    q_value_path = os.path.join(dir, "q_values.npy")
    np.save(action_path, my_actions)
    np.save(reward_path, rewards)
    np.save(q_value_path, q_values)
    # np.save("actions.npy", my_actions)
    # np.save("rewards.npy", rewards)
    # np.save("q_values.npy", q_values)

def diffusion_training():
    import torch
    import torch.optim as optim
    from agents.diffusion import Diffusion
    from agents.model import MLP
    state_dim = 2
    action_dim = 5
    max_action = 1.0
    scale = 1.0
    # n_timesteps = 1
    n_timestep_list = [1, 5, 10, 100]
    device = 'cuda'
    plt.figure(figsize=(10, 10))
    for n_timesteps in n_timestep_list:
        model = MLP(state_dim=state_dim,
                            action_dim=action_dim, device=device)
        beta_schedule = 'vp'

        diffusion_model = Diffusion(state_dim=state_dim, action_dim=action_dim, model=model, max_action=max_action,
                                beta_schedule=beta_schedule, n_timesteps=n_timesteps, scale=scale).to(device)

        # Set up your diffusion model architecture and parameters
        num_steps = 500
        batch_size = 32
        learning_rate = 0.001

        # Set up optimizer
        optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)

        states = torch.rand(10000, 2, device=device)
        actions = torch.rand(10000, 5, device=device) * states[:, 0].unsqueeze(1)  
        losses = []

        # Training loop
        for step in range(num_steps):
            # Generate a batch of samples from the diffusion model
            indices = np.random.randint(0, 10000, batch_size)
            state = states[indices]
            action = actions[indices]
            # Clear the gradients
            optimizer.zero_grad()
            
            # Backpropagation
            loss = diffusion_model.loss(action, state)
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Report the loss
            if (step + 1) % 10 == 0:
                print(f"n_timesteps: {n_timesteps} Step [{step+1}/{num_steps}], Loss: {loss.item()}")
            losses.append(loss.item())
        plt.plot(losses, label=f'n_timesteps={n_timesteps}')
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Loss')
    plt.title('Loss vs. Episode')
    plt.legend()
    plt.savefig(f'loss-plot.png')
    plt.close()

if __name__ == "__main__":
    infer()
    # diffusion_training()    