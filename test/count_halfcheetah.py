import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import gym
import d4rl

def compute_interval_lengths(arr):
    interval_lengths = []
    start = 0

    for i in range(len(arr)):
        if arr[i]:
            interval_lengths.append(i - start + 1)
            start = i
    interval_lengths.append(len(arr) - start + 1)
    return interval_lengths

def vis_dataset():
    # Create the HalfCheetah environment
    env_name = 'halfcheetah-medium-v2'
    # env_name = 'halfcheetah-medium-expert-v2'
    # env_name = 'hopper-medium-v2'
    # env_name = 'walker2d-medium-v2'
    # env = gym.make('halfcheetah-medium-v2')
    env=gym.make(env_name)

    # Set the random seed for reproducibility
    np.random.seed(42)


    dataset = d4rl.qlearning_dataset(env)
    episode_rewards = dataset['rewards']
    episode_dones = dataset['terminals']
    interval_lengths = compute_interval_lengths(episode_dones)
    interval_lengths = np.array(interval_lengths)
    print('Number of episodes:', len(episode_rewards))
    print('Average episode reward:', np.mean(episode_rewards))

    # Plot the change of reward with time step
    plt.plot(episode_rewards[:8000])
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Reward vs. Episode')
    plt.savefig(f'{env_name}-reward-plot.png')
    plt.close()

    # Histogram of the distribution of rewards
    plt.hist(episode_rewards, bins=30)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rewards')
    plt.savefig(f'{env_name}-reward-histogram.png')
    plt.close()

    # Plot the change of reward with time step
    plt.hist(interval_lengths, bins=30)
    plt.xlabel('Interval Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Interval Lengths')
    plt.savefig(f'{env_name}-interval-length-histogram.png')
    plt.close()

def vis_impl():
    filename = "/home/kairong/Diffusion-Policies-for-Offline-RL"
    # dir = "/home/kairong/Diffusion-Policies-for-Offline-RL/test/checkpoints/halfcheetah/"
    dir = "/home/kairong/Diffusion-Policies-for-Offline-RL/test/checkpoints/ql/"
    action_path = os.path.join(dir, "actions.npy")
    reward_path = os.path.join(dir, "rewards.npy")
    q_value_path = os.path.join(dir, "q_values.npy")
    # actions = np.load(os.path.join(filename, "actions.npy"))
    # rewards = np.load(os.path.join(filename, "rewards.npy"))
    # q_values = np.load(os.path.join(filename, "q_values.npy"))
    actions = np.load(action_path)
    rewards = np.load(reward_path)
    q_values = np.load(q_value_path)
    print('Number of episodes:', len(rewards))
    print('Average episode reward:', np.mean(rewards))

    # Plot the change of reward with time step
    # plt.plot(rewards[:2000])
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Reward vs. Episode')
    plt.savefig(os.path.join(dir, 'impl-reward-plot.png'))
    plt.close()

    # Histogram of the distribution of rewards
    plt.hist(rewards, bins=30)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rewards')
    plt.savefig(os.path.join(dir, 'impl-reward-histogram.png'))
    plt.close()

    # Plot the change of reward with time step
    # plt.plot(q_values[:2000])
    plt.plot(q_values)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Reward vs. Episode')
    plt.savefig(os.path.join(dir, 'impl-q_values-plot.png'))
    plt.close()

    # Histogram of the distribution of rewards
    plt.hist(q_values, bins=30)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rewards')
    plt.savefig(os.path.join(dir, 'impl-q_values-histogram.png'))
    plt.close()

if __name__ == "__main__":
    vis_impl()
    # vis_dataset()