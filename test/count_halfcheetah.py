import numpy as np
import matplotlib.pyplot as plt
import gym
import d4rl

# Create the HalfCheetah environment
# env_name = 'halfcheetah-medium-v2'
# env_name = 'halfcheetah-medium-expert-v2'
env_name = 'hopper-medium-v2'
# env = gym.make('halfcheetah-medium-v2')
env=gym.make(env_name)

# Set the random seed for reproducibility
np.random.seed(42)


dataset = d4rl.qlearning_dataset(env)
episode_rewards = dataset['rewards']
print('Number of episodes:', len(episode_rewards))
print('Average episode reward:', np.mean(episode_rewards))

# Plot the change of reward with time step
plt.plot(episode_rewards[:2000])
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
