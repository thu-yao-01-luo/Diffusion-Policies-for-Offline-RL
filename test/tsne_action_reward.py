import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import gym
import d4rl


def visualize_tsne(actions, rewards):
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    actions_tsne = tsne.fit_transform(actions)

    # Normalize rewards to range [0, 1]
    normalized_rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))

    # Create scatter plot
    plt.scatter(actions_tsne[:, 0], actions_tsne[:, 1], c=normalized_rewards, cmap='cool', alpha=0.5)
    plt.colorbar(label='Normalized Rewards')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Actions with Rewards')
    # plt.show()
    plt.savefig('tsne.png')


# Create the HalfCheetah environment
env = gym.make('halfcheetah-medium-v2')

# Set the random seed for reproducibility
np.random.seed(42)


dataset = d4rl.qlearning_dataset(env)
episode_rewards = dataset['rewards'][:2000]
actions = dataset['actions'][:2000]
print('Number of episodes:', len(episode_rewards))
print('Average episode reward:', np.mean(episode_rewards))
visualize_tsne(actions, episode_rewards)