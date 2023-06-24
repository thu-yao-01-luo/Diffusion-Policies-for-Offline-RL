import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import gym
import d4rl


def visualize_tsne(actions, rewards, prefix='halfcheetah'):
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
    plt.savefig(prefix + '-tsne.png')

def visualize_tsne_double(actions, rewards, impl_actions, impl_rewards, prefix='halfcheetah'):
    # Assuming you have two numpy arrays: array1 and array2

    # Generate t-SNE embeddings for array1
    # tsne1 = TSNE(n_components=2, random_state=42)
    # embeddings1 = tsne1.fit_transform(actions)

    # # Generate t-SNE embeddings for array2
    # tsne2 = TSNE(n_components=2, random_state=42)
    # embeddings2 = tsne2.fit_transform(impl_actions)

    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(np.concatenate((actions, impl_actions), axis=0))
    
    plt.figure(figsize=(10, 5))
    
    plt.scatter(embedding[:len(actions), 0], embedding[:len(actions), 1], c=rewards, cmap='YlGnBu', alpha=0.7)
    plt.colorbar(label='Normalized Rewards Data')
    plt.scatter(embedding[len(actions):, 0], embedding[len(actions):, 1], c=impl_rewards, cmap='YlOrBr', alpha=0.7)

    # # Plot the embeddings for array1 with lighter color indicating higher values
    # plt.scatter(embeddings1[:, 0], embeddings1[:, 1], c=rewards, cmap='YlGnBu', alpha=0.7)

    # # Plot the embeddings for array2 with darker color indicating higher values
    # plt.scatter(embeddings2[:, 0], embeddings2[:, 1], c=impl_rewards, cmap='YlOrBr', alpha=0.7)

    plt.colorbar(label='Normalized Rewards Impl')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Actions with Rewards')
    plt.savefig(prefix + '-double-tsne.png')

def dataset_tsne(dataset):
    # Create the HalfCheetah environment
    # env = gym.make('halfcheetah-medium-v2')
    env_name = 'hopper-medium-v2'
    env = gym.make(env_name)

    # Set the random seed for reproducibility

    dataset = d4rl.qlearning_dataset(env)
    episode_rewards = dataset['rewards'][:2000]
    actions = dataset['actions'][:2000]
    print('Number of episodes:', len(episode_rewards))
    print('Average episode reward:', np.mean(episode_rewards))
    visualize_tsne(actions, episode_rewards, prefix=env_name)
    
def impl_tsne():
    filename = "/home/kairong/Diffusion-Policies-for-Offline-RL"
    actions = np.load(os.path.join(filename, "actions.npy"))
    rewards = np.load(os.path.join(filename, "rewards.npy"))
    visualize_tsne(actions, rewards, prefix='hopper-impl')

def double_tsne():
    filename = "/home/kairong/Diffusion-Policies-for-Offline-RL/test/checkpoints/halfcheetah"
    impl_actions = np.load(os.path.join(filename, "actions.npy"))[:2000]
    impl_rewards = np.load(os.path.join(filename, "rewards.npy"))[:2000]
    impl_q_values = np.load(os.path.join(filename, "q_values.npy"))[:2000]
    # env_name = 'hopper-medium-v2'
    env_name = 'halfcheetah-medium-v2'
    env = gym.make(env_name)

    # Set the random seed for reproducibility

    dataset = d4rl.qlearning_dataset(env)
    episode_rewards = dataset['rewards'][:2000]
    actions = dataset['actions'][:2000]

    visualize_tsne_double(actions, episode_rewards, impl_actions, impl_rewards, prefix=env_name + '-rewards')
    visualize_tsne_double(actions, episode_rewards, impl_actions, impl_q_values, prefix=env_name + '-q_values')


if __name__ == "__main__":
    # dataset_tsne()
    # impl_tsne()
    double_tsne()