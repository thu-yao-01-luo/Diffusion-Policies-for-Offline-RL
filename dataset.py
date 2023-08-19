import d4rl 
import gym
import torch 
import numpy as np

def build_dataset(env_name, is_d4rl):
    if is_d4rl:
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        return dataset
    else:
        raise NotImplementedError

class DatasetSampler:
    def __init__(self, dataset, device) -> None:
        self.dataset = dataset
        self.device = device
        self.observations = np.array(dataset['observations'])
        self.actions = np.array(dataset['actions'])
        self.rewards = np.array(dataset['rewards'])
        self.terminals = np.array(dataset['terminals'])
        self.next_observations = np.array(dataset['next_observations'])
        self.size = len(self.observations)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.observations[idxs], dtype=torch.float32).to(self.device)
        act = torch.as_tensor(self.actions[idxs], dtype=torch.float32).to(self.device)
        obs2 = torch.as_tensor(self.next_observations[idxs], dtype=torch.float32).to(self.device)
        done = torch.as_tensor(self.terminals[idxs], dtype=torch.float32).to(self.device)
        rew = torch.as_tensor(self.rewards[idxs], dtype=torch.float32).to(self.device)
        return obs, act, obs2, rew, 1-done