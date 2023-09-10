import d4rl 
import gym
import torch 
import numpy as np
from demo_data_generator import data_generation

def build_dataset(env_name, is_d4rl):
    if is_d4rl:
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        return dataset
    elif env_name == "Demo-v0":
        dataset = data_generation(action_type="medium", save_path="data/Demo-v0.hdf5")
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
        # obs = torch.as_tensor(self.observations[idxs], dtype=torch.float32).to(self.device)
        # act = torch.as_tensor(self.actions[idxs], dtype=torch.float32).to(self.device)
        # obs2 = torch.as_tensor(self.next_observations[idxs], dtype=torch.float32).to(self.device)
        # done = torch.as_tensor(self.terminals[idxs], dtype=torch.float32).reshape(-1, 1).to(self.device)
        # rew = torch.as_tensor(self.rewards[idxs], dtype=torch.float32).reshape(-1, 1).to(self.device)
        obs = torch.from_numpy(self.observations[idxs]).float().to(self.device)
        act = torch.from_numpy(self.actions[idxs]).float().to(self.device)
        obs2 = torch.from_numpy(self.next_observations[idxs]).float().to(self.device)
        done = torch.from_numpy(self.terminals[idxs]).float().reshape(-1, 1).to(self.device)
        rew = torch.from_numpy(self.rewards[idxs]).float().reshape(-1, 1).to(self.device)
        return obs, act, obs2, rew, 1-done

class DatasetSamplerNP(object):
	def __init__(self, data, device):
		# self.state = torch.from_numpy(data['observations'])
		# self.action = torch.from_numpy(data['actions'])
		# self.next_state = torch.from_numpy(data['next_observations'])
		# reward = torch.from_numpy(data['rewards']).view(-1, 1)
        # self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1)
		self.state = torch.as_tensor(data['observations'], dtype=torch.float32)
		self.action = torch.as_tensor(data['actions'], dtype=torch.float32)
		self.next_state = torch.as_tensor(data['next_observations'], dtype=torch.float32)
		self.reward = torch.as_tensor(data['rewards'], dtype=torch.float32).view(-1, 1)
		self.not_done = 1. - torch.as_tensor(data['terminals'], dtype=torch.float32).view(-1, 1)
		self.size = self.state.shape[0]
		self.state_dim = self.state.shape[1]
		self.action_dim = self.action.shape[1]
		self.device = device

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))

		return (
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.not_done[ind].to(self.device)
		)
