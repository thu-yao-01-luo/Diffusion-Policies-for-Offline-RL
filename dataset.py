import d4rl 
import gym
import torch 
import numpy as np
from demo_data_generator import data_generation

def build_dataset(env_name, is_d4rl):
	if is_d4rl:
		env = gym.make(env_name)
		dataset = d4rl.qlearning_dataset(env)
		dataset_ = env.get_dataset() # for timeouts
		dataset["timeouts"] = dataset_["timeouts"]
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

def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward

class DatasetSamplerNP(object):
	def __init__(self, data, device, reward_tune='no'):
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
		reward = self.reward
		if reward_tune == 'normalize':
			reward = (reward - reward.mean()) / reward.std()
		elif reward_tune == 'iql_antmaze':
			reward = reward - 1.0
		elif reward_tune == 'iql_locomotion':
			reward = iql_normalize(reward, self.not_done)
		elif reward_tune == 'cql_antmaze':
			reward = (reward - 0.5) * 4.0
		elif reward_tune == 'antmaze':
			reward = (reward - 0.25) * 2.0
		self.reward = reward

		if reward_tune == 'normalize':
			reward = (reward - reward.mean()) / reward.std()
		elif reward_tune == 'iql_antmaze':
			reward = reward - 1.0
		elif reward_tune == 'iql_locomotion':
			reward = iql_normalize(reward, self.not_done)
		elif reward_tune == 'cql_antmaze':
			reward = (reward - 0.5) * 4.0
		elif reward_tune == 'antmaze':
			reward = (reward - 0.25) * 2.0
		self.reward = reward

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))

		return (
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.not_done[ind].to(self.device)
		)

class DatasetTrajectorySampler(object):
	def __init__(self, data, device, state_len=1, action_len=1, eval_steps=1, gamma=0.99):
		self.state = torch.as_tensor(data['observations'], dtype=torch.float32)
		self.action = torch.as_tensor(data['actions'], dtype=torch.float32)
		self.next_state = torch.as_tensor(data['next_observations'], dtype=torch.float32)
		self.reward = torch.as_tensor(data['rewards'], dtype=torch.float32).view(-1, 1)
		self.terminals = torch.as_tensor(data['terminals'], dtype=torch.float32).view(-1, 1)
		self.not_done = 1. - self.terminals
		self.timeouts = torch.as_tensor(data['timeouts'], dtype=torch.float32).view(-1, 1)
		self.size = self.state.shape[0]
		self.state_dim = self.state.shape[1]
		self.action_dim = self.action.shape[1]
		self.device = device
		self.state_len = state_len
		self.action_len = action_len
		self.trajectory_indices = self.get_trajectory_indices()	# list of tuples (start, end)
		self.action_indices = self.get_action_indices()	# list of tuples (start, end)
		self.merged_action = self.merge_action()
		self.merged_action_len = self.merged_action.shape[0]
		self.merged_action_dim = self.merged_action.shape[1]
		self.merged_state = self.merge_state()
		self.merged_state_len = self.merged_state.shape[0]
		self.merged_state_dim = self.merged_state.shape[1]
		assert self.merged_action_len == self.merged_state_len
		self.eval_steps = eval_steps
		assert self.eval_steps <= self.state_len and self.eval_steps <= self.action_len
		self.merged_next_state = self.merge_next_state()
		self.gamma = gamma
		self.merged_reward = self.merge_reward()
		self.merged_terminal = self.merge_terminal()
		self.merged_not_done = 1. - self.merged_terminal
		
	def get_trajectory_indices(self):
		trajectory_indices = []
		trajectory_start = 0
		print(torch.where(self.terminals == 1))
		print(torch.where(self.timeouts == 1))
		for i in range(self.size):
			if self.terminals[i] == 0 and self.timeouts[i] == 0:
				continue
			else:
				trajectory_indices.append((trajectory_start, i+1)) # [start, end)
				trajectory_start = i+1
		return trajectory_indices
  
	def get_action_indices(self):
		assert self.trajectory_indices is not None
		action_indices = []
		print("len(self.trajectory_indices)", len(self.trajectory_indices))
		for i in range(len(self.trajectory_indices)):
			start = self.trajectory_indices[i][0]
			end = self.trajectory_indices[i][1]
			print("start", start, " end", end)
			if end - start > self.action_len - 1:
				action_indices.append((start, end - self.action_len + 1)) # [start, end)
		return action_indices
  
	def merge_action(self):
		assert self.action_indices is not None
		# merged_action = torch.zeros((len(self.action_indices), self.action_len, self.action_dim))
		merged_action = []
		print("len(self.action_indices)", len(self.action_indices))
		for i in range(len(self.action_indices)):
			print("i", i)
			print("self.action_indices[i]", self.action_indices[i])
			for j in range(self.action_indices[i][0], self.action_indices[i][1]):
				start = j
				end = j + self.action_len
				merged_action.append(self.action[start:end].view(-1))
		merged_action = torch.stack(merged_action, dim=0)
		return merged_action

	def merge_state(self):
		assert self.action_indices is not None
		merged_state = []
		for i in range(len(self.action_indices)):
			for j in range(self.action_indices[i][0], self.action_indices[i][1]):
				# begin = j - self.state_len + 1
				start = max(self.action_indices[i][0], j - self.state_len + 1)
				end = j	+ 1 # open
				if end - start < self.state_len:
					repeated_start = self.state[start].repeat(self.state_len - (end - start), 1)
					merged_state.append(torch.cat((repeated_start, self.state[start:end]), dim=0).view(-1))
				else:
					merged_state.append(self.state[start:end].view(-1))
		merged_state = torch.stack(merged_state, dim=0)
		return merged_state

	def merge_next_state(self):
		assert self.action_indices is not None
		merged_next_state = []
		for i in range(len(self.action_indices)):
			for j in range(self.action_indices[i][0], self.action_indices[i][1]):
				end = j + self.eval_steps # open
				start = max(self.action_indices[i][0], end - self.state_len)
				if end - start < self.state_len:
					repeated_start = self.state[start].repeat(self.state_len - (end - start), 1)
					merged_next_state.append(torch.cat((repeated_start, self.state[start:end]), dim=0).view(-1))
				else:
					merged_next_state.append(self.next_state[start:end].view(-1))
		merged_next_state = torch.stack(merged_next_state, dim=0)
		return merged_next_state

	def merge_reward(self):
		assert self.action_indices is not None
		merged_reward = []
		coef = torch.tensor([self.gamma**k for k in range(self.eval_steps)])
		for i in range(len(self.action_indices)):
			for j in range(self.action_indices[i][0], self.action_indices[i][1]):
				start = j
				end = j + self.eval_steps 	
				merged_reward.append((self.reward[start:end].view(-1) * coef).sum())
		merged_reward = torch.stack(merged_reward, dim=0).reshape(-1, 1)
		return merged_reward

	def merge_terminal(self):
		tmp = torch.zeros((self.merged_action.shape[0],))
		index = 0
		for i in range(len(self.action_indices)):
			span = self.action_indices[i][1] - self.action_indices[i][0]
			index = index + span
			tmp[index - 1] = 1
		return tmp.reshape(-1, 1)

	def sample(self, batch_size):
		ind = torch.randint(0, len(self.action_indices), size=(batch_size,))
		return (
			self.merged_state[ind].to(self.device),
			self.merged_action[ind].to(self.device),
			self.merged_next_state[ind].to(self.device),
			self.merged_reward[ind].to(self.device),
			self.merged_not_done[ind].to(self.device)
		)
		# else:
		# 	ind = torch.randint(0, self.size, size=(batch_size,))
		# 	return (
		# 		self.state[ind].to(self.device),
		# 		self.action[ind].to(self.device),
		# 		self.next_state[ind].to(self.device),
		# 		self.reward[ind].to(self.device),
		# 		self.not_done[ind].to(self.device)
		# 	)

import unittest
import torch
import numpy as np

# Import the DatasetTrajectorySampler class here

class TestDatasetTrajectorySampler(unittest.TestCase):
    def setUp(self):
        # Create a sample data dictionary
        data = {
            'observations': np.random.rand(100, 5),         # Sample state data
            'actions': np.random.rand(100, 2),             # Sample action data
            'next_observations': np.random.rand(100, 5),    # Sample next state data
            'rewards': np.random.rand(100),                # Sample reward data
            'terminals': np.random.randint(2, size=100),   # Sample terminal data (0 or 1)
        }

        device = 'cpu'  # You can use 'cuda' if you have a GPU

        # Initialize the DatasetTrajectorySampler with sample data
        self.sampler = DatasetTrajectorySampler(data, device=device)

    def test_trajectory_indices(self):
        # Test if trajectory_indices returns a list of tuples
        self.assertTrue(isinstance(self.sampler.trajectory_indices, list))
        self.assertTrue(isinstance(self.sampler.trajectory_indices[0], tuple))

    def test_action_indices(self):
        # Test if action_indices returns a list of tuples
        self.assertTrue(isinstance(self.sampler.action_indices, list))
        self.assertTrue(isinstance(self.sampler.action_indices[0], tuple))

    def test_merge_action(self):
        # Test if merge_action returns a torch.Tensor
        merged_action = self.sampler.merge_action()
        self.assertTrue(isinstance(merged_action, torch.Tensor))

    def test_merge_state(self):
        # Test if merge_state returns a torch.Tensor
        merged_state = self.sampler.merge_state()
        self.assertTrue(isinstance(merged_state, torch.Tensor))

    def test_merge_next_state(self):
        # Test if merge_next_state returns a torch.Tensor
        merged_next_state = self.sampler.merge_next_state()
        self.assertTrue(isinstance(merged_next_state, torch.Tensor))

    def test_merge_reward(self):
        # Test if merge_reward returns a torch.Tensor
        merged_reward = self.sampler.merge_reward()
        self.assertTrue(isinstance(merged_reward, torch.Tensor))

    def test_merge_terminal(self):
        # Test if merge_terminal returns a torch.Tensor
        merged_terminal = self.sampler.merge_terminal()
        self.assertTrue(isinstance(merged_terminal, torch.Tensor))

    def test_sample_trajectory(self):
        # Test if sample method with trajectory=True returns the expected tensors
        batch_size = 5
        sampled_data = self.sampler.sample(batch_size)

        # Check the shapes of the sampled tensors
        self.assertEqual(sampled_data[0].shape, (batch_size, self.sampler.merged_state_dim))
        self.assertEqual(sampled_data[1].shape, (batch_size, self.sampler.merged_action_dim))
        self.assertEqual(sampled_data[2].shape, (batch_size, self.sampler.merged_state_dim))
        self.assertEqual(sampled_data[3].shape, (batch_size, 1))
        self.assertEqual(sampled_data[4].shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()
