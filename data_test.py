from dataset import build_dataset, DatasetSampler
import torch
import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
import numpy as np
import gym

def main():
    env_name = "halfcheetah-medium-v2"
    env = gym.make(env_name)
    dataset = build_dataset(env_name, is_d4rl=True)
    dataset2 = d4rl.qlearning_dataset(env)
    print(dataset['observations'].shape)    
    print(dataset['actions'].shape)
    print(dataset['next_observations'].shape)
    print(dataset['rewards'].shape)
    print(dataset['terminals'].shape)
    print(dataset2['observations'].shape)
    print(dataset2['actions'].shape)
    print(dataset2['next_observations'].shape)
    print(dataset2['rewards'].shape)
    print(dataset2['terminals'].shape)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    data_sampler = DatasetSampler(dataset, device=device)
    data_sampler2 = Data_Sampler(dataset, device=device)
    print(data_sampler.sample(3))
    print(data_sampler2.sample(3))

if __name__ == "__main__":
    main()