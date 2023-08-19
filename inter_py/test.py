import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer

if __name__ == "__main__":
    num_envs = 8
    env_name = "HalfCheetah-v2"

    envs = [lambda: gym.make(env_name) for i in range(num_envs)]
    env = SubprocVecEnv(envs, start_method="spawn", )

    buffer_size = 1000
    obs_space = env.observation_space
    action_space = env.action_space
    print(obs_space, action_space)
    buffer = ReplayBuffer(buffer_size, obs_space, action_space, n_envs=num_envs)
    obs = env.reset()
    print(obs.shape)
    for _ in range(10):
        actions = np.stack([env.action_space.sample()] * num_envs, axis=0)
        print(actions.shape)
        next_obs, reward, done, info = env.step(actions)
        buffer.add(obs, next_obs, actions, reward, done, info)
    obs_batch, act_batch, next_obs_batch, done_batch, rew_batch = buffer.sample(8)
    print(obs_batch.shape, act_batch.shape, rew_batch.shape, next_obs_batch.shape, done_batch.shape)     

