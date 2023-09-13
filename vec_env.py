from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
import numpy as np
import torch
import d4rl

if __name__ == "__main__":
    env_fn = lambda: gym.make("halfcheetah-medium-v2")
    vec_env = SubprocVecEnv([env_fn for _ in range(4)])
    output1 = vec_env.reset()
    print(output1)
    action = np.array([vec_env.action_space.sample() for _ in range(4)])    
    output2 = vec_env.step(action)
    print(output2)
    vec_env.close()
