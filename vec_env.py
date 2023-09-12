import gym
import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            observation, reward, done, info = env.step(data)
            if done:
                observation = env.reset()
            remote.send((observation, reward, done, info))
        elif cmd == 'reset':
            observation = env.reset()
            remote.send(observation)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError

class VectorizedEnv:
    def __init__(self, env_fn, num_envs):
        self.remotes, self.workers = zip(*[Pipe() for _ in range(num_envs)])
        self.processes = [Process(target=worker, args=(remote, parent_remote, env_fn)) for remote, parent_remote in zip(self.remotes, self.workers)]
        
        for process in self.processes:
            process.start()
        for remote in self.remotes:
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        observations, rewards, dones, infos = zip(*results)
        return np.stack(observations), np.array(rewards), np.array(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()

# Example usage:
if __name__ == '__main__':
    def make_env():
        return gym.make('CartPole-v1')
    
    num_envs = 4
    vector_env = VectorizedEnv(make_env, num_envs)

    observations = vector_env.reset()
    for _ in range(10):
        actions = np.random.randint(0, 2, size=num_envs)  # Random actions
        observations, rewards, dones, infos = vector_env.step(actions)
        print(observations, rewards, dones, infos)

    vector_env.close()
