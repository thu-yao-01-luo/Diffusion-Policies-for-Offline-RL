import numpy as np
import torch
from torch import nn
import numpy as np
from sklearn.neighbors import KernelDensity

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class BufferWrapper:
    def __init__(self, buffer, device) -> None:
        self.buffer = buffer 
        self.device = device

    def sample(self, batch_size):
        sample = self.buffer.sample(batch_size)
        obs = torch.tensor(sample[0], dtype=torch.float32).to(self.device)
        act = torch.tensor(sample[1], dtype=torch.float32).to(self.device)
        obs2 = torch.tensor(sample[2], dtype=torch.float32).to(self.device)
        done = torch.tensor(sample[3], dtype=torch.float32).to(self.device)
        rew = torch.tensor(sample[4], dtype=torch.float32).to(self.device)
        return obs, act, obs2, rew, done

class Buffer:
    def __init__(self, buffer, device) -> None:
        self.buffer = buffer 
        self.device = device

    def sample(self, batch_size):
        sample = self.buffer.sample_batch(batch_size)
        obs = sample['obs'].to(self.device)
        act = sample['act'].to(self.device)
        obs2 = sample['obs2'].to(self.device)
        done = sample['done'].to(self.device)
        rew = sample['rew'].to(self.device)
        return obs, act, obs2, rew, done

class BufferNotDone:
    def __init__(self, buffer, device) -> None:
        self.buffer = buffer 
        self.device = device

    def sample(self, batch_size):
        sample = self.buffer.sample_batch(batch_size)
        obs = sample['obs'].to(self.device)
        act = sample['act'].to(self.device)
        obs2 = sample['obs2'].to(self.device)
        done = sample['done'].to(self.device)
        rew = sample['rew'].to(self.device)
        return obs, act, obs2, rew, 1-done
    
class SACBufferNotDone:
    def __init__(self, buffer, device) -> None:
        self.buffer = buffer 
        self.device = device
        
    def sample(self, batch_size):
        sample = self.buffer.sample_batch(batch_size)
        obs = sample['obs'].to(self.device)
        act = sample['act'].to(self.device)
        obs2 = sample['obs2'].to(self.device)
        done = sample['done'].to(self.device).unsqueeze(1)
        rew = sample['rew'].to(self.device).unsqueeze(1)
        return obs, act, rew, obs2, 1-done

def compute_entropy(samples):
    # samples: list of numpy array
    # Convert samples to a numpy array
    samples_array = np.array(samples)
    # print(samples_array)
    # Fit a kernel density estimator to the samples
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0)  # You can adjust the kernel and bandwidth
    kde.fit(samples_array)

    # Estimate the log probability of the given sample
    assert samples_array.ndim == 2, "samples shape must be (n, f)!"
    log_prob = kde.score_samples(samples_array)

    # Convert log probability to actual probability
    return -np.mean(log_prob)

class sac_args_type:
    def __init__(self, args):
        self.gamma = args.discount
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.update_ema_every
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.hidden_size = 256
        self.lr = args.lr
        self.cuda = torch.cuda.is_available()
        self.determine = args.determine