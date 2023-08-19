import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
import utils.logger_zhiao as logger_zhiao
from agents.diffusion import Diffusion
from agents.model import MLP
import time
from agents.helpers import EMA, SinusoidalPosEmb
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import animation
import numpy as np
from scipy.stats import multivariate_normal

"""
enviroment
"""
def compute_gaussian_density(s):
    mean1 = np.array([-5, 0])
    mean2 = np.array([5, 0])
    height1 = 3
    height2 = 6

    if len(s.shape) == 2:
        density1 = multivariate_normal.pdf(s, mean1, 0.5 * np.eye(2)) * height1
        density2 = multivariate_normal.pdf(s, mean2, 0.5 * np.eye(2)) * height2
        return density1 + density2
    elif len(s.shape) == 1:
        density1 = multivariate_normal.pdf(s, mean1, 0.5 * np.eye(2)) * height1
        density2 = multivariate_normal.pdf(s, mean2, 0.5 * np.eye(2)) * height2
        return density1 + density2
    else:
        raise ValueError("Invalid shape for input 's'. It should be (b, 2) or (2).")

class CustomEnvironment(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.random.uniform(low=-10, high=10, size=(2,))
        return self.state

    def step(self, action):
        self.state = np.clip(self.state + action, -10, 10)
        # reward = self.calculate_reward(self.state)
        reward = compute_gaussian_density(self.state)
        # done = False  # You can define a termination condition here if needed
        done = reward > 6 - 1e-3
        info = {}  # Additional information, if any
        return self.state, reward, done, info

    def render(self, mode='human', **kwargs):
        # Generate grid points within the range [-10, 10]
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        # Calculate function values for each point in the grid
        # Z = self.calculate_reward(np.stack((X, Y), axis=2))
        Z = compute_gaussian_density(np.stack((X, Y), axis=2).reshape(-1, 2)).reshape(100, 100,)

        # Plot the function values as a color map
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        cmap = plt.get_cmap('Blues')  # Set the colormap to blue shades
        norm = plt.Normalize(vmin=0, vmax=np.max(Z))
        ax.scatter(self.state[0], self.state[1], c='gold', marker='*', s=100)
        im = ax.imshow(Z, extent=[-10, 10, -10, 10], cmap=cmap, origin='lower', norm=norm)
        fig.colorbar(im)

        # Render the figure to an array
        canvas.draw()
        data = np.array(canvas.buffer_rgba())

        # Return the NumPy array representation of the figure
        plt.close(fig)
        data_as_numpy = np.asarray(data)
        return data_as_numpy

gym.register(id='Demo-v0', entry_point=CustomEnvironment, max_episode_steps=20)

"""
critic network, includes Q and V
"""
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q_network(x)

class VNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(VNetwork, self).__init__()
        self.state_dim = state_dim

        self.v_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.v_network(state)

"""
adapt to DAC style
"""
class TestCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TestCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_network2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.v_network1 = VNetwork(state_dim, hidden_dim)
        self.v_network2 = VNetwork(state_dim, hidden_dim)

    def forward(self, state, noisy_action, t):
        if torch.all(t == 0): 
            return self.q_network1(state, noisy_action), self.q_network2(state, noisy_action)
        elif torch.all(t == 1):
            return self.v_network1(state), self.v_network2(state)
        else:
            raise ValueError("t must be either 0 or 1")

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

