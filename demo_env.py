import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import animation

import numpy as np

from scipy.stats import multivariate_normal

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
        done = False  # You can define a termination condition here if needed
        # done = reward > 6 - 1e-3
        info = {}  # Additional information, if any
        return self.state, reward, done, info

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

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

gym.register(id='Demo-v0', entry_point=CustomEnvironment)
    # def __init__(self):
    #     self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    #     self.observation_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
    #     self.state = None

    # def reset(self):
    #     self.state = np.random.uniform(low=-10, high=10, size=(2,))
    #     return self.state

    # def step(self, action):
    #     self.state = np.clip(self.state + action, -10, 10)
    #     # reward = self.calculate_reward(self.state)
    #     reward = compute_gaussian_density(self.state)
    #     done = False  # You can define a termination condition here if needed
    #     info = {}  # Additional information, if any
    #     return self.state, reward, done, info

    # def render(self):
    #     # Generate grid points within the range [-10, 10]
    #     x = np.linspace(-10, 10, 100)
    #     y = np.linspace(-10, 10, 100)
    #     X, Y = np.meshgrid(x, y)

    #     # Calculate function values for each point in the grid
    #     # Z = self.calculate_reward(np.stack((X, Y), axis=2))
    #     Z = compute_gaussian_density(np.stack((X, Y), axis=2).reshape(-1, 2)).reshape(100, 100,)

    #     # Plot the function values as a color map
    #     fig = Figure()
    #     canvas = FigureCanvas(fig)
    #     ax = fig.add_subplot(111)
    #     im = ax.imshow(Z, extent=[-10, 10, -10, 10], cmap='hot', origin='lower', vmin=0, vmax=6)
    #     fig.colorbar(im)

    #     # Render the figure to an array
    #     canvas.draw()
    #     data = np.array(canvas.buffer_rgba())

    #     # Return the NumPy array representation of the figure
    #     plt.close(fig)
    #     data_as_numpy = np.asarray(data)
    #     return data_as_numpy
gym.register(id='Demo-v0', entry_point=CustomEnvironment, max_episode_steps=20)

if __name__ == '__main__':
    env = CustomEnvironment()
    state = env.reset()
    ims = []    
    fig = plt.figure()
    for _ in range(100):
        action = np.random.uniform(low=-1, high=1, size=(2,))
        next_state, reward, done, _ = env.step(action)
        im = env.render()
        im = plt.imshow(im, animated=True)
        ims.append([im])
    vis = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    vis.save('animation.mp4')
    env.close()