import h5py
from matplotlib import animation
import gym
import numpy as np
import matplotlib.pyplot as plt


def custom_func(x):
    return 5.0 - 0.2 * (x ** 2)


def episodes(env, init_state):  # init_state is (b,) shape numpy array
    state_list = np.zeros(
        (init_state.shape[0], env.max_episode_steps))  # (b, t)
    action_list = np.zeros(
        (init_state.shape[0], env.max_episode_steps))  # (b, t)
    next_state_list = np.zeros(
        (init_state.shape[0], env.max_episode_steps))  # (b, t)
    terminal_list = np.zeros(
        (init_state.shape[0], env.max_episode_steps))  # (b, t)
    reward_list = np.zeros(
        (init_state.shape[0], env.max_episode_steps))  # (b, t)
    state = init_state
    for t in range(env.max_episode_steps):
        action = best_action(state)
        next_state = state + action
        state_list[:, t] = state
        action_list[:, t] = action
        next_state_list[:, t] = next_state
        state = next_state
        # reward_list[:, t] = -0.2 * (next_state ** 2) + 0.2 * (state ** 2)
        reward_list[:, t] = custom_func(next_state) - custom_func(state)
        terminal_list[:, t] = (state < env.threshold) * \
            (state > -env.threshold)
    terminal_list[:, -1] = True
    result = dict(
        observations=state_list,
        actions=action_list,
        next_observations=next_state_list,
        terminals=terminal_list,
        rewards=reward_list
    )
    return result


class SquareFunctionToyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.state = None
        self._max_episode_steps = 15
        self.current_step = 0
        self.threshold = 0.01

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.state += action
        self.current_step += 1

        # state_valuation = 5.0 - 0.2 * (self.state ** 2)
        # reward = state_valuation - (5.0 - 0.2 * ((self.state - action) ** 2))
        reward = custom_func(self.state) - custom_func(self.state + action)

        done = self.current_step >= self._max_episode_steps or self.state < self.threshold and self.state > -self.threshold
        info = {}
        info["expectation"] = custom_func(self.state)
        info["action"] = action

        return self.state, reward, done, info

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def reset(self):
        # self.state = np.random.uniform(low=-5.0, high=5.0)
        self.state = 5.0
        self.current_step = 0
        return self.state

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.tight_layout()

            # Plot current position
            axs[0, 0].scatter(self.state, 0, color='r')
            axs[0, 0].set_xlim([-5, 5])
            axs[0, 0].set_ylim([-1, 1])
            axs[0, 0].set_title('Current Position')

            # Plot valuation function
            x = np.linspace(-5, 5, 100)
            # valuation = 5.0 - 0.2 * (x ** 2)
            valuation = custom_func(x)

            axs[0, 1].plot(x, valuation)
            # axs[0, 1].plot(self.state, 5.0 - 0.2 * (self.state ** 2))
            axs[0, 1].scatter(self.state, custom_func(self.state), color='r')
            axs[0, 1].set_xlim([-5, 5])
            axs[0, 1].set_ylim([0, 5])
            axs[0, 1].set_title('Valuation Function')

            # Plot best action distribution (uniform in this case)
            actions = np.linspace(-1, 1, 100)
            distribution = np.ones_like(actions) / len(actions)
            axs[1, 0].plot(actions, distribution)
            axs[1, 0].set_xlim([-1, 1])
            axs[1, 0].set_ylim([0, 1])
            axs[1, 0].set_title('Best Action Distribution')

            # Plot trajectory of the current point
            trajectory = np.zeros((self._max_episode_steps + 1, 2))
            trajectory[:self.current_step, 0] = np.arange(self.current_step)
            trajectory[:self.current_step, 1] = np.linspace(
                0, self.state, self.current_step).flatten()
            axs[1, 1].plot(trajectory[:, 0], trajectory[:, 1])
            axs[1, 1].set_xlim([0, self._max_episode_steps])
            axs[1, 1].set_ylim([-5, 5])
            axs[1, 1].set_title('Trajectory')

            plt.setp(axs, xticks=[], yticks=[])
            plt.subplots_adjust(wspace=0.2, hspace=0.4)

            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            return data
        else:
            super(SquareFunctionToyEnv, self).render(mode=mode)

    def qlearning_dataset(self, size=10000):
        return episodes(self, np.random.uniform(low=-5.0, high=5.0, size=(10000,)))


# Register the environment with the Gym library
gym.register(
    id='square-function-toy-env-v0',
    entry_point=SquareFunctionToyEnv,
)


def test():
    # Usage example:
    env = gym.make('square-function-toy-env-v0')
    state = env.reset()
    done = False

    ims = []
    fig = plt.figure()

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        im = plt.imshow(np.array(env.render(mode='rgb_array')), animated=True)
        ims.append([im])
        animate = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=1000)
        animate.save('animation.mp4')

    env.close()


def best_action(state):
    return np.clip(-state, -1.0, 1.0)


def generate_data():
    env = gym.make('square-function-toy-env-v0')

    result = episodes(env, np.random.uniform(
        low=-5.0, high=5.0, size=(10000,)))
    f = h5py.File('sq_data.hdf5', 'w')
    for k, v in result.items():
        f.create_dataset(k, data=v)


def generate_data_flatten():
    env = gym.make('square-function-toy-env-v0')

    result = episodes(env, np.random.uniform(
        low=-5.0, high=5.0, size=(10000,)))
    f = h5py.File('sq_data_flatten.hdf5', 'w')
    for k, v in result.items():
        if k in ['observations', 'next_observations', 'actions']:
            v = v.reshape(-1, 1)
        else:
            v = v.reshape(-1,)
        f.create_dataset(k, data=v)


def vis_with_states(state):
    env = gym.make('square-function-toy-env-v0')
    ims = []
    fig = plt.figure()
    for s in state:
        env.set_state(s)
        ims.append([plt.imshow(env.render(mode='rgb_array'), animated=True)])
    animate = animation.ArtistAnimation(
        fig, ims, interval=200, blit=True, repeat_delay=1000)
    animate.save('vis_animation_sq.mp4')


def vis():
    f = h5py.File('sq_data.hdf5', 'r')
    state = np.array(f['observations'][:])  # (b, t)
    vis_with_states(state[:10].reshape(-1,))


if __name__ == '__main__':
    # generate_data()
    # vis()
    # generate_data_flatten()
    test()
