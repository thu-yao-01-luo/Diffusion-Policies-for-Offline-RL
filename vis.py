import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import numpy as np
import torch
import itertools
import numpy as np
import torch
import gym
from demo_env import CustomEnvironment, compute_gaussian_density

"""
visualize the value function for the demo environment
"""
def value_figure(
        new_action,  
        value,
        state_list,
        with_action=True,
        fit_min=False,
    ):
        x = np.linspace(-10, 10, 25)
        y = np.linspace(-10, 10, 25)
        X, Y = np.meshgrid(x, y)
        vector_array = new_action.cpu().detach().numpy().reshape(50, 50, 2)
        # Get the x and y components of the vectors
        u = vector_array[::2, ::2, 0]
        v = vector_array[::2, ::2, 1]

        # Create a quiver plot to show the distribution of the vectors
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        # plt.quiver(X, Y, u, v)
        ax = fig.add_subplot(111)
        # cmap = plt.get_cmap('Blues')  # Set the colormap to blue shades
        cmap = plt.get_cmap('Reds')  # Set the colormap to blue shades # type: ignore
        norm = plt.Normalize(vmin=0, vmax=np.max(value)) # type: ignore
        if fit_min:
            norm = plt.Normalize(vmin=np.min(value), vmax=np.max(value)) # type: ignore
        for state in state_list:
            ax.scatter(state[0], state[1], c='gold', marker='*', s=100) # type: ignore
        if with_action:
            ax.quiver(X, Y, u, v, scale=25) 
        im = ax.imshow(value, extent=[-10, 10, -10, 10], cmap=cmap, origin='lower', norm=norm)
        fig.colorbar(im)
        # Render the figure to an array
        canvas.draw()
        data = np.array(canvas.buffer_rgba())
        # Return the NumPy array representation of the figure
        plt.close(fig)
        data_as_numpy = np.asarray(data) 
        return data_as_numpy

def merge_dicts(images_dict):
    ims = []
    length = len(images_dict)
    h = int(np.sqrt(length))
    w = length // h
    if h * w < length:
        w += 1
    titles = list(images_dict.keys()) # list of titles
    assert "reference" in titles, "reference images is not in the dict"
    times = len(images_dict["reference"])
    fig, axs = plt.subplots(h, w, figsize=(6 * w, 6 * h))
    if length >= 4:
        for t in range(times):
            for i in range(length):
                title = titles[i]
                if title == "reference":
                    axs[i // w][i % w].imshow(images_dict[title][t])
                else:
                    axs[i // w][i % w].imshow(images_dict[title])
                axs[i // w][i % w].set_title(title)
            plt.tight_layout()
            # Render the figure on a canvas
            canvas = FigureCanvas(fig)
            # Convert the canvas to a numpy array
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            data_ = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3) # type: ignore
            ims.append(data_)
            plt.close(fig)
    else:
        for t in range(times):
            for i in range(length):
                title = titles[i]
                if title == "reference":
                    axs[i % w].imshow(images_dict[title][t])
                else:
                    axs[i % w].imshow(images_dict[title])
                axs[i % w].set_title(title)
            plt.tight_layout()
            # Render the figure on a canvas
            canvas = FigureCanvas(fig)
            # Convert the canvas to a numpy array
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            data_ = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3) # type: ignore
            ims.append(data_)
            plt.close(fig)
    return ims

def animation(
    eval_env,
    vis_q, 
    policy,
    algo,
):
    traj_return = 0
    state, done = eval_env.reset(), False
    ims = []
    state_list = []
    while not done:
        action = policy.sample_action(np.array(state))
        state, reward, done, _ = eval_env.step(action)
        traj_return += reward
        im_np = eval_env.render(mode='rgb_array')
        ims.append(im_np)
        state_list.append(state)
    if vis_q:
        x = np.linspace(-10, 10, 50)
        y = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(x, y)
        coords = np.stack((X, Y), axis=2).reshape(-1, 2)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords = torch.tensor(coords, dtype=torch.float32, requires_grad=True, device=device)
        if algo == 'dac':
            # coords and actions are of the same shapes
            t = torch.zeros((coords.shape[0],), device=device).long()
            # new_action = policy.model(coords, torch.randn_like(coords))
            # new_action = policy.model(coords)
            # new_action = policy.actor.sample(coords)
            new_action = policy.actor.model(coords, t, torch.randn_like(coords))
            # Calculate function values for each point in the grid
            # value = policy.critic(coords, new_action, t)[0]
            q = policy.critic.q1(coords, new_action)
            q = q.cpu().detach().numpy().reshape(50, 50)
            print("max value:", np.max(q))
            v = policy.critic.v(coords)
            v = v.cpu().detach().numpy().reshape(50, 50)
            consistency_error = q - v
            next_state = coords + new_action
            next_state = torch.clamp(next_state, -10, 10)
            next_state_value = policy.critic.v(next_state)
            next_state_value_as_numpy = next_state_value.cpu().detach().numpy().reshape(50, 50)
            # next_action = policy.model(next_state, torch.randn_like(coords))
            # next_action = policy.sample_action(next_state)
            # next_action = policy.actor.sample(next_state)
            next_action = policy.actor.model(next_state, t, torch.randn_like(coords))
            next_state_q = policy.critic.q1(next_state, next_action)
            next_state_q_as_numpy = next_state_q.cpu().detach().numpy().reshape(50, 50)
            real_msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * next_state_q.cpu().detach().numpy().reshape(-1)) - v.reshape(-1)).reshape(50, 50)
            msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * next_state_value.cpu().detach().numpy().reshape(-1)) - v.reshape(-1)).reshape(50, 50)
            real_error_minus_msbe_error = real_msbe_error - msbe_error
            # output numpy images 
            data_as_numpy = value_figure(new_action, q, state_list) 
            consist_data_as_numpy = value_figure(new_action, v, state_list, with_action=False)
            consist_error_as_numpy = value_figure(new_action, consistency_error, state_list, with_action=False, fit_min=True)
            next_state_value_as_numpy = value_figure(new_action, next_state_value_as_numpy, state_list, with_action=False)
            msbe_error_as_numpy = value_figure(new_action, msbe_error, state_list, with_action=False, fit_min=True)
            real_next_state_value_as_numpy = value_figure(next_action, next_state_q_as_numpy, state_list, with_action=True)
            real_msbe_error_as_numpy = value_figure(next_action, real_msbe_error, state_list, with_action=False, fit_min=True)
            real_error_minus_msbe_error_as_numpy = value_figure(next_action, real_error_minus_msbe_error, state_list, with_action=False, fit_min=True)
            # merge images
            dicts = {
                "reference": ims,
                "action and q value Q(s, a)": data_as_numpy,
                "q value on random V(s)": consist_data_as_numpy,
                "consistency error: Q(s, a)-V(s)": consist_error_as_numpy,
                "next_state value s'": next_state_value_as_numpy,
                "msbe error r+V(s')-Q(s, a)": msbe_error_as_numpy,
                "real next_state value Q(s', a')": real_next_state_value_as_numpy,
                "real msbe error r+Q(s', a')-Q(s,a)": real_msbe_error_as_numpy,
                "real error minus msbe error": real_error_minus_msbe_error_as_numpy,
            }
            ims = merge_dicts(dicts)
        else:
            raise NotImplementedError
    return ims

