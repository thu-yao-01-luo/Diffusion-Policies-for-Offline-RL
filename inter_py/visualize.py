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
        cmap = plt.get_cmap('Reds')  # Set the colormap to blue shades
        norm = plt.Normalize(vmin=0, vmax=np.max(value))
        if fit_min:
            norm = plt.Normalize(vmin=np.min(value), vmax=np.max(value))
        for state in state_list:
            ax.scatter(state[0], state[1], c='gold', marker='*', s=100)
        if with_action:
            ax.quiver(X, Y, u, v, scale=30) 
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
            data_ = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
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
            data_ = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            ims.append(data_)
            plt.close(fig)
    return ims

@DeprecationWarning
def merge_images(image1_list, image2_list, image3_list, image4_list, image5_list, image6_list, image7_list, image8_list, image9_list):
    ims = []
    for im1, im2, im3, im4, im5, im6, im7, im8, im9 in zip(image1_list, image2_list, image3_list, image4_list, image5_list, image6_list, image7_list, image8_list, image9_list):
        # Create a figure with two subplots
        fig, axs = plt.subplots(3, 3, figsize=(20, 18))

        # Plot the first figure on the first subplot
        axs[0][0].imshow(im1)  # 'gray' colormap for grayscale images
        axs[0][0].set_title('real reward')

        # Plot the second figure on the second subplot
        axs[1][0].imshow(im2)  # 'gray' colormap for grayscale images
        axs[1][0].set_title('action and q value')

        axs[2][0].imshow(im3)  # 'gray' colormap for grayscale images
        axs[2][0].set_title('q value on random')

                # Plot the first figure on the first subplot
        axs[0][1].imshow(im4)  # 'gray' colormap for grayscale images
        axs[0][1].set_title('consistency error')

        # Plot the second figure on the second subplot
        axs[1][1].imshow(im5)  # 'gray' colormap for grayscale images
        axs[1][1].set_title('next_state')
        axs[2][1].imshow(im6)  # 'gray' colormap for grayscale images
        axs[2][1].set_title('msbe error')
        axs[0][2].imshow(im7)  # 'gray' colormap for grayscale images
        # axs[2][0].set_title('next_state real msbe error')
        axs[0][2].set_title('real next_state')

        # Plot the second figure on the second subplot
        axs[1][2].imshow(im8)  # 'gray' colormap for grayscale images
        # axs[2][1].set_title('real next_state')
        axs[1][2].set_title('next_state real msbe error')
        # Adjust layout to prevent overlapping of titles and axes

        axs[2][2].imshow(im9)  # 'gray' colormap for grayscale images
        axs[2][2].set_title('real msbe error - msbe error')
        plt.tight_layout()

        # Render the figure on a canvas
        canvas = FigureCanvas(fig)

        # Convert the canvas to a numpy array
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        data_ = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
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
            t = torch.zeros((coords.shape[0],), device=device).long()
            # new_action = policy.actor.p_sample(torch.randn_like(coords), t, coords)
            new_action = policy.model(torch.randn_like(coords), coords)
            random_action = torch.zeros_like(new_action, device=device)
            # Calculate function values for each point in the grid
            value = policy.critic(coords, new_action, t)[0]
            print("max value:", max(value.cpu().detach().numpy().reshape(-1)))
            value = value.cpu().detach().numpy().reshape(50, 50)
            consist_value = policy.critic(coords, random_action, t+1)[0]
            consist_value = consist_value.cpu().detach().numpy().reshape(50, 50)
            consistency_error = value - consist_value
            next_state = coords + new_action
            next_state = torch.clamp(next_state, -10, 10)
            next_state_value = policy.critic(next_state, random_action, t+1)[0]
            next_state_value_as_numpy = next_state_value.cpu().detach().numpy().reshape(50, 50)
            # real_next_action = policy.actor.p_sample(torch.randn_like(coords), t, next_state)
            # real_next_action = policy.model(torch.randn_like(coords), next_state)
            real_next_action = policy.actor(next_state)
            real_next_state_value = policy.critic(next_state, real_next_action, t)[0]
            real_next_state_value_as_numpy = real_next_state_value.cpu().detach().numpy().reshape(50, 50)
            real_msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * real_next_state_value.cpu().detach().numpy().reshape(-1)) - value.reshape(-1)).reshape(50, 50)
            msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * next_state_value.cpu().detach().numpy().reshape(-1)) - value.reshape(-1)).reshape(50, 50)
            real_error_minus_msbe_error = real_msbe_error - msbe_error
            # output numpy images 
            data_as_numpy = value_figure(new_action, value, state_list) 
            consist_data_as_numpy = value_figure(new_action, consist_value, state_list, with_action=False)
            consist_error_as_numpy = value_figure(new_action, consistency_error, state_list, with_action=False, fit_min=True)
            next_state_value_as_numpy = value_figure(new_action, next_state_value_as_numpy, state_list, with_action=False)
            msbe_error_as_numpy = value_figure(new_action, msbe_error, state_list, with_action=False, fit_min=True)
            real_next_state_value_as_numpy = value_figure(real_next_action, real_next_state_value_as_numpy, state_list, with_action=True)
            real_msbe_error_as_numpy = value_figure(real_next_action, real_msbe_error, state_list, with_action=False, fit_min=True)
            real_error_minus_msbe_error_as_numpy = value_figure(real_next_action, real_error_minus_msbe_error, state_list, with_action=False, fit_min=True)
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
            # ims = merge_images(ims, [data_as_numpy] * len(ims), 
            #                    [consist_data_as_numpy] * len(ims), 
            #                    [consist_error_as_numpy] * len(ims), 
            #                    [next_state_value_as_numpy] * len(ims), 
            #                    [msbe_error_as_numpy] * len(ims), 
            #                    [real_next_state_value_as_numpy] * len(ims), 
            #                    [real_msbe_error_as_numpy] * len(ims), 
            #                    [real_error_minus_msbe_error_as_numpy] * len(ims))
        elif algo == 't1dac':
            # new_action = policy.model(torch.randn_like(coords), coords)
            # new_action = policy.model(coords)
            new_action = policy.policy(coords)
            random_action = torch.zeros_like(new_action, device=device)
            # Calculate function values for each point in the grid
            value = policy.critic.q1(coords, new_action)
            # value = policy.critic(coords, new_action)[0]
            print("max value:", max(value.cpu().detach().numpy().reshape(-1)))
            value = value.cpu().detach().numpy().reshape(50, 50)
            # consist_value = policy.critic.v(coords)
            consist_value = policy.value(coords)
            consist_value = consist_value.cpu().detach().numpy().reshape(50, 50)
            consistency_error = value - consist_value
            next_state = coords + new_action
            next_state = torch.clamp(next_state, -10, 10)
            # next_state_value = policy.critic.v(next_state)
            next_state_value = policy.value(next_state)
            next_state_value_as_numpy = next_state_value.cpu().detach().numpy().reshape(50, 50)
            # real_next_action = policy.model(torch.randn_like(coords), next_state)
            real_next_action = policy.model(next_state)
            # real_next_action = policy.policy(next_state)
            # real_next_state_value = policy.critic.v(next_state)
            real_next_state_value = policy.value(next_state)
            real_next_state_value_as_numpy = real_next_state_value.cpu().detach().numpy().reshape(50, 50)
            real_msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * real_next_state_value.cpu().detach().numpy().reshape(-1)) - value.reshape(-1)).reshape(50, 50)
            msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * next_state_value.cpu().detach().numpy().reshape(-1)) - value.reshape(-1)).reshape(50, 50)
            real_error_minus_msbe_error = real_msbe_error - msbe_error
            # output numpy images 
            data_as_numpy = value_figure(new_action, value, state_list) 
            consist_data_as_numpy = value_figure(new_action, consist_value, state_list, with_action=False)
            consist_error_as_numpy = value_figure(new_action, consistency_error, state_list, with_action=False, fit_min=True)
            next_state_value_as_numpy = value_figure(new_action, next_state_value_as_numpy, state_list, with_action=False)
            msbe_error_as_numpy = value_figure(new_action, msbe_error, state_list, with_action=False, fit_min=True)
            real_next_state_value_as_numpy = value_figure(real_next_action, real_next_state_value_as_numpy, state_list, with_action=True)
            real_msbe_error_as_numpy = value_figure(real_next_action, real_msbe_error, state_list, with_action=False, fit_min=True)
            real_error_minus_msbe_error_as_numpy = value_figure(real_next_action, real_error_minus_msbe_error, state_list, with_action=False, fit_min=True)
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
        elif algo == 'dac2':
            # new_action = policy.model(torch.randn_like(coords), coords)
            new_action = policy.model(coords)
            random_action = torch.zeros_like(new_action, device=device)
            # Calculate function values for each point in the grid
            value = policy.critic.q1(coords, new_action)
            # value = policy.critic(coords, new_action)[0]
            print("max value:", max(value.cpu().detach().numpy().reshape(-1)))
            value = value.cpu().detach().numpy().reshape(50, 50)
            consist_value = policy.critic.v(coords)
            # consist_value = policy.value(coords)
            consist_value = consist_value.cpu().detach().numpy().reshape(50, 50)
            consistency_error = value - consist_value
            next_state = coords + new_action
            next_state = torch.clamp(next_state, -10, 10)
            next_state_value = policy.critic.v(next_state)
            # next_state_value = policy.value(next_state)
            next_state_value_as_numpy = next_state_value.cpu().detach().numpy().reshape(50, 50)
            # real_next_action = policy.model(torch.randn_like(coords), next_state)
            real_next_action = policy.model(next_state)
            # real_next_action = policy.policy(next_state)
            real_next_state_value = policy.critic.v(next_state)
            # real_next_state_value = policy.value(next_state)
            real_next_state_value_as_numpy = real_next_state_value.cpu().detach().numpy().reshape(50, 50)
            real_msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * real_next_state_value.cpu().detach().numpy().reshape(-1)) - value.reshape(-1)).reshape(50, 50)
            msbe_error = ((compute_gaussian_density(next_state.cpu().detach().numpy()).reshape(-1) + policy.discount * next_state_value.cpu().detach().numpy().reshape(-1)) - value.reshape(-1)).reshape(50, 50)
            real_error_minus_msbe_error = real_msbe_error - msbe_error
            # output numpy images 
            data_as_numpy = value_figure(new_action, value, state_list) 
            consist_data_as_numpy = value_figure(new_action, consist_value, state_list, with_action=False)
            consist_error_as_numpy = value_figure(new_action, consistency_error, state_list, with_action=False, fit_min=True)
            next_state_value_as_numpy = value_figure(new_action, next_state_value_as_numpy, state_list, with_action=False)
            msbe_error_as_numpy = value_figure(new_action, msbe_error, state_list, with_action=False, fit_min=True)
            real_next_state_value_as_numpy = value_figure(real_next_action, real_next_state_value_as_numpy, state_list, with_action=True)
            real_msbe_error_as_numpy = value_figure(real_next_action, real_msbe_error, state_list, with_action=False, fit_min=True)
            real_error_minus_msbe_error_as_numpy = value_figure(real_next_action, real_error_minus_msbe_error, state_list, with_action=False, fit_min=True)
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
        elif algo == 'sac':
            new_action = policy.policy.sample(coords)[2]
            new_action = torch.tensor(new_action, dtype=torch.float32, device=device)
            val = policy.critic(coords, new_action)[0]
            value = val.cpu().detach().numpy().reshape(50, 50)
            state_val = policy.value(coords)
            state_value = state_val.cpu().detach().numpy().reshape(50, 50)
            data_as_numpy = value_figure(new_action, value, state_list)
            state_value_as_numpy = value_figure(new_action, state_value, state_list, with_action=False)
            ims = merge_dicts({"reference": ims, 
                               "action and q value Q(s, a)": data_as_numpy,
                               "state value V(s)": state_value_as_numpy})
        elif algo == 'td3':
            new_action = policy.ac.pi(coords)
            val = policy.ac.q1(coords, new_action)
            value = val.cpu().detach().numpy().reshape(50, 50)
            data_as_numpy = value_figure(new_action, value, state_list)
            ims = merge_images(ims, data_as_numpy, data_as_numpy, data_as_numpy, data_as_numpy, data_as_numpy, data_as_numpy, data_as_numpy, data_as_numpy)
        else:
            raise NotImplementedError
    return ims

