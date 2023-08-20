from demo_env import CustomEnvironment, compute_gaussian_density 
import numpy as np
import h5py 
from moviepy.editor import VideoFileClip
from matplotlib import animation
import matplotlib.pyplot as plt
import argparse

def generate_action(state, type="random"):
    batch_size = state.shape[0]
    target_pos = np.array([5, 0])
    if type == "random":
        random_actions = np.random.uniform(low=-1, high=1, size=(batch_size, 2))
        return random_actions
    else: 
        optimal = target_pos[None, :] - state # batch_size , 2 
        norm = np.linalg.norm(optimal, axis=1, keepdims=True) # batch_size, 1   
        normalized_opt = optimal / norm # batch_size, 2
        if type == "expert":
            return np.where(norm > 1, normalized_opt, optimal)
        elif type == "medium":
            noise = np.random.normal(loc=0, scale=0.2, size=(batch_size, 2)) # batch_size, 2
            noisy_scalar = np.random.uniform(low=0.5, high=1.5, size=(batch_size, 1)) # batch_size, 1
            noisy_action = noisy_scalar * normalized_opt + noise # batch_size, 2
            return np.clip(noisy_action, -1, 1)

def data_generation(size = 1e6, action_type="random", save_path="demoenv_data.hdf5"):
    env = CustomEnvironment()
    num_trajectory = int(size) // 20    
    states = np.random.uniform(low=-10, high=10, size=(num_trajectory, 2)) # init state, uniform from -10 to 10
    states_list = []
    next_states_list = []
    actions_list = []
    rewards_list = []
    for t in range(20): # run 20 steps
        states_list.append(states)
        actions = generate_action(states, action_type)
        assert type(actions) is np.ndarray, type(actions)
        states = states + actions
        actions_list.append(actions)
        next_states_list.append(states)
        rewards_list.append(compute_gaussian_density(states))
    states_ = np.stack(states_list, axis=0).transpose(1, 0, 2).reshape(-1, 2)   
    actions_ = np.stack(actions_list, axis=0).transpose(1, 0, 2).reshape(-1, 2)
    next_states_ = np.stack(next_states_list, axis=0).transpose(1, 0, 2).reshape(-1, 2)
    rewards_ = np.stack(rewards_list, axis=0).transpose(1, 0).reshape(-1,)
    dones_ = np.zeros_like(rewards_list).transpose(1, 0).reshape(-1,)
    data = {
        "observations": states_,
        "actions": actions_,
        "next_observations": next_states_,
        "rewards": rewards_,
        "terminals": dones_,
    }
    f = h5py.File(save_path, "w")
    for key in data.keys():
        f.create_dataset(key, data=data[key])
    f.close()
    return data

def save_animation(images, output_file):  
   images = [np.array(img).reshape([1, -1, 3]) for img in images]
   video = VideoFileClip(images)
   video.write_videofile(output_file, codec="libx264", audio_codec="aac")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1e6)
    parser.add_argument("--action_type", type=str, default="random")
    parser.add_argument("--save_path", type=str, default="data/demoenv_data.hdf5")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """
    test the sanity of the generated dataset
    """
    args = parse_args()
    data = data_generation(size=args.size, action_type=args.action_type, save_path=args.save_path)
    print("observation shapes:", data["observations"].shape)
    print("action shapes:", data["actions"].shape)
    print("next observation shapes:", data["next_observations"].shape)
    print("reward shapes:", data["rewards"].shape)
    print("terminal shapes:", data["terminals"].shape)
    env = CustomEnvironment()
    obs = env.reset()
    ims = []
    fig = plt.figure()
    sub_obs = data["observations"][:20]
    for i in range(20):
        env.set_state(sub_obs[i])
        im_np = env.render(mode='rgb_array')
        ims.append([plt.imshow(im_np, animated=True)])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
    ani.save('data/demoenv.mp4')
    # save_animation(images, "data/demoenv.mp4")

