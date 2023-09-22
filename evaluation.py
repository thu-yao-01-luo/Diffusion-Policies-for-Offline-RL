import numpy as np
import gym
from config import Config
from helpers import compute_entropy
import utils.logger_zhiao as logger_zhiao
import utils.utils as utils   
from vis import animation
import time
from stable_baselines3.common.vec_env import SubprocVecEnv

def eval_policy(args:Config, policy):
    # initialize
    d4rl: bool = args.d4rl
    need_animation: bool = args.need_animation
    vis_q: bool = args.vis_q 
    env_name: str = args.env_name
    need_entropy_test: bool = args.need_entropy_test
    algo = args.algo
    eval_episodes = args.eval_episodes
    if not args.vec_env_eval:
        eval_env = gym.make(env_name)
        scores = []
        lengths = []
        actions_abs = []
        actions = []
        inference_time = []
        ret = {}
        action_pool = []
        # test scores and compute normalized scores
        for _ in range(eval_episodes):
            traj_return = 0.
            traj_length = 0
            state, done = eval_env.reset(), False
            sampled_action = None
            while not done:
                if args.algo == "tmac":
                    while action_pool == [] or sampled_action is None:
                        starting_time = time.time()
                        sampled_action = policy.sample_action(np.array(state))
                        inference_time.append(time.time() - starting_time)
                        if type(sampled_action) is np.ndarray:
                            action_pool.append(sampled_action)
                        elif type(sampled_action) is list:
                            action_pool.extend(sampled_action)
                        elif sampled_action is None:
                            continue
                        else:
                            raise NotImplementedError
                    action = action_pool.pop(0)
                else:
                    starting_time = time.time()
                    action = policy.sample_action(np.array(state))
                    inference_time.append(time.time() - starting_time)
                actions_abs.append(np.mean(np.abs(action)))
                state, reward, done, _ = eval_env.step(action)
                if args.algo == "tmac" and done:
                    # policy.sample_count = 0
                    policy.sample_states = []
                traj_return += reward
                traj_length += 1
            scores.append(traj_return)
            lengths.append(traj_length)
        print("scores", scores)
        print("lengths", lengths)
        avg_reward = np.mean(scores)
        std_reward = np.std(scores)
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        avg_inference_time = np.mean(inference_time)
        std_inference_time = np.std(inference_time)
        ret["avg_reward"] = avg_reward
        ret["std_reward"] = std_reward
        ret["avg_length"] = avg_length
        ret["std_length"] = std_length
        ret["avg_inference_time"] = avg_inference_time
        ret["std_inference_time"] = std_inference_time
        if len(actions) > 0:
            avg_action = np.mean(actions_abs)
            std_action = np.std(actions_abs)
            ret["avg_action"] = avg_action
            ret["std_action"] = std_action
        if d4rl:
            normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
            avg_norm_score = eval_env.get_normalized_score(avg_reward)
            std_norm_score = np.std(normalized_scores)
            ret["avg_norm_score"] = avg_norm_score
            ret["std_norm_score"] = std_norm_score
            utils.print_banner("Normalized score: {:.2f} +/- {:.2f}".format(
                avg_norm_score, std_norm_score))
        # check the local optimality
        if env_name == 'Demo-v0':
            raise NotImplementedError
            local_opt = True
            state, done = eval_env.reset(), False
            eval_env.set_state(np.array([-5, 0]))
            while not done:
                action = policy.sample_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
            if np.linalg.norm(eval_env.state - np.array([5, 0])) < 2:
                local_opt = False
            ret["local_opt"] = local_opt
        utils.print_banner(
            f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f}")
        # animation 
        if need_animation:
            raise NotImplementedError
            ims = animation(eval_env, vis_q, policy, algo)
            logger_zhiao.animate(ims, f'{args.algo}_{args.T}_{args.env_name}_bcw{args.bc_weight}.mp4')
        if need_entropy_test:
            raise NotImplementedError
            # the state is the last state of test(for mujoco, it is suitable)
            if args.env_name == 'Demo-v0':
                state = eval_env.reset()
            elif args.d4rl == True:
                pass
            else:
                raise NotImplementedError
            action_list = []
            for i in range(args.num_entropy_samples):
                # TODO: state is not the last state of test(for mujoco, it is not suitable)
                action = policy.sample_action(np.array(state), noise_scale=args.act_noise) # type:ignore
                action_list.append(action)
            entropy = compute_entropy(action_list)
            ret["entropy"] = entropy
        return ret
    else: 
        env_fn = lambda: gym.make(env_name)
        cpus = args.num_cpu if args.num_cpu is not None else 2
        assert cpus > 1
        assert args.trajectory == False
        eval_env = env_fn()
        vec_eval_env = SubprocVecEnv([env_fn for i in range(cpus)])
        # max_len = env_fn().max_episode_steps
        max_len = 1000
        scores = []
        lengths = []
        actions_abs = []
        inference_time = []
        ret = {}
        traj_return = np.zeros(cpus) 
        traj_length = np.zeros(cpus) 
        state, done = vec_eval_env.reset(), np.array([False] * cpus) # state: (num_cpu, obs_dim), done: (num_cpu, )
        for _ in range((eval_episodes + cpus - 1) // cpus * max_len):
            starting_time = time.time()
            action = policy.sample_action(np.array(state)) # action: (num_cpu, act_dim)
            inference_time.append(time.time() - starting_time)
            actions_abs.append(np.mean(np.abs(action)))
            state, reward, done, _ = vec_eval_env.step(action)
            traj_return += reward
            traj_length += np.ones(cpus, dtype=np.int32)
            if any(done):
                for i in range(cpus):
                    if done[i]:
                        scores.append(traj_return[i])
                        lengths.append(traj_length[i])
                        traj_return[i] = 0.
                        traj_length[i] = 0
        print("scores", scores)
        print("lengths", lengths)
        avg_reward = np.mean(scores)
        std_reward = np.std(scores)
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        avg_inference_time = np.mean(inference_time)
        std_inference_time = np.std(inference_time)
        ret["avg_reward"] = avg_reward
        ret["std_reward"] = std_reward
        ret["avg_length"] = avg_length
        ret["std_length"] = std_length
        ret["avg_inference_time"] = avg_inference_time
        ret["std_inference_time"] = std_inference_time
        if len(actions_abs) > 0:
            avg_action = np.mean(actions_abs)
            std_action = np.std(actions_abs)
            ret["avg_action"] = avg_action
            ret["std_action"] = std_action
        if d4rl:
            normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
            avg_norm_score = eval_env.get_normalized_score(avg_reward)
            std_norm_score = np.std(normalized_scores)
            ret["avg_norm_score"] = avg_norm_score
            ret["std_norm_score"] = std_norm_score
            utils.print_banner("Normalized score: {:.2f} +/- {:.2f}".format(
                avg_norm_score, std_norm_score))
        # check the local optimality
        if env_name == 'Demo-v0':
            local_opt = True
            state, done = eval_env.reset(), False
            eval_env.set_state(np.array([-5, 0]))
            while not done:
                action = policy.sample_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
            if np.linalg.norm(eval_env.state - np.array([5, 0])) < 2:
                local_opt = False
            ret["local_opt"] = local_opt
        utils.print_banner(
            f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f}")
        # animation 
        if need_animation:
            ims = animation(eval_env, vis_q, policy, algo)
            logger_zhiao.animate(ims, f'{args.algo}_{args.T}_{args.env_name}_bcw{args.bc_weight}.mp4')
        if need_entropy_test:
            # the state is the last state of test(for mujoco, it is suitable)
            if args.env_name == 'Demo-v0':
                state = eval_env.reset()
            elif args.d4rl == True:
                pass
            else:
                raise NotImplementedError
            action_list = []
            for i in range(args.num_entropy_samples):
                # TODO: state is not the last state of test(for mujoco, it is not suitable)
                action = policy.sample_action(np.array(state), noise_scale=args.act_noise) # type:ignore
                action_list.append(action)
            entropy = compute_entropy(action_list)
            ret["entropy"] = entropy
        return ret