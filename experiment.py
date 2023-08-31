import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from copy import deepcopy
import d4rl
import os
import numpy as np
import torch
from torch.optim import Adam
import gym
import numpy as np
import torch
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.utils_zhiao import load_config
import utils.logger_zhiao as logger_zhiao
from utils.logger import logger
from dataclasses import dataclass, field
from torch import nn
from copy import deepcopy
import itertools
import time
import numpy as np
import torch
from torch.optim import Adam
import gym
from demo_env import CustomEnvironment, compute_gaussian_density
# from visualize import animation
from vis import animation
from helpers import BufferNotDone, ReplayBuffer, Buffer, SACBufferNotDone, compute_entropy, sac_args_type     
from config import Config
from dataset import build_dataset, DatasetSampler

def eval_policy(args, policy, eval_env, algo, eval_episodes=10,):
    d4rl: bool = args.d4rl
    need_animation: bool = args.need_animation
    vis_q: bool = args.vis_q 
    env_name: str = args.env_name
    need_entropy_test: bool = args.need_entropy_test
    scores = []
    lengths = []
    actions_abs = []
    actions = []
    for _ in range(eval_episodes):
        traj_return = 0.
        traj_length = 0
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            actions_abs.append(np.mean(np.abs(action)))
            actions.append(action)
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
            traj_length += 1
        scores.append(traj_return)
        lengths.append(traj_length)
    # check the local optimality
    local_opt = True
    if env_name == 'Demo-v0':
        state, done = eval_env.reset(), False
        eval_env.set_state(np.array([-5, 0]))
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
        if np.linalg.norm(eval_env.state - np.array([5, 0])) < 2:
            local_opt = False
    print("scores", scores)
    # print("lengths", lengths)
    avg_reward = np.mean(scores)
    std_reward = np.std(scores)
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    if len(actions) > 0:
        # print("actions", actions)
        avg_action = np.mean(actions_abs)
        std_action = np.std(actions_abs)
        logger_zhiao.logkv('AvgAction', avg_action)
        logger_zhiao.logkv('StdAction', std_action)
    if d4rl:
        normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
        avg_norm_score = eval_env.get_normalized_score(avg_reward)
        std_norm_score = np.std(normalized_scores)
    else:
        normalized_scores = 0
        avg_norm_score = 0
        std_norm_score = 0

    utils.print_banner(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    if need_animation:
        ims = animation(eval_env, vis_q, policy, algo)
        logger_zhiao.animate(ims, f'{args.algo}_{args.T}_{args.env_name}_bcw{args.bc_weight}.mp4')
    entropy = 0
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
    return avg_reward, std_reward, avg_norm_score, std_norm_score, avg_length, std_length, local_opt, entropy

def online_train(args, env_fn):
    # parameters
    num_envs = args.num_envs
    seed = args.seed
    num_steps_per_epoch = args.num_steps_per_epoch
    replay_size = args.replay_size
    start_steps = args.start_steps
    update_after = args.update_after
    update_every = args.update_every
    max_ep_len = args.max_ep_len
    output_dir = args.output_dir
    torch.manual_seed(seed)
    np.random.seed(seed)

    # environment
    env = env_fn()
    test_env = env_fn()
    # observation and action dimension
    obs_dim = env.observation_space.shape[-1] # type:ignore
    act_dim = env.action_space.shape[-1] # type:ignore
    print("obs_dim", obs_dim) 
    print("act_dim", act_dim)
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = float(env.action_space.high[0]) # type:ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = None  # SummaryWriter(output_dir)
    # buffer and evaluation
    evaluations = []
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    buffer_size = replay_size
    best_nreward = -np.inf
    obs_space = env.observation_space
    action_space = env.action_space
    buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    if args.algo == 'dac':
        from agents.ac import Diffusion_AC as Agent
        agent = Agent(state_dim=obs_dim,
            action_dim=act_dim,
            max_action=act_limit,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
            MSBE_coef=args.MSBE_coef,
            discount2=args.discount2,
            compute_consistency=args.compute_consistency,
            iql_style=args.iql_style,
            expectile=args.expectile,
            quantile=args.quantile,
            temperature=args.temperature,
            bc_weight=args.bc_weight,
            tune_bc_weight=args.tune_bc_weight,
            std_threshold=args.std_threshold,
            bc_lower_bound=args.bc_lower_bound,
            bc_decay=args.bc_decay,
            bc_upper_bound=args.bc_upper_bound,
            value_threshold=args.value_threshold,
            consistency=args.consistency,
            scale=args.scale,
            predict_epsilon=args.predict_epsilon,
            debug=args.debug,
            g_mdp=args.g_mdp,
            policy_freq=args.policy_delay,
            norm_q=args.norm_q,
            consistency_coef=args.consistency_coef,
            target_noise=args.target_noise, 
            noise_clip=args.noise_clip,
            add_noise=args.add_noise,
            update_ema_every=args.update_ema_every,
            test_critic=args.test_critic,
            resample=args.resample,
            )
    elif args.algo == 'dql':
        from agents.ql import Diffusion_QL as Agent
        agent = Agent(state_dim=obs_dim,
            action_dim=act_dim,
            max_action=act_limit,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
            MSBE_coef=args.MSBE_coef,
            discount2=args.discount2,
            compute_consistency=args.compute_consistency,
            iql_style=args.iql_style,
            expectile=args.expectile,
            quantile=args.quantile,
            temperature=args.temperature,
            bc_weight=args.bc_weight,
            tune_bc_weight=args.tune_bc_weight,
            std_threshold=args.std_threshold,
            bc_lower_bound=args.bc_lower_bound,
            bc_decay=args.bc_decay,
            bc_upper_bound=args.bc_upper_bound,
            value_threshold=args.value_threshold,
            consistency=args.consistency,
            scale=args.scale,
            predict_epsilon=args.predict_epsilon,
            debug=args.debug,
            g_mdp=args.g_mdp,
            policy_freq=args.policy_delay,
            norm_q=args.norm_q,
            consistency_coef=args.consistency_coef,
            target_noise=args.target_noise, 
            noise_clip=args.noise_clip,
            add_noise=args.add_noise,
            update_ema_every=args.update_ema_every,
            test_critic=args.test_critic,
            resample=args.resample,
            )
    elif args.algo == "sac": 
        from sac import SAC
        sac_args = sac_args_type(args)
        agent = SAC(
            num_inputs=obs_dim,
            action_space=action_space,
            args=sac_args,
        )
    else:
        raise NotImplementedError

    o, ep_ret, ep_len = env.reset(), 0, 0
    assert num_steps_per_epoch % update_every == 0, "num_steps_per_epoch must be a multiple of update_every"
    starting_time = time.time()
    for t in range(max_timesteps):
        if t >= start_steps:
            a = np.array(agent.sample_action(o, noise_scale=args.act_noise))
        else:   
            a = np.array(env.action_space.sample())

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret = ep_ret + r
        ep_len = ep_len + 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)

        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        # assert o is np.array and o2 is np.array, "o and o2 must be np.array"
        o = np.array(o)
        o2 = np.array(o2)
        buffer.store(o, a, r, o2, d)
        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger_zhiao.logkv_mean('EpRet', ep_ret)
            logger_zhiao.logkv_mean('EpLen', ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # replay_buffer, iterations, batch_size=100, log_writer=None):
        if t >= update_after and t % update_every == 0:
            if args.algo == 'dac' or args.algo == 'dql':
                data_sampler = BufferNotDone(buffer, device)
                # torch.autograd.set_detect_anomaly(True)
                loss_metric = agent.train(
                                replay_buffer=data_sampler,
                                iterations=update_every,
                                batch_size=args.batch_size,
                                log_writer=writer) # type:ignore
                for k, v in loss_metric.items():
                    if v == []:
                        continue    
                    try:
                        logger_zhiao.logkv(k, np.mean(v))
                        logger_zhiao.logkv(k + '_std', np.std(v))
                    except:
                        print("problem", k, v)
                        raise NotImplementedError
            elif args.algo == 'sac':
                data_sampler = SACBufferNotDone(buffer, device)
                loss_metric = agent.train(
                    replay_buffer=data_sampler,
                    iterations=update_every,
                    batch_size=args.batch_size,
                    log_writer=writer,
                    t=t,) # type:ignore
                for k, v in loss_metric.items():
                    if v == []:
                        continue    
                    logger_zhiao.logkv(k, np.mean(v))
                    logger_zhiao.logkv(k + '_std', np.std(v))
            else: 
                raise NotImplementedError
            if t % args.num_steps_per_epoch == 0:
                eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, eval_len, eval_len_std, local_opt, entropy = \
                eval_policy(args, agent, test_env, algo=args.algo, eval_episodes=args.eval_episodes)
                current_time = time.time()
                time_span = current_time - starting_time
                logger_zhiao.logkvs({'eval_reward': eval_res, 'eval_nreward': eval_norm_res,
                                    'eval_reward_std': eval_res_std, 'eval_nreward_std': eval_norm_res_std,
                                    'eval_len': eval_len, 'eval_len_std': eval_len_std, "time": time_span, 
                                    "local_opt": local_opt, "entropy": entropy})
                if args.algo == 'dac':
                    print("bc_loss", np.mean(loss_metric['bc_loss']))
                    print("ql_loss", np.mean(loss_metric['ql_loss']))
                    print("actor_loss", np.mean(loss_metric['actor_loss']))
                    print("critic_loss", np.mean(loss_metric['critic_loss']))
                    evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                                        np.mean(loss_metric['bc_loss']), np.mean(
                                            loss_metric['ql_loss']),
                                        np.mean(loss_metric['actor_loss']), np.mean(
                                            loss_metric['critic_loss']),
                                        t // update_every])

                    if args.save_best_model and eval_norm_res > best_nreward:
                        best_nreward = eval_norm_res
                        agent.save_model(output_dir, t // update_every)
                logger_zhiao.dumpkvs()

def offline_train(args, env_fn):
    # parameters
    seed = args.seed
    update_every = args.update_every
    output_dir = args.output_dir
    torch.manual_seed(seed)
    np.random.seed(seed)

    # environment
    test_env = env_fn()
    # observation and action dimension
    obs_dim = test_env.observation_space.shape[-1] # type:ignore
    act_dim = test_env.action_space.shape[-1] # type:ignore
    print("obs_dim", obs_dim) 
    print("act_dim", act_dim)
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = float(test_env.action_space.high[0]) # type:ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = None  # SummaryWriter(output_dir)
    evaluations = []
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    best_nreward = -np.inf
    dataset = build_dataset(env_name=args.env_name, is_d4rl=args.d4rl) # TODO: build dataset 
    data_sampler = DatasetSampler(dataset, device)
    if args.algo == 'dac':
        from agents.ac import Diffusion_AC as Agent
        agent = Agent(state_dim=obs_dim,
            action_dim=act_dim,
            max_action=act_limit,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
            MSBE_coef=args.MSBE_coef,
            discount2=args.discount2,
            compute_consistency=args.compute_consistency,
            iql_style=args.iql_style,
            expectile=args.expectile,
            quantile=args.quantile,
            temperature=args.temperature,
            bc_weight=args.bc_weight,
            tune_bc_weight=args.tune_bc_weight,
            std_threshold=args.std_threshold,
            bc_lower_bound=args.bc_lower_bound,
            bc_decay=args.bc_decay,
            bc_upper_bound=args.bc_upper_bound,
            value_threshold=args.value_threshold,
            consistency=args.consistency,
            scale=args.scale,
            predict_epsilon=args.predict_epsilon,
            debug=args.debug,
            g_mdp=args.g_mdp,
            policy_freq=args.policy_delay,
            norm_q=args.norm_q,
            consistency_coef=args.consistency_coef,
            target_noise=args.target_noise, 
            noise_clip=args.noise_clip,
            add_noise=args.add_noise,
            update_ema_every=args.update_ema_every,
            test_critic=args.test_critic,
            resample=args.resample,
            )
    elif args.algo == 'dql':
        from agents.ql import Diffusion_QL as Agent
        agent = Agent(state_dim=obs_dim,
            action_dim=act_dim,
            max_action=act_limit,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
            MSBE_coef=args.MSBE_coef,
            discount2=args.discount2,
            compute_consistency=args.compute_consistency,
            iql_style=args.iql_style,
            expectile=args.expectile,
            quantile=args.quantile,
            temperature=args.temperature,
            bc_weight=args.bc_weight,
            tune_bc_weight=args.tune_bc_weight,
            std_threshold=args.std_threshold,
            bc_lower_bound=args.bc_lower_bound,
            bc_decay=args.bc_decay,
            bc_upper_bound=args.bc_upper_bound,
            value_threshold=args.value_threshold,
            consistency=args.consistency,
            scale=args.scale,
            predict_epsilon=args.predict_epsilon,
            debug=args.debug,
            g_mdp=args.g_mdp,
            policy_freq=args.policy_delay,
            norm_q=args.norm_q,
            consistency_coef=args.consistency_coef,
            target_noise=args.target_noise, 
            noise_clip=args.noise_clip,
            add_noise=args.add_noise,
            update_ema_every=args.update_ema_every,
            test_critic=args.test_critic,
            resample=args.resample,
            )
    elif args.algo == "bc":
        from agents.bc import Diffusion_BC as Agent
        agent = Agent(state_dim=obs_dim,
            action_dim=act_dim,
            max_action=act_limit,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
            MSBE_coef=args.MSBE_coef,
            discount2=args.discount2,
            compute_consistency=args.compute_consistency,
            iql_style=args.iql_style,
            expectile=args.expectile,
            quantile=args.quantile,
            temperature=args.temperature,
            bc_weight=args.bc_weight,
            tune_bc_weight=args.tune_bc_weight,
            std_threshold=args.std_threshold,
            bc_lower_bound=args.bc_lower_bound,
            bc_decay=args.bc_decay,
            bc_upper_bound=args.bc_upper_bound,
            value_threshold=args.value_threshold,
            consistency=args.consistency,
            scale=args.scale,
            predict_epsilon=args.predict_epsilon,
            debug=args.debug,
            g_mdp=args.g_mdp,
            policy_freq=args.policy_delay,
            norm_q=args.norm_q,
            consistency_coef=args.consistency_coef,
            target_noise=args.target_noise, 
            noise_clip=args.noise_clip,
            add_noise=args.add_noise,
            update_ema_every=args.update_ema_every,
            test_critic=args.test_critic,
            )
    else:
        raise NotImplementedError

    starting_time = time.time()
    last_eval_time = starting_time  
    for t in range(max_timesteps):
        if args.algo == 'dac' or args.algo == 'dql' or args.algo == 'bc':
            loss_metric = agent.train(
                        replay_buffer=data_sampler,
                        iterations=update_every,
                        batch_size=args.batch_size,
                        log_writer=writer)
            for k, v in loss_metric.items():
                if v == []:
                    continue    
                try:
                    logger_zhiao.logkv(k, np.mean(v))
                    logger_zhiao.logkv(k + '_std', np.std(v))
                except:
                    print("problem", k, v)
                    raise NotImplementedError
        else:
            raise NotImplementedError
        train_time = time.time() - last_eval_time 
        if t % args.num_steps_per_epoch == 0 and args.with_eval:
            infer_begin = time.time()
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, eval_len, eval_len_std, local_opt, entropy = eval_policy(args, agent, test_env, algo=args.algo, 
            eval_episodes=args.eval_episodes)
            current_time = time.time()
            time_span = current_time - starting_time
            infer_time = time.time() - infer_begin
            logger_zhiao.logkvs({'eval_reward': eval_res, 'eval_nreward': eval_norm_res,
                                'eval_reward_std': eval_res_std, 'eval_nreward_std': eval_norm_res_std,
                                'eval_len': eval_len, 'eval_len_std': eval_len_std, "time": time_span, 
                                "local_opt": local_opt, "entropy": entropy, "train_time": train_time, "infer_time": infer_time})
            if args.algo == 'dac':
                print("bc_loss", np.mean(loss_metric['bc_loss']))
                print("ql_loss", np.mean(loss_metric['ql_loss']))
                print("actor_loss", np.mean(loss_metric['actor_loss']))
                print("critic_loss", np.mean(loss_metric['critic_loss']))
                evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                                    np.mean(loss_metric['bc_loss']), np.mean(
                                        loss_metric['ql_loss']),
                                    np.mean(loss_metric['actor_loss']), np.mean(
                                        loss_metric['critic_loss']),
                                    t // update_every])

                if args.save_best_model and eval_norm_res > best_nreward:
                    best_nreward = eval_norm_res
                    agent.save_model(output_dir, t // update_every)
            logger_zhiao.dumpkvs()
            # last_eval_time = time.time()

if __name__ == '__main__':
    args = load_config(Config)
    logger_zhiao.configure(
        "logs",
        format_strs=args.format,
        config=args,
        project="online-dream-ac",
        name=args.name,
        id=args.id,
    )  # type: ignore

    # args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir = os.path.join(os.environ['MODEL_DIR'], f'{args.dir}')
    args.eval_episodes = 10 if 'v2' in args.env_name else 5

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay:
        file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline':
        file_name += f'|k-{args.top_k}'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    if args.online:
        online_train(args, lambda: gym.make(args.env_name))
    else:
        offline_train(args, lambda: gym.make(args.env_name))