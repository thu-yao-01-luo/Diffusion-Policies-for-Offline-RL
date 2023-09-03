from evaluation import eval_policy
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
              args=args)
    elif args.algo == 'dql':
        from agents.ql import Diffusion_QL as Agent
        agent = Agent(state_dim=obs_dim,
              action_dim=act_dim,
              max_action=act_limit,
              device=device,
              args=args)
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
    # assert num_steps_per_epoch % update_every == 0, "num_steps_per_epoch must be a multiple of update_every"
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
                # eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, eval_len, eval_len_std, local_opt, entropy = \
                eval_ret = eval_policy(args, agent, test_env, algo=args.algo, eval_episodes=args.eval_episodes)
                logger_zhiao.logkvs(eval_ret)
                if args.algo == 'dac':
                    print("bc_loss", np.mean(loss_metric['bc_loss']))
                    print("ql_loss", np.mean(loss_metric['ql_loss']))
                    print("actor_loss", np.mean(loss_metric['actor_loss']))
                    print("critic_loss", np.mean(loss_metric['critic_loss']))
                    if args.save_best_model and eval_ret["avg_norm_score"] > best_nreward:
                        best_nreward = eval_ret["avg_norm_score"] 
            logger_zhiao.dumpkvs()

