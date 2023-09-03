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
from evaluation import eval_policy

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
    # max_timesteps = args.num_epochs * args.num_steps_per_epoch
    max_timesteps = args.num_epochs // args.eval_freq
    best_nreward = -np.inf
    dataset = build_dataset(env_name=args.env_name, is_d4rl=args.d4rl) # TODO: build dataset 
    data_sampler = DatasetSampler(dataset, device)
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
    elif args.algo == "bc":
        from agents.bc import Diffusion_BC as Agent
        agent = Agent(state_dim=obs_dim,
              action_dim=act_dim,
              max_action=act_limit,
              device=device,
              args=args)
    else:
        raise NotImplementedError

    starting_time = time.time()
    last_eval_time = starting_time  
    for t in range(max_timesteps):
        steps = args.num_steps_per_epoch * args.eval_freq
        if args.algo == 'dac' or args.algo == 'dql' or args.algo == 'bc':
            loss_metric = agent.train(
                        replay_buffer=data_sampler,
                        iterations=steps,
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
        # if t % args.num_steps_per_epoch == 0 and args.with_eval:
        if args.with_eval:
            infer_begin = time.time()
            eval_ret = eval_policy(args, agent, test_env, algo=args.algo, eval_episodes=args.eval_episodes)
            logger_zhiao.logkvs(eval_ret)
            current_time = time.time()
            time_span = current_time - starting_time
            infer_time = time.time() - infer_begin
            if args.algo == 'dac':
                print("bc_loss", np.mean(loss_metric['bc_loss']))
                print("ql_loss", np.mean(loss_metric['ql_loss']))
                print("actor_loss", np.mean(loss_metric['actor_loss']))
                print("critic_loss", np.mean(loss_metric['critic_loss']))
                if args.save_best_model and eval_ret["avg_norm_score"] > best_nreward:
                    best_nreward = eval_ret["avg_norm_score"] 
                    agent.save_model(output_dir, t)
            logger_zhiao.dumpkvs()

