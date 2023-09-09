# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gym
import numpy as np
import os
import torch
import json
import time
import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from dataset import DatasetSampler
from utils.utils_zhiao import load_config
import utils.logger_zhiao as logger_zhiao
from utils.logger import logger, setup_logger
from dataclasses import dataclass, field
from config import Config

def train_agent(env, state_dim, action_dim, max_action, device, args, output_dir=None, using_server=True):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    if args.pre_dataset:
        data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    else:
        data_sampler = DatasetSampler(dataset, device)

    if args.algo == 'ql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
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
                      bc_weight=args.bc_weight,
                      tune_bc_weight=args.tune_bc_weight,
                      bc_lower_bound=args.bc_lower_bound,
                      bc_decay=args.bc_decay,
                      bc_upper_bound=args.bc_upper_bound,
                      value_threshold=args.value_threshold,
                      scale=args.scale,
                      predict_epsilon=args.predict_epsilon,
                      debug=args.debug,
                      )
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr,
                      )
    if args.algo == 'ddd':
        from agents.dd_diffusion import Diffusion_DD as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
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
                      bc_weight=args.bc_weight,
                      tune_bc_weight=args.tune_bc_weight,
                      bc_lower_bound=args.bc_lower_bound,
                      bc_decay=args.bc_decay,
                      bc_upper_bound=args.bc_upper_bound,
                      value_threshold=args.value_threshold,
                      scale=args.scale,
                      predict_epsilon=args.predict_epsilon,
                      debug=args.debug,
                      )
        agent.fit_dataset(data_sampler)
    elif args.algo == 'dac':
        # from agents.ac_diffusion import Diffusion_AC as Agent
        from agents.pre_ac import Diffusion_AC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
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
                      MSBE_coef=args.coef,
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
                      )
        # from agents.pre_ac import Diffusion_AC as Agent
        # agent = Agent(state_dim=state_dim,
        #         action_dim=action_dim,
        #         max_action=max_action,
        #         device=device,
        #         args=args)
    elif args.algo == 'td3':         
        from agents.td3_diffusion import Diffusion_TD3 as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
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
                      bc_weight=args.bc_weight,
                      tune_bc_weight=args.tune_bc_weight,
                      bc_lower_bound=args.bc_lower_bound,
                      bc_decay=args.bc_decay,
                      bc_upper_bound=args.bc_upper_bound,
                      value_threshold=args.value_threshold,
                      scale=args.scale,
                      predict_epsilon=args.predict_epsilon,
                      debug=args.debug,
                      fast=args.fast,
                      )
    else:
        raise NotImplementedError
    writer = None  # SummaryWriter(output_dir)

    for i in range(args.num_epochs):
    #    iterations = int(args.eval_freq * args.num_steps_per_epoch)
        iterations = args.num_steps_per_epoch
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)
        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, eval_len, eval_len_std = eval_policy(agent, args.env_name, args.seed,
                                                                                                       eval_episodes=args.eval_episodes, eval_seed=args.eval_seed)
        logger_zhiao.logkvs({'eval_reward': eval_res, 'eval_nreward': eval_norm_res,
                            'eval_reward_std': eval_res_std, 'eval_nreward_std': eval_norm_res_std,
                             'eval_len': eval_len, 'eval_len_std': eval_len_std, })

        for k, v in loss_metric.items():
            if v == []:
                continue    
            logger_zhiao.logkv(k, np.mean(v))
            logger_zhiao.logkv(k + '_std', np.std(v))
        logger_zhiao.dumpkvs()

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, need_animation=False, eval_seed=100):
    eval_env = gym.make(env_name)
    eval_env.seed(eval_seed)

    scores = []
    lengths = []
    for _ in range(eval_episodes):
        traj_return = 0.
        traj_length = 0
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
            traj_length += 1
        scores.append(traj_return)
        lengths.append(traj_length)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)
    return avg_reward, std_reward, avg_norm_score, std_norm_score, avg_length, std_length


if __name__ == "__main__":
    args = load_config(Config)
    logger_zhiao.configure(
        "logs",
        format_strs=args.format,
        config=args,
        project="online-dream-ac",
        name=args.name,
        id=args.id,
    )  
    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                args)