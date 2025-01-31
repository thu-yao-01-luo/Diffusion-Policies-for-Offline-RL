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
from utils.utils_zhiao import load_config
import utils.logger_zhiao as logger_zhiao
# from dreamfuser.logger import logger as logger_zhiao
# from dreamfuser.configs import load_config
from utils.logger import logger, setup_logger
from dataclasses import dataclass, field
# from torch.utils.tensorboard import SummaryWriter

hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 12000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 24000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 24000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250, 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}


@dataclass
class Config:
    # experiment
    exp: str = 'exp_1'
    device: int = 0
    env_name: str = 'halfcheetah-medium-v2'
    dir: str = 'results'
    seed: int = 0
    num_steps_per_epoch: int = 100
    format: list = field(default_factory=lambda: ['stdout', 'wandb', 'csv'])
    # optimization
    batch_size: int = 256
    lr_decay: bool = False
    early_stop: bool = False
    save_best_model: bool = True
    # rl parameters
    discount: float = 0.99
    discount2: float = 1.0
    tau: float = 0.005
    # diffusion
    T: int = 5
    beta_schedule: str = 'vp'
    # algo
    algo: str = 'dac'
    ms: str = 'offline'
    coef: float = 0.2
    eta: float = 1.0
    compute_consistency: bool = True
    iql_style: str = "discount"
    expectile: float = 0.7
    quantile: float = 0.6
    temperature: float = 1.0
    bc_weight: float = 1.0
    name: str = 'dac'
    id: str = 'dac'
    tune_bc_weight: bool = False
    std_threshold: float = 1e-4
    bc_lower_bound: float = 1e-2
    bc_decay: float = 0.995
    bc_upper_bound: float = 1e2
    value_threshold: float = 2.5e-4
    consistency: bool = True
    scale: float = 1.0
    predict_epsilon: bool = True
    debug: bool = False
    fast: bool = False

def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args, using_server=True):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')

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
                      predict_epsilon=args.predict_epsilon,
                      )
    elif args.algo == 'ddd':
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
        from agents.ac_diffusion import Diffusion_AC as Agent
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

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    best_nreward = -np.inf
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))
        # Logging
        utils.print_banner(
            f"Train step: {training_iters}", separator="*", num_star=90)
        if not using_server:
            logger.record_tabular('Trained Epochs', curr_epoch)
            logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
            logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
            logger.record_tabular(
                'Actor Loss', np.mean(loss_metric['actor_loss']))
            logger.record_tabular(
                'Critic Loss', np.mean(loss_metric['critic_loss']))
            logger.dump_tabular()

        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, eval_len, eval_len_std = eval_policy(agent, args.env_name, args.seed,
                                                                                                       eval_episodes=args.eval_episodes)
        logger_zhiao.logkvs({'eval_reward': eval_res, 'eval_nreward': eval_norm_res,
                            'eval_reward_std': eval_res_std, 'eval_nreward_std': eval_norm_res_std,
                             'eval_len': eval_len, 'eval_len_std': eval_len_std, })

        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            np.mean(loss_metric['bc_loss']), np.mean(
                                loss_metric['ql_loss']),
                            np.mean(loss_metric['actor_loss']), np.mean(
                                loss_metric['critic_loss']),
                            curr_epoch])
        if not using_server:
            np.save(os.path.join(output_dir, "eval"), evaluations)
            logger.record_tabular('Average Episodic Reward', eval_res)
            logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
            logger.dump_tabular()

        bc_loss = np.mean(loss_metric['bc_loss'])
        for k, v in loss_metric.items():
            if v == []:
                continue    
            logger_zhiao.logkv(k, np.mean(v))
            logger_zhiao.logkv(k + '_std', np.std(v))
            logger_zhiao.logkv(k + '_max', np.max(v))
            logger_zhiao.logkv(k + '_min', np.min(v))
        logger_zhiao.dumpkvs()
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model and eval_norm_res > best_nreward:
            best_nreward = eval_norm_res
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    # writer.close()


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, need_animation=False):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

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

    utils.print_banner(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    if need_animation:
        state, done = eval_env.reset(), False
        ims = []
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
            ims.append(eval_env.render(mode='rgb_array'))
        logger_zhiao.animate(ims, f'{args.env_name}_{args.algo}_{args.T}.mp4')
        scores.append(traj_return)

    return avg_reward, std_reward, avg_norm_score, std_norm_score, avg_length, std_length


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # ### Experimental Setups ###
    # parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    # parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    # parser.add_argument("--env_name", default="walker2d-medium-expert-v2", type=str)  # OpenAI gym environment name
    # parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    # parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    # ### Optimization Setups ###
    # parser.add_argument("--batch_size", default=256, type=int)
    # parser.add_argument("--lr_decay", action='store_true')
    # parser.add_argument('--early_stop', action='store_true')
    # parser.add_argument('--save_best_model', action='store_true')

    # ### RL Parameters ###
    # parser.add_argument("--discount", default=0.99, type=float)
    # parser.add_argument("--tau", default=0.005, type=float)

    # ### Diffusion Setting ###
    # parser.add_argument("--T", default=5, type=int)
    # parser.add_argument("--beta_schedule", default='vp', type=str)
    # ### Algo Choice ###
    # parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    # parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    # parser.add_argument("--format", default=["stdout", "csv", "wandb"], type=list, help="format to log")
    # parser.add_argument("--coef", default=0.2, type=float)
    # # parser.add_argument("--top_k", default=1, type=int)

    # # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--eta", default=1.0, type=float)
    # # parser.add_argument("--max_q_backup", action='store_true')
    # # parser.add_argument("--reward_tune", default='no', type=str)
    # # parser.add_argument("--gn", default=-1.0, type=float)

    # args = parser.parse_args()

    args = load_config(Config)
    logger_zhiao.configure(
        "logs",
        format_strs=args.format,
        config=args,
        project="online-dream-ac",
        # name=f"Discount{args.discount2}-T{args.T}-Coef{args.coef}-{args.algo}-{args.env_name}-lrd{args.lr_decay}-cc{args.compute_consistency}-iql{args.iql_style}-{time.time()}",
        # id=f"Discount{args.discount2}-T{args.T}-Coef{args.coef}-{args.algo}-{args.env_name}-lrd{args.lr_decay}-cc{args.compute_consistency}-iql{args.iql_style}-{time.time()}",
        name=args.name,
        id=args.id,
    )  # type: ignore

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = os.path.join(os.environ['MODEL_DIR'], f'{args.dir}')

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta'] if args.eta == 1.0 else args.eta
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

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
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)

    variant.update(version=f"Diffusion-Policies-RL")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    # setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(
        f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
