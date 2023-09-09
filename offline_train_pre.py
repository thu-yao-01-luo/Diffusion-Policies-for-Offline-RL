from evaluation import eval_policy as pre_eval_policy
import gym
import d4rl
from utils.data_sampler import Data_Sampler
import utils.logger_zhiao as logger_zhiao
import time
import torch
import numpy as np

def eval_policy(args, agent, eval_episodes=10):
    eval_env = gym.make(args.env_name)
    return pre_eval_policy(args, agent, eval_env, args.algo, eval_episodes=eval_episodes)

def train_agent(args):
    # Load buffer
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)

    if args.algo == 'dac':
        # from agents.pre_ac import Diffusion_AC as Agent
        # agent = Agent(state_dim=state_dim,
        #               action_dim=action_dim,
        #               max_action=max_action,
        #               device=device,
        #               discount=args.discount,
        #               tau=args.tau,
        #               max_q_backup=args.max_q_backup,
        #               beta_schedule=args.beta_schedule,
        #               n_timesteps=args.T,
        #               eta=args.eta,
        #               lr=args.lr,
        #               lr_decay=args.lr_decay,
        #               lr_maxt=args.num_epochs,
        #               grad_norm=args.gn,
        #               MSBE_coef=args.coef,
        #               discount2=args.discount2,
        #               compute_consistency=args.compute_consistency,
        #               iql_style=args.iql_style,
        #               expectile=args.expectile,
        #               quantile=args.quantile,
        #               temperature=args.temperature,
        #               bc_weight=args.bc_weight,
        #               tune_bc_weight=args.tune_bc_weight,
        #               std_threshold=args.std_threshold,
        #               bc_lower_bound=args.bc_lower_bound,
        #               bc_decay=args.bc_decay,
        #               bc_upper_bound=args.bc_upper_bound,
        #               value_threshold=args.value_threshold,
        #               consistency=args.consistency,
        #               scale=args.scale,
        #               predict_epsilon=args.predict_epsilon,
        #               debug=args.debug,
        #               )
        from agents.ac import Diffusion_AC as Agent
        agent = Agent(state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                device=device,
                args=args)
    else:
        raise NotImplementedError

    for i in range(args.num_epochs):
        iterations = args.num_steps_per_epoch
        starting_time = time.time()
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=None)
        ending_time = time.time()
        logger_zhiao.logkv('train_time', ending_time - starting_time)
        for k, v in loss_metric.items():
            if v == []:
                continue    
            try:
                logger_zhiao.logkv(k, np.mean(v))
                logger_zhiao.logkv(k + '_std', np.std(v))
            except:
                print("can not log", k, v)
                raise NotImplementedError
        if args.with_eval:
            eval_ret = eval_policy(args, agent)
            logger_zhiao.logkvs(eval_ret)
        logger_zhiao.dumpkvs()