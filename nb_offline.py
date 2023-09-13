import numpy as np
import os
import gym
import torch
import utils.logger_zhiao as logger_zhiao
import time
from dataset import build_dataset
from dataset import DatasetSamplerNP as DatasetSampler
from evaluation import eval_policy
from config import Config

def offline_train(args: Config):
    # parameters
    args.output_dir = os.path.join(os.environ['MODEL_DIR'], f'{args.dir}')
    seed = args.seed
    output_dir = args.output_dir
    torch.manual_seed(seed)
    np.random.seed(seed)

    # environment
    test_env = gym.make(args.env_name)
    # observation and action dimension
    obs_dim = test_env.observation_space.shape[-1] # type:ignore
    act_dim = test_env.action_space.shape[-1] # type:ignore
    print("obs_dim", obs_dim) 
    print("act_dim", act_dim)
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = float(test_env.action_space.high[0]) # type:ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = None  # SummaryWriter(output_dir)
    best_nreward = -np.inf

    dataset = build_dataset(env_name=args.env_name, is_d4rl=args.d4rl) # TODO: build dataset 
    data_sampler = DatasetSampler(dataset, device)
    if args.algo == 'dac':
        from agents.nbac import Diffusion_AC as Agent
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
    elif args.algo == "pac":
        from agents.pac import Diffusion_AC as Agent
        agent = Agent(state_dim=obs_dim,
                action_dim=act_dim,
                max_action=act_limit,
                device=device,
                args=args)
    else:
        raise NotImplementedError
    for t in range(args.num_epochs):
        steps = args.num_steps_per_epoch
        if args.algo == 'dac' or args.algo == 'dql' or args.algo == 'bc' or args.algo == 'pac':
            starting_time = time.time()
            loss_metric = agent.train(
                        replay_buffer=data_sampler,
                        iterations=steps,
                        batch_size=args.batch_size,
                        log_writer=writer)
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
        else:
            raise NotImplementedError
        if args.with_eval:
            eval_ret = eval_policy(args, agent)
            logger_zhiao.logkvs(eval_ret)
            if args.save_best_model and eval_ret["avg_norm_score"] > best_nreward:
                best_nreward = eval_ret["avg_norm_score"] 
                agent.save_model(output_dir, t)
        logger_zhiao.dumpkvs()