import numpy as np
import torch
import utils.logger_zhiao as logger_zhiao
import time
# from visualize import animation
from dataset import build_dataset, DatasetSampler
from evaluation import eval_policy
from pre_main import eval_policy as pre_eval_policy

def pre_eval_policy_wrapper(agent, env_name, seed, eval_episodes=10):
    result = pre_eval_policy(agent, env_name, seed, eval_episodes)
    result_dict = {
        "avg_reward": result[0],
        "std_reward": result[1],
        "avg_norm_score": result[2],
        "std_norm_score": result[3],
        "avg_length": result[4],
        "std_length": result[5],
    }
    return result_dict

def offline_train(args, env_fn):
    # parameters
    seed = args.seed
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
    elif args.algo == "pre-dac":
        from agents.pre_ac import Diffusion_AC as Agent
        # agent = Agent(state_dim=obs_dim,
        #         action_dim=act_dim,
        #         max_action=act_limit,
        #         device=device,
        #         args=args)
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
    else:
        raise NotImplementedError
    for t in range(max_timesteps):
        steps = args.num_steps_per_epoch
        if args.algo == 'dac' or args.algo == 'dql' or args.algo == 'bc' or args.algo == 'pre-dac':
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
            if args.pre_eval:
                eval_ret = pre_eval_policy_wrapper(agent, args.env_name, args.seed, eval_episodes=args.eval_episodes)
            else:
                eval_ret = eval_policy(args, agent, test_env, algo=args.algo, eval_episodes=args.eval_episodes)
            logger_zhiao.logkvs(eval_ret)
            if args.algo == 'dac':
                print("bc_loss", np.mean(loss_metric['bc_loss']))
                print("ql_loss", np.mean(loss_metric['ql_loss']))
                print("actor_loss", np.mean(loss_metric['actor_loss']))
                print("critic_loss", np.mean(loss_metric['critic_loss']))
                if args.save_best_model and eval_ret["avg_norm_score"] > best_nreward:
                    best_nreward = eval_ret["avg_norm_score"] 
                    agent.save_model(output_dir, t)
            logger_zhiao.dumpkvs()