from copy import deepcopy
import d4rl
import os
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.utils_zhiao import load_config
import utils.logger_zhiao as logger_zhiao
from utils.logger import logger
from dataclasses import dataclass, field
from torch import nn

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

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

class td3:
    def __init__(self, env_fn, actor_critic, pi_lr, q_lr, target_noise, noise_clip, policy_delay, args) -> None:
            # Set up optimizers for policy and q-function
            actor_critic = MLPActorCritic
            env = env_fn()
            self.ac = actor_critic(env.observation_space, env.action_space, hidden_sizes=[args.hid]*args.l,)
            ac = self.ac
            self.pi_lr = pi_lr
            self.q_lr = q_lr
            q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
            self.q_params = q_params
            self.pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
            self.q_optimizer = Adam(q_params, lr=q_lr) 
            self.ac_targ = deepcopy(ac) 
            self.target_noise = target_noise    
            self.noise_clip = noise_clip
            self.act_limit = env.action_space.high[0]
            self.gamma = args.discount
            self.act_dim = env.action_space.shape[0]
            self.polyak = args.tau
            self.policy_delay = policy_delay
            for p in self.ac_targ.parameters():
                p.requires_grad = False
        
    # Set up function for computing TD3 Q-losses
    def compute_loss_q(self, data):
        ac = self.ac
        ac_targ = self.ac_targ
        target_noise = self.target_noise
        noise_clip = self.noise_clip
        act_limit = self.act_limit
        gamma = self.gamma

        # o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, o2, r, d = data
        o = o.to(torch.float32)
        a = a.to(torch.float32)
        o2 = o2.to(torch.float32)
        r = r.to(torch.float32)
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, data):
        ac = self.ac
        # o = data['obs']
        o = data[0]
        # o = o.astype(np.float32)
        o = o.to(torch.float32)
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # logger.setup_pytorch_saver(ac)

    def update(self, data, timer):
        policy_delay = self.policy_delay
        ac = self.ac
        ac_targ = self.ac_targ
        polyak = self.polyak
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **loss_info)
        logger_zhiao.logkv_mean('LossQ', loss_q.item())


        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            # logger.store(LossPi=loss_pi.item())
            logger_zhiao.logkv_mean('LossPi', loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(self, o, noise_scale):
        ac = self.ac
        act_dim = self.act_dim
        act_limit = self.act_limit
        
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)  

    def sample_action(self, o):
        return self.get_action(o, 0.1)

def eval_policy(policy, eval_env, eval_episodes=10, need_animation=False):
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
        traj_return = 0
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

class BufferWrapper:
    def __init__(self, buffer, device) -> None:
        self.buffer = buffer 
        self.device = device

    def sample(self, batch_size):
        sample = self.buffer.sample(batch_size)
        obs = torch.tensor(sample[0], dtype=torch.float32).to(self.device)
        act = torch.tensor(sample[1], dtype=torch.float32).to(self.device)
        obs2 = torch.tensor(sample[2], dtype=torch.float32).to(self.device)
        done = torch.tensor(sample[3], dtype=torch.float32).to(self.device)
        rew = torch.tensor(sample[4], dtype=torch.float32).to(self.device)
        return obs, act, obs2, rew, done

@dataclass
class Config:
    # experiment
    exp: str = 'exp_1'
    device: int = 0
    env_name: str = 'halfcheetah-medium-v2'
    # online_env: str = 'HalfCheetah-v2'
    dir: str = 'results'
    seed: int = 0
    format: list = field(default_factory=lambda: ['stdout', "wandb"])
    # format: list = field(default_factory=lambda: ['stdout'])
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
    # bc_weight: float = 1.0
    bc_weight: float = 0.0
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
    seed: int =0
    num_steps_per_epoch: int = 4000
    replay_size: int = int(1e6)
    start_steps: int = 10000
    update_after: int = 1000
    update_every: int = 500
    num_envs: int = 8
    max_ep_len: int = 1000
    hid: int = 256
    l: int = 2
    init: str = "random"

def online_train(args, env_fn):
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
    env = [env_fn for _ in range(num_envs)]
    env = SubprocVecEnv(env, start_method="spawn",)
    test_env = env_fn()
    obs_dim = env.observation_space.shape[-1] # type:ignore
    act_dim = env.action_space.shape[-1] # type:ignore
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = float(env.action_space.high[0]) # type:ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    buffer_size = replay_size
    best_nreward = -np.inf
    obs_space = env.observation_space
    action_space = env.action_space
    
    buffer = ReplayBuffer(buffer_size, obs_space, action_space, n_envs=num_envs, device="cpu")
    if args.algo == 'td3':    
        agent = td3(env_fn, MLPActorCritic, args.lr, args.lr, 0.2, 0.5, 2, args)
    elif args.algo == 'dac':
        from agents.ac_diffusion import Diffusion_AC as Agent
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
    o, ep_ret, ep_len = env.reset(), np.zeros(num_envs, dtype=np.float32), np.zeros(num_envs, dtype=np.int32)
    assert num_steps_per_epoch % update_every == 0, "num_steps_per_epoch must be a multiple of update_every"
    if args.init == "dataset":
        dataset = d4rl.qlearning_dataset(env_fn())
        for i in range(len(dataset['observations']) // 8):   
            buffer.add(dataset['observations'][8 * i: 8 * i + 8], dataset['next_observations'][8 * i: 8 * i + 8],
                        dataset['actions'][8 * i: 8 * i + 8], dataset['rewards'][8 * i: 8 * i + 8],
                        dataset['terminals'][8 * i: 8 * i + 8], [{} for _ in range(8)])
    for t in range(max_timesteps):
        if args.init == "random":
            if t > start_steps:
                a = np.array([agent.sample_action(o[i]) for i in range(num_envs)])
            # a = np.array([agent.sample_action(o[i]) for i in range(num_envs)])
            else:
                a = np.array([env.action_space.sample() for i in range(num_envs)])
        else:
            a = np.array([agent.sample_action(o[i]) for i in range(num_envs)])

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret = ep_ret + r
        ep_len = ep_len + 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len==max_ep_len else d
        # d = np.array([False if ep_len[i] == max_ep_len else d[i] for i in range(num_envs)])
        d = ~(ep_len == max_ep_len) & d

        # Store experience to replay buffer
        buffer.add(o, o2, a, r, d, info)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        # if d or (ep_len == max_ep_len):
        indices = np.where(d | (ep_len == max_ep_len))[0]
        for i in indices:
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            logger_zhiao.logkv_mean('EpRet', ep_ret[i])
            logger_zhiao.logkv_mean('EpLen', ep_len[i])
            o[i], ep_ret[i], ep_len[i] = env.env_method("reset", indices=[i])[0], 0, 0

        # Update handling
        # data_sampler = BufferWrapper(buffer, device)
        data_sampler = buffer
        
        if t >= update_after and t % update_every == 0:
            if args.algo == 'dac':
                loss_metric = agent.train(data_sampler,
                                iterations=update_every,
                                batch_size=args.batch_size,
                                log_writer=writer)
                for k, v in loss_metric.items():
                    if v == []:
                        continue    
                    logger_zhiao.logkv(k, np.mean(v))
                    logger_zhiao.logkv(k + '_std', np.std(v))
                    logger_zhiao.logkv(k + '_max', np.max(v))
                    logger_zhiao.logkv(k + '_min', np.min(v))
            elif args.algo == 'td3':
                batch = data_sampler.sample(args.batch_size)
                agent.update(batch, t)
            else: 
                raise NotImplementedError
            if t % args.num_steps_per_epoch == 0:
                eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, eval_len, eval_len_std = eval_policy(agent, test_env,
                                                                                                            eval_episodes=args.eval_episodes)
                logger_zhiao.logkvs({'eval_reward': eval_res, 'eval_nreward': eval_norm_res,
                                    'eval_reward_std': eval_res_std, 'eval_nreward_std': eval_norm_res_std,
                                    'eval_len': eval_len, 'eval_len_std': eval_len_std, })
                if args.algo == 'dac':
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
    online_train(args, lambda: gym.make(args.env_name))

