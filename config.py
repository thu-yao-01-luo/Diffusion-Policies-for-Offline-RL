from dataclasses import dataclass, field

@dataclass
class Config:
    # experiment
    exp: str = 'exp_1'
    # device: int = 0
    device: str = 'cuda:0'
    output_dir: str = 'results'
    env_name: str = 'halfcheetah-medium-v2'
    # env_name: str = 'Demo-v0'
    dir: str = 'results'
    seed: int = 0
    # format: list = field(default_factory=lambda: ['stdout', "wandb"])
    format: list = field(default_factory=lambda: ['stdout'])
    # optimization
    batch_size: int = 100
    lr_decay: bool = False
    early_stop: bool = False
    save_best_model: bool = True
    # rl parameters
    discount: float = 0.99
    discount2: float = 1.0
    tau: float = 0.005
    ema_decay: float = 0.995
    # diffusion
    target_noise: float = 0.2
    noise_clip: float = 0.5
    T: int = 100
    beta_schedule: str = 'vp'
    # algo
    algo: str = 'dac'
    ms: str = 'offline'
    # coef: float = 0.2
    coef: float = 1.0
    MSBE_coef: float = 1.0
    eta: float = 1.0
    compute_consistency: bool = True
    iql_style: str = "discount"
    expectile: float = 0.7
    quantile: float = 0.6
    temperature: float = 1.0
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
    num_steps_per_epoch: int = 5000
    replay_size: int = int(1e6)
    start_steps: int = 10000
    update_after: int = 1000
    update_every: int = 50
    num_envs: int = 8
    max_ep_len: int = 1000
    hid: int = 256
    l: int = 2
    init: str = "random"
    policy_delay: int = 2
    act_noise: float = 0.1
    g_mdp: bool = True # only used in the debug phase with T=1
    online: bool = True    
    norm_q: bool = True
    consistency_coef: float = 1.0
    add_noise: bool = False
    update_ema_every: int = 5
    step_start_ema: int = 1000
    need_animation: bool = False
    num_epochs: int = 1000
    eval_freq: int = 50
    eval_episodes: int = 10
    lr: float = 3e-4
    lr_maxt: int = 1000
    eta: float = 1.0
    max_q_backup: bool = False
    reward_tune: str = "no"
    gn: float = 9.0
    top_k: int = 1
    d4rl: bool = True
    vis_q: bool = False
    test_critic: bool = False
    alpha: float = 0.2
    automatic_entropy_tuning: bool = False
    determine: bool = True
    need_entropy_test: bool = False
    num_entropy_samples: int = 100
    with_eval: bool = True
    resample: bool = False
    grad_norm: float = 9.0
    ablation: bool = False
    mdp: bool = False
    critic_ema: int = 2
    pre_eval: bool = False