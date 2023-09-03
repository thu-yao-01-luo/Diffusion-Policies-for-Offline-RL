import os
import gym
import torch
from utils import utils
from utils.utils_zhiao import load_config
import utils.logger_zhiao as logger_zhiao
import torch
import gym
from config import Config
from online_train import online_train
from offline_train import offline_train

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