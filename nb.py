from utils.utils_zhiao import load_config
import os
import d4rl
import utils.logger_zhiao as logger_zhiao
from config import Config
from nb_online import online_train
from nb_offline import offline_train    
import random
import numpy as np
import torch

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
    config = args
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    args.num_cpu = min(os.cpu_count(), args.num_cpu)
    if args.online:
        online_train(args)
    else:
        offline_train(args)