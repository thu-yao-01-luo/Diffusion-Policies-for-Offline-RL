from utils.utils_zhiao import load_config
import torch
import os
import utils.logger_zhiao as logger_zhiao
from config import Config
from nb_online import online_train
from nb_offline import offline_train    

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

    args.num_cpu = min(os.cpu_count(), args.num_cpu)
    if args.online:
        online_train(args)
    else:
        offline_train(args)