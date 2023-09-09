import gym 
import os
import torch
import numpy as np
from utils.utils_zhiao import load_config
import utils.logger_zhiao as logger_zhiao
from config import Config
from offline_train_pre import train_agent

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
    
    train_agent(args)