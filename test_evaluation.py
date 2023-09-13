from config import Config
from utils.utils_zhiao import load_config
from evaluation import eval_policy 
import numpy as np

class policy:
    def sample_action(self, state):
        return np.random.rand(state.shape[0], 6)

if __name__ == "__main__":
    args = load_config(Config)
    eval_policy(args, policy()) 