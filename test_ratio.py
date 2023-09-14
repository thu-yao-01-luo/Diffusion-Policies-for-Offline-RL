from agents.diffusion import Diffusion
from agents.model import MLP
from config import Config
from utils.utils_zhiao import load_config

if __name__ == "__main__":
    args = load_config(Config)
    state_dim = 3
    action_dim = 2
    device = "cpu"
    mlp = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
    model = Diffusion(3, 2, mlp, 1,
                 beta_schedule='linear', n_timesteps=8,
                 loss_type='l2', clip_denoised=True, predict_epsilon=False, scale=1.0)
    
    print(model.posterior_mean_coef1) # x_start  
    print(model.posterior_mean_coef2) # x_t
    #         extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
    #         extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        