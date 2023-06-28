from agents.helpers import vp_beta_schedule
import torch

betas = vp_beta_schedule(2)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
one_minus_alphas_cumprod = 1 - alphas_cumprod
alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
bc_weight_coef = one_minus_alphas_cumprod / alphas_cumprod

print(betas)
print(alphas)
print(alphas_cumprod)
print(alphas_cumprod_prev)
print(bc_weight_coef)
# for t in range(10):
#     print(f"x_{t} = {torch.sqrt(alphas_cumprod[t])}x_0 + {torch.sqrt(one_minus_alphas_cumprod[t])}*epsilon_{t}")