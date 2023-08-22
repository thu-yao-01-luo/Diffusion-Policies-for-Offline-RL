from agents.helpers import (cosine_beta_schedule,
                     linear_beta_schedule,
                        vp_beta_schedule,
                        extract,    
)

import torch

print(cosine_beta_schedule(10))
print(linear_beta_schedule(10))
print(vp_beta_schedule(10))
