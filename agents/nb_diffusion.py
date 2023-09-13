import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from agents.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
from utils.utils import Progress, Silent

class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action, sampler_type="ddim", eta=0.0, 
                 beta_schedule='linear', n_timesteps=100, n_inf_steps=10, device="cpu", use_clipped_model_output=False,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True, scale=1.0):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.scale = scale
        self.eta = eta

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise NotImplementedError

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

        self.scheduler_type = sampler_type
        self.n_inf_steps = n_inf_steps
        self.use_clipped_model_output = use_clipped_model_output
        predict_type = "epsilon" if predict_epsilon else "sample" 
        if sampler_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=n_timesteps, 
                # beta_schedule=beta_schedule, 
                trained_betas=betas.cpu().numpy(),
                clip_sample=clip_denoised, 
                clip_sample_range=max_action, 
                prediction_type=predict_type
            )
        elif sampler_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=n_timesteps,    
                # beta_schedule=beta_schedule,
                trained_betas=betas.cpu().numpy(),
                clip_sample=clip_denoised,
                clip_sample_range=max_action,
                prediction_type=predict_type
            )
        elif sampler_type == "dpm_multistep":
            self.scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=n_timesteps,
                # beta_schedule=beta_schedule,
                trained_betas=betas.cpu().numpy(),
            )
        elif sampler_type == "origin":
            self.scheduler = None
        else:
            raise NotImplementedError
        if self.scheduler is not None:
            self.scheduler.set_timesteps(
                num_inference_steps=self.n_inf_steps,
                device=device,
            )

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x) * self.scale
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) * self.scale

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def diff_sample(self, state, shape):
        input = torch.randn(shape, device=state.device, dtype=torch.float32)
        for i in self.scheduler.timesteps:
            with torch.no_grad():
                batch_size = shape[0]
                timesteps = torch.full(
                    (batch_size,), i.item(), device=state.device, dtype=torch.long)
            model_output = self.model(input, timesteps, state)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            if self.diff_type == "ddim":
                input = self.scheduler.step(
                    model_output, int(i.item()), input, eta=self.eta, use_clipped_model_output=self.use_clipped_model_output
                ).prev_sample
            else:
                input = self.scheduler.step(
                    model_output, int(i.item()), input).prev_sample   
        return input

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        if self.scheduler_type == "origin":
            action = self.p_sample_loop(state, shape, *args, **kwargs)
        else:
            action = self.diff_sample(state, shape)
        return action.clamp_(-self.max_action, self.max_action)  # important

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start) * self.scale

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start) * self.scale

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,),
                          device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

    def loss_to_verify(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,),
                          device=x.device).long()
        noise = torch.randn_like(x) * self.scale

        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            # loss = self.loss_fn(x_recon, noise, weights)
            loss_vec = ((x_recon - noise) ** 2).mean(dim=-1)
            loss = loss_vec * (1.0 - extract(self.alphas_cumprod, t, loss_vec.shape)) / extract(
                self.alphas_cumprod, t, loss_vec.shape)
            loss = loss.mean()
        else:
            # loss = self.loss_fn(x_recon, x, weights)
            loss_vec = ((x_recon - x) ** 2).mean(dim=-1)
            loss = loss_vec * (1.0 - extract(self.alphas_cumprod, t, loss_vec.shape)) / extract(
                self.alphas_cumprod, t, loss_vec.shape)
            loss = loss.mean()

        return loss




        #     self.scheduler_type = sampler_type
        #     self.n_timesteps = n_timesteps
        #     predict_type = "epsilon" if predict_epsilon else "sample" 
        #     if sampler_type == "ddpm":
        #         self.scheduler = DDPMScheduler(
        #             num_train_timesteps=n_timesteps, 
        #             beta_schedule=beta_schedule, 
        #             clip_sample=clip_denoised, 
        #             clip_sample_range=max_action, 
        #             prediction_type=predict_type
        #         )
        #     elif sampler_type == "ddim":
        #         self.scheduler = DDIMScheduler(
        #             num_train_timesteps=n_timesteps,    
        #             beta_schedule=beta_schedule,
        #             clip_sample=clip_denoised,
        #             clip_sample_range=max_action,
        #             prediction_type=predict_type
        #         )
        #     elif sampler_type == "dpm_multistep":
        #         self.scheduler = DPMSolverMultistepScheduler(
        #             num_train_timesteps=n_timesteps,
        #             beta_schedule=beta_schedule,
        #         )
        #     else:
        #         raise NotImplementedError
        #     self.scheduler.set_timesteps(
        #         num_inference_steps=n_inf_steps,
        #         device=device,
        #     )

        # def q_sample(self, action, t, noise):
        #     return self.scheduler.add_noise(action, noise, t)

        # def p_sample(self, noisy_action, t, state):
        #     self.scheduler.step