import torch
import torchvision.transforms as transforms
import numpy as np
import torchsde
import random

def _extract_into_tensor(arr_or_func, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array or a func.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if callable(arr_or_func):
        res = arr_or_func(timesteps).float()
    else:
        res = arr_or_func.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class RevVPSDE(torch.nn.Module):

    def __init__(self, model, beta_min=0.02, beta_max=10, N=200):
        super(RevVPSDE,self).__init__()
        self.model = model
        self.beta_0 = beta_min 
        self.beta_1 = beta_max 
        self.img_shape = (2048,3)#(1024,3)
        self.N = N
        self.context = None
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def _scale_timesteps(self, t):
        assert torch.all(t <= 1) and torch.all(t >= 0), f't has to be in [0, 1], but get {t} with shape {t.shape}'
        return (t.float() * self.N).long()

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def rvpsde_fn(self, t, x, return_type='drift'):
        """Create the drift and diffusion functions for the reverse SDE"""

        drift, diffusion = self.vpsde_fn(t, x)

        if return_type == 'drift':

            x_img = x.view(-1, *self.img_shape)
            disc_steps = self._scale_timesteps(t)  # (batch_size, ), from float in [0,1] to int in [0, 1000]
            model_output = self.model.score(x_img, int(disc_steps[0]),context=self.context)
            # with learned sigma, so model_output contains (mean, val)
            # model_output, _ = torch.split(model_output, self.img_shape[0], dim=1)
            # assert x_img.shape == model_output.shape, f'{x_img.shape}, {model_output.shape}'
            model_output = model_output.view(x.shape[0], -1)
            score = _extract_into_tensor(self.sqrt_1m_alphas_cumprod_neg_recip_cont, t, x.shape) * model_output

            drift = drift - diffusion[:, None] ** 2 * score
            return drift

        else:
            return diffusion


    def f(self, t, x):
        """Create the drift function -f(x, 1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        drift = self.rvpsde_fn(1 - t, x, return_type='drift')
        # assert drift.shape == x.shape
        return -drift

    def g(self, t, x):
        """Create the diffusion function g(1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        # assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)



class RevGuidedDiffusion(torch.nn.Module):
    def __init__(self, model, N=200, device=None):
        super(RevGuidedDiffusion,self).__init__()
        self.N = N
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        model.eval().to(self.device)

        self.model = model
        self.rev_vpsde = RevVPSDE(model=model).to(self.device)
        self.betas = self.rev_vpsde.discrete_betas.float().to(self.device)


    def truncated_sample(self, img, T, context):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        img = img.to(self.device)
        x0 = img

        e = torch.randn_like(x0).to(self.device)
        total_noise_levels = T

        a = (1 - self.betas).cumprod(dim=0).to(self.device)
        x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
        # print( a[total_noise_levels - 1].sqrt())
        epsilon_dt0, epsilon_dt1 = 0, 1e-5
        t0, t1 = 1 - T * 1. / self.N + epsilon_dt0, 1 - epsilon_dt1
        t_size = 2
        ts = torch.linspace(t0, t1, t_size).to(self.device)
        x_ = x.reshape(batch_size, -1)  # (batch_size, state_size)
        self.rev_vpsde.context = context
        xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler',dt=5e-3)
        x0 = xs_[-1].reshape(x.shape)  # (batch_size, c, h, w)
        return x0