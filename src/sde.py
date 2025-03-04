import torch
from .sde_base import SDEBase

class VP_SDE(SDEBase):
    '''
    An SDE version of DDPM.
    '''
    def __init__(self, beta_min=0.1, beta_max=20., eps=1e-5, rescale=True):
        super().__init__(eps, rescale)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift_coef(self, x, t):
        drift = self.beta_t(t)
        drift = self.match_dim(drift, x)
        drift = - drift * x / 2
        return drift
    
    def diffusion_coef(self, t):
        return torch.sqrt(self.beta_t(t))
    
    def x0_coef(self, t):
        x = - t**2 * (self.beta_max - self.beta_min) / 4
        x = x - t * self.beta_min / 2
        return torch.exp(x)
    
    def sigma_t(self, t):
        x = self.x0_coef(t)
        return torch.sqrt(1 - x**2)