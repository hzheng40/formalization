import torch
import torch.nn as nn
import torch.functional as F

from utils import on_cuda

class LinearShift(nn.Module):
    """
    Linear layers that only shifts the distribution's mean and variance, still outputs a gaussian
    """
    def __init__(self, config):
        super(LinearShift, self).__init__()
        self.n_layers = config.n_shift_layers
        self.linear_mu = nn.ModuleList([nn.Linear(config.n_z, config.n_z) for _ in range(self.n_layers)])
        self.linear_logvar = nn.ModuleList([nn.Linear(config.n_z, config.n_z) for _ in range(self.n_layers)])

    def forward(self, mu, logvar):
        for layer in range(self.n_layers):
            mu = self.linear_mu[layer](mu)
            logvar = self.linear_logvar[layer](logvar)
        return mu, logvar