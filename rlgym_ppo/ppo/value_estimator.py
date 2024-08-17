"""
File: value_estimator.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which models the value function of a policy.
"""

import torch.nn as nn
import torch
import numpy as np

from rlgym_ppo.models import FFN


class ValueEstimator(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device

        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        
        self.model = FFN(input_shape, 1, layer_sizes, objective="value").to(device)

    def forward(self, x):
        t = type(x)
        if t != torch.Tensor:
            if t != np.array:
                x = np.asarray(x)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return self.model(x)
