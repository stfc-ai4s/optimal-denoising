"""Gaussian fidelity terms using various robust penalty shapes."""

from __future__ import annotations
from torch import Tensor
import torch
from penalties.base import FidelityTerm
from penalties.shapes import charbonnier, welsch, log_penalty

__all__ = [
    "FidelityCharbonnierGaussian",
    "FidelityWelschGaussian",
    "FidelityLogGaussian"
]



# Charbonnier variant
class FidelityCharbonnierGaussian(FidelityTerm):
    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = epsilon
        self.y = None

    def set_observation(self, y: Tensor, **kwargs):
        self.y = y

    def value(self, x: Tensor) -> Tensor:
        diff = x - self.y
        return charbonnier(diff, self.epsilon).sum()

    def grad(self, x: Tensor) -> Tensor:
        diff = x - self.y
        return diff / torch.sqrt(diff**2 + self.epsilon**2)


# Welsch variant
class FidelityWelschGaussian(FidelityTerm):
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.y = None

    def set_observation(self, y: Tensor, **kwargs):
        self.y = y

    def value(self, x: Tensor) -> Tensor:
        diff = x - self.y
        return welsch(diff, self.beta).sum()

    def grad(self, x: Tensor) -> Tensor:
        diff = x - self.y
        w = 2. / self.beta**2 * torch.exp(-diff**2 / self.beta**2)
        return w * diff


# Log penalty variant
class FidelityLogGaussian(FidelityTerm):
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.y = None

    def set_observation(self, y: Tensor, **kwargs):
        self.y = y

    def value(self, x: Tensor) -> Tensor:
        diff = x - self.y
        return log_penalty(diff, self.beta).sum()

    def grad(self, x: Tensor) -> Tensor:
        diff = x - self.y
        denom = 1 + (diff**2 / self.beta**2)
        return 2 * diff / (self.beta**2 * denom)
    
# and finally, a plain MSE variant
class FidelityMSE(FidelityTerm):
    def set_observation(self, y: Tensor, **kwargs):
        self.y = y

    def value(self, x: Tensor):
        return ((x - self.y) ** 2).sum()

    def grad(self, x: Tensor):
        return 2 * (x - self.y)