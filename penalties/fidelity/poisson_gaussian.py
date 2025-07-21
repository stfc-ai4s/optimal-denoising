"""Poisson–Gaussian fidelity terms using various robust penalty shapes."""

from __future__ import annotations
from torch import Tensor
import torch
from penalties.base import FidelityTerm
from penalties.shapes import charbonnier, welsch, log_penalty
from ..shapes import charbonnier, welsch, log_penalty
from utils.image_ops import _gradient

__all__ = [
    "FidelityCharbonnierPG",
    "FidelityWelschPG",
    "FidelityLogPG",
    "FidelityWelschGradModPG",
    "FidelityWelschContrastModPG"
]


# Charbonnier variant
class FidelityCharbonnierPG(FidelityTerm):
    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = epsilon
        self.y = self.sigma = None

    def set_observation(self, y: Tensor, sigma: Tensor):
        self.y = y
        self.sigma = sigma

    def value(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / self.sigma
        return charbonnier(diff, self.epsilon).sum()

    def grad(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / self.sigma
        return diff / torch.sqrt(diff**2 + self.epsilon**2) / self.sigma


# Welsch variant
class FidelityWelschPG(FidelityTerm):
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.y = self.sigma = None

    def set_observation(self, y: Tensor, sigma: Tensor):
        self.y = y
        self.sigma = sigma

    def value(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / self.sigma
        return welsch(diff, self.beta).sum()

    def grad(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / self.sigma
        w = 2. / self.beta**2 * torch.exp(-diff**2 / self.beta**2)
        return w * diff / self.sigma


# Log penalty variant
class FidelityLogPG(FidelityTerm):
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.y = self.sigma = None

    def set_observation(self, y: Tensor, sigma: Tensor):
        self.y = y
        self.sigma = sigma

    def value(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / self.sigma
        return log_penalty(diff, self.beta).sum()

    def grad(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / self.sigma
        denom = 1 + (diff**2 / self.beta**2)
        return (2 * diff / (self.beta**2 * denom)) / self.sigma
    


# Gradient-Weighted Welsch Fidelity
class FidelityWelschGradModPG(FidelityTerm):
    """
    Gradient-weighted Welsch fidelity:
        D(x, y) = sum(1 - exp( -0.5 * ((x - y) / sigma)^2 * |grad y|^2 ))
    """
    def __init__(self, c: float = 1.0):
        self.c = c
        self.y = self.sigma = None

    def set_observation(self, y: Tensor, sigma: Tensor):
        self.y = y
        self.sigma = sigma
        self.grad_y2 = (_gradient(y)**2).sum(0).clamp(min=1e-6)

    def value(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / (self.c * self.sigma)
        val = (1 - torch.exp(-0.5 * diff**2 * self.grad_y2)).sum()
        return val

    def grad(self, x: Tensor) -> Tensor:
        diff = (x - self.y) / (self.c * self.sigma)
        w = torch.exp(-0.5 * diff**2 * self.grad_y2)
        return (diff * w * self.grad_y2) / (self.c * self.sigma)
    


# Contrast-Aware Welsch Fidelity
class FidelityWelschContrastModPG(FidelityTerm):
    """
    Contrast-aware Welsch fidelity:
        D(x, y) = sum(1 - exp( -0.5 * ((x - y) / (c * sigma + gamma * | grad y|))^2 ))
    """
    def __init__(self, c: float = 1.0, gamma: float = 1.0):
        self.c = c
        self.gamma = gamma
        self.y = self.sigma = None

    def set_observation(self, y: Tensor, sigma: Tensor):
        self.y = y
        self.sigma = sigma
        self.grad_y = torch.sqrt((_gradient(y)**2).sum(0) + 1e-6)

    def value(self, x: Tensor) -> Tensor:
        denom = self.c * self.sigma + self.gamma * self.grad_y
        diff = (x - self.y) / denom
        return (1 - torch.exp(-0.5 * diff**2)).sum()

    def grad(self, x: Tensor) -> Tensor:
        denom = self.c * self.sigma + self.gamma * self.grad_y
        diff = (x - self.y) / denom
        w = torch.exp(-0.5 * diff**2)
        return (diff * w) / denom
