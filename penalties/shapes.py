"""
Shared penalty shapes used in fidelity and regulariser terms.

These functions take a tensor (e.g., gradient, residual) and apply a robust
nonlinear transformation such as Charbonnier, Log, or Welsch.
"""

from torch import Tensor
import torch

__all__ = [
    "charbonnier",
    "log_penalty",
    "welsch",
    "smooth_abs"
]

# Charbonnier: sqrt(x^2 + epsilon^2)
def charbonnier(x: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt(x**2 + epsilon**2)

# Log penalty: log(1 + x^2 / beta^2)
def log_penalty(x: Tensor, beta: float = 1.0) -> Tensor:
    return torch.log1p(x**2 / beta**2)

# Welsch: 1 - exp(-x^2 / beta^2)
def welsch(x: Tensor, beta: float = 1.0) -> Tensor:
    return 1.0 - torch.exp(-x**2 / beta**2)

# Optional alias: smoothed |x|
def smooth_abs(x: Tensor, epsilon: float = 1e-3) -> Tensor:
    return charbonnier(x, epsilon)