"""
Regulariser wrapper for external denoisers, enabling Plug-and-Play Priors or RED-style regularisation.

This allows models like BM3D, AutoEncoders, or CNNs to be used as regularisers, in addition 
to traditional TV or Hessian-based methods.

References:
    - Romano et al. "The little engine that could: Regularization by Denoising (RED)", SIAM 2017.
    - Venkatakrishnan et al. "Plug-and-Play Priors for Model Based Reconstruction", IEEE TCI, 2013.
"""

from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor
from ..base import RegulariserTerm

__all__ = ["RegulariserDenoiser"]

class RegulariserDenoiser(RegulariserTerm):
    def __init__(self, denoiser: Callable[[Tensor], Tensor], weight: float = 1.0):
        """
        Args:
            denoiser: A callable function that takes a noisy tensor and returns a denoised tensor.
                      Must be differentiable if used with RED-style gradients.
            weight:   Regularisation weight \lamba.
        """
        self.denoiser = denoiser
        self.weight = weight

    def value(self, x: Tensor) -> Tensor:
        """
        Optional: Proxy value of R(x) = 0.5 * ||x - f(x)||^2
        Useful for monitoring or debugging.
        """
        with torch.no_grad():
            fx = self.denoiser(x)
            return 0.5 * self.weight * torch.norm(x - fx)**2

    def grad(self, x: Tensor) -> Tensor:
        """
        RED-style gradient: grad R(x) = \lamba(x - f(x))
        Assumes f(x) is a (possibly black-box) denoiser.
        """
        fx = self.denoiser(x)
        return self.weight * (x - fx)

    def prox(self, v: Tensor, tau: float) -> Tensor:
        """
        Optional: Plug-and-play proximal operator:
            prox_{\tau R}(v) ≈ f(v), assuming denoiser acts like proximal.
        """
        return self.denoiser(v)