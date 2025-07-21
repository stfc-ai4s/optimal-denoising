"""Invex regulariser penalty: Hessian-based second-order edge-preserving regulariser."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from ..base import RegulariserTerm

__all__ = ['RegulariserHessianLog']


def _expand_for_pad(x: Tensor) -> tuple[Tensor, bool]:
    """Ensure input has shape [N, C, H, W] for padding. Return reshaped and flag."""
    if x.ndim == 2:
        return x[None, None], True
    elif x.ndim == 3:
        return x[None], True
    return x, False


def _hessian(x: Tensor) -> tuple[Tensor, Tensor]:
    """Compute diagonal Hessian components (dxx, dyy)."""
    x, squeezed = _expand_for_pad(x)

    dxx = F.pad(x, (1, 1, 0, 0), mode='replicate')[:, :, :, 2:] - 2 * x + F.pad(x, (1, 1, 0, 0), mode='replicate')[:, :, :, :-2]
    dyy = F.pad(x, (0, 0, 1, 1), mode='replicate')[:, :, 2:, :] - 2 * x + F.pad(x, (0, 0, 1, 1), mode='replicate')[:, :, :-2, :]

    if squeezed:
        return dxx[0, 0], dyy[0, 0]
    return dxx, dyy


def safe_pad(x: Tensor, pad: tuple[int, int, int, int], mode: str = 'replicate') -> Tensor:
    """Apply safe padding compatible with 2D gradients."""
    return F.pad(x[None, None], pad, mode=mode)[0, 0]


class RegulariserHessianLog(RegulariserTerm):
    """
    Second-Order Edge-Preserving Regulariser (HessianLog).

    This regulariser applies a logarithmic penalty to the sum of squared second derivatives:

        R(x) = \lambda \sum log(1 + (grad^2 x)^2 / \beta^2)

    Where:
        - grad^2 x is the diagonal Hessian (approximating Laplacian curvature),
        - \beta controls curvature sensitivity,
        - \lambda governs the strength of smoothing.
    """

    def __init__(self, lambda_: float = 0.1, beta: float = 2.0):
        self.lambda_ = lambda_
        self.beta2 = beta ** 2

    def value(self, x: Tensor) -> Tensor:
        dxx, dyy = _hessian(x)
        return self.lambda_ * torch.log1p((dxx ** 2 + dyy ** 2) / self.beta2).sum()

    def grad(self, x: Tensor) -> Tensor:
        dxx, dyy = _hessian(x)
        hess_norm2 = dxx ** 2 + dyy ** 2
        weight = 2. / self.beta2 / (1 + hess_norm2 / self.beta2)

        dxx_w = weight * dxx
        dyy_w = weight * dyy

        gxx = safe_pad(dxx_w, (1, 1, 0, 0), mode='replicate')
        gyy = safe_pad(dyy_w, (0, 0, 1, 1), mode='replicate')

        ddx = gxx[:, 2:] - 2 * gxx[:, 1:-1] + gxx[:, :-2]
        ddy = gyy[2:, :] - 2 * gyy[1:-1, :] + gyy[:-2, :]

        return self.lambda_ * (ddx + ddy)

    def prox(self, v: Tensor, tau: float) -> Tensor:
        raise NotImplementedError("HessianLog has no efficient closed-form prox.")