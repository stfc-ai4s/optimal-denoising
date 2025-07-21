"""Invex regulariser penalty: LogTV Regulariser."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from ..base import InvexPenalty
from ..base import RegulariserTerm
from utils.image_ops import _gradient

__all__ = ['RegulariserLogTV']


class RegulariserLogTV(RegulariserTerm):
    """
    Log-Total Variation (LogTV) Regulariser.

    This regulariser penalises image gradient magnitudes via a logarithmic transform:

        R(x) = \lambda sum log(1 + |\nabal x|^2 / \beta^2)

    Where:
        - \nabal x is the image gradient (first-order derivatives),
        - \beta controls edge sensitivity (lower \beta means sharper edge retention),
        - \lambda balances the regularisation strength.

    LogTV is a smooth, nonconvex alternative to classical Total Variation (TV), offering:
        - Edge preservation without staircasing artefacts.
        - Smooth differentiability (enables gradient-based optimisation).
        - Resistance to over-smoothing in high-contrast areas.

    """

    def __init__(self, lambda_: float = 0.5, beta: float = 2.0):
        self.lambda_ = lambda_
        self.beta2 = beta * beta

    def value(self, x: Tensor) -> Tensor:
        grad = _gradient(x)
        return self.lambda_ * torch.log1p((grad ** 2).sum(0) / self.beta2).sum()

    def grad(self, x: Tensor) -> Tensor:
        grad = _gradient(x)
        mag2 = (grad ** 2).sum(0)
        w = 2. / self.beta2 / (1 + mag2 / self.beta2)
        gx, gy = grad

        div = (
            F.pad(w * gx, (1, 0))[:, :-1] - F.pad(w * gx, (0, 1))[:, 1:] +
            F.pad(w * gy, (0, 0, 1, 0))[1:, :] - F.pad(w * gy, (0, 0, 0, 1))[:-1, :]
        )
        return self.lambda_ * div

    def prox(self, v: Tensor, tau: float) -> Tensor:
        """
        Placeholder proximal operator for LogTV.

        No closed-form solution is known. Approximating this would require
        nested optimisation, so gradient descent is typically preferred.
        """
        raise NotImplementedError("LogTV does not support a tractable closed-form prox.")