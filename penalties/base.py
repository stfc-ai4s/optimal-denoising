"""Baseline class for Invex Penalties (Fidelity or Regulariser)"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor

__all__ = ['InvexPenalty', 'FidelityTerm', 'RegulariserTerm']

# Base interface for all invex penalties
class InvexPenalty(ABC):
    @abstractmethod
    def value(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def prox(self, v: Tensor, tau: float) -> Tensor:
        """Optional proximal operator (used in ADMM or primal-dual solvers)."""
        raise NotImplementedError


# Fidelity term (e.g., data consistency)
class FidelityTerm(InvexPenalty):
    def set_observation(self, y: Tensor, **kwargs):
        """
        Set observed data (e.g., low-dose input).
        `kwargs` may include noise parameters (e.g., sigma) or modality-specific info.
        """
        self.y = y  # Base assignment; subclasses may override or extend this
        


# Regulariser term (e.g., TV, Hessian, Log)
class RegulariserTerm(InvexPenalty):
    """
    Base class for invex regularisers (e.g., TV, LogTV, Hessian).
    Inherits from InvexPenalty (value, grad, optional prox).
    """

    def grad_from_dual(self, p: Tensor) -> Tensor | None:
        """
        Optional: primal update using dual variable p.
        If not implemented by subclass, returns None.
        """
        return None

    def supports_dual(self) -> bool:
        """Returns True if grad_from_dual is overridden."""
        return type(self).grad_from_dual != RegulariserTerm.grad_from_dual