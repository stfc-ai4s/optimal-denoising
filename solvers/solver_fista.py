from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import List, Optional
from penalties.base import FidelityTerm, RegulariserTerm
from .base import OptimisationSolver


__all__ = ['FISTA_Base', 'VanillaFISTA', 'FasterFISTA', 'FreeFISTA']


# ---------------------------------------------------------------------
# Abstract Base: Shared interface for all FISTA variants
# ---------------------------------------------------------------------
class FISTA_Base(OptimisationSolver, ABC):
    """
    Abstract base class for FISTA-style solvers.
    All variants must implement `optimise(...)`.
    """
    @abstractmethod
    def optimise(self,
                 x0: Tensor,
                 fidelity: FidelityTerm,
                 regularisers: List[RegulariserTerm]) -> Tensor:
        pass


# ---------------------------------------------------------------------
# Vanilla FISTA: Smooth + differentiable only
# ---------------------------------------------------------------------
class VanillaFISTA(FISTA_Base):
    """
    Vanilla FISTA for smooth unconstrained optimisation with additive differentiable regularisation.

    Minimises: D(x, y) + sum(R_i(x))
    where D is a differentiable fidelity term and Rᵢ are invex regularisers.
    """

    def __init__(self, stepsize: float = 1e-2, max_iter: int = 100, tol: Optional[float] = 1e-5):
        self.stepsize = stepsize
        self.max_iter = max_iter
        self.tol = tol

    def optimise(self,
                 x0: Tensor,
                 fidelity: FidelityTerm,
                 regularisers: List[RegulariserTerm]) -> Tensor:
        x = x0.clone()
        y = x.clone()
        t = 1.0

        for _ in range(self.max_iter):
            grad_total = fidelity.grad(y)
            for reg in regularisers:
                grad_total += reg.grad(y)

            x_new = y - self.stepsize * grad_total
            t_new = (1 + (1 + 4 * t**2)**0.5) / 2
            y = x_new + ((t - 1) / t_new) * (x_new - x)

            if self.tol is not None and torch.norm(x_new - x) < self.tol:
                break

            x, t = x_new, t_new

        return x.detach()


# ---------------------------------------------------------------------
# Faster FISTA: Learned version (Liang et al.)
# ---------------------------------------------------------------------
class FasterFISTA(FISTA_Base):
    """
    Faster-FISTA (Liang et al., 2021): Learns step sizes, momentum, and stopping.

    - Learns per-iteration parameters
    - Uses greedy early stopping
    - Suitable for unrolled training (requires training loop)
    """

    def __init__(self, T: int = 20, learnable: bool = True):
        self.T = T
        self.learnable = learnable
        # Placeholder: implement learned schedule, stopping rule, etc.

    def optimise(self,
                 x0: Tensor,
                 fidelity: FidelityTerm,
                 regularisers: List[RegulariserTerm]) -> Tensor:
        raise NotImplementedError("FasterFISTA is not yet implemented. Placeholder class.")
    

