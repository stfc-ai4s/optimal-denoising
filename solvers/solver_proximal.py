import torch
from torch import Tensor
from typing import List
from .base import OptimisationSolver
from penalties.base import FidelityTerm, RegulariserTerm

__all__ = ['ProximalGradientSolver']

class ProximalGradientSolver(OptimisationSolver):
    """
    Basic Proximal Gradient Descent Solver.

    Solves:
        min_x  D(x) + sum(R_i(x))

    Assumes:
    - D is smooth and differentiable.
    - Each R_ has a known proximal operator (or fallback to grad-based descent).

    Update:
        x <- prox_{\tau R}(x - \tau grad D(x))
    """

    def __init__(self, stepsize: float = 1e-2, lam: float = 1.0, max_iter: int = 100):
        self.stepsize = stepsize
        self.lam = lam
        self.max_iter = max_iter

    def optimise(
        self,
        x0: Tensor,
        fidelity: FidelityTerm,
        regularisers: List[RegulariserTerm]
    ) -> Tensor:
        x = x0.clone()

        for _ in range(self.max_iter):
            grad = fidelity.grad(x)
            x_new = x - self.stepsize * grad

            for reg in regularisers:
                try:
                    x_new = reg.prox(x_new, self.stepsize * self.lam)
                except NotImplementedError:
                    x_new = x_new.clone().detach().requires_grad_(True)
                    reg_loss = reg.value(x_new)
                    grad_r = torch.autograd.grad(reg_loss, x_new)[0]
                    x_new = (x_new - self.stepsize * self.lam * grad_r).detach()

            x = x_new

        return x.detach()