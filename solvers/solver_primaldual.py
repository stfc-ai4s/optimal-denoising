import torch
from torch import Tensor
from typing import List
from penalties.base import FidelityTerm, RegulariserTerm
from .base import OptimisationSolver

__all__ = ['PrimalDualSolver']


class PrimalDualSolver(OptimisationSolver):
    """
    Primal-Dual Hybrid Gradient (PDHG / Chambolle–Pock) Solver.

    Minimises composite objectives of the form:
        min_x  D(x) + sum(R_i(x))

    where:
        - D(x): fidelity term (differentiable),
        - R_i(x): regularisers (may be smooth or nonsmooth).

    Algorithm:
        - Dual ascent using Rᵢ.prox() if available, otherwise projected gradient fallback.
        - Primal descent using ∇D(x) and dual-based regularisation updates.
        - Extrapolation (momentum-like step) for accelerated convergence.

    Features:
        - Supports multiple regularisers with or without `prox()` and `grad_from_dual()`.
        - Falls back to `.grad()` if no dual-based gradient is defined.
        - Robust and adaptable for convex and invex regularisers.

    Reference:
        Chambolle & Pock (2011), "A first-order primal-dual algorithm for convex problems
        with applications to imaging."

    Suitable for:
        - Regularisers with known proximal operators (e.g., TV, entropy).
        - Imaging applications with non-differentiable or edge-aware priors.
    """

    def __init__(self, tau: float = 0.01, sigma: float = 0.01, theta: float = 1.0, max_iter: int = 100):
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.max_iter = max_iter

    def optimise(
        self,
        x0: Tensor,
        fidelity: FidelityTerm,
        regularisers: List[RegulariserTerm]
    ) -> Tensor:
        x = x0.clone()
        x_bar = x.clone()

        duals = [torch.zeros_like(x) for _ in regularisers]

        for _ in range(self.max_iter):
            # --- Dual update ---
            for i, reg in enumerate(regularisers):
                dual_tmp = duals[i] + self.sigma * reg.grad(x_bar)
                try:
                    duals[i] = reg.prox(dual_tmp, self.sigma)
                except NotImplementedError:
                    duals[i] = torch.clamp(dual_tmp, -1.0, 1.0)

            # --- Primal update ---
            # grad = fidelity.grad(x)
            # for i, reg in enumerate(regularisers):
            #     if hasattr(reg, "grad_from_dual"):
            #         g = reg.grad_from_dual(duals[i])
            #     else:
            #         g = reg.grad(x)
            #     grad += g

            grad = fidelity.grad(x)
            for i, reg in enumerate(regularisers):
                if reg.supports_dual():
                    g = reg.grad_from_dual(duals[i])
                else:
                    g = reg.grad(x)
                grad += g

            x_new = x - self.tau * grad
            x_bar = x_new + self.theta * (x_new - x)
            x = x_new

        return x.detach()