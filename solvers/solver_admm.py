import torch
from torch import Tensor
from typing import List
from .base import OptimisationSolver
from penalties.base import FidelityTerm, RegulariserTerm

class ADMMSolver(OptimisationSolver):
    """
    Alternating Direction Method of Multipliers (ADMM) for composite invex optimisation.

    Solves problems of the form:
        min_x  D(x, y) + sum(R_i(x)

    by splitting the fidelity and regularisers using an auxiliary variable z, such that:
        min_x,z  D(x, y) + sum(R_i(z)   s.t.  x = z

    ADMM iteratively updates:
      - x-step: gradient descent on fidelity + quadratic penalty
      - z-step: proximal operator (or fallback to gradient descent)
      - u-step: dual variable update (scaled Lagrange multiplier)

    Notes:
    - Supports additive regularisers.
    - If `reg.prox()` is not implemented, falls back to gradient descent.
    - Joint proximal support is noted but not implemented (see TODO).

    Recommended for:
    - Cases where regularisers are non-differentiable or prox-friendly.
    - Welsch, entropy, or TV-type terms.

    TODO:
    - Add joint proximal support for sum(R_i(x)) when applicable.
    """

    def __init__(self, rho: float = 1.0, max_iter: int = 100, step_size: float = 0.1):
        self.rho = rho
        self.max_iter = max_iter
        self.step_size = step_size

    def optimise(
        self,
        x0: Tensor,
        fidelity: FidelityTerm,
        regularisers: List[RegulariserTerm]
    ) -> Tensor:

        x = x0.clone().detach().requires_grad_(True)
        z = x0.clone().detach()
        u = torch.zeros_like(x)

        for _ in range(self.max_iter):
            # ----- x-update -----
            x.requires_grad_(True)
            fidelity_val = fidelity.value(x)
            penalty = 0.5 * self.rho * torch.norm(x - z + u)**2
            loss = fidelity_val + penalty
            grad = torch.autograd.grad(loss, x)[0]
            x = (x - self.step_size * grad).detach()

            # ----- z-update -----
            v = x + u
            z_temp = v.clone()
            for reg in regularisers:
                try:
                    z_temp = reg.prox(z_temp, 1.0 / self.rho)
                except NotImplementedError:
                    z_temp = z_temp.clone().detach().requires_grad_(True)
                    reg_loss = reg.value(z_temp) + 0.5 * self.rho * torch.norm(z_temp - v)**2
                    grad_z = torch.autograd.grad(reg_loss, z_temp)[0]
                    z_temp = (z_temp - self.step_size * grad_z).detach()

            z = z_temp

            # ----- u-update (dual ascent) -----
            u = u + x - z

        return x.detach()