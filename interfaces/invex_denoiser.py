# interfaces/invex_denoiser.py

from torch import Tensor
import inspect
from typing import Optional, Sequence, Union
from solvers.base import OptimisationSolver
from penalties.base import FidelityTerm, RegulariserTerm
from interfaces.composite_objective import CompositeObjective


class InvexDenoiser:
    """
    Generic invex-based solver interface for 2D denoising.

    Supports:
    - FidelityTerm (e.g., Welsch, Charbonnier)
    - One or more RegulariserTerm(s)
    - Any OptimisationSolver (L-BFGS, ADMM, PDHG, FISTA, etc.)
    """
    def __init__(self,
                 y: Tensor,
                 sigma: Optional[Tensor],
                 fidelity: FidelityTerm,
                 regulariser: Union[RegulariserTerm, Sequence[RegulariserTerm]],
                 solver: OptimisationSolver):
        self.y = y
        self.sigma = sigma
        self.fidelity = fidelity
        self.regulariser = [regulariser] if isinstance(regulariser, RegulariserTerm) else list(regulariser)
        self.solver = solver

        # Meke the following safe for Gaussian and Poisson-Gaussian fidelity terms
        sig = inspect.signature(self.fidelity.set_observation)
        if 'sigma' in sig.parameters:
            self.fidelity.set_observation(y, sigma=sigma)
        else:
            self.fidelity.set_observation(y)

    def run(self, x0: Optional[Tensor] = None) -> Tensor:
        x0 = x0 or self.y.clone()

        if self.solver.__class__.__name__.lower().startswith("admm"):
            return self.solver.optimise(x0, self.fidelity, self.regulariser).cpu()

        obj = CompositeObjective([self.fidelity] + self.regulariser)
        return self.solver.optimise(x0, obj).cpu()