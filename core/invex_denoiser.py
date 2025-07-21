from __future__ import annotations
from typing import Sequence, Optional, Union
import torch
from torch import Tensor

from penalties.base import FidelityTerm, RegulariserTerm
from solvers.base import OptimisationSolver


class InvexDenoiser:
    """
    Generic modality-agnostic denoiser using invex fidelity and regularisation.

    Supports additive regularisers and any solver implementing the OptimisationSolver interface.
    """

    def __init__(
        self,
        fidelity: FidelityTerm,
        regulariser: Union[RegulariserTerm, Sequence[RegulariserTerm]],
        solver: OptimisationSolver
    ):
        self.fidelity = fidelity
        self.regulariser = (
            [regulariser] if isinstance(regulariser, RegulariserTerm) else list(regulariser)
        )
        self.solver = solver

        self.y: Optional[Tensor] = None
        self.sigma: Optional[Tensor] = None

    def set_observation(self, y: Tensor, sigma: Optional[Tensor] = None):
        """
        Assign observed image and optional noise map (modality-specific).
        """
        self.y = y
        self.sigma = sigma
        self.fidelity.set_observation(y, sigma=sigma)

    def run(self, x0: Optional[Tensor] = None) -> Tensor:
        """
        Denoise image via solver: solves

            min_x D(x, y) + sum (R_i(x))

        Returns:
            \hat{x} (denoised result on CPU)
        """
        if self.y is None:
            raise ValueError("Observation not set. Call `set_observation(y, sigma)` first.")

        x0 = x0 if x0 is not None else self.y.clone()

        # Generic solve step
        return self.solver.optimise(x0, self.fidelity, self.regulariser).cpu()