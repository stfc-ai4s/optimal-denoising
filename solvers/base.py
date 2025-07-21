from abc import ABC, abstractmethod
from torch import Tensor
from typing import List
from penalties.base import FidelityTerm, RegulariserTerm

class OptimisationSolver(ABC):
    @abstractmethod
    def optimise(self, 
                 x0: Tensor, 
                 fidelity: FidelityTerm, 
                 regularisers: List[RegulariserTerm]) -> Tensor:
        """
        Run optimisation and return denoised result.

        Args:
            x0: Initial guess.
            fidelity: Fidelity term (e.g., data consistency).
            regularisers: One or more regularisation penalties.

        Returns:
            Denoised image or volume. 
        """
        pass