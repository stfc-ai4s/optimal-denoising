"""
Abstract base class for all denoising models.

Defines the common interface expected from denoisers, including:
- A unified `run()` method to trigger denoising
- Optional warm-starting with `x0`
- Optional observation injection with `set_observation()`
"""

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Optional


class DenoiserABC(ABC):
    @abstractmethod
    def set_observation(self, y: Tensor, sigma: Optional[Tensor] = None):
        """
        Sets the input observation (e.g., noisy image) and optional noise estimate.

        This allows decoupling observation injection from denoising execution, enabling:
        - modality-specific sigma estimates (e.g., Cryo-EM)
        - reusable solver/fidelity pipelines with multiple datasets
        """
        pass

    @abstractmethod
    def run(self, x0: Optional[Tensor] = None) -> Tensor:
        """
        Runs the denoising algorithm and returns the denoised image.

        Args:
            x0: Optional warm-start / initial guess (default = noisy image)

        Returns:
            Tensor: The denoised output
        """
        pass