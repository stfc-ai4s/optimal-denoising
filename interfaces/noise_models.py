"""
noise_models.py

Abstract and concrete noise models to encapsulate modality-specific noise estimation
for use in invex-based denoising pipelines.

Supports:
- CryoEM (floating point)
- X-ray (Poisson-dominated or log-transformed)

All models return a per-image or per-pixel sigma (noise level) tensor.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
import math
import numpy as np

__all__ = ['NoiseModel', 'CryoEMNoiseModel', 'FixedNoiseModel' ,'get_noise_model']



class NoiseModel(ABC):
    @abstractmethod
    def estimate(self, y: Tensor) -> Tensor:
        """
        Estimate per-pixel or global noise level from the noisy observation y.
        Returns:
            Tensor of shape (H, W) or scalar tensor with same dtype/device as y.
        """
        pass


class CryoEMNoiseModel(NoiseModel):
    """
    Assumes floating point CryoEM density maps with zero-centered signal.
    Uses global MAD or STD depending on choice.
    """
    def __init__(self, use_mad: bool = True):
        self.use_mad = use_mad

    def estimate(self, y: Tensor) -> Tensor:
        if self.use_mad:
            med = y.median()
            mad = (y - med).abs().median()
            s = 1.4826 * mad.item() #this magic number comes from the MAD definition
        else:
            s = y.std().item()
        return torch.full_like(y, s)


class FixedNoiseModel(NoiseModel):
    """
    Simple wrapper to return a fixed σ value everywhere (modality-agnostic).
    """
    def __init__(self, sigma_val: float):
        self.sigma_val = sigma_val

    def estimate(self, y: Tensor) -> Tensor:
        return torch.full_like(y, self.sigma_val)


def get_noise_model(name: str, **kwargs) -> NoiseModel:
    """
    Returns an instantiated noise model based on name.

    Args:
        name: One of ['cryoem', 'fixed'].
        kwargs: Parameters to pass to the noise model.

    Returns:
        An instance of NoiseModel subclass.
    """
    name = name.lower()
    if name == 'cryoem':
        return CryoEMNoiseModel(**kwargs)
    elif name == 'fixed':
        return FixedNoiseModel(**kwargs)
    else:
        raise ValueError(f"Unknown noise model '{name}'. Supported: cryoem & fixed")
    