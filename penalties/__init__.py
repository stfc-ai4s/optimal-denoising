"""
Invex-based penalties for image denoising.

This package defines:
- Fidelity terms (data consistency)
- Regularisation terms (structure, smoothness, prior)
- Shape functions (Charbonnier, Welsch, Log, etc.)

Each component follows a unified interface and is usable in any solver.
"""

from .base import FidelityTerm, RegulariserTerm
from .shapes import charbonnier, welsch, log_penalty

# Fidelity Terms
from .fidelity.poisson_gaussian import (
    FidelityCharbonnierPG,
    FidelityWelschPG,
    FidelityLogPG,
    FidelityWelschGradModPG,
    FidelityWelschContrastModPG,
)

# Regulariser Terms
from .regulariser.logtv import RegulariserLogTV
from .regulariser.hessian_log import RegulariserHessianLog
from .regulariser.denoiser import RegulariserDenoiser

__all__ = [
    # Core base types
    "FidelityTerm", "RegulariserTerm",

    # Shape functions
    "charbonnier", "welsch", "log_penalty",

    # Fidelity variants
    "FidelityCharbonnierPG",
    "FidelityWelschPG",
    "FidelityLogPG",
    "FidelityWelschGradModPG",
    "FidelityWelschContrastModPG",

    # Regularisers
    "RegulariserLogTV",
    "RegulariserHessianLog",
    "RegulariserDenoiser",
]