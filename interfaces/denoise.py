from __future__ import annotations
import torch
import numpy as np
from torch import Tensor
from typing import Optional, Sequence, Union

from interfaces.invex_denoiser import InvexDenoiser
from interfaces.noise_models import NoiseModel, get_noise_model
from penalties.base import FidelityTerm, RegulariserTerm
from solvers.solver_lbfgs import LBFGSSolver
from utils.image_ops import _gradient
from solvers.base import OptimisationSolver
from interfaces.invex_denoiser import InvexDenoiser
from interfaces.noise_models import get_noise_model
from solvers.solver_lbfgs import LBFGSSolver  # fallback default

__all__ = ['denoise', 'denoise_slice', 'denoise_volume']


def denoise(y: Union[Tensor, 'np.ndarray'],
            fidelity: FidelityTerm,
            regulariser: Union[RegulariserTerm, Sequence[RegulariserTerm]],
            sigma: Optional[Tensor] = None,
            solver=None,
            noise_model: Optional[Union[str, NoiseModel]] = 'ct',
            **kwargs) -> Tensor:
    """
    Main entry point for denoising 2D or 3D images.

    Dispatches to slice-wise or volume-aware denoiser based on `y.ndim`.

    Args:
        y: Input image (2D or 3D tensor or NumPy array)
        fidelity: Fidelity term
        regulariser: One or more regulariser terms
        sigma: Optional noise std tensor
        solver: Optimiser (e.g. FISTA, ADMM)
        noise_model: Optional string or NoiseModel instance
        **kwargs: Extra args forwarded to denoiser

    Returns:
        Denoised image (same shape as input)
    """
    if not torch.is_tensor(y):
        y = torch.as_tensor(y).float()
    
    if solver is None:
        solver = LBFGSSolver(max_iter=500)  

    if y.ndim == 2:
        return denoise_slice(y, fidelity, regulariser, sigma, solver, noise_model, **kwargs)
    elif y.ndim == 3:
        return denoise_volume(y, fidelity, regulariser, sigma, solver, noise_model, **kwargs)
    else:
        raise ValueError(f"Unsupported input shape: {y.shape}")


def denoise_slice(y: Union[Tensor, np.ndarray],
                  fidelity: FidelityTerm,
                  regulariser: Union[RegulariserTerm, Sequence[RegulariserTerm]],
                  sigma: Optional[Tensor] = None,
                  solver: Optional[OptimisationSolver] = None,
                  noise_model: Optional[Union[str, NoiseModel]] = 'ct',
                  return_sigma: bool = False,
                  **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Denoise a 2D slice using the InvexDenoiser interface.

    Args:
        y             : 2D image tensor or NumPy array
        fidelity      : FidelityTerm (e.g., FidelityWelschPG)
        regulariser   : RegulariserTerm or list of terms
        sigma         : Optional tensor of noise std values
        solver        : Optional solver (e.g., LBFGSSolver)
        noise_model   : NoiseModel instance or name ('ct' by default)
        return_sigma  : Whether to return sigma in addition to the denoised image
        **kwargs      : Extra args passed to the noise model

    Returns:
        x_hat         : Denoised image (torch.Tensor)
        sigma         : (optional) estimated noise level (torch.Tensor)
    """


    # Convert input to tensor if needed
    if isinstance(y, np.ndarray):
        y = torch.as_tensor(y).float()

    if y.ndim != 2:
        raise ValueError("denoise_slice expects a 2D tensor")

    # Set defaults
    if solver is None:
        solver = LBFGSSolver(max_iter=500)

    if isinstance(noise_model, str):
        noise_model = get_noise_model(noise_model)

    if sigma is None:
        sigma = noise_model.estimate(y, **kwargs)

    # Instantiate and run the denoiser
    denoiser = InvexDenoiser(
        y=y,
        sigma=sigma,
        fidelity=fidelity,
        regulariser=regulariser,
        solver=solver
    )
    x_hat = denoiser.run()

    return (x_hat, sigma) if return_sigma else x_hat


def denoise_volume(y: Tensor,
                   fidelity: FidelityTerm,
                   regulariser: Union[RegulariserTerm, Sequence[RegulariserTerm]],
                   sigma: Optional[Tensor] = None,
                   solver=None,
                   noise_model: Optional[Union[str, NoiseModel]] = 'ct',
                   **kwargs) -> Tensor:
    """
    Volume-level denoising stub.
    Current implementation applies slice-wise denoising.

    Future version can support full 3D models and regularisers.
    """
    return torch.stack([
        denoise_slice(y[i], fidelity, regulariser, sigma=sigma, solver=solver, noise_model=noise_model, **kwargs)
        for i in range(y.shape[0])
    ])