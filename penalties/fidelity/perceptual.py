from torch import Tensor
import torch
from model.penalties.base import FidelityTerm

__all__ = ['FidelitySSIM']

class FidelitySSIM(FidelityTerm):
    """(1 - SSIM) computed using torchmetrics)."""
    def __init__(self, win_size: int = 7):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0,
                                                      kernel_size=win_size)
    def set_observation(self, y: Tensor, **kwargs):
        self.y = (y - y.min()) / (y.max() - y.min() + 1e-3)

    def value(self, x: Tensor):
        x_n = (x - x.min()) / (x.max() - x.min() + 1e-3)
        return 1.0 - self._ssim(x_n[None, None], self.y[None, None])

    def grad(self, x: Tensor):
        x = x.clone().requires_grad_(True)
        loss = self.value(x)
        loss.backward()
        return x.grad.detach()