from torch import Tensor
from model.penalties.base import FidelityTerm
from model.penalties.fidelity.gaussian import FidelityMSE
from model.penalties.fidelity.perceptual import FidelitySSIM

__all__ = ['FidelityMSE_SSIM_Mixed']

class FidelityMSE_SSIM_Mixed(FidelityTerm):
    """\lambda_1 x MSE + \lambda_1·(1 - SSIM) hybrid fidelity."""
    def __init__(self, lam_mse: float = 0.5, lam_ssim: float = 0.5):
        self.mse = FidelityMSE()
        self.ssim = FidelitySSIM()
        self.lm, self.ls = lam_mse, lam_ssim

    def set_observation(self, y: Tensor, **kwargs):
        self.mse.set_observation(y)
        self.ssim.set_observation(y)

    def value(self, x: Tensor):
        return self.lm * self.mse.value(x) + self.ls * self.ssim.value(x)

    def grad(self, x: Tensor):
        return self.lm * self.mse.grad(x) + self.ls * self.ssim.grad(x)