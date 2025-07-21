import torch
from torch import Tensor
from torch.optim import LBFGS
from .base import OptimisationSolver

__all__ = ['LBFGSSolver']

class LBFGSSolver(OptimisationSolver):
    def __init__(self, max_iter: int = 100, line_search_fn: str = 'strong_wolfe'):
        self.max_iter = max_iter
        self.line_search_fn = line_search_fn

    def optimise(self, x0: Tensor, objective) -> Tensor:
        """
        Minimises a CompositeObjective via L-BFGS.

        Args:
            x0         : Initial guess (Tensor with grad)
            objective  : Must implement `.value(x)` and `.grad(x)`

        Returns:
            x (Tensor): Denoised solution
        """
        x = x0.clone().detach().requires_grad_(True)

        def closure():
            if torch.is_grad_enabled():
                if x.grad is not None:
                    x.grad.zero_()
            loss = objective.value(x)
            loss.backward()
            return loss

        optimizer = LBFGS([x], max_iter=self.max_iter, line_search_fn=self.line_search_fn)
        optimizer.step(closure)
        return x.detach()