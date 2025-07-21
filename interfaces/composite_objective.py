from torch import Tensor
from typing import List
from penalties.base import InvexPenalty

class CompositeObjective(InvexPenalty):
    """
    Composite objective that sums multiple invex penalty terms.

    Used by solvers expecting a single objective (e.g., LBFGS).
    """
    def __init__(self, terms: List[InvexPenalty]):
        self.terms = terms

    def value(self, x: Tensor) -> Tensor:
        return sum(term.value(x) for term in self.terms)

    def grad(self, x: Tensor) -> Tensor:
        return sum(term.grad(x) for term in self.terms)