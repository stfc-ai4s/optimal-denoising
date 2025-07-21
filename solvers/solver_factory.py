from .solver_admm import ADMMSolver
from .solver_fista import VanillaFISTA
from .solver_primaldual import PrimalDualSolver
from .solver_lbfgs import LBFGSSolver
from .solver_proximal import ProximalGradientSolver
from .solver_fista import VanillaFISTA, FasterFISTA, FreeFISTA


SOLVER_REGISTRY = {
    "admm": ADMMSolver,
    "fista": VanillaFISTA,
    "fista_fast": FasterFISTA,
    "fista_free": FreeFISTA,
    "pdhg": PrimalDualSolver,
    "lbfgs": LBFGSSolver,
    "prox": ProximalGradientSolver,
}

def get_solver(name: str, **kwargs):
    try:
        return SOLVER_REGISTRY[name.lower()](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown solver '{name}'. Available: {list(SOLVER_REGISTRY)}")