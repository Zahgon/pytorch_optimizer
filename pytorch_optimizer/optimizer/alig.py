from typing import Callable, Optional

import torch

from pytorch_optimizer.base.exception import NoClosureError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


@torch.no_grad()
def l2_projection(parameters: ParamsT, max_norm: float = 1e2) -> None:
    r"""Get l2 normalized parameter."""
    global_norm = torch.sqrt(sum(p.norm().pow(2) for p in parameters or []))
    if global_norm > max_norm:
        ratio = max_norm / global_norm
        for param in parameters or []:
            param.mul_(ratio)


class AliG(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        max_lr: Optional[float] = None,
        projection_fn: Optional[Callable] = None,
        momentum: float = 0.0,
        adjusted_momentum: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(max_lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)

        self.projection_fn = projection_fn
        self.maximize = maximize

        defaults: Defaults = {'max_lr': max_lr, 'adjusted_momentum': adjusted_momentum, 'momentum': momentum}

        super().__init__(params, defaults)

        if self.projection_fn is not None:
            self.projection_fn()

    def __str__(self) -> str:
        return 'AliG'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def compute_step_size(self, loss: float) -> float:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
