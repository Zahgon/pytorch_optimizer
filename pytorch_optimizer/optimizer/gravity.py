import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class Gravity(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        alpha: float = 0.01,
        beta: float = 0.9,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'alpha': alpha, 'beta': beta}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Gravity'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
