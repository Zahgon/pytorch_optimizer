import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class FOCUS(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: Betas = (0.9, 0.999),
        gamma: float = 0.1,
        weight_decay: float = 0.0,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(gamma, 'gamma', 0.0, 1.0, '[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'betas': betas, 'gamma': gamma, 'weight_decay': weight_decay}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FOCUS'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
