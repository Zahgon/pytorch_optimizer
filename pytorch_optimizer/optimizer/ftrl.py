import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class FTRL(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        lr_power: float = -0.5,
        beta: float = 0.0,
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(beta, 'beta')
        self.validate_non_positive(lr_power, 'lr_power')
        self.validate_non_negative(lambda_1, 'lambda_1')
        self.validate_non_negative(lambda_2, 'lambda_2')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'lr_power': lr_power, 'beta': beta, 'lambda_1': lambda_1, 'lambda_2': lambda_2}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FTRL'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
