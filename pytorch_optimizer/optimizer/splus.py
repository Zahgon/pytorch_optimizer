from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import (
    Betas,
    Closure,
    Defaults,
    Loss,
    ParamGroup,
    ParamsT,
)


class SPlus(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ema_rate: float = 0.999,
        inverse_steps: int = 100,
        nonstandard_constant: float = 1e-3,
        max_dim: int = 10000,
        eps: float = 1e-30,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(ema_rate, 'ema_rate', 0.0, 1.0)
        self.validate_positive(inverse_steps, 'inverse_steps')
        self.validate_positive(max_dim, 'max_dim')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ema_rate': ema_rate,
            'inverse_steps': inverse_steps,
            'max_dim': max_dim,
            'nonstandard_constant': nonstandard_constant,
            'eps': eps,
            'train_mode': True,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SPlus'

    @torch.no_grad()
    def eval(self):
        pass

    @torch.no_grad()
    def train(self):
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def get_scaled_lr(shape: Tuple[int, int], lr: float, nonstandard_constant: float, max_dim: int = 10000) -> float:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
