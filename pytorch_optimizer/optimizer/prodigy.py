import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class Prodigy(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.9, 0.999),
        beta3: Optional[float] = None,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        bias_correction: bool = False,
        safeguard_warmup: bool = False,
        eps: Optional[float] = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas((*betas, beta3))
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'beta3': beta3,
            'd': d0,
            'd0': d0,
            'd_max': d0,
            'd_coef': d_coef,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'bias_correction': bias_correction,
            'safeguard_warmup': safeguard_warmup,
            'step': 1,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Prodigy'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
