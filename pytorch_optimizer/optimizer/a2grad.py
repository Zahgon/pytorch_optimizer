import math
from typing import Literal, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT

VARIANTS = Literal['uni', 'inc', 'exp']


class A2Grad(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: Optional[float] = None,
        beta: float = 10.0,
        lips: float = 10.0,
        rho: float = 0.5,
        variant: VARIANTS = 'uni',
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(lips, 'lips')
        self.validate_non_negative(rho, 'rho')
        self.validate_options(variant, 'variant', ['uni', 'inc', 'exp'])

        self.variant = variant
        self.maximize = maximize

        defaults: Defaults = {'beta': beta, 'lips': lips}
        if variant == 'exp':
            defaults.update({'rho': rho})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'A2Grad'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
