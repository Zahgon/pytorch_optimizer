import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class AdamG(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.95, 0.999, 0.95),
        p: float = 0.2,
        q: float = 0.24,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_positive(p, 'p')
        self.validate_positive(q, 'q')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.p = p
        self.q = q
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamG'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def s(self, p: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
