from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class LARS(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        dampening: float = 0.0,
        trust_coefficient: float = 1e-3,
        nesterov: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(dampening, 'dampening', 0.0, 1.0)
        self.validate_non_negative(trust_coefficient, 'trust_coefficient')

        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'dampening': dampening,
            'trust_coefficient': trust_coefficient,
            'nesterov': nesterov,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Lars'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_buffers: List[torch.Tensor],
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
