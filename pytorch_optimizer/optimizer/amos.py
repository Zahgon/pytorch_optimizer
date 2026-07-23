import math
from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.foreach_utils import foreach_rsqrt_


class Amos(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.999,
        momentum: float = 0.0,
        extra_l2: float = 0.0,
        c_coef: float = 0.25,
        d_coef: float = 0.25,
        foreach: Optional[bool] = None,
        eps: float = 1e-18,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(extra_l2, 'extra_l2')
        self.validate_non_negative(eps, 'eps')

        self.c_coef = c_coef
        self.d_coef = d_coef
        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'momentum': momentum,
            'extra_l2': extra_l2,
            'foreach': foreach,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Amos'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    @staticmethod
    def get_scale(p: torch.Tensor) -> float:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        decays: List[torch.Tensor],
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
