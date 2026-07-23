import math
from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class StableAdamW(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.99),
        kahan_sum: bool = True,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'kahan_sum': kahan_sum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableAdamW'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        kahan_comps: List[torch.Tensor],
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
