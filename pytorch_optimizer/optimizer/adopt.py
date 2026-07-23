import math
from typing import Callable, List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class ADOPT(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.9999),
        clip_lambda: Callable[[float], float] = lambda step: math.pow(step, 0.25),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        foreach: Optional[bool] = None,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.clip_lambda = clip_lambda
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'foreach': foreach,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPT'

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
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
