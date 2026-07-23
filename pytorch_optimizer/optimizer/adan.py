import math
from typing import List, Optional, Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class Adan(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.98, 0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        max_grad_norm: float = 0.0,
        foreach: Optional[bool] = None,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_grad_norm, 'max_grad_norm')
        self.validate_non_negative(eps, 'eps')

        self.max_grad_norm = max_grad_norm
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'max_grad_norm': max_grad_norm,
            'foreach': foreach,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Adan'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.sqrt_().add_(self.defaults['eps'])

        return torch.clamp(self.defaults['max_grad_norm'] / global_grad_norm, max=1.0)

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        exp_avg_diffs: List[torch.Tensor],
        prev_grads: List[torch.Tensor],
        clip_global_grad_norm: Union[torch.Tensor, float],
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup, clip_global_grad_norm: Union[torch.Tensor, float]) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
