from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, HutchinsonG, Loss, ParamGroup, ParamsT


class AdaHessian(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        hessian_power: float = 1.0,
        update_period: int = 1,
        num_samples: int = 1,
        hessian_distribution: HutchinsonG = 'rademacher',
        eps: float = 1e-16,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_range(hessian_power, 'Hessian Power', 0, 1, range_type='(]')

        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'hessian_power': hessian_power,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaHessian'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None, hessian: Optional[List[torch.Tensor]] = None) -> Loss:
        pass
