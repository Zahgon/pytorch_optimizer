from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, HutchinsonG, Loss, ParamGroup, ParamsT


class SophiaH(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 6e-2,
        betas: Betas = (0.96, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        p: float = 1e-2,
        update_period: int = 10,
        num_samples: int = 1,
        hessian_distribution: HutchinsonG = 'gaussian',
        eps: float = 1e-12,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(p, 'p (gradient clip)')
        self.validate_step(update_period, 'update_period')
        self.validate_positive(num_samples, 'num_samples')
        self.validate_options(hessian_distribution, 'hessian_distribution', ['gaussian', 'rademacher'])
        self.validate_non_negative(eps, 'eps')

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
            'p': p,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SophiaH'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None, hessian: Optional[List[torch.Tensor]] = None) -> Loss:
        pass
