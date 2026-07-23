from typing import Literal, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT

Mode = Literal['g', 'm', 'c']


class BCOS(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.9,
        beta2: Optional[float] = None,
        mode: Mode = 'c',
        simple_cond: bool = False,
        weight_decay: float = 0.1,
        weight_decouple: bool = True,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0)
        self.validate_options(mode, 'mode', ['g', 'm', 'c'])
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.mode = mode
        self.simple_cond = simple_cond
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'beta2': beta2,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            **kwargs,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'BCOS'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def compute_v(self, grad: torch.Tensor, m: torch.Tensor, beta: float, beta2: Optional[float]) -> torch.Tensor:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
