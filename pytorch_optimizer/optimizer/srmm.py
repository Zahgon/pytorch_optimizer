from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class SRMM(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.01,
        beta: float = 0.5,
        memory_length: Optional[int] = 100,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'beta': beta, 'memory_length': memory_length}

        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'SRMM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
