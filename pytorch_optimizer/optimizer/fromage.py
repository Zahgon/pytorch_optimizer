
import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class Fromage(BaseOptimizer):

    def __init__(
        self, params: ParamsT, lr: float = 1e-2, p_bound: Optional[float] = None, maximize: bool = False, **kwargs
    ):
        self.validate_learning_rate(lr)

        self.p_bound = p_bound
        self.maximize = maximize

        defaults: Defaults = {'lr': lr}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Fromage'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
