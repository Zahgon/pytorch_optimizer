from collections import deque
from typing import Callable, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class AdaShift(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        keep_num: int = 10,
        reduce_func: Optional[Callable] = torch.max,
        eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_positive(keep_num, 'keep_num')
        self.validate_non_negative(eps, 'eps')

        self.reduce_func: Callable = reduce_func if reduce_func is not None else lambda x: x
        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'betas': betas, 'keep_num': keep_num, 'eps': eps, **kwargs}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaShift'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
