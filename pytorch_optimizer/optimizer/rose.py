from typing import Union

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils import copy_stochastic


class ROSE(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        wd_schedule: Union[bool, float] = False,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        centralize: bool = True,
        stabilize: bool = True,
        bf16_sr: bool = True,
        compute_dtype: torch.dtype = torch.float64,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        if bf16_sr and compute_dtype not in (torch.float32, torch.float64, None):
            raise ValueError(f'bf16_sr=True has no useful effect when compute_dtype is {compute_dtype}.')

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'wd_schedule': wd_schedule,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'centralize': centralize,
            'stabilize': stabilize,
            'bf16_sr': bf16_sr,
            'compute_dtype': compute_dtype,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ROSE'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
