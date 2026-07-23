import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class CAME(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-4,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-16,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps1': eps1,
            'eps2': eps2,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CAME'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        pass

    @staticmethod
    def get_rms(x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ):
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
