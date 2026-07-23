import math
from typing import Literal, Optional, Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5

MARS_TYPE = Literal['adamw', 'lion', 'shampoo']


class MARS(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        betas: Betas = (0.95, 0.99),
        gamma: float = 0.025,
        mars_type: MARS_TYPE = 'adamw',
        optimize_1d: bool = False,
        lr_1d: bool = 3e-3,
        betas_1d: Betas = (0.9, 0.95),
        weight_decay: float = 0.0,
        weight_decay_1d: float = 1e-1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        cautious: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(lr_1d)
        self.validate_betas(betas)
        self.validate_betas(betas_1d)
        self.validate_options(mars_type, 'mars_type', ['adamw', 'lion', 'shampoo'])
        self.validate_non_negative(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(weight_decay_1d, 'weight_decay_1d')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'lr_1d': lr_1d,
            'lr_1d_factor': lr_1d / lr,
            'betas': betas,
            'betas_1d': betas_1d,
            'mars_type': mars_type,
            'gamma': gamma,
            'optimize_1d': optimize_1d,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'cautious': cautious,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'MARS'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def optimize_mixed(
        self,
        grad: torch.Tensor,
        last_grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        betas: Tuple[int, int],
        gamma: float,
        mars_type: MARS_TYPE,
        is_grad_2d: bool,
        step: int,
        ams_bound: bool,
        cautious: bool,
        eps: float,
    ) -> torch.Tensor:
        pass

    def optimize_1d(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        betas: Tuple[int, int],
        step: int,
        ams_bound: bool,
        cautious: bool,
        eps: float,
    ) -> torch.Tensor:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
