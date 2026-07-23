import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.agc import agc


class Ranger25(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.98, 0.9999),
        weight_decay: float = 1e-3,
        alpha: float = 5.0,
        t_alpha_beta3: Optional[float] = None,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        cautious: bool = True,
        stable_adamw: bool = True,
        orthograd: bool = True,
        eps: Optional[float] = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(t_alpha_beta3, 't_alpha_beta3')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(lookahead_merge_time, 'lookahead_merge_time')
        self.validate_range(lookahead_blending_alpha, 'lookahead_blending_alpha', 0.0, 1.0, '[]')
        self.validate_non_negative(eps, 'eps')

        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.cautious = cautious
        self.stable_adamw: bool = stable_adamw if isinstance(eps, float) else False
        self.orthograd = orthograd
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'eps': eps if (eps is not None) or (eps is None and not stable_adamw) else 1e-8,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Ranger25'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        pass

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float) -> float:
        pass

    @torch.no_grad()
    def apply_orthogonal_gradients(self, params, eps: float = 1e-16) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
