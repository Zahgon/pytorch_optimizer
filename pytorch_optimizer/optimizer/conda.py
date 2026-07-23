import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.galore_utils import PROJECTION_TYPE, GaLoreProjector


class Conda(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        update_proj_gap: int = 2000,
        scale: float = 1.0,
        projection_type: PROJECTION_TYPE = 'std',
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_positive(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'update_proj_gap': update_proj_gap,
            'scale': scale,
            'projection_type': projection_type,
            'eps': eps,
            **kwargs,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Conda'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
