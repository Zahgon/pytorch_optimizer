import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class AdaGC(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        beta: float = 0.98,
        lambda_abs: float = 1.0,
        lambda_rel: float = 1.05,
        warmup_steps: int = 100,
        weight_decay: float = 1e-1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(beta, 'beta', 0.0, 1.0, '[)')
        self.validate_positive(lambda_abs, 'lambda_abs')
        self.validate_positive(lambda_rel, 'lambda_rel')
        self.validate_non_negative(warmup_steps, 'warmup_steps')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'beta': beta,
            'lambda_abs': lambda_abs,
            'lambda_rel': lambda_rel,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaGC'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
