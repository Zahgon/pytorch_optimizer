import torch
from torch.nn.functional import softmax

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class Adalite(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-2,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        g_norm_min: float = 1e-10,
        ratio_min: float = 1e-4,
        tau: float = 1.0,
        eps1: float = 1e-6,
        eps2: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'g_norm_min': g_norm_min,
            'ratio_min': ratio_min,
            'tau': tau,
            'eps1': eps1,
            'eps2': eps2,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Adalite'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
