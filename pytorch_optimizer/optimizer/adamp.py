import math
from typing import Callable, List, Tuple

import torch
from torch.nn.functional import cosine_similarity

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient


def channel_view(x: torch.Tensor) -> torch.Tensor:
    """Do channel view."""
    return x.view(x.size()[0], -1)


def layer_view(x: torch.Tensor) -> torch.Tensor:
    pass


def cosine_similarity_by_view(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    view_func: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    pass


def projection(
    p: torch.Tensor,
    grad: torch.Tensor,
    perturb: torch.Tensor,
    delta: float,
    wd_ratio: float,
    eps: float,
) -> Tuple[torch.Tensor, float]:
    pass


class SGDP(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(wd_ratio, 'wd_ratio', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'momentum': momentum,
            'dampening': dampening,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDP'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class AdamP(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(wd_ratio, 'wd_ratio', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamP'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
