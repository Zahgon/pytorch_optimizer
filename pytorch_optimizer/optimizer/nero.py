from typing import List

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


def channel_view(x: torch.Tensor) -> torch.Tensor:
    """Do channel view."""
    return x.view(x.size()[0], -1)


def neuron_norm(x: torch.Tensor) -> torch.Tensor:
    """Get norm of the tensor."""
    if x.dim() <= 1:
        return x.abs()

    view_shape: List[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).norm(dim=1).view(*view_shape)


def neuron_mean(x: torch.Tensor) -> torch.Tensor:
    """Get mean of the tensor."""
    if x.dim() <= 1:
        raise ValueError('[-] neuron_mean not defined on 1D tensors.')

    view_shape: List[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).mean(dim=1).view(*view_shape)


class Nero(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.01,
        beta: float = 0.999,
        constraints: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'beta': beta, 'constraints': constraints, 'eps': eps}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Nero'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
