import math
from enum import IntEnum
from typing import Dict, List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5


class LMONorm(IntEnum):

    NONE = 0
    AUTO = 1
    SPECTRAL = 2
    SPECTRALCONV = 3
    SIGN = 4
    BIAS = 5
    COL = 6
    ROW = 7


class Norm:

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor) -> torch.Tensor:
        pass


class Col(Norm):

    def __init__(self, normalized: bool = False, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pass


class Row(Norm):

    def __init__(self, normalized: bool = True, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pass


class BiasRMS(Norm):

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pass


class SpectralConv(Norm):

    def __init__(self, num_steps: int = 5) -> None:
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pass


class Spectral(Norm):

    def __init__(self, max_scale: bool = False, normalize: bool = True, num_steps: int = 5) -> None:
        self.max_scale = max_scale
        self.normalize = normalize
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pass


class Sign(Norm):

    def __init__(self, zero_init: bool = False, normalize: bool = True) -> None:
        self.zero_init = zero_init
        self.normalize = normalize

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pass


class Auto(Norm):

    def init(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def lmo(self, grad: torch.Tensor) -> torch.Tensor:
        pass


def build_lmo_norm(norm_type: int, **kwargs) -> Norm:  # noqa: PLR0911
    """Build LMONorm by given norm_type."""
    if norm_type == LMONorm.AUTO:
        return Auto()
    if norm_type == LMONorm.SPECTRAL:
        return Spectral(**kwargs)
    if norm_type == LMONorm.SPECTRALCONV:
        return SpectralConv(**kwargs)
    if norm_type == LMONorm.SIGN:
        return Sign(**kwargs)
    if norm_type == LMONorm.BIAS:
        return BiasRMS()
    if norm_type == LMONorm.COL:
        return Col(**kwargs)
    if norm_type == LMONorm.ROW:
        return Row(**kwargs)
    return Norm()


class SCION(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.1,
        constraint: bool = False,
        norm_type: int = LMONorm.AUTO,
        norm_kwargs: Optional[Dict] = None,
        scale: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '(]')
        self.validate_positive(scale, 'scale')

        self.foreach = foreach
        self.maximize = maximize

        if norm_kwargs is None:
            norm_kwargs = {}

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'constraint': constraint,
            'norm_type': norm_type,
            'norm_kwargs': norm_kwargs,
            'scale': scale,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SCION'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def init(self):
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        norm: Norm,
        ds: List[torch.Tensor],
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup, norm: Norm) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class SCIONLight(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.1,
        constraint: bool = False,
        norm_type: int = LMONorm.AUTO,
        norm_kwargs: Optional[Dict] = None,
        scale: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '(]')
        self.validate_positive(scale, 'scale')

        self.foreach = foreach
        self.maximize = maximize

        if norm_kwargs is None:
            norm_kwargs = {}

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'constraint': constraint,
            'norm_type': norm_type,
            'norm_kwargs': norm_kwargs,
            'scale': scale,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'foreach': foreach,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SCIONLight'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def init(self):
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        norm: Norm,
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup, norm: Norm) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
