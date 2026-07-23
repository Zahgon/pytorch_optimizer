
import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm, to_real


class DAdaptAdaGrad(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.0,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 0.0,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'k': 0,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdaGrad'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:  # noqa: PLR0912, PLR0915
        pass


class DAdaptAdam(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.9, 0.999),
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        bias_correction: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'bias_correction': bias_correction,
            'step': 0,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdam'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class DAdaptSGD(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.9,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'd': d0,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'step': 0,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptSGD'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class DAdaptAdan(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.98, 0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        d0: float = 1e-6,
        growth_rate: float = float('inf'),
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'd': d0,
            'growth_rate': growth_rate,
            'k': 0,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptAdan'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class DAdaptLion(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.9, 0.999),
        d0: float = 1e-6,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'd': d0,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'step': 0,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'DAdaptLion'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
