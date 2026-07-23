import math
from typing import List, Optional, Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class AccSGD(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        kappa: float = 1000.0,
        xi: float = 10.0,
        constant: float = 0.7,
        weight_decay: float = 0.0,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(kappa, 'kappa')
        self.validate_non_negative(xi, 'xi')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_boundary(constant, boundary=1.0, bound_type='upper')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'kappa': kappa,
            'xi': xi,
            'constant': constant,
            'weight_decay': weight_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AccSGD'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class SGDW(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        dampening: float = 0.0,
        nesterov: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'dampening': dampening,
            'nesterov': nesterov,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDW'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_buffers: List[torch.Tensor],
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class ASGD(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        amplifier: float = 0.02,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        theta: float = 1.0,
        dampening: float = 1.0,
        eps: float = 1e-5,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(amplifier, 'amplifier')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'amplifier': amplifier,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'theta': theta,
            'dampening': dampening,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ASGD'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def get_norms_by_group(group: ParamGroup, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class SignSGD(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'beta', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SignSGD'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_buffers: List[torch.Tensor],
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class SGDSaI(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.has_warmup: bool = False
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SGDSaI'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def warmup_step(self, closure: Closure = None) -> Loss:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class VSGD(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        ghattg: float = 30.0,
        ps: float = 1e-8,
        tau1: float = 0.81,
        tau2: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(ghattg, 'ghattg')
        self.validate_non_negative(ps, 'ps')
        self.validate_non_negative(tau1, 'tau1')
        self.validate_non_negative(tau2, 'tau2')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'tau1': tau1,
            'tau2': tau2,
            'pa2': 2.0 * ps + 1.0 + 1e-4,
            'pbg2': 2.0 * ps,
            'pbhg2': 2.0 * ghattg * ps,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'VSGD'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
