from collections import defaultdict
from typing import Callable, Dict, List

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import (
    Betas,
    Closure,
    Defaults,
    Loss,
    OptimizerInstanceOrClass,
    ParamGroup,
    ParamsT,
    State,
)


class ScheduleFreeSGD(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'warmup_steps': warmup_steps,
            'eps': eps,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
        }

        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'ScheduleFreeSGD'

    def eval(self):
        pass

    def train(self):
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class ScheduleFreeAdamW(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        decoupling_c: int = 0,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(decoupling_c, 'decoupling_c')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'warmup_steps': warmup_steps,
            'decoupling_c': decoupling_c,
            'ams_bound': ams_bound,
            'eps': eps,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'use_palm': kwargs.get('use_palm', False),
        }

        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'ScheduleFreeAdamW'

    def eval(self):
        pass

    def train(self):
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class ScheduleFreeRAdam(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2.5e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        silent_sgd_phase: bool = True,
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
            'silent_sgd_phase': silent_sgd_phase,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'eps': eps,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'use_palm': kwargs.get('use_palm', False),
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ScheduleFreeRAdam'

    def eval(self):
        pass

    def train(self):
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class ScheduleFreeWrapper(BaseOptimizer):

    def __init__(
        self,
        optimizer: OptimizerInstanceOrClass,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '[)')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.r = r
        self.weight_lr_power = weight_lr_power
        self.train_mode: bool = False
        self.maximize = maximize

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.state: State = defaultdict(dict)
        self.defaults: Defaults = self.optimizer.defaults

    def __str__(self) -> str:
        return 'ScheduleFree'

    @property
    def param_groups(self):
        pass

    def __getstate__(self):
        return {'state': self.state, 'optimizer': self.optimizer}

    def add_param_group(self, param_group):
        pass

    def state_dict(self) -> State:
        pass

    def load_state_dict(self, state: State) -> None:
        pass

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    @torch.no_grad()
    def eval(self):
        pass

    @torch.no_grad()
    def train(self):
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
