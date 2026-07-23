import math
from typing import Literal, Optional

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.galore_utils import GaLoreProjector

SCALE_TYPE = Literal['channel', 'tensor']


class ApolloDQN(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        init_lr: Optional[float] = 1e-5,
        beta: float = 0.9,
        rebound: str = 'constant',
        weight_decay: float = 0.0,
        weight_decay_type: str = 'l2',
        warmup_steps: int = 500,
        eps: float = 1e-4,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_options(rebound, 'rebound', ['constant', 'belief'])
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_options(weight_decay_type, 'weight_decay_type', ['l2', 'decoupled', 'stable'])
        self.validate_non_negative(eps, 'eps')

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.init_lr: float = init_lr if init_lr is not None else lr / 1000.0
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'init_lr': self.init_lr,
            'beta': beta,
            'rebound': rebound,
            'weight_decay': weight_decay,
            'weight_decay_type': weight_decay_type,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ApolloDQN'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class APOLLO(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: Betas = (0.9, 0.999),
        scale_type: SCALE_TYPE = 'tensor',
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        correct_bias: bool = True,
        eps: float = 1e-6,
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
            'scale_type': scale_type,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'correct_bias': correct_bias,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'APOLLO'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
