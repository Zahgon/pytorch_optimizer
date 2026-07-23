import math
from typing import Optional

import torch
from torch.nn import Parameter, ParameterList
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class CosineDecay:

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd: Optimizer = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper: LRScheduler = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        pass

    def get_death_rate(self, current_step: int) -> float:
        pass


class SPAM(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        density: float = 1.0,
        weight_decay: float = 0.0,
        warmup_epoch: int = 50,
        threshold: int = 5000,
        grad_accu_steps: int = 20,
        update_proj_gap: int = 500,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(warmup_epoch, 'warmup_epoch')
        self.validate_non_negative(density, 'density')
        self.validate_non_negative(threshold, 'threshold')
        self.validate_non_negative(grad_accu_steps, 'grad_accu_steps')
        self.validate_positive(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(eps, 'eps')

        self.density = density
        self.warmup_epoch = warmup_epoch
        self.threshold = threshold
        self.grad_accu_steps = grad_accu_steps
        self.update_proj_gap = update_proj_gap
        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps, **kwargs}

        super().__init__(params, defaults)

        self.warmup = CosineDecay(0.99, self.warmup_epoch)

        self.init_masks()

        self.state['total_step'] = 0
        self.state['current_step'] = self.warmup_epoch + 1

    @staticmethod
    def initialize_random_rank_boolean_tensor(m: int, n: int, density: float, device: torch.device) -> torch.Tensor:
        pass

    def update_mask_random(self, p: torch.Tensor, old_mask: torch.Tensor) -> torch.Tensor:
        pass

    def update_masks(self) -> None:
        pass

    def init_masks(self) -> None:
        pass

    def __str__(self) -> str:
        return 'SPAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class StableSPAM(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        gamma1: float = 0.7,
        gamma2: float = 0.9,
        theta: float = 0.999,
        t_max: Optional[int] = None,
        eta_min: float = 0.5,
        weight_decay: float = 0.0,
        update_proj_gap: int = 1000,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(eps, 'eps')

        self.gamma1: float = betas[0] if gamma1 == -1.0 else gamma1
        self.gamma2: float = gamma2
        self.theta: float = theta
        self.t_max = t_max
        self.update_proj_gap = update_proj_gap
        self.warmup = CosineDecay(1.0, t_max, eta_min=eta_min) if t_max is not None else None
        self.maximize = maximize

        self.total_step: int = 0

        defaults: Defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps, **kwargs}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableSPAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
