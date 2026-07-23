import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class RACS(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.9,
        alpha: float = 0.05,
        gamma: float = 1.01,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_positive(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RACS'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class Alice(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        betas: Betas = (0.9, 0.9, 0.999),
        alpha: float = 0.3,
        alpha_c: float = 0.4,
        update_interval: int = 200,
        rank: int = 256,
        gamma: float = 1.01,
        leading_basis: int = 40,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_range(alpha_c, 'alpha_c', 0.0, 1.0)
        self.validate_positive(update_interval, 'update_interval')
        self.validate_positive(rank, 'rank')
        self.validate_positive(gamma, 'gamma')
        self.validate_positive(leading_basis, 'leading_basis')
        self.validate_non_negative(rank - leading_basis, 'rank - leading_basis')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'alpha_c': alpha_c,
            'update_interval': update_interval,
            'rank': rank,
            'gamma': gamma,
            'leading_basis': leading_basis,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Alice'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def subspace_iteration(
        a: torch.Tensor, mat: torch.Tensor, num_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def switch(self, q: torch.Tensor, u_prev: torch.Tensor, rank: int, leading_basis: int) -> torch.Tensor:
        pass

    @staticmethod
    def compensation(
        grad: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        gamma: float,
        decay_rate: float,
        rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
