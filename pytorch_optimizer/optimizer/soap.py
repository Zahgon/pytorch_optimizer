import math
from itertools import chain
from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, DataFormat, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.shampoo_utils import merge_small_dims


class SOAP(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        betas: Betas = (0.95, 0.95),
        shampoo_beta: Optional[float] = None,
        weight_decay: float = 1e-2,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        correct_bias: bool = True,
        normalize_gradient: bool = False,
        data_format: DataFormat = 'channels_first',
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(shampoo_beta, 'shampoo_beta')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(precondition_frequency, 'precondition_frequency')
        self.validate_positive(max_precondition_dim, 'max_precondition_dim')
        self.validate_non_negative(eps, 'eps')

        self.data_format = data_format
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'shampoo_beta': shampoo_beta,
            'weight_decay': weight_decay,
            'precondition_frequency': precondition_frequency,
            'max_precondition_dim': max_precondition_dim,
            'merge_dims': merge_dims,
            'precondition_1d': precondition_1d,
            'correct_bias': correct_bias,
            'normalize_gradient': normalize_gradient,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SOAP'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def project(
        self,
        grad: torch.Tensor,
        state,
        merge_dims: bool = False,
        max_precondition_dim: int = 10000,
        project_type: str = 'forward',
    ) -> torch.Tensor:
        pass

    @staticmethod
    def get_orthogonal_matrix(mat: torch.Tensor) -> List[torch.Tensor]:
        pass

    def get_orthogonal_matrix_qr(self, state, max_precondition_dim: int = 10000, merge_dims: bool = False):
        pass

    @staticmethod
    def init_pre_conditioner(
        grad,
        state,
        precondition_frequency: int = 10,
        shampoo_beta: float = 0.95,
        max_precondition_dim: int = 10000,
        precondition_1d: bool = False,
        merge_dims: bool = False,
    ) -> None:
        pass

    def update_pre_conditioner(
        self,
        grad,
        state,
        step: int,
        max_precondition_dim: int = 10000,
        precondition_1d: bool = False,
        merge_dims: bool = False,
    ) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
