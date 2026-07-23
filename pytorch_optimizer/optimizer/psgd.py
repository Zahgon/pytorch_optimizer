import math
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.psgd_utils import norm_lower_bound

MEMORY_SAVE_MODE_TYPE = Literal['one_diag', 'smart_one_diag', 'all_diag']


def precondition_update_prob_schedule(
    max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.001, flat_start: int = 500
) -> Callable[[int], torch.Tensor]:
    pass


class Kron(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        pre_conditioner_update_probability: Optional[Callable[[int], torch.Tensor]] = None,
        max_size_triangular: int = 8192,
        min_ndim_triangular: int = 2,
        memory_save_mode: Optional[MEMORY_SAVE_MODE_TYPE] = None,
        momentum_into_precondition_update: bool = True,
        mu_dtype: Optional[torch.dtype] = None,
        precondition_dtype: Optional[torch.dtype] = torch.float32,
        balance_prob: float = 0.01,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        if pre_conditioner_update_probability is None:
            pre_conditioner_update_probability = precondition_update_prob_schedule()

        self.balance_prob: float = balance_prob
        self.eps: float = torch.finfo(torch.bfloat16).tiny
        self.prob_step: int = 0
        self.update_counter: int = 0
        self.maximize = maximize

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'pre_conditioner_update_probability': pre_conditioner_update_probability,
            'max_size_triangular': max_size_triangular,
            'min_ndim_triangular': min_ndim_triangular,
            'memory_save_mode': memory_save_mode,
            'momentum_into_precondition_update': momentum_into_precondition_update,
            'precondition_lr': 1e-1,
            'precondition_init_scale': 1.0,
            'mu_dtype': mu_dtype,
            'precondition_dtype': precondition_dtype,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Kron'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


def initialize_q_expressions(
    t: torch.Tensor,
    scale: float,
    max_size: int,
    min_ndim_triangular: int,
    memory_save_mode: Optional[MEMORY_SAVE_MODE_TYPE],
    dtype: Optional[torch.dtype] = None,
) -> Tuple[List[torch.Tensor], Tuple[str, List[str], str]]:
    """Initialize Q expressions.

    For a scalar or tensor t, we initialize its pre-conditioner Q and reusable einsum expressions for updating Q and
    pre-conditioning gradient.
    """
    letters: str = ascii_lowercase + ascii_uppercase

    t_dtype: torch.dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:
        qs: List[torch.Tensor] = [scale * torch.ones_like(t, dtype=t_dtype)]
        expressions_a: str = ',->'
        expression_gr: List[str] = [',->']
        expression_r: str = ',,->'

        return qs, (expressions_a, expression_gr, expression_r)

    if len(shape) > 13:
        raise ValueError(f'got tensor with dim {len(t.shape)}. Einstein runs out of letters!')

    scale = math.pow(scale, 1.0 / len(shape))

    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == 'one_diag':
        dim_diag = [False for _ in shape]
        dim_diag[np.argsort(shape)[::-1][0]] = True
    elif memory_save_mode == 'smart_one_diag':
        dim_diag = [False for _ in shape]
        sorted_shape = sorted(shape)
        if len(shape) >= 2 and sorted_shape[-1] > sorted_shape[-2]:
            dim_diag[np.argsort(shape)[::-1][0]] = True
    elif memory_save_mode == 'all_diag':
        dim_diag = [True for _ in shape]
    else:
        raise NotImplementedError(
            f'invalid memory_save_mode {memory_save_mode}. '
            'it must be one of [None, `one_diag`, `smart_one_diag`, `all_diag`]'
        )

    qs: List[torch.Tensor] = []
    expr_gr = []
    piece_1a, piece_2a, piece_3a = [], '', ''
    piece_1p, piece_2p, piece_3p, piece_4p = [], [], '', ''
    for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
            qs.append(scale * torch.ones(size, dtype=t_dtype, device=t.device))

            piece_1a.append(letters[i])
            piece_2a += letters[i]
            piece_3a += letters[i]

            piece1: str = ''.join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            expr_gr.append(f'{piece1},{piece1}->{letters[i + 13]}')

            piece_1p.append(letters[i + 13])
            piece_2p.append(letters[i + 13])
            piece_3p += letters[i + 13]
            piece_4p += letters[i + 13]
        else:
            qs.append(scale * torch.eye(size, dtype=t_dtype, device=t.device))

            piece_1a.append(letters[i] + letters[i + 13])
            piece_2a += letters[i + 13]
            piece_3a += letters[i]

            piece1: str = ''.join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            piece2: str = ''.join([(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))])
            expr_gr.append(f'{piece1},{piece2}->{letters[i + 13]}{letters[i + 26]}')

            a, b, c = letters[i], letters[i + 13], letters[i + 26]
            piece_1p.append(a + b)
            piece_2p.append(a + c)
            piece_3p += c
            piece_4p += b

    expr_a: str = ','.join(piece_1a) + f',{piece_2a}->{piece_3a}'
    expr_r: str = ','.join(piece_1p) + ',' + ','.join(piece_2p) + f',{piece_3p}->{piece_4p}'

    return qs, (expr_a, expr_gr, expr_r)


def balance_q(q_in: List[torch.Tensor]) -> None:
    pass


def solve_triangular_right(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    pass


def get_a_and_conj_b(
    expr_a: List[str], g: torch.Tensor, qs: List[torch.Tensor], v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass


def get_q_terms(expr_gs: List[str], a: torch.Tensor, conj_b: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pass


def update_precondition(
    qs: List[torch.Tensor],
    expressions: List[Tuple[str, List[str], str]],
    v: torch.Tensor,
    g: torch.Tensor,
    step: int,
    eps: float,
) -> None:
    pass


def get_precondition_grad(qs: List[torch.Tensor], expressions: List[str], g: torch.Tensor) -> torch.Tensor:
    pass
