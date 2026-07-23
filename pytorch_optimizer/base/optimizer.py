import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.base.type import (
    Betas,
    Closure,
    Defaults,
    HutchinsonG,
    Loss,
    OptimizerInstanceOrClass,
    ParamGroup,
    ParamsT,
    State,
)
from pytorch_optimizer.optimizer.foreach_utils import foreach_rsqrt_


class BaseOptimizer(ABC, Optimizer):

    def __init__(self, params: ParamsT, defaults: Defaults) -> None:
        super().__init__(params, defaults)

    @staticmethod
    def load_optimizer(optimizer: OptimizerInstanceOrClass, **kwargs) -> Optimizer:
        """Build torch.optim.Optimizer class."""
        if isinstance(optimizer, Optimizer):
            return optimizer

        if 'params' in kwargs:
            params = kwargs.pop('params')
            return optimizer(params, **kwargs)

        raise ValueError('need to pass `params` when you pass the `torch.optim.Optimizer` instance.')

    @staticmethod
    @torch.no_grad()
    def set_hessian(param_groups: ParamsT, state: State, hessian: List[torch.Tensor]) -> None:
        pass

    @staticmethod
    def zero_hessian(param_groups: ParamsT, state: State, pre_zero: bool = True) -> None:
        pass

    @staticmethod
    @torch.no_grad()
    def compute_hutchinson_hessian(
        param_groups: ParamsT,
        state: State,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: HutchinsonG = 'gaussian',
    ) -> None:
        pass

    @staticmethod
    def apply_weight_decay(
        p: torch.Tensor,
        grad: Optional[torch.Tensor],
        lr: float,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
        ratio: Optional[float] = None,
    ) -> None:
        """Apply weight decay in an in-place manner.

        Args:
            p (torch.Tensor): Parameter tensor to apply weight decay to.
            grad (torch.Tensor): Gradient tensor of parameter p.
            lr (float): Learning rate to scale the update.
            weight_decay (float): Weight decay coefficient (L2 penalty).
            weight_decouple (bool): If True, applies decoupled weight decay as in AdamW.
            fixed_decay (bool): If True, fixes weight decay to not depend on learning rate.
            ratio (Optional[float]): Optional scaling factor for weight decay.

        """
        if weight_decouple:
            p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) * (ratio if ratio is not None else 1.0))
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)

    @staticmethod
    def apply_cautious_weight_decay(
        p: torch.Tensor,
        update: torch.Tensor,
        lr: float,
        weight_decay: float,
    ) -> None:
        pass

    @staticmethod
    def apply_ams_bound(
        ams_bound: bool,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        eps: float,
        exp_avg_sq_eps: float = 1e-15,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def debias(beta: float, step: int) -> float:
        pass

    @staticmethod
    def debias_beta(beta: float, step: int) -> float:
        pass

    @staticmethod
    def apply_adam_debias(adam_debias: bool, step_size: float, bias_correction1: float) -> float:
        pass

    @staticmethod
    def get_rectify_step_size(
        is_rectify: bool,
        step: int,
        lr: float,
        beta2: float,
        n_sma_threshold: int,
        degenerated_to_sgd: bool,
    ) -> Tuple[float, float]:
        pass

    @staticmethod
    def get_adanorm_gradient(
        grad: torch.Tensor, adanorm: bool, exp_grad_norm: Optional[torch.Tensor] = None, r: Optional[float] = 0.95
    ) -> torch.Tensor:
        r"""Get AdaNorm gradient.

        Args:
            grad (torch.Tensor): Gradient.
            adanorm (bool): Whether to use the AdaNorm variant.
            exp_grad_norm (Optional[torch.Tensor]): Exponential moving average of gradient norm.
            r (Optional[float]): EMA factor; between 0.9 and 0.99 is preferred.

        """
        if not adanorm or exp_grad_norm is None:
            return grad

        if r is None:
            r = 0.95

        grad_norm = torch.linalg.norm(grad)

        exp_grad_norm.mul(r).add_(grad_norm, alpha=1.0 - r)

        return grad.mul(exp_grad_norm).div_(grad_norm) if exp_grad_norm > grad_norm else grad

    @staticmethod
    def get_rms(x: Union[List[torch.Tensor], torch.Tensor]) -> Union[List[torch.Tensor], torch.Tensor]:
        pass

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: Union[List[torch.Tensor], torch.Tensor],
        exp_avg_sq_col: Union[List[torch.Tensor], torch.Tensor],
        output: Union[List[torch.Tensor], torch.Tensor],
    ) -> None:
        pass

    @staticmethod
    def apply_cautious(update: torch.Tensor, grad: torch.Tensor) -> None:
        pass

    @staticmethod
    def can_use_foreach(group: ParamGroup, foreach: Optional[bool]) -> bool:
        pass

    @staticmethod
    def collect_trainable_params(
        group: ParamGroup,
        state: State,
        state_keys: Optional[List[str]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        pass

    @staticmethod
    def apply_weight_decay_foreach(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        lr: Union[List[float], List[torch.Tensor], float, torch.Tensor],
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
    ) -> None:
        pass

    @staticmethod
    def get_stable_adamw_rms(grad: torch.Tensor, exp_avg_sq: torch.Tensor, eps: float = 1e-16) -> float:
        pass

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = '[)') -> None:
        pass

    @staticmethod
    def validate_non_negative(x: Optional[float], name: str) -> None:
        pass

    @staticmethod
    def validate_non_positive(x: Optional[float], name: str) -> None:
        pass

    @staticmethod
    def validate_positive(x: Union[float, int], name: str) -> None:
        pass

    @staticmethod
    def validate_boundary(constant: float, boundary: float, bound_type: str = 'upper') -> None:
        pass

    @staticmethod
    def validate_step(step: int, step_type: str) -> None:
        pass

    @staticmethod
    def validate_options(x: str, name: str, options: List[str]) -> None:
        pass

    @staticmethod
    def validate_learning_rate(learning_rate: Optional[float]) -> None:
        pass

    @staticmethod
    def validate_mod(x: int, y: int) -> None:
        pass

    def validate_betas(
        self,
        betas: Union[Betas, Tuple[None, float]],
        beta_range_type: str = '[)',
        beta3_range_type: str = '[]',
    ) -> None:
        pass

    def validate_nus(self, nus: Union[float, Tuple[float, float]]) -> None:
        pass

    @abstractmethod
    def init_group(self, group: ParamGroup, **kwargs) -> None:  # pragma: no cover
        """Initialize the group of the optimizer and return is_complex."""
        return

    @staticmethod
    def view_as_real(param, *state_and_grads) -> tuple:
        """View imaginary tensors as real tensors."""
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            state_and_grads = tuple(
                torch.view_as_real(s) if (s is not None and torch.is_complex(s)) else s if s is not None else None
                for s in state_and_grads
            )

        return param, *state_and_grads

    @staticmethod
    def maximize_gradient(grad: torch.Tensor, maximize: bool = False) -> None:
        """Maximize the objective with respect to the params, instead of minimizing."""
        if maximize:
            grad.neg_()

    def step(self, closure: Closure = None) -> Loss:  # pragma: no cover
        raise NotImplementedError
