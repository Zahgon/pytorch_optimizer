import math
from typing import List, Optional, Sequence, Tuple, Union

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.foreach_utils import foreach_rsqrt


class AdaFactor(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: Optional[float] = 1e-3,
        betas: Union[Tuple[None, float], Tuple[float, float], Tuple[float, float, float]] = (0.9, 0.999),
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        momentum_dtype: torch.dtype = torch.bfloat16,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.decay_rate = decay_rate
        self.clip_threshold = clip_threshold
        self.eps1: float = eps1 if momentum_dtype != torch.float16 else 1e-7
        self.eps2 = eps2
        self.momentum_dtype = momentum_dtype
        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'scale_parameter': scale_parameter,
            'relative_step': relative_step,
            'warmup_init': warmup_init,
            'eps1': eps1,
            'eps2': eps2,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaFactor'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def get_relative_step_size(lr: float, step: int, relative_step: bool, warmup_init: bool) -> float:
        pass

    def get_lr(
        self,
        relative_step_size: Union[torch.Tensor, float],
        rms: Union[List[torch.Tensor], torch.Tensor, float],
        scale_parameter: bool,
    ) -> Union[Sequence[torch.Tensor], torch.Tensor, float]:
        pass

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sq_rows: List[torch.Tensor],
        exp_avg_sq_cols: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        exp_avg_sq_hats: List[torch.Tensor],
        beta1: float,
        beta2_t: float,
        relative_step_size: float,
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup, beta1: float, beta2_t: float, relative_step_size: float) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
