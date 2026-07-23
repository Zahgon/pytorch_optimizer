import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT

GROUP_SIZE: int = 32
VALID_MASTER_WEIGHT_BITS: Tuple[Optional[int], ...] = (None, 24, 32)
BITS_TO_BYTES: Dict[Optional[int], int] = {None: 0, 24: 3, 32: 4}
DTYPE_WIDTHS: Dict[torch.dtype, int] = {
    torch.int8: 1,
    torch.int16: 2,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}


def _quantized_key(name: str) -> str:
    pass


def _scales_key(name: str) -> str:
    pass


def quantize_state(
    tensor: torch.Tensor,
    signed: bool = True,
    sqrt: bool = False,
    softsign: bool = True,
    group_size: int = GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass


def dequantize_state(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    signed: bool = True,
    sqrt: bool = False,
    softsign: bool = True,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    pass


def _state_spec(name: str) -> Tuple[bool, bool, bool]:
    pass


def materialize_state(state: Dict[str, Any], name: str) -> torch.Tensor:
    pass


def store_state(state: Dict[str, Any], name: str, tensor: torch.Tensor, quantize: bool, dtype: torch.dtype) -> None:
    pass


def ulp_scale(narrow: torch.Tensor) -> torch.Tensor:
    """Scale the parameter."""
    next_values = torch.nextafter(narrow.abs(), torch.full_like(narrow, float('inf')))
    return next_values.sub(narrow.abs()).to(torch.float32).mul_(0.5).clamp_min_(torch.finfo(torch.float32).tiny)


def compute_ecc_bits(fp32_param: torch.Tensor, narrow_param: torch.Tensor, master_byte_width: int) -> torch.Tensor:
    """Compute ECC bits."""
    if fp32_param.dtype != torch.float32:
        raise ValueError(f'fp32_param must be float32, got {fp32_param.dtype}')
    if narrow_param.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f'narrow_param must be bf16 or fp16, got {narrow_param.dtype}')

    error_bytes = master_byte_width - narrow_param.element_size()
    if error_bytes == 1:
        error_dtype, signed_max = torch.int8, 127.0
    elif error_bytes == 2:
        error_dtype, signed_max = torch.int16, 32767.0
    else:
        raise ValueError(f'unsupported master byte width: {master_byte_width}')

    normalized_error = (fp32_param - narrow_param.to(torch.float32)) / ulp_scale(narrow_param)
    return torch.round(normalized_error.clamp_(-1.0, 1.0) * signed_max).to(error_dtype)


def reconstruct_fp32_param(param: torch.Tensor, error_bits: torch.Tensor) -> torch.Tensor:
    """Reconstruct fp32 parameters."""
    if param.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f'param must be bf16 or fp16, got {param.dtype}')
    if error_bits.dtype == torch.int8:
        signed_max = 127.0
    elif error_bits.dtype == torch.int16:
        signed_max = 32767.0
    else:
        raise ValueError(f'error_bits must be int8 or int16, got {error_bits.dtype}')

    return param.to(torch.float32).add(error_bits.to(torch.float32).div(signed_max).mul(ulp_scale(param)))


class FlashAdamW(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-2,
        decouple_lr: bool = False,
        quantize: bool = True,
        compress_state_dict: bool = True,
        master_weight_bits: Optional[int] = None,
        check_numerics: bool = False,
        fused: bool = False,
        maximize: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(weight_decay, 'weight_decay')

        if master_weight_bits not in VALID_MASTER_WEIGHT_BITS:
            raise ValueError(f'master_weight_bits must be one of {VALID_MASTER_WEIGHT_BITS}')

        if fused:
            raise NotImplementedError('FlashAdamW fused Triton kernels are not available in this portable backend')

        self.maximize = maximize
        self.compress_state_dict = compress_state_dict
        self.check_numerics = check_numerics
        self.master_byte_width = BITS_TO_BYTES[master_weight_bits]
        self.param_absmax: Dict[int, float] = {}

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'decouple_lr': decouple_lr,
            'quantize': quantize,
            'master_byte_width': self.master_byte_width,
            **kwargs,
        }

        super().__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('initial_lr', group['lr'])

        if master_weight_bits is not None and all(
            p.dtype == torch.float32 for group in self.param_groups for p in group['params']
        ):
            raise ValueError('master_weight_bits has no effect when all parameters are fp32')

    def __str__(self) -> str:
        return 'FlashAdamW'

    def maybe_check_numerics(self, p: torch.Tensor, lr: float, master_byte_width: int) -> None:
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def get_param_fp32(p: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        pass

    @staticmethod
    def set_param_fp32(p: torch.Tensor, state: Dict[str, Any], value: torch.Tensor, master_byte_width: int) -> None:
        pass

    def recompute_param_stats(self) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass

    def state_dict(self) -> Dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def get_fp32_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        pass

    @torch.inference_mode()
    def set_fp32_model_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        pass
