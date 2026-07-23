import functools
import math
import operator
import re
import warnings
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn
from torch.distributed import all_reduce
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.type import Closure, Loss, ParamsT


def parse_pytorch_version(version_string: str) -> List[int]:
    """Parse a PyTorch version string."""
    match = re.match(r'(\d+\.\d+\.\d+)', version_string)
    if not match:
        raise ValueError(f'invalid version string format: {version_string}')

    return [int(x) for x in match.group(1).split('.')]


def compare_versions(v1: str, v2: str) -> bool:
    """Compare two PyTorch versions."""
    return parse_pytorch_version(v1) >= parse_pytorch_version(v2)


HAS_TRANSFORMERS: bool = find_spec('transformers') is not None
TORCH_VERSION_AT_LEAST_2_4: bool = compare_versions(torch.__version__, '2.4.0')
TORCH_VERSION_AT_LEAST_2_8: bool = compare_versions(torch.__version__, '2.8.0')

if HAS_TRANSFORMERS:  # pragma: no cover
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
    except ImportError:
        from transformers.deepspeed import is_deepspeed_zero3_enabled
else:

    def is_deepspeed_zero3_enabled() -> bool:
        """Check if DeepSpeed zero3 is enabled."""
        if HAS_TRANSFORMERS:
            return is_deepspeed_zero3_enabled()  # pragma: no cover

        warnings.warn(
            'you need to install `transformers` to use `is_deepspeed_zero3_enabled` function. it will return False.',
            category=ImportWarning,
            stacklevel=2,
        )

        return False


class CPUOffloadOptimizer:  # pragma: no cover

    def __init__(
        self,
        params: ParamsT,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        *,
        offload_gradients: bool = False,
        **kwargs,
    ) -> None:
        if optimizer_class is torch.optim.AdamW and TORCH_VERSION_AT_LEAST_2_4 and 'fused' not in kwargs:
            kwargs.update(fused=True)

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError('optimizer got an empty parameter list')

        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        self.param_cuda2cpu_map: Dict[torch.Tensor, torch.Tensor] = {}
        self.optim_dict: Dict[torch.Tensor, Optimizer] = {}
        self.stream = torch.cuda.Stream()

        self.queue = {}

        def backward_hook(p_cuda: torch.Tensor) -> None:
            pass

        for param_group in param_groups:
            params = param_group.get('params', None)  # type: ignore
            if params is None:
                continue

            for p_cuda in params:
                p_cpu = torch.empty_like(p_cuda, device='cpu', pin_memory=True)
                p_cpu.grad = torch.empty_like(p_cpu, pin_memory=True)

                p_cpu.copy_(p_cuda.detach(), non_blocking=True)
                self.param_cuda2cpu_map[p_cuda] = p_cpu

                p_cuda.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[p_cuda] = optimizer_class([{'params': p_cpu, **param_group}], **kwargs)  # type: ignore

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass

    def zero_grad(self, _: bool = True) -> None:
        pass

    @property
    def param_groups(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass


class StochasticAccumulator:

    @staticmethod
    def stochastic_grad_accum(p: torch.Tensor) -> None:
        pass

    @staticmethod
    def reassign_grad_buffer(model: nn.Module) -> None:
        pass

    @staticmethod
    def assign_hooks(model: nn.Module) -> List:
        pass


def is_valid_parameters(parameters: ParamsT) -> bool:
    """Check where the parameters are valid."""
    return isinstance(parameters, (list, tuple)) and len(parameters) > 0 and isinstance(parameters[0], dict)


def has_overflow(grad_norm: torch.Tensor) -> bool:
    """Detect inf and NaN in grad_norm."""
    return bool(torch.logical_or(torch.isnan(grad_norm), torch.isinf(grad_norm)).any())


def to_real(x: torch.Tensor) -> torch.Tensor:
    """Return real value of tensor."""
    return x.real if torch.is_complex(x) else x


def normalize_gradient(x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8) -> None:
    """Normalize gradient with stddev.

    Args:
        x (torch.Tensor): Gradient tensor to normalize.
        use_channels (bool): If True, perform channel-wise normalization.
        epsilon (float): Small constant added for numerical stability.

    """
    size: int = x.dim()
    if size > 1 and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True).add_(epsilon)
        x.div_(s)
    elif torch.numel(x) > 2:
        s = x.std().add_(epsilon)
        x.div_(s)


def clip_grad_norm(
    parameters: Union[ParamsT, torch.Tensor],
    max_norm: float = 0.0,
    sync: bool = False,
) -> Union[torch.Tensor, float]:
    """Clip gradient norms.

    During combination with FSDP, will also ensure that grad norms are aggregated across all workers,
    since each worker only stores their shard of the gradients.

    Args:
        parameters (ParamsT): ParamsT whose gradients we wish to clip.
        max_norm (float): Maximum norm we wish the gradients to have. If non-positive,
            then we will not perform clipping.
        sync (bool): Boolean indicating whether we should aggregate across the distributed group.
            Used only in combination with FSDP.

    Returns:
        float: The gradient norm across all parameters, before clipping.

    """
    if parameters is None:
        raise ValueError('ParamsT cannot be None.')

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = cast(List, list(parameters))

    if max_norm > 0 and not sync:
        return clip_grad_norm_(parameters, max_norm)

    norm_sq = sum(p.grad.norm() ** 2 for p in parameters if p.grad is not None)
    if sync:  # pragma: no cover
        all_reduce(norm_sq)

    grad_norm: float = math.sqrt(norm_sq)
    if max_norm > 0:  # pragma: no cover
        clip_coefficient = max_norm / (grad_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coefficient)

    return grad_norm


def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    """Get norm of unit."""
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)


def disable_running_stats(model: nn.Module):
    """Disable running stats (momentum) of BatchNorm."""

    def _disable(module):
        pass

    model.apply(_disable)


def enable_running_stats(model: nn.Module):
    """Enable running stats (momentum) of BatchNorm."""

    def _enable(module):
        pass

    model.apply(_enable)


@torch.no_grad()
def get_global_gradient_norm(param_groups: List[Dict]) -> torch.Tensor:
    """Get global gradient norm."""
    global_grad_norm = torch.zeros(1, dtype=torch.float32, device=param_groups[0]['params'][0].device)

    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                global_grad_norm.add_(p.grad.norm().pow(2))

    return global_grad_norm


@torch.no_grad()
def reg_noise(
    network1: nn.Module, network2: nn.Module, num_data: int, lr: float, eta: float = 8e-3, temperature: float = 1e-4
) -> Union[torch.Tensor, float]:
    """Entropy-MCMC: Sampling from flat basins with ease.

    Usage example and detailed implementation can be found at:
    https://github.com/lblaoke/EMCMC/blob/master/exp/cifar10_emcmc.py

    Args:
        network1 (nn.Module): First neural network.
        network2 (nn.Module): Second neural network.
        num_data (int): Number of training data points.
        lr (float): Learning rate.
        eta (float): Eta parameter controlling auxiliary guiding variable.
        temperature (float): Temperature parameter for sampling.

    """
    reg_coef: float = 0.5 / (eta * num_data)
    noise_coef: float = math.sqrt(2.0 / lr / num_data * temperature)

    loss = torch.tensor(0.0, device=next(network1.parameters()).device)

    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        reg = (param1 - param2).pow_(2).mul_(reg_coef).sum()

        noise = param1 * torch.randn_like(param1)
        noise.add_(param2 * torch.randn_like(param2))

        loss.add_(reg - noise.mul_(noise_coef).sum())

    return loss


@torch.no_grad()
def copy_stochastic(target: torch.Tensor, source: torch.Tensor) -> None:
    r"""Copy stochastic.

    reference: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905

    Args:
        target (torch.Tensor): A tensor in bfloat16 format to copy to.
        source (torch.Tensor): A tensor in float32 format to copy from.

    """
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=1 << 16,
    )

    result.add_(source.view(dtype=torch.int32))

    result.bitwise_and_(-65536)

    target.copy_(result.view(dtype=torch.float32))
