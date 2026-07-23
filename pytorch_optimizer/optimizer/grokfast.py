import math
from collections import deque
from typing import Dict, List, Literal, Optional, cast

import torch
from torch import nn

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT

FILTER_TYPE = Literal['mean', 'sum']


@torch.no_grad()
def gradfilter_ma(
    model: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: FILTER_TYPE = 'mean',
    warmup: bool = True,
) -> Dict[str, deque]:
    """Grokfast-MA.

    Args:
        model (nn.Module): Model that contains every trainable parameters.
        grads (Optional[Dict[str, deque]]): Running memory (queue for windowed moving average).
            Initialize by setting  it to None.
            Feed the output of the method recursively after one call.
        window_size (int): The width of the filter window.
            Additional memory requirements increase linearly with window size.
        lamb (float): Amplifying factor hyperparameter of the filter.
        filter_type (FILTER_TYPE): Aggregation method for the running queue.
        warmup (bool): If true, the filter is not applied until the queue is filled.

    Example:
        loss.backwards()  # Calculate the gradients.

        grads = gradfilter_ma(model, grads=grads, window_size=window_size, lamb=lamb)

        optimizer.step()  # Call the optimizer.

    """
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad:
            grads[n].append(p.grad)

            if not warmup or len(grads[n]) == window_size:
                if filter_type == 'mean':
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == 'sum':
                    avg = sum(grads[n])
                else:
                    raise NotImplementedError(f'not supported filter_type {filter_type}')

                p.grad.add_(avg, alpha=lamb)

    return grads


@torch.no_grad()
def gradfilter_ema(
    model: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Grokfast.

    Args:
        model (nn.Module): Model that contains every trainable parameters.
        grads (Optional[Dict[str, deque]]): Running memory (EMA). Initialize by setting it to None.
            Feed the output of the method recursively after one call.
        alpha (int): Momentum hyperparameter of the EMA.
        lamb (float): Amplifying factor hyperparameter of the filter.

    Example:
        loss.backwards()  # Calculate the gradients.

        grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)

        optimizer.step()  # Call the optimizer.

    """
    if grads is None:
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}

    grads = cast(Dict[str, torch.Tensor], grads)

    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].mul_(alpha).add_(p.grad, alpha=1.0 - alpha)
            p.grad.add_(grads[n], alpha=lamb)

    return grads


class GrokFastAdamW(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: Betas = (0.9, 0.99),
        grokfast: bool = True,
        grokfast_alpha: float = 0.98,
        grokfast_lamb: float = 2.0,
        grokfast_after_step: int = 0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        normalize_lr: bool = True,
        eps: float = 1e-8,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(grokfast_alpha, 'grokfast_alpha', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.foreach = foreach
        self.maximize = maximize

        if grokfast and normalize_lr:
            lr /= 1.0 + grokfast_lamb

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'grokfast': grokfast,
            'grokfast_alpha': grokfast_alpha,
            'grokfast_lamb': grokfast_lamb,
            'grokfast_after_step': grokfast_after_step,
            'foreach': foreach,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'GrokFastAdamW'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        grok_exp_avgs: List[torch.Tensor],
        should_grokfast: bool,
    ) -> None:
        pass

    def _step_per_param(self, group: ParamGroup, should_grokfast: bool) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
