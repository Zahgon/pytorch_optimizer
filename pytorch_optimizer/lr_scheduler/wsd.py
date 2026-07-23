import math
from functools import partial
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

COOLDOWN_TYPE = Literal['cosine', '1-sqrt', 'linear', '1-square']


def get_cosine_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float,
    num_cycles: float,
) -> float:
    pass


def get_1sqrt_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
) -> float:
    pass


def get_1square_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
) -> float:
    pass


def get_linear_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
) -> float:
    pass


def get_wsd_scheduler_lambda(  # noqa: PLR0911
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float,
    num_cycles: float,
    cooldown_type: COOLDOWN_TYPE,
) -> float:
    pass


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    cooldown_type: COOLDOWN_TYPE = '1-sqrt',
    last_epoch: int = -1,
) -> LRScheduler:
    r"""Get Warmup-Stable-Decay (WSD) learning rate scheduler.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of warmup steps.
        num_stable_steps (int): The number of stable steps.
        num_decay_steps (int): The number of decay steps.
        min_lr_ratio (float): The minimum learning rate as a ratio of the initial learning rate.
        num_cycles (float): The number of waves in the cosine schedule (default is a half-cosine decay).
        cooldown_type (COOLDOWN_TYPE): Cooldown type of the learning rate scheduler.
        last_epoch (int): The index of the last epoch when resuming training.

    """
    lr_scheduler = partial(
        get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
        cooldown_type=cooldown_type,
    )

    return LambdaLR(optimizer, lr_scheduler, last_epoch)
