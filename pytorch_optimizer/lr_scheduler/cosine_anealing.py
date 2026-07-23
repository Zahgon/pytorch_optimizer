import math
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmupRestarts(LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 0.9,
        last_epoch: int = -1,
    ):
        if warmup_steps >= first_cycle_steps:
            raise ValueError(
                f'warmup_steps must be smaller than first_cycle_steps. {warmup_steps} < {first_cycle_steps}'
            )

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.step_in_cycle = last_epoch
        self.last_epoch = last_epoch

        self.cycle: int = 0
        self.base_lrs: List[float] = []

        super().__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self) -> None:
        pass

    def get_lr(self) -> List[float]:
        pass

    def step(self, epoch: Optional[int] = None):
        pass
