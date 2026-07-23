from typing import List

from torch.optim.lr_scheduler import LRScheduler


class ProportionScheduler:

    def __init__(
        self,
        lr_scheduler: LRScheduler,
        max_lr: float,
        min_lr: float = 0.0,
        max_value: float = 2.0,
        min_value: float = 2.0,
    ):
        self.lr_scheduler = lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value

        self.step_t: int = 0
        self.last_lr: List[float] = []

        self.step()

    def get_lr(self) -> float:
        pass

    def step(self) -> float:
        pass
