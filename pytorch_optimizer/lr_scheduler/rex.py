from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class REXScheduler(LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_lr: float = 1.0,
        min_lr: float = 0.0,
    ):
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.step_t: int = 0
        self.base_lrs: List[float] = []

        self.last_lr: List[float] = [self.max_lr]

        super().__init__(optimizer)

        self.init_lr()

    def init_lr(self) -> None:
        pass

    def get_lr(self) -> float:
        pass

    def get_linear_lr(self) -> float:
        pass

    def step(self, epoch: Optional[int] = None) -> float:
        pass
