from abc import ABC, abstractmethod
from typing import List

from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError


class BaseLinearWarmupScheduler(ABC):

    def __init__(
        self,
        optimizer: Optimizer,
        t_max: int,
        max_lr: float,
        min_lr: float = 0.0,
        init_lr: float = 0.0,
        warmup_steps: int = 0,
    ):
        self.optimizer = optimizer
        self.total_steps = t_max
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

        self.step_t: int = 0
        self.base_lrs: List[float] = []

        self.last_lr: List[float] = [init_lr]

        self.validate_parameters()

        self._init_lr()

    def validate_parameters(self):
        pass

    def _init_lr(self):
        pass

    def step(self):
        pass

    @abstractmethod
    def _step(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def get_lr(self) -> float:
        pass
