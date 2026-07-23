import math

import numpy as np

from pytorch_optimizer.base.scheduler import BaseLinearWarmupScheduler


class LinearScheduler(BaseLinearWarmupScheduler):

    def _step(self) -> float:
        pass


class CosineScheduler(BaseLinearWarmupScheduler):

    def _step(self) -> float:
        pass


class PolyScheduler(BaseLinearWarmupScheduler):

    def __init__(self, optimizer, poly_order: float = 0.5, **kwargs):
        self.poly_order = poly_order

        if poly_order <= 0:
            raise ValueError(f'poly_order must be positive. {poly_order}')

        super().__init__(optimizer, **kwargs)

    def _step(self) -> float:
        pass
