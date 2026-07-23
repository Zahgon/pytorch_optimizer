from typing import Dict, List, Optional, Union, cast

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_optimizer.base.type import Closure, ParamsT
from pytorch_optimizer.optimizer.utils import clip_grad_norm, has_overflow


class DynamicLossScaler:

    def __init__(
        self,
        init_scale: float = 2.0 ** 15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        tolerance: float = 0.00,
        threshold: Optional[float] = None,
    ):  # fmt: skip
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold

        self.iter: int = 0
        self.last_overflow_iter: int = -1
        self.last_rescale_iter: int = -1
        self.overflows_since_rescale: int = 0
        self.has_overflow_serial: bool = False

    def update_scale(self, overflow: bool):
        pass

    def decrease_loss_scale(self):
        pass


class SafeFP16Optimizer(Optimizer):  # pragma: no cover

    def __init__(
        self,
        optimizer: Optimizer,
        aggregate_g_norms: bool = False,
        min_loss_scale: float = 2 ** -5,
    ) -> None:  # fmt: skip
        self.optimizer = optimizer
        self.aggregate_g_norms = aggregate_g_norms
        self.min_loss_scale = min_loss_scale

        self.fp16_params = self.get_parameters(optimizer)
        self.fp32_params = self.build_fp32_params(self.fp16_params, flatten=False)

        if len(optimizer.param_groups) != 1:
            raise NotImplementedError('Need to implement the parameter group transfer.')

        optimizer.param_groups[0]['params'] = self.fp32_params

        self.scaler: DynamicLossScaler = DynamicLossScaler(2.0 ** 15)  # fmt: skip
        self.needs_sync: bool = True

    @classmethod
    def get_parameters(cls, optimizer: Optimizer) -> List:
        pass

    @classmethod
    def build_fp32_params(cls, parameters: ParamsT, flatten: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        pass

    def state_dict(self) -> Dict:
        pass

    def load_state_dict(self, state_dict: Dict):
        pass

    def backward(self, loss, update_main_grads: bool = False):
        pass

    def sync_fp16_grads_to_fp32(self, multiply_grads: float = 1.0) -> None:
        pass

    def multiply_grads(self, c: float) -> None:
        pass

    def update_main_grads(self) -> None:
        pass

    def clip_main_grads(self, max_norm: float):
        pass

    def step(self, closure: Closure = None):
        pass

    def zero_grad(self) -> None:
        pass

    def get_lr(self) -> float:
        pass

    def set_lr(self, lr: float):
        pass

    @property
    def loss_scale(self) -> float:
        pass
