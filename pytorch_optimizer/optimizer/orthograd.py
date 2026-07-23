from typing import Callable, Dict

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, OptimizerInstanceOrClass, ParamGroup, State


class OrthoGrad(BaseOptimizer):

    def __init__(self, optimizer: OptimizerInstanceOrClass, **kwargs) -> None:
        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}
        self.eps: float = 1e-30

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self.defaults: Defaults = self.optimizer.defaults

    def __str__(self) -> str:
        return 'OrthoGrad'

    @property
    def param_groups(self):
        pass

    @property
    def state(self) -> State:
        pass

    def state_dict(self) -> State:
        pass

    def load_state_dict(self, state_dict: State) -> None:
        pass

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def apply_orthogonal_gradients(self, params) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
