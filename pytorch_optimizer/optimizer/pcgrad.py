import random
from copy import deepcopy
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer


def flatten_grad(grads: List[torch.Tensor]) -> torch.Tensor:
    pass


def un_flatten_grad(grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
    pass


class PCGrad(BaseOptimizer):

    def __init__(self, optimizer: Optimizer, reduction: str = 'mean'):
        self.validate_options(reduction, 'reduction', ['mean', 'sum'])

        self.optimizer = optimizer
        self.reduction = reduction

    @torch.no_grad()
    def init_group(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def set_grad(self, grads: List[torch.Tensor]) -> None:
        pass

    def retrieve_grad(self) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
        pass

    def pack_grad(self, objectives: Iterable) -> Tuple[List[torch.Tensor], List[List[int]], List[torch.Tensor]]:
        pass

    def project_conflicting(self, grads: List[torch.Tensor], has_grads: List[torch.Tensor]) -> torch.Tensor:
        pass

    def pc_backward(self, objectives: Iterable[nn.Module]) -> None:
        pass
