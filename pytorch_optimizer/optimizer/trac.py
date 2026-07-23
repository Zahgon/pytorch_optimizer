from typing import Callable, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, OptimizerInstanceOrClass, ParamGroup, State


def polyval(x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    pass


class ERF1994(nn.Module):

    def __init__(self, num_coefs: int = 128) -> None:
        super().__init__()

        self.n: int = num_coefs

        self.i: torch.Tensor = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        self.m = 2 * self.n
        self.m2 = 2 * self.m
        self.k = torch.linspace(-self.m + 1, self.m - 1, self.m2 - 1)
        self.l = torch.sqrt(self.n / torch.sqrt(torch.tensor(2.0)))
        self.theta = self.k * torch.pi / self.m
        self.t = self.l * torch.tan(self.theta / 2.0)
        self.f = torch.exp(-self.t ** 2) * (self.l ** 2 + self.t ** 2)  # fmt: skip
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real / self.m2
        self.a = torch.flipud(self.a[1:self.n + 1])  # fmt: skip

    def w_algorithm(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass


class TRAC(BaseOptimizer):

    def __init__(
        self,
        optimizer: OptimizerInstanceOrClass,
        betas: List[float] = (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999),
        num_coefs: int = 128,
        s_prev: float = 1e-8,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_positive(num_coefs, 'num_coefs')
        self.validate_non_negative(s_prev, 's_prev')
        self.validate_non_negative(eps, 'eps')

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self.betas = betas
        self.s_prev = s_prev
        self.eps = eps

        self.erf: nn.Module = ERF1994(num_coefs=num_coefs)
        self.f_term: torch.Tensor = self.s_prev / self.erf_imag(1.0 / torch.sqrt(torch.tensor(2.0)))

        self.defaults: Defaults = self.optimizer.defaults

    def __str__(self) -> str:
        return 'TRAC'

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

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    @torch.no_grad()
    def erf_imag(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def backup_params_and_grads(self) -> Tuple[Dict, Dict]:
        pass

    @torch.no_grad()
    def trac_step(self, updates: Dict, grads: Dict) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
