from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn
from torch.distributed import ReduceOp, all_reduce, get_world_size, is_initialized
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoClosureError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, OptimizerType, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import disable_running_stats, enable_running_stats


def get_global_gradient_norm(param_groups: ParamsT, device: torch.device) -> torch.Tensor:
    """Get global gradient norm."""
    norms: List[torch.Tensor] = []
    for group in param_groups or []:
        params: List[torch.Tensor] = group.get('params', []) or []
        adaptive: bool = group.get('adaptive', False)
        for p in params:
            if p.grad is not None:
                norm = ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2).to(device)
                norms.append(norm)

    if not norms:
        return torch.tensor(0.0, device=device)

    return torch.norm(torch.stack(norms), p=2)


class SAM(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.05,
        adaptive: bool = False,
        use_gc: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.use_gc = use_gc
        self.perturb_eps = perturb_eps

        defaults: Defaults = {'rho': rho, 'adaptive': adaptive, **kwargs}

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'SAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        pass

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None):
        pass

    def load_state_dict(self, state_dict: Dict):
        pass


class GSAM(BaseOptimizer):  # pragma: no cover

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: Optimizer,
        model: nn.Module,
        rho_scheduler,
        alpha: float = 0.4,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_range(alpha, 'alpha', 0.0, 1.0)

        self.model = model
        self.rho_scheduler = rho_scheduler
        self.alpha = alpha
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps

        self.rho_t: float = 0.0
        self.forward_backward_func: Optional[Callable] = None

        if hasattr(ReduceOp, 'AVG'):
            self.grad_reduce = ReduceOp.AVG
            self.manual_average: bool = False
        else:
            self.grad_reduce = ReduceOp.SUM
            self.manual_average: bool = True

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        defaults: Defaults = {'adaptive': adaptive, **kwargs}

        super().__init__(params, defaults)

        self.update_rho_t()

    def __str__(self) -> str:
        return 'GSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def update_rho_t(self) -> float:
        pass

    @torch.no_grad()
    def perturb_weights(self, rho: float):
        pass

    @torch.no_grad()
    def un_perturb(self):
        pass

    @torch.no_grad()
    def gradient_decompose(self, alpha: float = 0.0):
        pass

    @torch.no_grad()
    def sync_grad(self):
        pass

    @torch.no_grad()
    def grad_norm(self, by: Optional[str] = None, weight_adaptive: bool = False) -> torch.Tensor:
        pass

    def maybe_no_sync(self):
        pass

    @torch.no_grad()
    def set_closure(self, loss_fn: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Tuple[Any, torch.Tensor]:
        pass

    def load_state_dict(self, state_dict: Dict):
        pass


class WSAM(BaseOptimizer):

    def __init__(
        self,
        model: Union[nn.Module, DistributedDataParallel],
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.05,
        gamma: float = 0.9,
        adaptive: bool = False,
        decouple: bool = True,
        max_norm: Optional[float] = None,
        eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')

        self.model = model
        self.decouple = decouple
        self.max_norm = max_norm

        alpha: float = gamma / (1.0 - gamma)

        defaults: Defaults = {'rho': rho, 'alpha': alpha, 'adaptive': adaptive, 'sam_eps': eps, **kwargs}

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'WSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        pass

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None):
        pass

    def load_state_dict(self, state_dict: Dict):
        pass


class BSAM(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        num_data: int,
        lr: float = 5e-1,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 1e-4,
        rho: float = 0.05,
        adaptive: bool = False,
        damping: float = 0.1,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(num_data, 'num_data')
        self.validate_non_negative(damping, 'damping')

        self.num_data = num_data
        self.damping = damping

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'rho': rho,
            'adaptive': adaptive,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'bSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def first_step(self):
        pass

    @torch.no_grad()
    def second_step(self):
        pass

    @torch.no_grad()
    def third_step(self):
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None):
        pass


class LookSAM(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.1,
        k: int = 10,
        alpha: float = 0.7,
        adaptive: bool = False,
        use_gc: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_positive(k, 'k')
        self.validate_range(alpha, 'alpha', 0.0, 1.0, '()')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.k = k
        self.alpha = alpha
        self.use_gc = use_gc
        self.perturb_eps = perturb_eps

        defaults: Defaults = {'rho': rho, 'adaptive': adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'LookSAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def get_step(self):
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        pass

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None):
        pass

    def load_state_dict(self, state_dict: Dict):
        pass


class FriendlySAM(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: OptimizerType,
        rho: float = 0.05,
        sigma: float = 1.0,
        lmbda: float = 0.9,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        self.validate_non_negative(rho, 'rho')
        self.validate_non_negative(sigma, 'sigma')
        self.validate_non_negative(lmbda, 'lmbda')
        self.validate_non_negative(perturb_eps, 'perturb_eps')

        self.perturb_eps = perturb_eps

        defaults: Defaults = {'rho': rho, 'sigma': sigma, 'lmbda': lmbda, 'adaptive': adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'FriendlySAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        pass

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None):
        pass

    def load_state_dict(self, state_dict: Dict):
        pass
