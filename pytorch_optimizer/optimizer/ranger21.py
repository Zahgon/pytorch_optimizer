import math
from typing import Optional

import torch
from torch.nn.functional import softplus

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm


class Ranger21(BaseOptimizer):

    def __init__(  # pylint: disable=R0913
        self,
        params: ParamsT,
        num_iterations: int,
        lr: float = 1e-3,
        beta0: float = 0.9,
        betas: Betas = (0.9, 0.999),
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        disable_lr_scheduler: bool = False,
        num_warm_up_iterations: Optional[int] = None,
        num_warm_down_iterations: Optional[int] = None,
        warm_down_min_lr: float = 3e-5,
        agc_clipping_value: float = 1e-2,
        agc_eps: float = 1e-3,
        centralize_gradients: bool = True,
        normalize_gradients: bool = True,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        weight_decay: float = 1e-4,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        norm_loss_factor: float = 1e-4,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(warm_down_min_lr)
        self.validate_betas(betas)
        self.validate_range(beta0, 'beta0', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(agc_clipping_value, 'agc_clipping_value')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(agc_eps, 'agc_eps')

        self.min_lr = warm_down_min_lr
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.disable_lr_scheduler = disable_lr_scheduler
        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.norm_loss_factor = norm_loss_factor
        self.maximize = maximize

        self.lookahead_step: int = 0
        self.starting_lr: float = lr
        self.current_lr: float = lr

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

        self.num_warm_up_iterations: int = (
            self.build_warm_up_iterations(num_iterations, betas[1])
            if num_warm_up_iterations is None
            else num_warm_up_iterations
        )
        self.num_warm_down_iterations: int = (
            self.build_warm_down_iterations(num_iterations)
            if num_warm_down_iterations is None
            else num_warm_down_iterations
        )
        self.start_warm_down: int = num_iterations - self.num_warm_down_iterations
        self.warm_down_lr_delta: float = self.starting_lr - self.min_lr

    def __str__(self) -> str:
        return 'Ranger21'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def build_warm_up_iterations(total_iterations: int, beta2: float, warm_up_pct: float = 0.22) -> int:
        pass

    @staticmethod
    def build_warm_down_iterations(total_iterations: int, warm_down_pct: float = 0.72) -> int:
        pass

    def warm_up_dampening(self, lr: float, step: int) -> float:
        pass

    def warm_down(self, lr: float, iteration: int) -> float:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass

    def lookahead_process_step(self):
        pass
