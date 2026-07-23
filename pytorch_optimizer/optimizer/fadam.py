import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class FAdam(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.1,
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 1e-8,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(clip, 'clip')
        self.validate_positive(p, 'p')
        self.validate_non_negative(eps, 'eps')

        self.momentum_dtype = momentum_dtype
        self.fim_dtype = fim_dtype
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'clip': clip,
            'p': p,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FAdam'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
