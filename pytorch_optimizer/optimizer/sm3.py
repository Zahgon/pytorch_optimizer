import torch

from pytorch_optimizer.base.exception import NoComplexParameterError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


@torch.no_grad()
def reduce_max_except_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Perform reduce-max along all dimensions except the given dim.

    Args:
        x (torch.Tensor): Tensor to reduce-max.
        dim (int): Dimension to exclude.

    """
    rank: int = len(x.shape)
    if rank == 0:
        return x

    if dim >= rank:
        raise ValueError(f'[-] given dim is bigger than rank. {dim} >= {rank}')

    for d in range(rank):
        if d != dim:
            x = x.max(dim=d, keepdim=True).values
    return x


class SM3(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        momentum: float = 0.0,
        beta: float = 0.0,
        eps: float = 1e-30,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SM3'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def make_sparse(grad: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
