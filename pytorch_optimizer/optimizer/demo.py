import math
from importlib.util import find_spec
from typing import List, Optional

import torch
from torch.distributed import ProcessGroup, all_gather, get_world_size

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Loss, ParamsT

HAS_EINOPS: bool = find_spec('einops') is not None

if HAS_EINOPS:  # pragma: ignore
    from einops import rearrange


class TransformDCT:

    @torch.no_grad()
    def __init__(self, param_groups, target_chunk, norm: str = 'ortho'):
        if not HAS_EINOPS:
            raise ImportError('You need to install `einops` to use `TransformDCT`')

        self.target_chunk = target_chunk

        self.shape_dict = {}
        self.f_dict = {}
        self.b_dict = {}

        for group in param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    sc = get_smaller_split(s, self.target_chunk)
                    self.shape_dict[s] = sc

                    if sc not in self.f_dict:
                        i = torch.eye(sc)
                        self.f_dict[sc] = dct(i, norm=norm).to(p.dtype).to(p.device)
                        self.b_dict[sc] = inverse_dct(i, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def einsum_2d(self, x: torch.Tensor, b: torch.Tensor, d: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @torch.no_grad()
    def einsum_2d_t(self, x: torch.Tensor, b: torch.Tensor, d: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        pass


class CompressDCT:

    @torch.no_grad()
    def __init__(self):
        if not HAS_EINOPS:
            raise ImportError('You need to install `einops` to use `CompressDCT`')

    @staticmethod
    def clamp_top_k(x: torch.Tensor, top_k: int) -> int:
        pass

    @torch.no_grad()
    def compress(self, x: torch.Tensor, top_k: int):
        pass

    @torch.no_grad()
    def decompress(self, p, idx, val, shape):
        pass

    @torch.no_grad()
    def batch_decompress(self, p, idx, val, shape) -> torch.Tensor:
        pass


def dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    pass


def inverse_dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    pass


def get_prime_divisors(n: int) -> List[int]:
    pass


def get_divisors(n: int) -> List[int]:
    pass


def get_smaller_split(n: int, close_to: int) -> int:
    pass


class DeMo(torch.optim.SGD, BaseOptimizer):  # pragma: no cover

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        compression_decay: float = 0.999,
        compression_top_k: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        process_group: Optional[ProcessGroup] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(compression_decay, 'compression_decay', 0.0, 1.0, range_type='[)')
        self.validate_positive(compression_top_k, 'compression_top_k')
        self.validate_positive(compression_chunk, 'compression_chunk')

        self.weight_decay = weight_decay

        self.compression_decay = compression_decay
        self.compression_top_k = compression_top_k
        self.compression_chunk = compression_chunk
        self.process_group = process_group

        self.data_transmit: int = 0
        self.data_receive: int = 0

        self.maximize = maximize

        super().__init__(
            params,
            lr=lr,
            foreach=False,
            momentum=0.0,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            weight_decay=0.0,
            **kwargs,
        )

        self.demo_state = {}
        self.init_demo_states()
        self.init_parameters()

        self.default_dtype: torch.dtype = self.find_dtype()
        self.transform = TransformDCT(self.param_groups, self.compression_chunk, norm='ortho')
        self.compress = CompressDCT()

    def __str__(self) -> str:
        return 'DeMo'

    def find_dtype(self) -> torch.dtype:
        pass

    def init_demo_states(self) -> None:
        pass

    def init_parameters(self) -> None:
        pass

    def demo_all_gather(self, sparse_idx, sparse_val):
        pass

    @torch.no_grad()
    def init_group(self):
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
