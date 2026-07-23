import math
from typing import Optional, Set

import torch
from torch import distributed as dist
from torch import nn

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup


class AdamMini(BaseOptimizer):  # pragma: no cover

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1.0,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.1,
        model_sharding: bool = False,
        num_embeds: int = 2048,
        num_heads: int = 32,
        num_query_groups: Optional[int] = None,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(num_embeds, 'num_embeds')
        self.validate_non_negative(num_heads, 'num_heads')
        self.validate_non_negative(eps, 'eps')

        self.num_query_groups: int = num_query_groups if num_query_groups is not None else num_embeds
        self.validate_mod(num_embeds, self.num_query_groups)

        self.world_size: int = torch.cuda.device_count()

        self.model = model
        self.model_sharding = model_sharding
        self.num_embeds = num_embeds
        self.num_heads = num_heads

        self.embed_blocks: Set[str] = {'embed', 'embd', 'wte', 'lm_head.weight', 'output.weight'}
        self.qk_blocks: Set[str] = {'k_proj.weight', 'q_proj.weight', 'wq.weight', 'wk.weight'}

        self.maximize = maximize

        groups = self.get_optimizer_groups(weight_decay)

        defaults: Defaults = {'lr': lr, 'betas': betas, 'eps': eps, **kwargs}

        super().__init__(groups, defaults)

    def __str__(self) -> str:
        return 'AdamMini'

    def get_optimizer_groups(self, weight_decay: float):
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @staticmethod
    def step_embed(
        p,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        pass

    @staticmethod
    def step_attn_proj(
        p,
        grad,
        state,
        parameter_per_head: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        pass

    @staticmethod
    def step_attn(
        p,
        grad,
        state,
        num_heads: int,
        q_per_kv: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        pass

    def step_lefts(
        self,
        p,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
