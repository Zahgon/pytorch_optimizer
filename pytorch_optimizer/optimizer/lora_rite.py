from typing import Any, Dict, List, Optional, Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class LoRARiteHelper:

    def __init__(self, maybe_inf_to_nan: bool = True):
        self.maybe_inf_to_nan = maybe_inf_to_nan

    def inf_to_nan(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def bias_corrected_decay(step: int, decay: float) -> float:
        pass

    @staticmethod
    def move_lora_dim_to_last(tensor: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Size]:
        pass

    @staticmethod
    def restore_original_shape_and_dim(tensor: torch.Tensor, dim: int, shape: torch.Size) -> torch.Tensor:
        pass

    def restore_param_shape(self, tensor: torch.Tensor, param: torch.Tensor, dim: int) -> torch.Tensor:
        pass

    @staticmethod
    def make_symmetric(tensor: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def create_preconditioner(tensor: torch.Tensor) -> torch.Tensor:
        pass

    def inverse_sqrt(
        self,
        tensor: torch.Tensor,
        escape: torch.Tensor,
        eps: float,
        eps_root: float,
        relative_epsilon: bool,
    ) -> torch.Tensor:
        pass

    def transform_second_moment_to_new_basis(self, moments: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def transform_first_moment_to_new_basis(moments: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def get_unmagnified_grad(grad: torch.Tensor, rotate_inv: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def rotate_update(update: torch.Tensor, rotate_inv: torch.Tensor) -> torch.Tensor:
        pass

    def get_unmagnified_rotate_second_escape(
        self, new_moments: torch.Tensor, old_moments: torch.Tensor
    ) -> torch.Tensor:
        pass

    def get_preconditioned_update(
        self,
        grad: torch.Tensor,
        moments: torch.Tensor,
        escape: torch.Tensor,
        eps: float,
        eps_root: float,
        relative_epsilon: bool,
        apply_escape: bool,
    ) -> torch.Tensor:
        pass

    def update_first_moment(
        self, step: int, update: torch.Tensor, moments: torch.Tensor, beta1: float
    ) -> torch.Tensor:
        pass

    def compute_second_moment(self, update: torch.Tensor) -> torch.Tensor:
        pass

    def update_second_moment(
        self, step: int, update: torch.Tensor, moments: torch.Tensor, beta2: float
    ) -> torch.Tensor:
        pass

    def update_second_escape(
        self, step: int, update: torch.Tensor, moments: torch.Tensor, beta2: float
    ) -> torch.Tensor:
        pass

    @staticmethod
    def get_rotation_and_basis(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @staticmethod
    def reduce_rms(tensor: torch.Tensor) -> torch.Tensor:
        pass

    def clip_update(self, update: torch.Tensor, clip_threshold: float) -> torch.Tensor:
        pass

    def skip_update(self, update: torch.Tensor, skip_threshold: float) -> torch.Tensor:
        pass


class LoRARite(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        eps: float = 1e-6,
        relative_epsilon: bool = False,
        clip_unmagnified_grad: float = 1.0,
        update_capping: float = 0.0,
        update_skipping: float = 1.0,
        weight_decay: float = 0.0,
        apply_escape: bool = False,
        lora_l_dim: int = 0,
        lora_r_dim: int = -1,
        maybe_inf_to_nan: bool = True,
        balance_param: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(clip_unmagnified_grad, 'clip_unmagnified_grad')
        self.validate_non_negative(update_capping, 'update_capping')
        self.validate_non_negative(update_skipping, 'update_skipping')
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.helper = LoRARiteHelper(maybe_inf_to_nan=maybe_inf_to_nan)
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'eps_root': eps**2,
            'relative_epsilon': relative_epsilon,
            'clip_unmagnified_grad': clip_unmagnified_grad,
            'update_capping': update_capping,
            'update_skipping': update_skipping,
            'weight_decay': weight_decay,
            'apply_escape': apply_escape,
            'lora_l_dim': lora_l_dim,
            'lora_r_dim': lora_r_dim,
            'balance_param': balance_param,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'LoRARite'

    @staticmethod
    def iter_lora_pairs(group: ParamGroup) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def init_pair_state(
        self, group: ParamGroup, state: Dict[str, Any], param_left: torch.Tensor, param_right: torch.Tensor
    ) -> None:
        pass

    def build_pair_info(
        self,
        group: ParamGroup,
        param_left: torch.Tensor,
        param_right: torch.Tensor,
    ) -> Dict[str, Any]:
        pass

    def apply_pair_update(
        self,
        group: ParamGroup,
        param_left: torch.Tensor,
        param_right: torch.Tensor,
        grad_norm: torch.Tensor,
    ) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
