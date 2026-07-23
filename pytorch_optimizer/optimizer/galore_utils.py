import math
from typing import Literal, Optional, Tuple, Union

import torch

PROJECTION_TYPE = Literal['std', 'reverse_std', 'right', 'left', 'full', 'random']


class GaLoreProjector:

    def __init__(
        self,
        rank: Optional[int] = 128,
        update_proj_gap: int = 50,
        scale: float = 1.0,
        projection_type: PROJECTION_TYPE = 'std',
        **kwargs,
    ) -> None:
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type

        self.ortho_matrix: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.last_svd_step: int = -1

    def get_orthogonal_matrix(
        self, weights: torch.Tensor, projection_type: str, from_random_matrix: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def get_low_rank_grad_std(self, grad: torch.Tensor) -> torch.Tensor:
        pass

    def get_low_rank_grad_reverse_std(self, grad: torch.Tensor) -> torch.Tensor:
        pass

    def get_low_rank_grad_right(self, grad: torch.Tensor) -> torch.Tensor:
        pass

    def get_low_rank_grad_left(self, grad: torch.Tensor) -> torch.Tensor:
        pass

    def get_low_rank_grad_full(self, grad: torch.Tensor) -> torch.Tensor:
        pass

    def get_low_rank_grad_random(self, grad: torch.Tensor) -> torch.Tensor:
        pass

    def update_ortho_matrix(self, x: torch.Tensor, from_random_matrix: bool) -> None:
        pass

    def project(
        self,
        grad: torch.Tensor,
        num_steps: int,
        svd_basis_matrix: Optional[torch.Tensor] = None,
        from_random_matrix: bool = False,
    ) -> torch.Tensor:
        pass

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        pass
