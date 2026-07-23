from importlib.util import find_spec
from typing import Any, List, Optional, Sequence

import torch
from torch import nn

HAS_GEOTORCH: bool = find_spec('geotorch') is not None

if HAS_GEOTORCH:
    from geotorch import orthogonal


def divide(numer: torch.Tensor, de_nom: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    pass


class VanillaMTL(nn.Module):

    def __init__(self, backbone, heads):
        super().__init__()
        self._backbone = [backbone]
        self.heads = heads

        self.rep = None
        self.grads: List = [None for _ in range(len(heads))]

    @property
    def backbone(self):
        pass

    def train(self, mode: bool = True) -> nn.Module:
        pass

    def to(self, *args, **kwargs):
        self.backbone.to(*args, **kwargs)
        for head in self.heads:
            head.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def _hook(self, index):
        pass

    def forward(self, x: torch.Tensor):
        pass

    def backward(self, losses, backbone_loss=None, **kwargs):
        pass

    def mtl_parameters(self, recurse=True):
        pass

    def model_parameters(self, recurse=True):
        pass


def rotate(points: torch.Tensor, rotation: torch.Tensor, total_size: int) -> torch.Tensor:
    pass


def rotate_back(points: torch.Tensor, rotation: torch.Tensor, total_size: int) -> torch.Tensor:
    pass


class RotateModule(nn.Module):

    def __init__(self, parent, item):
        super().__init__()

        self.parent = [parent]
        self.item = item

    def hook(self, grad: torch.Tensor):
        pass

    @property
    def p(self):
        pass

    @property
    def r(self):
        pass

    @property
    def weight(self):
        pass

    def rotate(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def rotate_back(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass


class RotateOnly(nn.Module):

    num_tasks: int
    backbone: nn.Module
    heads: Sequence[nn.Module]
    rep: Optional[torch.Tensor]

    def __init__(
        self,
        backbone: nn.Module,
        heads: Sequence[nn.Module],
        latent_size: int,
        *args,
        burn_in_period: int = 20,
        normalize_losses: bool = False,
    ):
        super().__init__()
        if not HAS_GEOTORCH:
            raise ImportError('[-] you need to install `geotorch` to use RotoGrad. `pip install geotorch`')

        self._backbone = [backbone]
        self.heads = heads

        self.num_tasks: int = len(heads)
        self.latent_size = latent_size
        self.burn_in_period = burn_in_period
        self.normalize_losses = normalize_losses

        for i in range(self.num_tasks):
            heads[i] = nn.Sequential(RotateModule(self, i), heads[i])

        for i in range(self.num_tasks):
            self.register_parameter(f'rotation_{i}', nn.Parameter(torch.eye(latent_size), requires_grad=True))
            orthogonal(self, f'rotation_{i}', triv='expm')  # uses exponential map (alternative: cayley)

        self.rep = None
        self.grads = [None for _ in range(self.num_tasks)]
        self.original_grads = [None for _ in range(self.num_tasks)]
        self.losses = [None for _ in range(self.num_tasks)]
        self.initial_losses = [None for _ in range(self.num_tasks)]
        self.initial_backbone_loss = None
        self.iteration_counter: int = 0

    @property
    def rotation(self) -> Sequence[torch.Tensor]:
        pass

    @property
    def backbone(self) -> nn.Module:
        pass

    def to(self, *args, **kwargs):
        self.backbone.to(*args, **kwargs)
        for head in self.heads:
            head.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True) -> nn.Module:
        pass

    def __len__(self) -> int:
        """Get the number of tasks."""
        return self.num_tasks

    def __getitem__(self, item) -> nn.Module:
        """Get an end-to-end model for the selected task."""
        return nn.Sequential(self.backbone, self.heads[item])

    def _hook(self, index):
        pass

    def forward(self, x: Any) -> Sequence[Any]:
        pass

    def backward(self, losses: Sequence[torch.Tensor], backbone_loss=None, **kwargs) -> None:
        pass

    def _rep_grad(self):
        pass

    def mtl_parameters(self, recurse: bool = True):
        pass

    def model_parameters(self, recurse=True):
        pass


class RotoGrad(RotateOnly):

    num_tasks: int
    backbone: nn.Module
    heads: Sequence[nn.Module]
    rep: torch.Tensor

    def __init__(
        self,
        backbone: nn.Module,
        heads: Sequence[nn.Module],
        latent_size: int,
        *args,
        burn_in_period: int = 20,
        normalize_losses: bool = False,
    ):
        super().__init__(backbone, heads, latent_size, burn_in_period, *args, normalize_losses=normalize_losses)

        self.initial_grads = None
        self.counter: int = 0

    def _rep_grad(self):
        pass


class RotoGradNorm(RotoGrad):

    def __init__(
        self,
        backbone: nn.Module,
        heads: Sequence[nn.Module],
        latent_size: int,
        *args,
        alpha: float,
        burn_in_period: int = 20,
        normalize_losses: bool = False,
    ):
        super().__init__(
            backbone, heads, latent_size, *args, burn_in_period=burn_in_period, normalize_losses=normalize_losses
        )
        self.alpha = alpha
        self.weight_ = nn.ParameterList([nn.Parameter(torch.ones([]), requires_grad=True) for _ in range(len(heads))])

    @property
    def weight(self) -> Sequence[torch.Tensor]:
        pass

    def _rep_grad(self):
        pass
