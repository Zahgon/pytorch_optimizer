import torch
from torch import nn
from torch.nn.functional import relu


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    pass


def lovasz_hinge_flat(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    pass


class LovaszHingeLoss(nn.Module):

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
