import torch
from torch import nn
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cosine_embedding_loss,
    cross_entropy,
    normalize,
    one_hot,
)

from pytorch_optimizer.loss.cross_entropy import BCELoss
from pytorch_optimizer.loss.tversky import TverskyLoss


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass


class FocalCosineLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, focal_weight: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass


class BCEFocalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smooth: float = 0.0,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.bce = BCELoss(label_smooth=label_smooth, eps=eps, reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass


class FocalTverskyLoss(nn.Module):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.gamma = gamma

        self.tversky = TverskyLoss(alpha, beta, smooth)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
