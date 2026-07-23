import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy


class BCELoss(nn.Module):

    def __init__(self, label_smooth: float = 0.0, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.label_smooth = label_smooth
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
