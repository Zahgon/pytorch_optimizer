import torch
from torch import nn


class SoftF1Loss(nn.Module):

    def __init__(self, beta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
