from typing import Optional, Tuple, cast

import torch
from torch import nn


def log_t(u: torch.Tensor, t: float) -> torch.Tensor:
    """Compute log_t for `u'."""
    return u.log() if t == 1.0 else (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u: torch.Tensor, t: float) -> torch.Tensor:
    """Compute exp_t for `u'."""
    return u.exp() if t == 1 else (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations: torch.Tensor, t: float, num_iters: int) -> torch.Tensor:
    pass


def compute_normalization_binary_search(activations: torch.Tensor, t: float, num_iters: int) -> torch.Tensor:
    pass


class ComputeNormalization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, activations: torch.Tensor, t: float, num_iters: int) -> torch.Tensor:
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


def compute_normalization(activations: torch.Tensor, t: float, num_iters: int = 5) -> torch.Tensor:
    r"""Compute normalization value for each example.

    Args:
        activations (torch.Tensor): A multi-dimensional tensor with the last dimension `num_classes`.
        t (float): Temperature parameter (> 1.0 for tail heaviness).
        num_iters (int): Number of iterations to run the method.

    """
    return cast(torch.Tensor, ComputeNormalization.apply(activations, t, num_iters))


def tempered_softmax(activations: torch.Tensor, t: float, num_iters: int = 5) -> torch.Tensor:
    """Tempered softmax function.

    Args:
        activations (torch.Tensor): A multidimensional tensor with last dimension `num_classes`.
        t (float): Temperature parameter (> 1.0 for tail heaviness).
        num_iters (int): Number of iterations to run the method.

    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants: torch.Tensor = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(
    activations: torch.Tensor,
    labels: torch.Tensor,
    t1: float,
    t2: float,
    label_smooth: float = 0.0,
    num_iters: int = 5,
    reduction: str = 'mean',
) -> torch.Tensor:
    r"""Bi-Tempered Logistic Loss.

    Args:
        activations (torch.Tensor): A multidimensional tensor with last dimension `num_classes`.
        labels (torch.Tensor): Tensor with the same shape and dtype as activations (one-hot encoded),
            or a long tensor with one dimension less (class indices).
        t1 (float): Temperature 1 (< 1.0 for boundedness of loss).
        t2 (float): Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
        label_smooth (float): Label smoothing parameter, between 0 and 1.
        num_iters (int): Number of iterations to run the normalization method.
        reduction (str): Specifies reduction method to apply to output: 'none', 'mean', or 'sum'.

    """
    if len(labels.shape) < len(activations.shape):
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smooth > 0:
        num_classes: int = labels_onehot.shape[-1]
        labels_onehot = (1.0 - label_smooth * num_classes / (num_classes - 1)) * labels_onehot + label_smooth / (
            num_classes - 1
        )

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = (
        labels_onehot * log_t(labels_onehot + 1e-10, t1)
        - labels_onehot * log_t(probabilities, t1)
        - labels_onehot.pow(2.0 - t1) / (2.0 - t1)
        + probabilities.pow(2.0 - t1) / (2.0 - t1)
    )
    loss_values = loss_values.sum(dim=-1)

    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()
    return loss_values


class BiTemperedLogisticLoss(nn.Module):

    def __init__(
        self,
        t1: float,
        t2: float,
        label_smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smooth = label_smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass


class BinaryBiTemperedLogisticLoss(nn.Module):

    def __init__(
        self,
        t1: float,
        t2: float,
        label_smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smooth = label_smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass
