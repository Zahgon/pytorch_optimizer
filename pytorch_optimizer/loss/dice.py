from typing import List, Optional, Tuple

import torch
from torch.nn.functional import logsigmoid, one_hot
from torch.nn.modules.loss import _Loss

from pytorch_optimizer.base.type import ClassMode


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    label_smooth: float = 0.0,
    eps: float = 1e-6,
    dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Get soft dice score.

    Args:
        output (torch.Tensor): Predicted segmentation probabilities.
        target (torch.Tensor): Ground truth segmentation masks.
        label_smooth (float): Label smoothing factor to avoid zero denominators.
        eps (float): Small epsilon for numerical stability.
        dims (Optional[Tuple[int, ...]]): Dimensions over which to reduce when computing score.

    """
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    return (2.0 * intersection + label_smooth) / (cardinality + label_smooth).clamp_min(eps)


class DiceLoss(_Loss):

    def __init__(
        self,
        mode: ClassMode = 'binary',
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        label_smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        if classes is not None and mode == 'binary':
            raise ValueError('masking classes is not supported with mode=binary')

        self.mode = mode
        self.classes = classes
        self.from_logits = from_logits
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def aggregate_loss(loss: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def compute_score(
        output: torch.Tensor,
        target: torch.Tensor,
        label_smooth: float = 0.0,
        eps: float = 1e-6,
        dims: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        pass
