from typing import List, Optional, Tuple

import torch
from torch.nn.functional import logsigmoid, one_hot
from torch.nn.modules.loss import _Loss

from pytorch_optimizer.base.type import ClassMode


def soft_jaccard_score(
    output: torch.Tensor,
    target: torch.Tensor,
    label_smooth: float = 0.0,
    eps: float = 1e-6,
    dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    r"""Get soft Jaccard score.

    Args:
        output (torch.Tensor): Predicted segments (probabilities or logits).
        target (torch.Tensor): Ground truth segments.
        label_smooth (float): Label smoothing factor to avoid zero denominators.
        eps (float): Small epsilon for numerical stability.
        dims (Optional[Tuple[int, ...]]): Dimensions to reduce over when computing the score.

    """
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    return (intersection + label_smooth) / (cardinality - intersection + label_smooth).clamp_min(eps)


class JaccardLoss(_Loss):

    def __init__(
        self,
        mode: ClassMode,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        label_smooth: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        if classes is not None and mode == 'binary':
            raise ValueError('masking classes is not supported with mode=binary')

        self.mode = mode
        self.classes = classes
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.label_smooth = label_smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
