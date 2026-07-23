from typing import List, Optional

import torch
from torch import nn
from torch.nn.functional import cross_entropy


class LDAMLoss(nn.Module):

    def __init__(
        self, num_class_list: List[int], max_m: float = 0.5, weight: Optional[torch.Tensor] = None, s: float = 30.0
    ):
        super().__init__()

        cls_num_list: torch.Tensor = torch.FloatTensor(num_class_list)
        m_list: torch.Tensor = 1.0 / cls_num_list.sqrt_().sqrt_()
        m_list *= max_m / max(m_list)

        self.m_list = m_list.unsqueeze(0)
        self.weight = weight
        self.s = s

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
