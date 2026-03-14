import torch
from torch.nn import CrossEntropyLoss as CELoss
from configs.base import Config


class CrossEntropyLoss(CELoss):
    def __init__(self, cfg: Config, **kwargs):
        super(CrossEntropyLoss, self).__init__(ignore_index=-100, **kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target) # logits already extracted by trainer
