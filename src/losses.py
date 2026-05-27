"""Loss functions for hard and soft targets."""

import torch
from torch import nn
from torch.nn import functional as F


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, target):
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=logits.size(-1)).float()
        return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.sum(-true_dist * log_probs, dim=-1).mean()


def build_criterion(args):
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        return SoftTargetCrossEntropy()
    if args.label_smoothing > 0:
        return LabelSmoothingCrossEntropy(args.label_smoothing)
    return nn.CrossEntropyLoss()
