"""MixUp and CutMix batch-level augmentation."""

import random

import numpy as np
import torch
from torch.nn import functional as F


def one_hot(targets, num_classes):
    if targets.ndim > 1:
        return targets.float()
    return F.one_hot(targets, num_classes=num_classes).float()


def rand_bbox(size, lam):
    _, _, height, width = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


class MixupCutmix:
    def __init__(self, num_classes, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0, switch_prob=0.5):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def _sample_lambda(self, alpha):
        return np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    def __call__(self, images, targets):
        targets = one_hot(targets, self.num_classes).to(images.device)
        if (self.mixup_alpha <= 0 and self.cutmix_alpha <= 0) or random.random() > self.prob:
            return images, targets

        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or random.random() < self.switch_prob
        )
        if use_cutmix:
            lam = self._sample_lambda(self.cutmix_alpha)
            x1, y1, x2, y2 = rand_bbox(images.size(), lam)
            mixed = images.clone()
            mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            area = images.size(-1) * images.size(-2)
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / area)
            mixed_targets = lam * targets + (1.0 - lam) * targets[index]
            return mixed, mixed_targets

        lam = self._sample_lambda(self.mixup_alpha)
        mixed = lam * images + (1.0 - lam) * images[index]
        mixed_targets = lam * targets + (1.0 - lam) * targets[index]
        return mixed, mixed_targets
