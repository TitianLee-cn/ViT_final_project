"""Metrics used during training and evaluation."""

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def _hard_target(target):
    if target.ndim > 1:
        return target.argmax(dim=1)
    return target


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    target = _hard_target(target)
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res


def compute_topk_metrics(output, target):
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    return {"acc1": acc1.item(), "acc5": acc5.item()}
