"""Validation and test evaluation."""

import torch

from src.metrics import AverageMeter, accuracy


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args, logger=None, split="val"):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        if args.dry_run and batch_idx >= 1:
            break

    metrics = {"loss": losses.avg, "acc1": top1.avg, "acc5": top5.avg}
    if logger is not None:
        logger.log(
            f"{split}: loss={metrics['loss']:.4f}, acc1={metrics['acc1']:.2f}, acc5={metrics['acc5']:.2f}"
        )
    return metrics
