"""Training loop."""

import torch

from src.checkpoint import save_last_and_best
from src.engine.evaluator import evaluate
from src.metrics import AverageMeter, accuracy
from src.utils import to_serializable


def _amp_enabled(args, device):
    return bool(args.amp and device.type == "cuda")


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    args,
    logger,
    mixup_fn=None,
):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled(args, device))

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=_amp_enabled(args, device)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if args.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)

        if batch_idx % args.log_interval == 0:
            logger.log(
                f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                f"loss={losses.avg:.4f} acc1={top1.avg:.2f}"
            )
        if args.dry_run and batch_idx >= 1:
            break

    return {"loss": losses.avg, "acc1": top1.avg}


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    args,
    logger,
    start_epoch=0,
    best_acc=0.0,
    mixup_fn=None,
):
    best_path = logger.checkpoint_dir / "best.pt"
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch,
            args,
            logger,
            mixup_fn=mixup_fn,
        )

        if (epoch + 1) % args.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, args, logger, split="val")
        else:
            val_metrics = {"loss": 0.0, "acc1": 0.0, "acc5": 0.0}

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        is_best = val_metrics["acc1"] > best_acc
        if is_best:
            best_acc = val_metrics["acc1"]

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
            "args": to_serializable(args),
        }
        last_path, new_best = save_last_and_best(state, logger.checkpoint_dir, is_best)
        if args.save:
            save_last_and_best(state, args.save, is_best)
        if new_best is not None:
            best_path = new_best

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc1": train_metrics["acc1"],
            "val_loss": val_metrics["loss"],
            "val_acc1": val_metrics["acc1"],
            "val_acc5": val_metrics["acc5"],
            "lr": lr,
        }
        logger.write_metrics(row)
        logger.log(
            f"epoch={epoch} done train_loss={train_metrics['loss']:.4f} "
            f"train_acc1={train_metrics['acc1']:.2f} val_acc1={val_metrics['acc1']:.2f} "
            f"lr={lr:.6g} last={last_path}"
        )
    return best_acc, best_path
