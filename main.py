"""Main entry point for ViT image classification experiments."""

from src.arguments import get_args
from src.augmentations.mixup_cutmix import MixupCutmix
from src.checkpoint import load_checkpoint
from src.datasets.builder import build_dataloaders
from src.engine.evaluator import evaluate
from src.engine.trainer import fit
from src.logger import ExperimentLogger
from src.losses import build_criterion
from src.models.builder import build_model
from src.optim import build_optimizer, build_scheduler
from src.utils import count_parameters, create_experiment_name, get_device, set_seed


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device(args)

    experiment_name = args.experiment_name or create_experiment_name(args)
    logger = ExperimentLogger(args, experiment_name)
    logger.save_args(args)

    logger.log(f"Experiment: {experiment_name}")
    logger.log(f"Device: {device}")

    data_info = build_dataloaders(args)
    args.num_classes = args.num_classes or data_info["num_classes"]
    args.image_size = data_info["image_size"]

    model, model_name = build_model(args, data_info["num_classes"], data_info["image_size"])
    model = model.to(device)
    logger.log(f"Model: {model_name}")
    logger.log(f"Trainable parameters: {count_parameters(model):,}")

    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, map_location=device
        )
        logger.log(f"Resumed from {args.resume}: start_epoch={start_epoch}, best_acc={best_acc:.4f}")

    mixup_fn = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup_fn = MixupCutmix(
            num_classes=data_info["num_classes"],
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
        )

    best_acc, best_path = fit(
        model=model,
        train_loader=data_info["train_loader"],
        val_loader=data_info["val_loader"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        logger=logger,
        start_epoch=start_epoch,
        best_acc=best_acc,
        mixup_fn=mixup_fn,
    )

    test_metrics = evaluate(
        model=model,
        data_loader=data_info["test_loader"],
        criterion=criterion,
        device=device,
        args=args,
        logger=logger,
        split="test",
    )
    final_result = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "best_val_acc1": best_acc,
        "best_checkpoint": str(best_path) if best_path else None,
        "test_loss": test_metrics["loss"],
        "test_acc1": test_metrics["acc1"],
        "test_acc5": test_metrics["acc5"],
    }
    logger.save_json(final_result, "final_result.json")
    logger.log(f"Final result: {final_result}")
    logger.close()


if __name__ == "__main__":
    main()
