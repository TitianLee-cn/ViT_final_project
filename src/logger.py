"""Experiment logging helpers."""

import csv
from pathlib import Path

from .utils import ensure_dir, save_json


class ExperimentLogger:
    def __init__(self, args, experiment_name):
        self.args = args
        self.experiment_name = experiment_name
        self.output_dir = Path(args.output_dir)
        self.log_dir = self.output_dir / "logs" / experiment_name
        self.checkpoint_dir = self.output_dir / "checkpoints" / experiment_name
        ensure_dir(self.log_dir)
        ensure_dir(self.checkpoint_dir)
        self.log_path = self.log_dir / "train_log.txt"
        self.metrics_path = self.log_dir / "metrics.csv"
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        self.init_metrics()

    def log(self, msg):
        print(msg, flush=True)
        self._log_file.write(str(msg) + "\n")
        self._log_file.flush()

    def save_args(self, args):
        save_json(args, self.log_dir / "args.json")

    def init_metrics(self):
        if self.metrics_path.exists():
            return
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc1", "val_loss", "val_acc1", "val_acc5", "lr"])

    def write_metrics(self, row):
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row.get("epoch"),
                    row.get("train_loss"),
                    row.get("train_acc1"),
                    row.get("val_loss"),
                    row.get("val_acc1"),
                    row.get("val_acc5"),
                    row.get("lr"),
                ]
            )

    def save_json(self, obj, filename):
        save_json(obj, self.log_dir / filename)

    def close(self):
        self._log_file.close()
