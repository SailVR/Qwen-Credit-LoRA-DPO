import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import TrainerCallback


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class LocalMetricCallback(TrainerCallback):
    def __init__(self, metrics_path):
        self.metrics_path = metrics_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step}
        record.update(logs)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def setup_local_run(run_name, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"{timestamp}_{run_name}")
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log_file = open(os.path.join(log_dir, "train.log"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"Local logs will be saved to: {os.path.abspath(log_dir)}")
    return log_dir


def save_predictions(log_dir, predictions):
    path = os.path.join(log_dir, "predictions.txt")
    with open(path, "w", encoding="utf-8") as f:
        for idx, text in enumerate(predictions, start=1):
            f.write(f"===== Prediction {idx} =====\n")
            f.write(text.strip() + "\n\n")
    print(f"Predictions saved to: {os.path.abspath(path)}")


def plot_loss_curve(log_dir, log_history):
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for item in log_history:
        if "loss" in item:
            train_steps.append(item.get("step", len(train_steps)))
            train_losses.append(item["loss"])
        if "eval_loss" in item:
            eval_steps.append(item.get("step", len(eval_steps)))
            eval_losses.append(item["eval_loss"])

    if not train_losses and not eval_losses:
        print("No loss values found, skipped loss curve.")
        return

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, label="train_loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(log_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {os.path.abspath(path)}")
