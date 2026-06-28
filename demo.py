import json
import os
import random

from local_logging import plot_loss_curve


def main():
    log_dir = os.path.join("logs", "demo_local_logging")
    os.makedirs(log_dir, exist_ok=True)

    epochs = 10
    offset = random.random() / 5
    log_history = []

    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for epoch in range(2, epochs):
            acc = 1 - 2**-epoch - random.random() / epoch - offset
            loss = 2**-epoch + random.random() / epoch + offset
            record = {"epoch": epoch, "step": epoch, "acc": acc, "loss": loss}
            log_history.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(record)

    plot_loss_curve(log_dir, log_history)
    print(f"Demo logs saved to: {os.path.abspath(log_dir)}")


if __name__ == "__main__":
    main()
