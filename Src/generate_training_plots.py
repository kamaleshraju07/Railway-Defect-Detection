
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"
REPORTS_DIR = BASE_DIR / "Reports"

HISTORY_FILE = ARTIFACTS_DIR / "transfer_training_history.json"


def load_training_history():
    """Load the saved MobileNetV2 training history."""

    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_FILE}"
        )

    with open(HISTORY_FILE, "r", encoding="utf-8") as file:
        history = json.load(file)

    required_keys = [
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss"
    ]

    for key in required_keys:
        if key not in history or not history[key]:
            raise ValueError(
                f"Missing or empty training history: {key}"
            )

    lengths = [len(history[key]) for key in required_keys]

    if len(set(lengths)) != 1:
        raise ValueError(
            "Training history arrays have different lengths."
        )

    return history


def generate_accuracy_curve(history):
    """Generate training and validation accuracy graph."""

    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.title("MobileNetV2 Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = REPORTS_DIR / "accuracy_curve.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Accuracy graph saved: {output_path}")


def generate_loss_curve(history):
    """Generate training and validation loss graph."""

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.title("MobileNetV2 Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = REPORTS_DIR / "loss_curve.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Loss graph saved: {output_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("GENERATING TRAINING GRAPHS")
    print("=" * 55)

    history = load_training_history()

    generate_accuracy_curve(history)
    generate_loss_curve(history)

    print("\nTRAINING GRAPHS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
