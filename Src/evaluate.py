
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from .config import (
    PROCESSED_DATASET_DIR,
    RESULTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    CLASS_NAMES
)

from .model import initialize_model
from .utils import setup_logger, save_json


# ============================================================
# 1. LOGGER CONFIGURATION
# ============================================================

logger = setup_logger("ModelEvaluation")


# ============================================================
# 2. LOAD EVALUATION DATASET
# ============================================================

def load_evaluation_dataset():

    dataset_path = Path(PROCESSED_DATASET_DIR)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}"
        )

    for class_name in CLASS_NAMES:

        class_path = dataset_path / class_name

        if not class_path.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_path}"
            )

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(dataset_path),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    logger.info("Evaluation dataset loaded successfully.")

    return dataset


# ============================================================
# 3. GENERATE MODEL PREDICTIONS
# ============================================================

def generate_predictions(model, dataset):

    actual_labels = []
    predicted_labels = []
    prediction_scores = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        if predictions.ndim != 2:
            raise ValueError(
                f"Expected two-dimensional predictions, "
                f"received shape {predictions.shape}"
            )

        if predictions.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                "Model output classes do not match CLASS_NAMES. "
                f"Model outputs: {predictions.shape[1]}, "
                f"configured classes: {len(CLASS_NAMES)}"
            )

        batch_predictions = np.argmax(
            predictions,
            axis=1
        )

        actual_labels.extend(
            labels.numpy().tolist()
        )

        predicted_labels.extend(
            batch_predictions.tolist()
        )

        prediction_scores.extend(
            predictions.tolist()
        )

    return (
        np.array(actual_labels),
        np.array(predicted_labels),
        np.array(prediction_scores)
    )


# ============================================================
# 4. CALCULATE EVALUATION METRICS
# ============================================================

def calculate_metrics(
    actual_labels,
    predicted_labels
):

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES)))
    )

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report
    }


# ============================================================
# 5. SAVE EVALUATION REPORT
# ============================================================

def save_evaluation_report(
    metrics,
    total_images
):

    evaluation_results = {
        "project": "Railway Defect Detection",
        "model": "MobileNetV2 Transfer Learning",
        "evaluation_type": "Full processed dataset assessment",
        "total_images": int(total_images),
        "accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics[
            "classification_report"
        ],
        "class_names": CLASS_NAMES
    }

    save_json(
        evaluation_results,
        "model_evaluation_results.json",
        RESULTS_DIR
    )

    return evaluation_results


# ============================================================
# 6. DISPLAY EVALUATION RESULTS
# ============================================================

def display_evaluation_results(results):

    print("\n" + "=" * 60)
    print("RAILWAY DEFECT DETECTION - MODEL EVALUATION")
    print("=" * 60)

    print(f"\nModel: {results['model']}")

    print(
        f"Total Images: {results['total_images']}"
    )

    print(
        f"Accuracy: {results['accuracy'] * 100:.2f}%"
    )

    print("\nConfusion Matrix:")

    for row in results["confusion_matrix"]:
        print(row)

    print("\nClassification Report:")

    print(
        json.dumps(
            results["classification_report"],
            indent=4
        )
    )

    print("\nEvaluation report saved successfully.")

    print("=" * 60)


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def evaluate_model():

    logger.info("Starting model evaluation.")

    print("=" * 60)
    print("LOADING TRAINED MOBILENETV2 MODEL")
    print("=" * 60)

    model = initialize_model("transfer")

    print("\nLoading processed dataset...")

    dataset = load_evaluation_dataset()

    print("\nGenerating predictions...")

    actual_labels, predicted_labels, scores = (
        generate_predictions(
            model,
            dataset
        )
    )

    metrics = calculate_metrics(
        actual_labels,
        predicted_labels
    )

    results = save_evaluation_report(
        metrics,
        len(actual_labels)
    )

    display_evaluation_results(results)

    logger.info("Model evaluation completed.")

    return results


# ============================================================
# 8. MODULE EXECUTION
# ============================================================

if __name__ == "__main__":

    evaluate_model()
