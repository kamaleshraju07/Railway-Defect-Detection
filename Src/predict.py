
from pathlib import Path
from datetime import datetime

import numpy as np

from .config import (
    DEFECTIVE_DIR,
    NON_DEFECTIVE_DIR,
    CLASS_NAMES
)

from .model import initialize_model
from .preprocessing import load_image
from .utils import setup_logger, save_json


# ============================================================
# 1. LOGGER CONFIGURATION
# ============================================================

logger = setup_logger("RailwayPrediction")


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

def load_prediction_model():

    logger.info("Loading trained MobileNetV2 model.")

    model = initialize_model("transfer")

    logger.info("Prediction model loaded successfully.")

    return model


# ============================================================
# 3. PREPARE IMAGE FOR PREDICTION
# ============================================================

def prepare_image(image_path):

    image_array = load_image(image_path)

    # Add batch dimension: (224, 224, 3)
    # becomes (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array.astype(np.float32)


# ============================================================
# 4. PREDICT IMAGE CLASS
# ============================================================

def predict_image(image_path, model=None):

    image_path = Path(image_path)

    if model is None:
        model = load_prediction_model()

    logger.info("Predicting image: %s", image_path)

    processed_image = prepare_image(image_path)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predictions = np.asarray(predictions)

    if predictions.ndim != 2:
        raise ValueError(
            f"Unexpected prediction shape: {predictions.shape}"
        )

    if predictions.shape[0] != 1:
        raise ValueError(
            "Expected predictions for exactly one image."
        )

    if predictions.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            "Model output classes do not match configured "
            "class names."
        )

    # Identify the predicted class
    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_NAMES[predicted_index]

    # Get confidence percentage
    confidence = float(
        predictions[0][predicted_index] * 100
    )

    result = {
        "image_name": image_path.name,
        "predicted_class": predicted_class,
        "confidence_percentage": round(confidence, 2),
        "class_probabilities": {
            class_name: round(
                float(predictions[0][index]) * 100,
                2
            )
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "prediction_time": datetime.now().isoformat()
    }

    logger.info(
        "Prediction completed: %s (%.2f%%)",
        predicted_class,
        confidence
    )

    return result


# ============================================================
# 5. DISPLAY PREDICTION RESULT
# ============================================================

def display_prediction(result):

    print("\n" + "=" * 60)
    print("RAILWAY DEFECT DETECTION - PREDICTION RESULT")
    print("=" * 60)

    print(f"\nImage Name: {result['image_name']}")

    print(
        f"Predicted Class: {result['predicted_class']}"
    )

    print(
        f"Confidence: {result['confidence_percentage']}%"
    )

    print("\nClass Probabilities:")

    for class_name, probability in (
        result["class_probabilities"].items()
    ):

        print(f"{class_name}: {probability}%")

    print("\n" + "=" * 60)


# ============================================================
# 6. SAVE PREDICTION RESULT
# ============================================================

def save_prediction_result(result):

    save_json(
        result,
        "latest_prediction.json"
    )

    logger.info("Prediction result saved successfully.")


# ============================================================
# 7. FIND SAMPLE IMAGE
# ============================================================

def find_sample_image():

    class_directories = [
        DEFECTIVE_DIR,
        NON_DEFECTIVE_DIR
    ]

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    for directory in class_directories:

        directory = Path(directory)

        if not directory.is_dir():
            continue

        image_files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported_extensions
        )

        if image_files:
            return image_files[0]

    raise FileNotFoundError(
        "No supported sample image found in the dataset."
    )


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY DEFECT DETECTION - PREDICTION MODULE")
    print("=" * 60)

    # Load model once
    trained_model = load_prediction_model()

    # Find one sample image
    sample_image = find_sample_image()

    print(f"\nTesting Image: {sample_image}")

    # Generate prediction
    prediction_result = predict_image(
        sample_image,
        trained_model
    )

    # Display prediction
    display_prediction(prediction_result)

    # Save prediction
    save_prediction_result(prediction_result)

    print("\nPREDICTION COMPLETED SUCCESSFULLY")
