
import tensorflow as tf
from pathlib import Path

from .config import (
    CNN_MODEL_PATH,
    TRANSFER_MODEL_PATH,
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    NUM_CLASSES
)

from .utils import setup_logger


# ============================================================
# 1. LOGGER CONFIGURATION
# ============================================================

logger = setup_logger("ModelManagement")


# ============================================================
# 2. GET MODEL PATH
# ============================================================

def get_model_path(model_type):

    model_type = model_type.lower().strip()

    if model_type == "cnn":
        return Path(CNN_MODEL_PATH)

    elif model_type in [
        "transfer",
        "mobilenet",
        "mobilenetv2"
    ]:
        return Path(TRANSFER_MODEL_PATH)

    else:
        raise ValueError(
            "Invalid model type. Choose 'cnn' or 'transfer'."
        )


# ============================================================
# 3. VALIDATE MODEL FILE
# ============================================================

def validate_model_file(model_path):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found: {model_path}"
        )

    if not model_path.is_file():
        raise ValueError(
            f"Model path is not a file: {model_path}"
        )

    if model_path.stat().st_size == 0:
        raise ValueError(
            f"Model file is empty: {model_path}"
        )

    return model_path


# ============================================================
# 4. LOAD TRAINED KERAS MODEL
# ============================================================

def load_trained_model(model_type="transfer"):

    model_path = get_model_path(model_type)

    validate_model_file(model_path)

    logger.info(
        "Loading %s model from %s",
        model_type,
        model_path
    )

    try:

        model = tf.keras.models.load_model(
            str(model_path),

            custom_objects={
                "preprocess_input":
                    tf.keras.applications.mobilenet_v2.preprocess_input
            },

            compile=False,

            safe_mode=False
        )

        logger.info(
            "%s model loaded successfully.",
            model_type
        )

        return model

    except Exception as error:

        logger.exception(
            "Failed to load %s model.",
            model_type
        )

        raise RuntimeError(
            f"Unable to load model '{model_type}': {error}"
        ) from error


# ============================================================
# 5. DISPLAY MODEL SUMMARY
# ============================================================

def display_model_summary(model):

    print("\n" + "=" * 60)
    print("TRAINED MODEL ARCHITECTURE")
    print("=" * 60)

    model.summary()

    print("=" * 60)


# ============================================================
# 6. VALIDATE MODEL INPUT SHAPE
# ============================================================

def validate_model_input(model):

    input_shape = model.input_shape

    if isinstance(input_shape, list):

        raise ValueError(
            "This module expects a single-input model."
        )

    expected_shape = (
        None,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        IMAGE_CHANNELS
    )

    if tuple(input_shape) != expected_shape:

        raise ValueError(
            f"Unexpected model input shape: {input_shape}. "
            f"Expected: {expected_shape}"
        )

    logger.info(
        "Model input shape validated: %s",
        input_shape
    )

    return True


# ============================================================
# 7. VALIDATE MODEL OUTPUT
# ============================================================

def validate_model_output(model):

    output_shape = model.output_shape

    if isinstance(output_shape, list):

        raise ValueError(
            "This module expects a single-output model."
        )

    if len(output_shape) != 2:

        raise ValueError(
            f"Unexpected model output shape: {output_shape}"
        )

    if output_shape[-1] != NUM_CLASSES:

        raise ValueError(
            f"Expected {NUM_CLASSES} output classes, "
            f"but model has {output_shape[-1]}."
        )

    logger.info(
        "Model output shape validated: %s",
        output_shape
    )

    return True


# ============================================================
# 8. INITIALIZE MODEL
# ============================================================

def initialize_model(model_type="transfer"):

    model = load_trained_model(model_type)

    validate_model_input(model)

    validate_model_output(model)

    display_model_summary(model)

    logger.info(
        "Model initialization completed successfully."
    )

    return model


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY DEFECT DETECTION - MODEL TEST")
    print("=" * 60)

    selected_model = "transfer"

    print(f"\nSelected Model: {selected_model}")

    trained_model = initialize_model(
        selected_model
    )

    print("\nModel Input Shape:")
    print(trained_model.input_shape)

    print("\nModel Output Shape:")
    print(trained_model.output_shape)

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
