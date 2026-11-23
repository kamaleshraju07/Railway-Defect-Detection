
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    IMAGE_SIZE,
    IMAGE_CHANNELS
)

from .utils import setup_logger


# ============================================================
# 1. CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

logger = setup_logger("ImagePreprocessing")


# ============================================================
# 2. VALIDATE IMAGE PATH
# ============================================================

def validate_image_path(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Provided path is not a file: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}"
        )

    return image_path


# ============================================================
# 3. LOAD AND RESIZE IMAGE
# ============================================================

def load_image(image_path):

    image_path = validate_image_path(image_path)

    try:
        with Image.open(image_path) as image:

            # Convert image to RGB
            image = image.convert("RGB")

            # Resize image
            image = image.resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0]),
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

        if image_array.shape != (
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            IMAGE_CHANNELS
        ):
            raise ValueError(
                f"Unexpected image shape: {image_array.shape}"
            )

        return image_array

    except Exception as error:

        logger.error(
            "Unable to load image %s: %s",
            image_path,
            error
        )

        raise


# ============================================================
# 4. NORMALIZE IMAGE
# ============================================================

def normalize_image(image_array, preprocessing_mode):

    image_array = np.asarray(
        image_array,
        dtype=np.float32
    )

    if image_array.ndim != 3:
        raise ValueError(
            "Expected a single image with three dimensions."
        )

    if image_array.shape != (
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        IMAGE_CHANNELS
    ):
        raise ValueError(
            f"Invalid image shape: {image_array.shape}"
        )

    if preprocessing_mode == "raw_0_255":

        processed_image = image_array

    elif preprocessing_mode == "rescale_0_1":

        processed_image = image_array / 255.0

    elif preprocessing_mode == "mobilenet_v2":

        processed_image = (image_array / 127.5) - 1.0

    else:

        raise ValueError(
            "Invalid preprocessing mode. "
            "Choose raw_0_255, rescale_0_1, "
            "or mobilenet_v2."
        )

    return processed_image.astype(np.float32)


# ============================================================
# 5. PREPROCESS SINGLE IMAGE
# ============================================================

def preprocess_image(image_path, preprocessing_mode):

    image_array = load_image(image_path)

    processed_image = normalize_image(
        image_array,
        preprocessing_mode
    )

    # Add batch dimension
    processed_image = np.expand_dims(
        processed_image,
        axis=0
    )

    logger.info(
        "Image preprocessing completed: %s",
        image_path
    )

    return processed_image


# ============================================================
# 6. PREPROCESS IMAGE BATCH
# ============================================================

def preprocess_batch(image_paths, preprocessing_mode):

    if not image_paths:
        raise ValueError(
            "Image path list cannot be empty."
        )

    processed_images = []

    for image_path in image_paths:

        image_array = load_image(image_path)

        normalized_image = normalize_image(
            image_array,
            preprocessing_mode
        )

        processed_images.append(normalized_image)

    batch_array = np.stack(
        processed_images,
        axis=0
    )

    logger.info(
        "Image batch preprocessing completed. "
        "Batch size: %d",
        len(processed_images)
    )

    return batch_array


# ============================================================
# 7. PREPROCESSING MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY DEFECT DETECTION - PREPROCESSING TEST")
    print("=" * 60)

    from .config import (
        DEFECTIVE_DIR,
        NON_DEFECTIVE_DIR
    )

    # Find one available image for testing
    sample_image = None

    for class_directory in [
        DEFECTIVE_DIR,
        NON_DEFECTIVE_DIR
    ]:

        class_directory = Path(class_directory)

        if class_directory.is_dir():

            candidates = sorted(
                path
                for path in class_directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            if candidates:
                sample_image = candidates[0]
                break

    if sample_image is None:

        raise FileNotFoundError(
            "No supported sample image found in the dataset."
        )

    print(f"\nSample Image: {sample_image}")

    # Test image loading
    image_array = load_image(sample_image)

    print("\nOriginal Processed Image Shape:")
    print(image_array.shape)

    # Test supported preprocessing modes
    modes = [
        "raw_0_255",
        "rescale_0_1",
        "mobilenet_v2"
    ]

    for mode in modes:

        result = preprocess_image(
            sample_image,
            mode
        )

        print(f"\nPreprocessing Mode: {mode}")
        print(f"Output Shape: {result.shape}")
        print(f"Minimum Value: {result.min():.4f}")
        print(f"Maximum Value: {result.max():.4f}")

    print("\n" + "=" * 60)
    print("PREPROCESSING TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
