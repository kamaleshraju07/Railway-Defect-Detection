
from pathlib import Path

from .config import (
    PROCESSED_DATASET_DIR,
    DEFECTIVE_DIR,
    NON_DEFECTIVE_DIR
)

from .utils import setup_logger


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

CLASS_MAPPING = {
    "Defective": 0,
    "Non defective": 1
}

CLASS_DIRECTORIES = {
    "Defective": DEFECTIVE_DIR,
    "Non defective": NON_DEFECTIVE_DIR
}


# ============================================================
# 2. VALIDATE DATASET DIRECTORIES
# ============================================================

def validate_dataset_directory():

    dataset_path = Path(PROCESSED_DATASET_DIR)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}"
        )

    for class_name, class_directory in CLASS_DIRECTORIES.items():

        if not Path(class_directory).is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_directory}"
            )

    return True


# ============================================================
# 3. LOAD IMAGE PATHS AND LABELS
# ============================================================

def load_dataset_paths():

    logger = setup_logger()

    validate_dataset_directory()

    samples = []

    for class_name, class_directory in CLASS_DIRECTORIES.items():

        class_directory = Path(class_directory)

        image_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        class_label = CLASS_MAPPING[class_name]

        for image_path in image_paths:

            samples.append({
                "image_path": str(image_path),
                "class_name": class_name,
                "label": class_label
            })

        logger.info(
            "%s: %d image files collected",
            class_name,
            len(image_paths)
        )

    if not samples:
        raise ValueError(
            "No supported image files found in the dataset."
        )

    logger.info(
        "Total dataset images collected: %d",
        len(samples)
    )

    return samples


# ============================================================
# 4. GET DATASET SUMMARY
# ============================================================

def get_dataset_summary(samples):

    summary = {
        "total_images": len(samples),
        "class_distribution": {}
    }

    for class_name in CLASS_MAPPING:

        count = sum(
            1 for sample in samples
            if sample["class_name"] == class_name
        )

        summary["class_distribution"][class_name] = count

    return summary


# ============================================================
# 5. LOAD COMPLETE DATASET
# ============================================================

def load_all_datasets():

    print("=" * 60)
    print("RAILWAY DEFECT DETECTION - DATA LOADER")
    print("=" * 60)

    samples = load_dataset_paths()

    summary = get_dataset_summary(samples)

    print("\nDATASET LOADED SUCCESSFULLY")
    print("-" * 60)

    print(f"Total Images: {summary['total_images']}")

    for class_name, count in summary["class_distribution"].items():

        print(f"{class_name}: {count}")

    print("-" * 60)

    print("CLASS LABEL MAPPING:")

    for class_name, label in CLASS_MAPPING.items():

        print(f"{class_name}: {label}")

    print("=" * 60)

    return samples, CLASS_MAPPING


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    samples, class_mapping = load_all_datasets()

