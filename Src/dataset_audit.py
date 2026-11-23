
from pathlib import Path
from PIL import Image

from .config import (
    PROCESSED_DATASET_DIR,
    DEFECTIVE_DIR,
    NON_DEFECTIVE_DIR,
    RESULTS_DIR
)

from .utils import (
    setup_logger,
    save_json
)


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

CLASS_DIRECTORIES = {
    "Defective": DEFECTIVE_DIR,
    "Non defective": NON_DEFECTIVE_DIR
}


# ============================================================
# 2. CHECK IMAGE VALIDITY
# ============================================================

def validate_image(image_path):

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except Exception:
        return False


# ============================================================
# 3. AUDIT INDIVIDUAL CLASS
# ============================================================

def audit_class(class_name, class_directory, logger):

    class_directory = Path(class_directory)

    result = {
        "class_name": class_name,
        "directory": str(class_directory),
        "total_files": 0,
        "valid_images": 0,
        "corrupted_images": 0,
        "unsupported_files": 0
    }

    if not class_directory.is_dir():
        raise FileNotFoundError(
            f"Class directory not found: {class_directory}"
        )

    files = [
        path for path in class_directory.rglob("*")
        if path.is_file()
    ]

    result["total_files"] = len(files)

    for file_path in files:

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            result["unsupported_files"] += 1
            continue

        if validate_image(file_path):
            result["valid_images"] += 1
        else:
            result["corrupted_images"] += 1

    logger.info(
        "%s: %d valid, %d corrupted, %d unsupported",
        class_name,
        result["valid_images"],
        result["corrupted_images"],
        result["unsupported_files"]
    )

    return result


# ============================================================
# 4. AUDIT COMPLETE DATASET
# ============================================================

def audit_dataset():

    logger = setup_logger()

    logger.info("Starting railway dataset audit.")

    if not Path(PROCESSED_DATASET_DIR).is_dir():
        raise FileNotFoundError(
            f"Processed dataset directory not found: "
            f"{PROCESSED_DATASET_DIR}"
        )

    class_results = {}

    for class_name, class_directory in CLASS_DIRECTORIES.items():

        class_results[class_name] = audit_class(
            class_name,
            class_directory,
            logger
        )

    total_images = sum(
        result["valid_images"] + result["corrupted_images"]
        for result in class_results.values()
    )

    total_valid = sum(
        result["valid_images"]
        for result in class_results.values()
    )

    total_corrupted = sum(
        result["corrupted_images"]
        for result in class_results.values()
    )

    total_unsupported = sum(
        result["unsupported_files"]
        for result in class_results.values()
    )

    report = {
        "project": "Railway Defect Detection",
        "dataset_directory": str(PROCESSED_DATASET_DIR),
        "classes": class_results,
        "summary": {
            "total_image_files": total_images,
            "valid_images": total_valid,
            "corrupted_images": total_corrupted,
            "unsupported_files": total_unsupported
        }
    }

    # Save audit report
    save_json(
        report,
        "dataset_audit.json",
        RESULTS_DIR
    )

    return report


# ============================================================
# 5. DISPLAY AUDIT RESULTS
# ============================================================

def display_audit_results(report):

    print("\n" + "=" * 60)
    print("RAILWAY DATASET AUDIT REPORT")
    print("=" * 60)

    for class_name, result in report["classes"].items():

        print(f"\nClass: {class_name}")
        print(f"Total Files: {result['total_files']}")
        print(f"Valid Images: {result['valid_images']}")
        print(f"Corrupted Images: {result['corrupted_images']}")
        print(f"Unsupported Files: {result['unsupported_files']}")

    summary = report["summary"]

    print("\n" + "-" * 60)
    print("OVERALL DATASET SUMMARY")
    print("-" * 60)

    print(f"Total Image Files: {summary['total_image_files']}")
    print(f"Valid Images: {summary['valid_images']}")
    print(f"Corrupted Images: {summary['corrupted_images']}")
    print(f"Unsupported Files: {summary['unsupported_files']}")

    print("=" * 60)
    print("DATASET AUDIT COMPLETED")
    print("=" * 60)


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    audit_report = audit_dataset()

    display_audit_results(audit_report)

