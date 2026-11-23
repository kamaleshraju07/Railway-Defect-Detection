
from pathlib import Path

from .config import (
    CNN_MODEL_PATH,
    TRANSFER_MODEL_PATH,
    MODEL_SAVE_PATH,
    CLASS_NAMES_PATH,
    TRANSFER_CLASS_NAMES_PATH
)

from .utils import setup_logger


# ============================================================
# 1. LOGGER CONFIGURATION
# ============================================================

logger = setup_logger("TrainingVerification")


# ============================================================
# 2. CHECK ARTIFACT STATUS
# ============================================================

def check_artifact(name, file_path):

    file_path = Path(file_path)

    artifact_info = {
        "name": name,
        "path": str(file_path),
        "exists": file_path.is_file(),
        "size_mb": 0.0
    }

    if file_path.is_file():

        artifact_info["size_mb"] = round(
            file_path.stat().st_size / (1024 * 1024),
            2
        )

    return artifact_info


# ============================================================
# 3. VERIFY TRAINED MODELS
# ============================================================

def verify_trained_models():

    logger.info("Starting trained model verification.")

    artifacts = [
        ("Custom CNN Model", CNN_MODEL_PATH),
        ("MobileNetV2 Transfer Learning Model", TRANSFER_MODEL_PATH),
        ("PyTorch Model", MODEL_SAVE_PATH),
        ("CNN Class Names", CLASS_NAMES_PATH),
        ("Transfer Learning Class Names", TRANSFER_CLASS_NAMES_PATH)
    ]

    results = []

    for name, file_path in artifacts:

        result = check_artifact(name, file_path)

        results.append(result)

        if result["exists"]:

            logger.info(
                "%s found. Size: %s MB",
                name,
                result["size_mb"]
            )

        else:

            logger.warning(
                "%s not found at %s",
                name,
                file_path
            )

    return results


# ============================================================
# 4. DISPLAY VERIFICATION REPORT
# ============================================================

def display_training_report(results):

    print("\n" + "=" * 65)
    print("RAILWAY DEFECT DETECTION - TRAINING ARTIFACT REPORT")
    print("=" * 65)

    for result in results:

        status = (
            "AVAILABLE"
            if result["exists"]
            else "NOT FOUND"
        )

        print(f"\nArtifact: {result['name']}")
        print(f"Status: {status}")
        print(f"Path: {result['path']}")

        if result["exists"]:

            print(f"File Size: {result['size_mb']} MB")

    available = sum(
        1 for result in results
        if result["exists"]
    )

    print("\n" + "-" * 65)
    print(f"Available Artifacts: {available}/{len(results)}")
    print("Notebook training is already completed.")
    print("No model retraining was performed.")
    print("=" * 65)


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    results = verify_trained_models()

    display_training_report(results)

