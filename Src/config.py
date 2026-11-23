
from pathlib import Path

# ============================================================
# 1. PROJECT DIRECTORY
# ============================================================

# Automatically identifies the project root.
# No hardcoded Windows or user-specific location is required.

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

# ============================================================
# 2. DATASET DIRECTORIES
# ============================================================

DATASET_DIR = BASE_DIR / "dataset"

RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

# Existing dataset classes
DEFECTIVE_DIR = PROCESSED_DATASET_DIR / "Defective"
NON_DEFECTIVE_DIR = PROCESSED_DATASET_DIR / "Non defective"

# Optional split directories, if created in the future
TRAIN_DIR = PROCESSED_DATASET_DIR
VALIDATION_DIR = None
TEST_DIR = None

# ============================================================
# 3. ARTIFACT DIRECTORIES
# ============================================================

ARTIFACT_DIR = BASE_DIR / "Artifacts"

MODEL_SAVE_PATH = ARTIFACT_DIR / "best_model.pth"

CNN_MODEL_PATH = ARTIFACT_DIR / "railway_cnn_model.keras"

TRANSFER_MODEL_PATH = (
    ARTIFACT_DIR / "railway_transfer_learning_model.keras"
)

CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"

TRANSFER_CLASS_NAMES_PATH = (
    ARTIFACT_DIR / "transfer_class_names.json"
)

MODEL_CONFIG_PATH = ARTIFACT_DIR / "model_config.json"

EVALUATION_REPORT_PATH = ARTIFACT_DIR / "evaluation_report.json"

# ============================================================
# 4. REPORT DIRECTORIES
# ============================================================

REPORT_DIR = BASE_DIR / "Reports"

RESULTS_DIR = REPORT_DIR / "results"

PLOTS_DIR = REPORT_DIR / "plots"

OUTPUT_DIR = REPORT_DIR

# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

IMG_SIZE = 224

IMAGE_SIZE = (224, 224)

IMAGE_CHANNELS = 3

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.001

RANDOM_SEED = 42

NUM_CLASSES = 2

CLASS_NAMES = ["Defective", "Non defective"]

MODEL_NAME = "Railway Defect Detection"

# ============================================================
# 6. API CONFIGURATION
# ============================================================

API_HOST = "127.0.0.1"

API_PORT = 8000

# ============================================================
# 7. CREATE REQUIRED OUTPUT DIRECTORIES
# ============================================================

for directory in [
    ARTIFACT_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    PLOTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 8. PROJECT CONFIGURATION VERIFICATION
# ============================================================

def display_configuration():

    print("=" * 60)
    print("RAILWAY DEFECT DETECTION CONFIGURATION")
    print("=" * 60)

    print(f"Project Directory: {BASE_DIR}")
    print(f"Source Directory: {SRC_DIR}")

    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Raw Dataset Directory: {RAW_DATASET_DIR}")
    print(f"Processed Dataset Directory: {PROCESSED_DATASET_DIR}")

    print(f"Defective Directory: {DEFECTIVE_DIR}")
    print(f"Non Defective Directory: {NON_DEFECTIVE_DIR}")

    print(f"Artifact Directory: {ARTIFACT_DIR}")
    print(f"Report Directory: {REPORT_DIR}")

    print("-" * 60)

    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Image Channels: {IMAGE_CHANNELS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Number of Classes: {NUM_CLASSES}")

    print("-" * 60)

    print(f"Dataset Exists: {DATASET_DIR.exists()}")
    print(f"Processed Dataset Exists: {PROCESSED_DATASET_DIR.exists()}")
    print(f"Defective Folder Exists: {DEFECTIVE_DIR.exists()}")
    print(f"Non Defective Folder Exists: {NON_DEFECTIVE_DIR.exists()}")

    print("=" * 60)
    print("CONFIGURATION LOADED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    display_configuration()

