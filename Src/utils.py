
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import (
    BASE_DIR,
    ARTIFACT_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    RANDOM_SEED
)


# ============================================================
# 1. CREATE PROJECT DIRECTORIES
# ============================================================

def create_directories():

    directories = [
        ARTIFACT_DIR,
        REPORT_DIR,
        RESULTS_DIR,
        PLOTS_DIR
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Project directories created successfully.")


# ============================================================
# 2. SET RANDOM SEED
# ============================================================

def set_random_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    print(f"Random seed set to: {seed}")

    return seed


# ============================================================
# 3. SET UP LOGGER
# ============================================================

def setup_logger(
    name="RailwayDefectDetection",
    log_file="project.log"
):

    create_directories()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    log_path = RESULTS_DIR / log_file

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ============================================================
# 4. SAVE DATA AS JSON
# ============================================================

def save_json(
    data,
    filename,
    directory=RESULTS_DIR
):

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / filename

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print(f"JSON file saved: {file_path}")

    return file_path


# ============================================================
# 5. LOAD DATA FROM JSON
# ============================================================

def load_json(
    filename,
    directory=ARTIFACT_DIR
):

    file_path = Path(directory) / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# ============================================================
# 6. GENERATE TIMESTAMP
# ============================================================

def get_timestamp():

    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 7. INITIALIZE PROJECT
# ============================================================

def initialize_project():

    create_directories()

    set_random_seed()

    logger = setup_logger()

    logger.info("Railway Defect Detection project initialized.")
    logger.info(f"Project directory: {BASE_DIR}")

    return logger


# ============================================================
# 8. MODULE TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY DEFECT DETECTION - UTILITY TEST")
    print("=" * 60)

    logger = initialize_project()

    logger.info("Testing logger functionality.")

    # Test JSON saving and loading
    test_data = {
        "project": "Railway Defect Detection",
        "classes": ["Defective", "Non defective"],
        "image_size": [224, 224],
        "status": "Utility test successful"
    }

    save_json(
        test_data,
        "utils_test.json",
        RESULTS_DIR
    )

    loaded_data = load_json(
        "utils_test.json",
        RESULTS_DIR
    )

    print("\nLoaded JSON Data:")
    print(json.dumps(loaded_data, indent=4))

    print("\nTimestamp:", get_timestamp())

    print("=" * 60)
    print("ALL UTILITY TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

