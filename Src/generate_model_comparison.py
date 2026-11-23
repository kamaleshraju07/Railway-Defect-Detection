
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "Reports" / "results"
OUTPUT_FILE = BASE_DIR / "Reports" / "model_comparison.csv"

EVALUATION_FILE = RESULTS_DIR / "model_evaluation_results.json"


def load_evaluation_results():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_comparison():
    results = load_evaluation_results()

    report = results.get("classification_report", {})

    weighted = report.get("weighted avg", {})

    accuracy = results.get("accuracy")

    if accuracy is None:
        accuracy = report.get("accuracy")

    if accuracy is None or not weighted:
        raise ValueError(
            "Required evaluation metrics are missing."
        )

    rows = [
        {
            "Model": "MobileNetV2 Transfer Learning",
            "Status": "Evaluated",
            "Evaluation Type": results.get(
                "evaluation_type",
                "Full processed dataset assessment"
            ),
            "Accuracy": accuracy,
            "Precision": weighted.get("precision"),
            "Recall": weighted.get("recall"),
            "F1 Score": weighted.get("f1-score"),
            "Total Images": results.get("total_images", "")
        },
        {
            "Model": "Custom CNN",
            "Status": "Evaluation not available",
            "Evaluation Type": "Not evaluated",
            "Accuracy": "",
            "Precision": "",
            "Recall": "",
            "F1 Score": "",
            "Total Images": ""
        }
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 55)
    print("MODEL COMPARISON REPORT GENERATED")
    print("=" * 55)

    for row in rows:
        print(f"\nModel: {row['Model']}")
        print(f"Status: {row['Status']}")

        if row["Accuracy"] != "":
            print(f"Accuracy: {float(row['Accuracy']) * 100:.2f}%")

    print(f"\nCSV saved successfully:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    generate_comparison()
