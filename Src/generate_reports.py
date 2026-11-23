
from pathlib import Path
import json
import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "Reports" / "results"
REPORTS_DIR = BASE_DIR / "Reports"

EVALUATION_FILE = RESULTS_DIR / "model_evaluation_results.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD EXISTING EVALUATION RESULTS
# --------------------------------------------------

if not EVALUATION_FILE.exists():
    raise FileNotFoundError(
        f"Evaluation file not found: {EVALUATION_FILE}"
    )

with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

class_names = results["class_names"]
confusion_matrix = results["confusion_matrix"]
class_report = results["classification_report"]

total_images = results["total_images"]
accuracy = results["accuracy"]


# --------------------------------------------------
# 1. GENERATE CLASSIFICATION REPORT
# --------------------------------------------------

report_lines = []

report_lines.append("RAILWAY DEFECT DETECTION")
report_lines.append("MOBILENETV2 CLASSIFICATION REPORT")
report_lines.append("=" * 65)
report_lines.append(f"Evaluation type: {results['evaluation_type']}")
report_lines.append(f"Total images: {total_images}")
report_lines.append(f"Accuracy: {accuracy * 100:.2f}%")
report_lines.append("")
report_lines.append(
    f"{'Class':<20}{'Precision':>12}{'Recall':>12}"
    f"{'F1-score':>12}{'Support':>10}"
)
report_lines.append("-" * 66)

for class_name in class_names:
    metrics = class_report[class_name]

    report_lines.append(
        f"{class_name:<20}"
        f"{metrics['precision']:>12.4f}"
        f"{metrics['recall']:>12.4f}"
        f"{metrics['f1-score']:>12.4f}"
        f"{int(metrics['support']):>10}"
    )

report_lines.append("-" * 66)

for average_name in ["macro avg", "weighted avg"]:
    metrics = class_report[average_name]

    report_lines.append(
        f"{average_name:<20}"
        f"{metrics['precision']:>12.4f}"
        f"{metrics['recall']:>12.4f}"
        f"{metrics['f1-score']:>12.4f}"
        f"{int(metrics['support']):>10}"
    )

report_lines.append("")
report_lines.append(
    "Note: This evaluation uses the full processed dataset."
)
report_lines.append(
    "It is not an independent test-set evaluation."
)

report_path = REPORTS_DIR / "classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))


# --------------------------------------------------
# 2. GENERATE CONFUSION MATRIX
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(confusion_matrix, cmap="Blues")

ax.set_title("Railway Defect Detection - Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))

ax.set_xticklabels(class_names, rotation=15)
ax.set_yticklabels(class_names)

for row in range(len(confusion_matrix)):
    for column in range(len(confusion_matrix[row])):
        ax.text(
            column,
            row,
            str(confusion_matrix[row][column]),
            ha="center",
            va="center",
            color="black",
            fontsize=14
        )

fig.colorbar(image, ax=ax)
fig.tight_layout()

fig.savefig(
    REPORTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# 3. GENERATE CLASS DISTRIBUTION
# --------------------------------------------------

class_counts = [
    int(class_report[name]["support"])
    for name in class_names
]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(class_names, class_counts)

ax.set_title("Railway Dataset - Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")

ax.bar_label(bars, padding=3)

fig.tight_layout()

fig.savefig(
    REPORTS_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# 4. GENERATE DATASET SUMMARY
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

summary_labels = class_names + ["Total Images"]
summary_values = class_counts + [total_images]

bars = ax.bar(summary_labels, summary_values)

ax.set_title("Railway Defect Detection - Dataset Summary")
ax.set_ylabel("Number of Images")

ax.bar_label(bars, padding=3)

fig.tight_layout()

fig.savefig(
    REPORTS_DIR / "dataset_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# COMPLETION MESSAGE
# --------------------------------------------------

print("\nREPORT GENERATION COMPLETED SUCCESSFULLY")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Total images: {total_images}")

print("\nGenerated files:")

print("1. classification_report.txt")
print("2. confusion_matrix.png")
print("3. class_distribution.png")
print("4. dataset_summary.png")

print(f"\nReports saved in: {REPORTS_DIR}")
