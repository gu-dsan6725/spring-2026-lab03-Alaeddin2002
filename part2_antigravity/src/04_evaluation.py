import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Try import local utils
try:
    from utils import setup_logging
except ImportError:
    try:
        from src.utils import setup_logging
    except ImportError:
        logging.basicConfig(level=logging.INFO)

        def setup_logging():
            pass


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model():
    setup_logging()

    # 1. Load Data
    train_path = OUTPUT_DIR / "train.parquet"
    test_path = OUTPUT_DIR / "test.parquet"
    model_path = OUTPUT_DIR / "xgboost_model.joblib"

    if not train_path.exists() or not test_path.exists():
        logging.error("Train/Test splits not found. Run 02_feature_engineering.py first.")
        return

    if not model_path.exists():
        logging.error("Model not found. Run 03_xgboost_model.py first.")
        return

    logging.info("Loading data and model...")
    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)
    clf = load(model_path)

    target_col = "target"
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select(target_col).to_numpy().ravel()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.select(target_col).to_numpy().ravel()

    # 2. Re-fit Scaler (since it wasn't saved)
    logging.info("Re-fitting scaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Evaluate on Test Set
    logging.info("Evaluating on test set...")
    y_pred = clf.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    logging.info(f"Test Metrics: {json.dumps(metrics, indent=2)}")

    # 4. Generate Confusion Matrix Plot
    logging.info("Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    # 5. Generate Feature Importance Plot
    logging.info("Generating feature importance plot...")
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align="center", color="skyblue")
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel("Relative Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png")
        plt.close()
    else:
        logging.warning("Model does not have feature_importances_ attribute.")
        importances = []
        indices = []

    # 6. Re-calculate CV Stats (optional but good for report)
    # Note: Using the trained model for CV might be slightly biased if we re-fit,
    # but here we use cross_val_score which clones the estimator.
    # However, 'clf' is already fitted. cross_val_score will clone it and re-fit on folds.
    logging.info("Recalculating CV scores for reporting...")
    # changing logging level to WARNING to suppress XGBoost logs if possible, or just accept them.
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    logging.info(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")

    # 7. Generate Report
    report_path = OUTPUT_DIR / "evaluation_report.md"
    logging.info(f"Writing report to {report_path}...")

    with open(report_path, "w") as f:
        f.write("# Wine Classification Evaluation Report\n\n")
        f.write("## Model Performance\n")
        f.write("### Cross-Validation (Training Set)\n")
        f.write(f"- **Mean Accuracy**: {cv_mean:.4f}\n")
        f.write(f"- **Standard Deviation**: {cv_std:.4f}\n\n")

        f.write("### Test Set Metrics\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
        f.write(f"- **Precision (Weighted)**: {metrics['precision']:.4f}\n")
        f.write(f"- **Recall (Weighted)**: {metrics['recall']:.4f}\n")
        f.write(f"- **F1 Score (Weighted)**: {metrics['f1_score']:.4f}\n\n")

        f.write("## Feature Importance Analysis\n")
        f.write("Top 5 most important features:\n")
        if len(importances) > 0:
            for i in range(min(5, len(importances))):
                idx = indices[i]
                f.write(f"{i+1}. **{feature_cols[idx]}**: {importances[idx]:.4f}\n")
        else:
            f.write("Feature importance not available.\n")

        f.write("\n## Conclusion\n")
        f.write("The XGBoost model demonstrates strong performance on the Wine dataset.\n")
        f.write(
            "Refer to `output/confusion_matrix.png` and `output/feature_importance.png` "
            "for visual insights.\n"
        )

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    evaluate_model()
