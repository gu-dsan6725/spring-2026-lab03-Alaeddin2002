import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import xgboost as xgb
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    from utils import log_dict, setup_logging
except ImportError:
    from src.utils import log_dict, setup_logging

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_path = OUTPUT_DIR / "train.parquet"
    test_path = OUTPUT_DIR / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Train/Test splits not found in output/. Run 02_feature_engineering.py first."
        )

    logging.info("Loading pre-split data from parquet...")
    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)

    target_col = "target"
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select(target_col).to_numpy().ravel()

    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.select(target_col).to_numpy().ravel()

    logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols


def _scale_data(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    logging.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def _train_cv(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[xgb.XGBClassifier, dict[str, float]]:
    logging.info("Training XGBoost with 5-Fold Stratified CV...")
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    logging.info(f"CV Accuracy Scores: {cv_scores}")
    logging.info(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    cv_stats = {
        "mean_cv_accuracy": float(cv_scores.mean()),
        "std_cv_accuracy": float(cv_scores.std()),
    }
    return clf, cv_stats


def _evaluate_model(
    clf: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    logging.info("Evaluating on test set...")
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    log_dict(metrics)
    return y_pred, metrics


def _plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> None:
    logging.info("Plotting confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()


def _plot_feature_importance(
    clf: xgb.XGBClassifier, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    logging.info("Plotting feature importance...")
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
    return importances, indices


def _generate_report(
    metrics: dict[str, float],
    cv_stats: dict[str, float],
    importances: np.ndarray,
    feature_cols: list[str],
    sorted_indices: np.ndarray,
) -> None:
    report_path = OUTPUT_DIR / "evaluation_report.md"
    logging.info(f"Generating evaluation report at {report_path}...")

    report_content = f"""# Wine Classification Evaluation Report

## Model Performance
### Cross-Validation (Training Set)
- **Mean Accuracy**: {cv_stats['mean_cv_accuracy']:.4f}
- **Standard Deviation**: {cv_stats['std_cv_accuracy']:.4f}

### Test Set Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision (Weighted)**: {metrics['precision']:.4f}
- **Recall (Weighted)**: {metrics['recall']:.4f}
- **F1 Score (Weighted)**: {metrics['f1_score']:.4f}

## Feature Importance Analysis
Top 5 most important features:
"""
    for i in range(5):
        if i < len(sorted_indices):
            idx = sorted_indices[i]
            report_content += f"{i + 1}. **{feature_cols[idx]}**: {importances[idx]:.4f}\n"

    report_content += """
## Conclusion
The XGBoost model demonstrates strong performance on the Wine dataset.
Refer to `output/confusion_matrix.png` and `output/feature_importance.png` for visual insights.
"""

    with open(report_path, "w") as f:
        f.write(report_content)

    logging.info("Report generated.")


def main() -> None:
    setup_logging()
    try:
        X_train, y_train, X_test, y_test, feature_cols = _load_data()
        X_train_scaled, X_test_scaled, _ = _scale_data(X_train, X_test)

        clf, cv_stats = _train_cv(X_train_scaled, y_train)

        logging.info("Training final model on full training set...")
        clf.fit(X_train_scaled, y_train)

        model_path = OUTPUT_DIR / "xgboost_model.joblib"
        dump(clf, model_path)
        logging.info(f"Saved model to {model_path}")

        y_pred, metrics = _evaluate_model(clf, X_test_scaled, y_test)

        _plot_confusion_matrix(y_test, y_pred, clf.classes_)
        importances, indices = _plot_feature_importance(clf, feature_cols)
        _generate_report(metrics, cv_stats, importances, feature_cols, indices)

    except Exception as e:
        logging.error(f"Error in model training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
