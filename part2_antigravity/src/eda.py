import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Ensure output directory exists (handled by run_command, but good practice in code too)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_balance(df: pl.DataFrame):
    """
    Plots the class balance of the target variable.
    Saves the plot to `output/class_balance.png`.
    """
    logging.info("Plotting class balance...")
    target_counts = df["target"].value_counts().sort("target")

    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_counts["target"], y=target_counts["count"], palette="viridis")
    plt.title("Class Balance of Wine Target")
    plt.xlabel("Target Class")
    plt.ylabel("Count")
    plt.savefig(OUTPUT_DIR / "class_balance.png")
    plt.close()
    logging.info("Saved class balance plot.")


def plot_distributions(df: pl.DataFrame):
    """
    Plots the distribution of each feature.
    Saves the plot to `output/distributions.png`.
    """
    logging.info("Plotting feature distributions...")
    features = [col for col in df.columns if col != "target"]
    n_features = len(features)
    n_rows = (n_features + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribution of {feature}")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distributions.png")
    plt.close()
    logging.info("Saved feature distribution plots.")


def plot_correlation_heatmap(df: pl.DataFrame):
    """
    Plots the correlation heatmap of the features.
    Saves the plot to `output/correlation_heatmap.png`.
    """
    logging.info("Plotting correlation heatmap...")
    # Polars correlation calculation
    matrix = df.corr()
    # Convert to pandas for easier plotting with seaborn heatmaps if needed,
    # but seaborn works nicely with pandas dataframes usually.
    # Note: polars .corr() returns a dataframe.
    # heatmap expects a matrix of numbers.

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix.to_pandas(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=df.columns,
        yticklabels=df.columns,
    )
    plt.title("Feature Correlation Heatmap")
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
    logging.info("Saved correlation heatmap.")


def detect_outliers(df: pl.DataFrame):
    """
    Detects outliers using the IQR method for each feature.
    Logs the number of outliers found per feature.
    """
    logging.info("Detecting outliers...")
    features = [col for col in df.columns if col != "target"]

    outlier_report = {}

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df.filter((pl.col(feature) < lower_bound) | (pl.col(feature) > upper_bound))
        count = outliers.height
        outlier_report[feature] = count

    logging.info("Outlier detection complete.")
    return outlier_report
