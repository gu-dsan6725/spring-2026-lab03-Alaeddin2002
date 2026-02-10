import logging
import sys

from data_loader import load_data
from eda import (
    detect_outliers,
    plot_class_balance,
    plot_correlation_heatmap,
    plot_distributions,
)
from utils import log_dict, setup_logging


def main():
    setup_logging()

    # Load Data
    try:
        df = load_data()
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Basic Stats
    logging.info("Generating summary statistics...")
    stats = df.describe()
    stats_dict = stats.to_pandas().to_dict(orient="records")
    log_dict(stats_dict)

    # Visualizations
    plot_class_balance(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)

    # Outliers
    outliers = detect_outliers(df)
    log_dict({"outlier_counts": outliers})


if __name__ == "__main__":
    main()
