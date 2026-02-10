import logging
import sys

import polars as pl

# Try to import from local modules if running as script
try:
    from data_loader import load_data
    from utils import setup_logging
except ImportError:
    # If imported as module in same package
    from src.data_loader import load_data
    from src.utils import setup_logging


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds derived features to the Wine dataset.

    1. magnesium_ash_ratio: magnesium / ash
    2. alcohol_color_ratio: alcohol / color_intensity
    3. log_proline: log(proline)

    Returns:
        pl.DataFrame: DataFrame with new features included.
    """
    logging.info("Adding derived features...")

    # Check if columns exist
    required_cols = ["magnesium", "ash", "alcohol", "color_intensity", "proline"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for feature engineering: {col}")

    df_features = df.with_columns(
        [
            (pl.col("magnesium") / pl.col("ash")).alias("magnesium_ash_ratio"),
            (pl.col("alcohol") / pl.col("color_intensity")).alias("alcohol_color_ratio"),
            (pl.col("proline").log()).alias("log_proline"),
        ]
    )

    logging.info(f"Added features. New shape: {df_features.shape}")
    return df_features


if __name__ == "__main__":
    setup_logging()
    try:
        df = load_data()
        df_feat = add_features(df)
        logging.info("Feature engineering demonstration complete.")
    except Exception as e:
        logging.error(f"Error during feature engineering demo: {e}")
        sys.exit(1)
