import logging
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

# Try to import from local modules if running as script
try:
    from data_loader import load_data
    from features import add_features
    from utils import setup_logging
except ImportError:
    from src.data_loader import load_data
    from src.features import add_features
    from src.utils import setup_logging

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    setup_logging()
    try:
        logging.info("Starting feature engineering...")

        # 1. Load Data
        df = load_data()

        # 2. Add Features
        df = add_features(df)

        # 3. Split Data
        logging.info("Splitting data into train/test (80/20 stratified)...")
        # We need to convert to pandas/numpy for stratified split usually,
        # or use indices. Polars doesn't have stratified split built-in yet (as of some versions).
        # Easiest way: Convert to pandas, split, convert back.
        # Or use sklearn train_test_split on indices.

        y = df["target"].to_numpy()
        indices = range(len(df))

        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

        train_df = df[train_idx]
        test_df = df[test_idx]

        logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # 4. Save to Parquet
        train_path = OUTPUT_DIR / "train.parquet"
        test_path = OUTPUT_DIR / "test.parquet"

        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)

        logging.info(f"Saved train split to {train_path}")
        logging.info(f"Saved test split to {test_path}")

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
