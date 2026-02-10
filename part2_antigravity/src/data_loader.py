import logging

import polars as pl
from sklearn.datasets import load_wine


def load_data() -> pl.DataFrame:
    """
    Loads the Wine dataset from sklearn and converts it to a Polars DataFrame.

    Returns:
        pl.DataFrame: The wine dataset including the target column.
    """
    logging.info("Loading Wine dataset...")
    wine = load_wine()

    # Create DataFrame from data
    df = pl.DataFrame(wine.data, schema=wine.feature_names, orient="row")

    # Add target column
    target_df = pl.DataFrame({"target": wine.target})
    df = pl.concat([df, target_df], how="horizontal")

    logging.info(f"Loaded dataset with shape: {df.shape}")
    return df
