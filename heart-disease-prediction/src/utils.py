"""Utility functions for logging, data ingestion, and inspection."""

import json
import logging
from pathlib import Path
from typing import Any, Dict
import urllib.request
import pandas as pd

from src.config import (
    COLUMN_NAMES,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    TARGET_COL,
    UCI_CLEVELAND_URL,
)


def setup_logger(name: str = "heart_disease_logger", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized logger with console formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger("utils")


def download_raw_data(destination_path: Path = RAW_DATA_PATH, url: str = UCI_CLEVELAND_URL) -> Path:
    """Download the Cleveland dataset from UCI repository if not already present."""
    destination_path = Path(destination_path)
    if destination_path.exists() and destination_path.stat().st_size > 0:
        logger.info(f"Raw dataset already present at {destination_path} ({destination_path.stat().st_size} bytes)")
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading UCI Cleveland data from {url} to {destination_path}...")
    try:
        urllib.request.urlretrieve(url, str(destination_path))
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download raw data: {e}")
        raise
    return destination_path


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw Cleveland heart disease data with assigned column names."""
    data_path = Path(data_path)
    if not data_path.exists():
        download_raw_data(data_path)

    # Note: Raw data has no header and contains '?' for missing entries
    df_raw = pd.read_csv(
        data_path,
        header=None,
        names=COLUMN_NAMES,
        dtype=str,  # Load as string first to preserve exact representations (e.g. '?')
    )
    return df_raw


def inspect_dataframe(df: pd.DataFrame, title: str = "Dataset Summary") -> None:
    """Print shape, dtypes, head(), describe(), and class distribution."""
    border = "=" * 80
    print(f"\n{border}")
    print(f" {title.upper()} ")
    print(f"{border}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Head (First 5 Rows) ---")
    print(df.head())
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all"))

    if TARGET_COL in df.columns:
        print(f"\n--- Target Distribution ('{TARGET_COL}') ---")
        val_counts = df[TARGET_COL].value_counts(dropna=False).sort_index()
        val_pcts = df[TARGET_COL].value_counts(normalize=True, dropna=False).sort_index() * 100
        dist_df = pd.DataFrame({"Count": val_counts, "Percentage (%)": val_pcts.round(2)})
        print(dist_df)
    print(f"{border}\n")


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save dictionary data to JSON using pathlib."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON metadata to {file_path}")


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load dictionary data from JSON using pathlib."""
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

