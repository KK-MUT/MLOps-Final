"""Functions for preprocessing the data."""

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from loguru import logger
import pandas as pd

from ARISA_DSML.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR, CSV_NAME


def get_raw_data() -> None:
    """Download and extract raw diabetes dataset from Kaggle."""
    dataset_name = DATASET
    download_folder = RAW_DATA_DIR
    download_folder.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading dataset {dataset_name}")
    api.dataset_download_files(DATASET, path=str(download_folder), unzip=True)

    logger.info("Download complete.")
    return RAW_DATA_DIR / CSV_NAME


def preprocess_df(file:str|Path) -> tuple[Path, Path]:
    """Preprocess data, split into train/test, save to disk, and return file paths."""
    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing numerical values with mean
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Fill missing categorical values with "unknown"
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col].fillna("unknown", inplace=True)

    # Split dataset
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Save to disk
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info("Train saved as train.csv, Test saved as test.csv")

    return train_path, test_path


if __name__ == "__main__":
    logger.info("Fetching raw dataset")
    raw_csv = get_raw_data()

    logger.info(f"Preprocessing and splitting data from {raw_csv}")
    train_path, test_path = preprocess_df(raw_csv)

    logger.info("Saving preprocessed data")
