import os
import pandas as pd

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "placement_predict_50k Dataset (3)(in).csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "processed_data", "preprocessed_dataset.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def load_preprocessed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Loads preprocessed dataset. If it does not exist yet,
    triggers preprocessing first to generate it.
    """
    if not os.path.exists(path):
        from preprocessing_data import run_preprocessing
        run_preprocessing()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")

    return pd.read_csv(path)


def get_data_summary(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = load_data()

    summary = {
        # Dataset Information
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "total_cells": int(df.size),
        "memory_usage_kb": float(round(df.memory_usage(deep=True).sum() / 1024, 2)),
        "duplicate_rows": int(df.duplicated().sum()),

        # Column Information
        "columns": list(df.columns),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        # Missing Values
        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },
        "missing_percentages": {
            col: float(round((df[col].isnull().sum() / len(df)) * 100, 2))
            for col in df.columns
        },

        # Unique Values
        "unique_values": {
            col: int(df[col].nunique())
            for col in df.columns
        },

        # Numerical Summary
        "numerical_summary": (
            df.describe(include="number")
            .round(2)
            .to_dict()
        ),

        # Categorical Summary
        "categorical_summary": (
            df.describe(include=["object", "string"])
            .fillna("")
            .to_dict()
        ),

        # Correlation Matrix
        "correlation": (
            df.select_dtypes(include="number")
            .corr()
            .round(2)
            .to_dict()
        ),

        # First 10 Rows
        "preview": df.head(10).to_dict(orient="records")
    }

    return summary


if __name__ == "__main__":
    from pprint import pprint
    pprint(get_data_summary())