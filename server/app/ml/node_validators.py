"""
Validation utilities for ML pipeline nodes and models.
"""

from typing import Any, List, Optional, Set
import pandas as pd
import numpy as np
from app.core.exceptions import ValidationError, InvalidDatasetError


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 10,
    required_columns: Optional[List[str]] = None,
    allow_missing: bool = False,
) -> None:
    """
    Validate a pandas DataFrame.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of required column names
        allow_missing: Whether to allow missing values

    Raises:
        InvalidDatasetError: If validation fails
    """
    # Check if DataFrame is empty
    if df.empty:
        raise InvalidDatasetError("DataFrame is empty", expected_format="Non-empty CSV")

    # Check minimum rows
    if len(df) < min_rows:
        raise InvalidDatasetError(
            f"Insufficient rows: {len(df)} (minimum {min_rows} required)",
            expected_format=f"CSV with at least {min_rows} rows",
        )

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise InvalidDatasetError(
                f"Missing required columns: {', '.join(missing_cols)}",
                expected_format=f"CSV with columns: {', '.join(required_columns)}",
            )

    # Check for missing values
    if not allow_missing:
        null_counts = df.isnull().sum()
        if null_counts.any():
            cols_with_nulls = null_counts[null_counts > 0].to_dict()
            raise InvalidDatasetError(
                f"Missing values found in columns: {cols_with_nulls}",
                expected_format="CSV without missing values (or use preprocessing node first)",
            )


def validate_target_column(
    df: pd.DataFrame, target_column: str, task_type: str = "regression"
) -> None:
    """
    Validate target column for ML task.

    Args:
        df: DataFrame containing target
        target_column: Name of target column
        task_type: Type of ML task (regression or classification)

    Raises:
        ValidationError: If validation fails
    """
    if target_column not in df.columns:
        raise ValidationError(
            field="target_column",
            message=f"Target column '{target_column}' not found in dataset",
            received_value=target_column,
        )

    # Check for missing values in target
    if df[target_column].isnull().any():
        null_count = df[target_column].isnull().sum()
        raise InvalidDatasetError(
            f"Target column '{target_column}' has {null_count} missing values",
            expected_format="Target column without missing values",
        )

    # Task-specific validation
    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValidationError(
                field="target_column",
                message=f"Regression target must be numeric, got {df[target_column].dtype}",
                received_value=str(df[target_column].dtype),
            )

    elif task_type == "classification":
        # For classification, can be categorical or numeric
        unique_values = df[target_column].nunique()
        if unique_values < 2:
            raise InvalidDatasetError(
                f"Classification target must have at least 2 classes, found {unique_values}",
                expected_format="Target column with 2+ unique values",
            )

        if unique_values > 100:
            raise InvalidDatasetError(
                f"Classification target has {unique_values} unique values (likely regression task?)",
                expected_format="Target column with reasonable number of classes (< 100)",
            )


def validate_feature_columns(
    df: pd.DataFrame, feature_columns: List[str], allow_categorical: bool = False
) -> None:
    """
    Validate feature columns.

    Args:
        df: DataFrame containing features
        feature_columns: List of feature column names
        allow_categorical: Whether to allow categorical features

    Raises:
        ValidationError: If validation fails
    """
    # Check if columns exist
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise ValidationError(
            field="feature_columns",
            message=f"Feature columns not found: {', '.join(missing_cols)}",
            received_value=feature_columns,
        )

    # Check data types
    for col in feature_columns:
        dtype = df[col].dtype

        if not allow_categorical:
            if dtype == "object" or pd.api.types.is_categorical_dtype(dtype):
                raise ValidationError(
                    field=col,
                    message=f"Feature '{col}' is categorical (type: {dtype}). Use preprocessing or encoding first.",
                    received_value=str(dtype),
                )


def validate_split_ratio(
    train_ratio: float, val_ratio: Optional[float] = None, test_ratio: Optional[float] = None
) -> None:
    """
    Validate train/val/test split ratios.

    Args:
        train_ratio: Training set ratio
        val_ratio: Validation set ratio (optional)
        test_ratio: Test set ratio (optional)

    Raises:
        ValidationError: If ratios are invalid
    """
    ratios = [train_ratio]
    if val_ratio is not None:
        ratios.append(val_ratio)
    if test_ratio is not None:
        ratios.append(test_ratio)

    # Check individual ratios
    for ratio in ratios:
        if not 0 < ratio < 1:
            raise ValidationError(
                field="split_ratio",
                message="Split ratios must be between 0 and 1",
                received_value=ratio,
            )

    # Check sum
    total = sum(ratios)
    if not np.isclose(total, 1.0, atol=0.01):
        raise ValidationError(
            field="split_ratio",
            message=f"Split ratios must sum to 1.0, got {total}",
            received_value=ratios,
        )


def validate_model_hyperparameters(algorithm: str, hyperparameters: dict) -> None:
    """
    Validate model hyperparameters.

    Args:
        algorithm: Algorithm name
        hyperparameters: Dictionary of hyperparameters

    Raises:
        ValidationError: If hyperparameters are invalid
    """
    if algorithm == "linear_regression":
        valid_params = {"fit_intercept", "normalize", "copy_X"}
        invalid = set(hyperparameters.keys()) - valid_params
        if invalid:
            raise ValidationError(
                field="hyperparameters",
                message=f"Invalid parameters for Linear Regression: {', '.join(invalid)}",
                received_value=list(hyperparameters.keys()),
            )

    elif algorithm == "logistic_regression":
        valid_params = {
            "C",
            "penalty",
            "solver",
            "max_iter",
            "tol",
            "fit_intercept",
            "class_weight",
            "random_state",
        }
        invalid = set(hyperparameters.keys()) - valid_params
        if invalid:
            raise ValidationError(
                field="hyperparameters",
                message=f"Invalid parameters for Logistic Regression: {', '.join(invalid)}",
                received_value=list(hyperparameters.keys()),
            )

        # Validate C
        if "C" in hyperparameters and hyperparameters["C"] <= 0:
            raise ValidationError(
                field="C",
                message="C (inverse regularization) must be positive",
                received_value=hyperparameters["C"],
            )

        # Validate max_iter
        if "max_iter" in hyperparameters and hyperparameters["max_iter"] < 1:
            raise ValidationError(
                field="max_iter",
                message="max_iter must be at least 1",
                received_value=hyperparameters["max_iter"],
            )
