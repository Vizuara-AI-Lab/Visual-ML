"""
Train/Val/Test Split Node - Splits dataset into training, validation, and test sets.
Supports configurable ratios and reproducible splits.
"""

from typing import Type, Optional, Dict, Any
import pandas as pd
from pydantic import Field, field_validator
from pathlib import Path
from sklearn.model_selection import train_test_split
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError
from app.core.config import settings
from app.core.logging import logger
from app.ml.node_validators import validate_split_ratio
from app.utils.ids import generate_id


class SplitInput(NodeInput):
    """Input schema for Split node."""

    dataset_path: str = Field(..., description="Path to dataset file")
    target_column: str = Field(..., description="Name of target column")

    # Split ratios
    train_ratio: float = Field(0.7, description="Training set ratio (0-1)")
    val_ratio: Optional[float] = Field(0.15, description="Validation set ratio (0-1)")
    test_ratio: float = Field(0.15, description="Test set ratio (0-1)")

    # Reproducibility
    random_seed: int = Field(42, description="Random seed for reproducibility")
    shuffle: bool = Field(True, description="Whether to shuffle before splitting")
    stratify: bool = Field(False, description="Stratify split by target (classification)")

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def validate_ratio(cls, v: Optional[float]) -> Optional[float]:
        """Validate split ratio."""
        if v is not None:
            if not 0 < v < 1:
                raise ValueError(f"Ratio must be between 0 and 1, got {v}")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate ratios sum to 1."""
        ratios = [self.train_ratio]
        if self.val_ratio:
            ratios.append(self.val_ratio)
        ratios.append(self.test_ratio)

        total = sum(ratios)
        if not abs(total - 1.0) < 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


class SplitOutput(NodeOutput):
    """Output schema for Split node."""

    train_path: str = Field(..., description="Path to training set")
    val_path: Optional[str] = Field(None, description="Path to validation set")
    test_path: str = Field(..., description="Path to test set")

    train_size: int = Field(..., description="Number of samples in training set")
    val_size: Optional[int] = Field(None, description="Number of samples in validation set")
    test_size: int = Field(..., description="Number of samples in test set")

    split_summary: Dict[str, Any] = Field(..., description="Summary of split operation")


class SplitNode(BaseNode):
    """
    Train/Val/Test Split Node - Dataset splitting for ML training.

    Responsibilities:
    - Split dataset into train/val/test sets
    - Configurable split ratios
    - Reproducible splits with random seed
    - Optional stratification for classification
    - Save split datasets separately
    - Return dataset references

    Production features:
    - Reproducible with random seed
    - Stratified splitting for imbalanced datasets
    - Validation set optional
    - Comprehensive split summary
    """

    node_type = "split"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return SplitInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return SplitOutput

    async def _execute(self, input_data: SplitInput) -> SplitOutput:
        """
        Execute dataset splitting.

        Args:
            input_data: Validated input data

        Returns:
            Split result with paths to split datasets
        """
        try:
            logger.info(f"Splitting dataset: {input_data.dataset_path}")

            # Load dataset
            df = pd.read_csv(input_data.dataset_path)

            # Validate target column
            if input_data.target_column not in df.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X = df.drop(columns=[input_data.target_column])
            y = df[input_data.target_column]

            # Prepare stratify parameter
            stratify_param = y if input_data.stratify else None

            # Perform splits
            if input_data.val_ratio and input_data.val_ratio > 0:
                # Three-way split (train/val/test)

                # First split: separate test set
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X,
                    y,
                    test_size=input_data.test_ratio,
                    random_state=input_data.random_seed,
                    shuffle=input_data.shuffle,
                    stratify=stratify_param,
                )

                # Second split: separate train and val from temp
                val_ratio_adjusted = input_data.val_ratio / (
                    input_data.train_ratio + input_data.val_ratio
                )

                stratify_param_temp = y_temp if input_data.stratify else None

                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=val_ratio_adjusted,
                    random_state=input_data.random_seed,
                    shuffle=input_data.shuffle,
                    stratify=stratify_param_temp,
                )

                # Combine features and target
                train_df = X_train.copy()
                train_df[input_data.target_column] = y_train

                val_df = X_val.copy()
                val_df[input_data.target_column] = y_val

                test_df = X_test.copy()
                test_df[input_data.target_column] = y_test

                # Save datasets
                split_id = generate_id("split")
                upload_dir = Path(settings.UPLOAD_DIR)
                upload_dir.mkdir(parents=True, exist_ok=True)

                train_path = upload_dir / f"{split_id}_train.csv"
                val_path = upload_dir / f"{split_id}_val.csv"
                test_path = upload_dir / f"{split_id}_test.csv"

                train_df.to_csv(train_path, index=False)
                val_df.to_csv(val_path, index=False)
                test_df.to_csv(test_path, index=False)

                logger.info(
                    f"Dataset split complete - Train: {len(train_df)}, "
                    f"Val: {len(val_df)}, Test: {len(test_df)}"
                )

                return SplitOutput(
                    node_type=self.node_type,
                    execution_time_ms=0,
                    train_path=str(train_path),
                    val_path=str(val_path),
                    test_path=str(test_path),
                    train_size=len(train_df),
                    val_size=len(val_df),
                    test_size=len(test_df),
                    split_summary={
                        "total_samples": len(df),
                        "train_ratio": input_data.train_ratio,
                        "val_ratio": input_data.val_ratio,
                        "test_ratio": input_data.test_ratio,
                        "random_seed": input_data.random_seed,
                        "stratified": input_data.stratify,
                        "shuffled": input_data.shuffle,
                    },
                )

            else:
                # Two-way split (train/test only)
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=input_data.test_ratio,
                    random_state=input_data.random_seed,
                    shuffle=input_data.shuffle,
                    stratify=stratify_param,
                )

                # Combine features and target
                train_df = X_train.copy()
                train_df[input_data.target_column] = y_train

                test_df = X_test.copy()
                test_df[input_data.target_column] = y_test

                # Save datasets
                split_id = generate_id("split")
                upload_dir = Path(settings.UPLOAD_DIR)
                upload_dir.mkdir(parents=True, exist_ok=True)

                train_path = upload_dir / f"{split_id}_train.csv"
                test_path = upload_dir / f"{split_id}_test.csv"

                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)

                logger.info(
                    f"Dataset split complete - Train: {len(train_df)}, Test: {len(test_df)}"
                )

                return SplitOutput(
                    node_type=self.node_type,
                    execution_time_ms=0,
                    train_path=str(train_path),
                    val_path=None,
                    test_path=str(test_path),
                    train_size=len(train_df),
                    val_size=None,
                    test_size=len(test_df),
                    split_summary={
                        "total_samples": len(df),
                        "train_ratio": input_data.train_ratio,
                        "test_ratio": input_data.test_ratio,
                        "random_seed": input_data.random_seed,
                        "stratified": input_data.stratify,
                        "shuffled": input_data.shuffle,
                    },
                )

        except Exception as e:
            logger.error(f"Dataset split failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
