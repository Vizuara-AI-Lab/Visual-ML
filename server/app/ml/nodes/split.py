"""
Train/Test Split Node - Splits dataset into training and test sets.
Supports target selection, configurable ratios, and reproducible splits.

"""

from typing import Type, Optional, Dict, Any
import pandas as pd
import io
from pydantic import Field, field_validator
from pathlib import Path
from sklearn.model_selection import train_test_split
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service
from typing import List, Any


class SplitInput(NodeInput):
    """Input schema for Split node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    target_column: str = Field(..., description="Name of target column")

    # Split ratios
    train_ratio: float = Field(0.8, description="Training set ratio (0-1)")
    test_ratio: float = Field(0.2, description="Test set ratio (0-1)")

    # Split type
    split_type: str = Field("random", description="Split type: random or stratified")

    # Reproducibility
    random_seed: int = Field(42, description="Random seed for reproducibility")
    shuffle: bool = Field(True, description="Whether to shuffle before splitting")

    @field_validator("train_ratio", "test_ratio")
    @classmethod
    def validate_ratio(cls, v: Optional[float]) -> Optional[float]:
        """Validate split ratio."""
        if v is not None:
            if not 0 < v < 1:
                raise ValueError(f"Ratio must be between 0 and 1, got {v}")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate ratios sum to 1."""
        total = self.train_ratio + self.test_ratio
        if not abs(total - 1.0) < 0.01:
            raise ValueError(f"Train and test ratios must sum to 1.0, got {total}")


class SplitOutput(NodeOutput):
    """Output schema for Split node."""

    train_dataset_id: str = Field(..., description="ID of training dataset")
    test_dataset_id: str = Field(..., description="ID of test dataset")

    train_path: str = Field(..., description="Path to training set")
    test_path: str = Field(..., description="Path to test set")

    train_size: int = Field(..., description="Number of samples in training set")
    test_size: int = Field(..., description="Number of samples in test set")

    target_column: str = Field(..., description="Target column name")
    columns: List[str] = Field(..., description="All column names in the dataset")
    feature_columns: List[str] = Field(..., description="Feature column names (excluding target)")

    split_summary: Dict[str, Any] = Field(..., description="Summary of split operation")


class SplitNode(BaseNode):
    """
    Target & Split Node - Select target column and split dataset for ML training.

    Responsibilities:
    - Select target column (y)
    - Treat remaining columns as features (X)
    - Split dataset into train/test sets (default 80/20)
    - Support random and stratified splitting
    - Reproducible splits with random seed
    - Save split datasets separately

    Production features:
    - Reproducible with random seed
    - Stratified splitting for classification tasks
    - Comprehensive split summary
    """

    node_type = "split"

    @property
    def metadata(self) -> NodeMetadata:
        """Define node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.DATA_TRANSFORM,
            primary_output_field="train_dataset_id",
            output_fields={
                "train_dataset_id": "Identifier for the training dataset",
                "test_dataset_id": "Identifier for the test dataset",
                "train_size": "Number of samples in training set",
                "test_size": "Number of samples in test set",
                "target_column": "Name of the target column",
                "columns": "All column names in the dataset",
                "feature_columns": "Feature column names (excluding target)",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.DATA_SOURCE,
                NodeCategory.PREPROCESSING,
                NodeCategory.DATA_TRANSFORM,
            ],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return SplitInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return SplitOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            # Recognize common missing value indicators: ?, NA, N/A, null, empty strings
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            # FIRST: Try to load from uploads folder (for preprocessed/engineered datasets)
            upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
            if upload_path.exists():
                logger.info(f"Loading dataset from uploads folder: {upload_path}")
                df = pd.read_csv(upload_path, na_values=missing_values, keep_default_na=True)
                return df

            # SECOND: Try to load from database (for original datasets)
            db = SessionLocal()
            dataset = (
                db.query(Dataset)
                .filter(Dataset.dataset_id == dataset_id, Dataset.is_deleted == False)
                .first()
            )

            if not dataset:
                logger.error(f"Dataset not found in uploads or database: {dataset_id}")
                db.close()
                return None

            if dataset.storage_backend == "s3" and dataset.s3_key:
                logger.info(f"Loading dataset from S3: {dataset.s3_key}")
                file_content = await s3_service.download_file(dataset.s3_key)
                df = pd.read_csv(
                    io.BytesIO(file_content), na_values=missing_values, keep_default_na=True
                )
            elif dataset.local_path:
                logger.info(f"Loading dataset from local: {dataset.local_path}")
                df = pd.read_csv(dataset.local_path, na_values=missing_values, keep_default_na=True)
            else:
                logger.error(f"No storage path found for dataset: {dataset_id}")
                db.close()
                return None

            db.close()
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None

    async def _execute(self, input_data: SplitInput) -> SplitOutput:
        """
        Execute dataset splitting.

        Args:
            input_data: Validated input data

        Returns:
            Split result with paths to split datasets
        """
        try:
            logger.info(f"Splitting dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            # Validate target column
            if input_data.target_column not in df.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X = df.drop(columns=[input_data.target_column])
            y = df[input_data.target_column]

            # Prepare stratify parameter
            stratify_param = y if input_data.split_type == "stratified" else None

            # Perform train/test split
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

            train_id = f"{split_id}_train"
            test_id = f"{split_id}_test"

            train_path = upload_dir / f"{train_id}.csv"
            test_path = upload_dir / f"{test_id}.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info(f"Dataset split complete - Train: {len(train_df)}, Test: {len(test_df)}")

            return SplitOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                train_dataset_id=train_id,
                test_dataset_id=test_id,
                train_path=str(train_path),
                test_path=str(test_path),
                train_size=len(train_df),
                test_size=len(test_df),
                target_column=input_data.target_column,
                columns=train_df.columns.tolist(),
                feature_columns=X.columns.tolist(),
                split_summary={
                    "total_samples": len(df),
                    "target_column": input_data.target_column,
                    "feature_columns": len(X.columns),
                    "train_ratio": input_data.train_ratio,
                    "test_ratio": input_data.test_ratio,
                    "split_type": input_data.split_type,
                    "random_seed": input_data.random_seed,
                    "shuffled": input_data.shuffle,
                },
            )

        except Exception as e:
            logger.error(f"Dataset split failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
