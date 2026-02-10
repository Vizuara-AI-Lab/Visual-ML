"""
Random Forest Node - Individual node wrapper for Random Forest algorithm.
Supports both classification and regression tasks with ensemble learning.
"""

from typing import Type, Optional, Dict, Any
import pandas as pd
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.ml.algorithms.classification.random_forest import RandomForestClassifier
from app.ml.algorithms.regression.random_forest_regressor import RandomForestRegressor
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class RandomForestInput(NodeInput):
    """Input schema for Random Forest node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: str = Field(..., description="Name of target column")
    task_type: str = Field(..., description="Task type: 'classification' or 'regression'")

    # Hyperparameters
    n_estimators: int = Field(100, description="Number of trees in the forest")
    max_depth: Optional[int] = Field(None, description="Maximum tree depth (None = unlimited)")
    min_samples_split: int = Field(2, description="Minimum samples required to split node")
    min_samples_leaf: int = Field(1, description="Minimum samples required in leaf node")
    random_state: int = Field(42, description="Random seed for reproducibility")


class RandomForestOutput(NodeOutput):
    """Output schema for Random Forest node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    task_type: str = Field(..., description="Task type used")
    n_estimators: int = Field(..., description="Number of trees in forest")

    training_metrics: Dict[str, Any] = Field(..., description="Training metrics")
    training_time_seconds: float = Field(..., description="Training duration")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")

    # Pass-through fields for downstream nodes (e.g., confusion_matrix)
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (passed from split node)"
    )
    target_column: Optional[str] = Field(None, description="Target column (passed from split node)")


class RandomForestNode(BaseNode):
    """
    Random Forest Node - Train random forest models for classification or regression.

    Responsibilities:
    - Load training dataset from database/S3
    - Train random forest model (classifier or regressor)
    - Calculate task-specific metrics
    - Save model artifacts
    - Return model metadata

    Production features:
    - Supports both classification and regression
    - Ensemble of decision trees
    - Configurable forest size and tree parameters
    - Feature importance from ensemble
    - Parallel training support
    - Model versioning
    """

    node_type = "random_forest"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return RandomForestInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return RandomForestOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            # Recognize common missing value indicators: ?, NA, N/A, null, empty strings
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            # FIRST: Try to load from uploads folder (for train/test datasets from split node)
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

    async def _execute(self, input_data: RandomForestInput) -> RandomForestOutput:
        """Execute random forest training."""
        try:
            logger.info(f"Training Random Forest model ({input_data.task_type})")

            # Load training data
            df_train = await self._load_dataset(input_data.train_dataset_id)
            if df_train is None or df_train.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.train_dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            # Validate target column
            if input_data.target_column not in df_train.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X_train = df_train.drop(columns=[input_data.target_column])
            y_train = df_train[input_data.target_column]

            # Initialize model based on task type
            if input_data.task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=input_data.n_estimators,
                    max_depth=input_data.max_depth,
                    min_samples_split=input_data.min_samples_split,
                    min_samples_leaf=input_data.min_samples_leaf,
                    random_state=input_data.random_state,
                )
            else:  # regression
                model = RandomForestRegressor(
                    n_estimators=input_data.n_estimators,
                    max_depth=input_data.max_depth,
                    min_samples_split=input_data.min_samples_split,
                    min_samples_leaf=input_data.min_samples_leaf,
                    random_state=input_data.random_state,
                )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id(f"model_random_forest_{input_data.task_type}")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "random_forest" / input_data.task_type
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename

            model.save(model_path)

            logger.info(f"Model saved to {model_path}")

            return RandomForestOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=len(X_train.columns),
                task_type=input_data.task_type,
                n_estimators=input_data.n_estimators,
                training_metrics=training_metadata.get("training_metrics", {}),
                training_time_seconds=training_time,
                metadata={
                    "target_column": input_data.target_column,
                    "feature_names": X_train.columns.tolist(),
                    "hyperparameters": {
                        "n_estimators": input_data.n_estimators,
                        "max_depth": input_data.max_depth,
                        "min_samples_split": input_data.min_samples_split,
                        "min_samples_leaf": input_data.min_samples_leaf,
                        "random_state": input_data.random_state,
                    },
                    "full_training_metadata": training_metadata,
                },
                # Pass through split node fields for downstream nodes
                test_dataset_id=input_data.test_dataset_id,
                target_column=input_data.target_column,
            )

        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
