"""
Linear Regression Node - Individual node wrapper for Linear Regression algorithm.
Supports training linear regression models with configurable hyperparameters.
"""

from typing import Type, Optional, Dict, Any
import pandas as pd
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.ml.algorithms.regression.linear_regression import LinearRegression
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class LinearRegressionInput(NodeInput):
    """Input schema for Linear Regression node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
    target_column: str = Field(..., description="Name of target column")
    
    # Hyperparameters
    fit_intercept: bool = Field(True, description="Calculate intercept for the model")
    random_state: int = Field(42, description="Random seed for reproducibility")


class LinearRegressionOutput(NodeOutput):
    """Output schema for Linear Regression node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")
    
    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    
    training_metrics: Dict[str, float] = Field(..., description="Training metrics (MAE, MSE, RMSE, R²)")
    training_time_seconds: float = Field(..., description="Training duration")
    
    coefficients: list = Field(..., description="Model coefficients")
    intercept: float = Field(..., description="Model intercept")
    
    metadata: Dict[str, Any] = Field(..., description="Training metadata")


class LinearRegressionNode(BaseNode):
    """
    Linear Regression Node - Train linear regression models.

    Responsibilities:
    - Load training dataset from database/S3
    - Train linear regression model
    - Calculate regression metrics
    - Save model artifacts
    - Return model metadata

    Production features:
    - Dataset loading from multiple sources
    - Hyperparameter configuration
    - Model versioning
    - Comprehensive metrics
    """

    node_type = "linear_regression"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return LinearRegressionInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return LinearRegressionOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage."""
        try:
            db = SessionLocal()
            dataset = (
                db.query(Dataset)
                .filter(Dataset.dataset_id == dataset_id, Dataset.is_deleted == False)
                .first()
            )

            if not dataset:
                logger.error(f"Dataset not found: {dataset_id}")
                db.close()
                return None

            if dataset.storage_backend == "s3" and dataset.s3_key:
                logger.info(f"Loading dataset from S3: {dataset.s3_key}")
                file_content = await s3_service.download_file(dataset.s3_key)
                df = pd.read_csv(io.BytesIO(file_content))
            elif dataset.local_path:
                logger.info(f"Loading dataset from local: {dataset.local_path}")
                df = pd.read_csv(dataset.local_path)
            else:
                logger.error(f"No storage path found for dataset: {dataset_id}")
                db.close()
                return None

            db.close()
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None

    async def _execute(self, input_data: LinearRegressionInput) -> LinearRegressionOutput:
        """
        Execute linear regression training.

        Args:
            input_data: Validated input data

        Returns:
            Training result with model metadata
        """
        try:
            logger.info(f"Training Linear Regression model")

            # Load training data
            df_train = await self._load_dataset(input_data.train_dataset_id)
            if df_train is None or df_train.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.train_dataset_id} not found or empty",
                    expected_format="Valid dataset ID"
                )

            # Validate target column
            if input_data.target_column not in df_train.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X_train = df_train.drop(columns=[input_data.target_column])
            y_train = df_train[input_data.target_column]

            # Initialize and train model
            model = LinearRegression(
                fit_intercept=input_data.fit_intercept,
            )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id("model_linear_regression")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "linear_regression"
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename

            model.save(model_path)

            logger.info(f"Model saved to {model_path}")

            return LinearRegressionOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=len(X_train.columns),
                training_metrics=training_metadata.get("training_metrics", {}),
                training_time_seconds=training_time,
                coefficients=training_metadata.get("coefficients", []),
                intercept=training_metadata.get("intercept", 0.0),
                metadata={
                    "target_column": input_data.target_column,
                    "feature_names": X_train.columns.tolist(),
                    "hyperparameters": {
                        "fit_intercept": input_data.fit_intercept,
                        "random_state": input_data.random_state,
                    },
                    "full_training_metadata": training_metadata,
                },
            )

        except Exception as e:
            logger.error(f"Linear Regression training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
