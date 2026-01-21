"""
Training Node - Trains ML models (Linear Regression, Logistic Regression).
Supports hyperparameter configuration and model versioning.
"""

from typing import Type, Optional, Dict, Any
import pandas as pd
from pydantic import Field, field_validator
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.ml.algorithms.regression.linear_regression import LinearRegression
from app.ml.algorithms.classification.logistic_regression import LogisticRegression
from app.core.exceptions import NodeExecutionError, ModelTrainingError
from app.core.config import settings
from app.core.logging import logger
from app.ml.node_validators import validate_model_hyperparameters
from app.utils.ids import generate_id


class TrainInput(NodeInput):
    """Input schema for Train node."""

    train_dataset_path: str = Field(..., description="Path to training dataset")
    target_column: str = Field(..., description="Name of target column")

    algorithm: str = Field(..., description="Algorithm to use")
    task_type: str = Field(..., description="Task type: 'regression' or 'classification'")

    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )

    model_name: Optional[str] = Field(None, description="Optional model name")

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate algorithm name."""
        valid_algorithms = ["linear_regression", "logistic_regression"]
        if v not in valid_algorithms:
            raise ValueError(f"Invalid algorithm. Choose from: {', '.join(valid_algorithms)}")
        return v

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task type."""
        valid_types = ["regression", "classification"]
        if v not in valid_types:
            raise ValueError(f"Invalid task type. Choose from: {', '.join(valid_types)}")
        return v


class TrainOutput(NodeOutput):
    """Output schema for Train node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")
    model_version: str = Field(..., description="Model version timestamp")

    algorithm: str = Field(..., description="Algorithm used")
    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")

    training_metrics: Dict[str, float] = Field(..., description="Training metrics")
    training_time_seconds: float = Field(..., description="Training duration")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")


class TrainNode(BaseNode):
    """
    Training Node - Train ML models.

    Responsibilities:
    - Load training dataset
    - Initialize model with hyperparameters
    - Train model on data
    - Calculate training metrics
    - Save model artifacts
    - Track training metadata
    - Version models

    Production features:
    - Hyperparameter support
    - Model versioning (timestamp-based)
    - Comprehensive metadata tracking
    - Automatic artifact saving
    - Multiple algorithm support
    """

    node_type = "train"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return TrainInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return TrainOutput

    async def _execute(self, input_data: TrainInput) -> TrainOutput:
        """
        Execute model training.

        Args:
            input_data: Validated input data

        Returns:
            Training result with model metadata
        """
        try:
            logger.info(f"Training {input_data.algorithm} model")

            # Validate hyperparameters
            if input_data.hyperparameters:
                validate_model_hyperparameters(input_data.algorithm, input_data.hyperparameters)

            # Load training data
            df_train = pd.read_csv(input_data.train_dataset_path)

            # Validate target column
            if input_data.target_column not in df_train.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X_train = df_train.drop(columns=[input_data.target_column])
            y_train = df_train[input_data.target_column]

            # Initialize and train model
            model = self._create_model(input_data.algorithm, input_data.hyperparameters)

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID and version
            model_id = generate_id(f"model_{input_data.algorithm}")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / input_data.algorithm
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename

            model.save(model_path)

            logger.info(f"Model saved to {model_path}")

            # Cleanup old model versions if needed
            self._cleanup_old_models(model_dir)

            return TrainOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                model_id=model_id,
                model_path=str(model_path),
                model_version=model_version,
                algorithm=input_data.algorithm,
                training_samples=len(X_train),
                n_features=len(X_train.columns),
                training_metrics=training_metadata.get("training_metrics", {}),
                training_time_seconds=training_time,
                metadata={
                    "model_name": input_data.model_name,
                    "target_column": input_data.target_column,
                    "feature_names": X_train.columns.tolist(),
                    "hyperparameters": input_data.hyperparameters,
                    "full_training_metadata": training_metadata,
                },
            )

        except ModelTrainingError:
            raise
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    def _create_model(self, algorithm: str, hyperparameters: Dict[str, Any]):
        """
        Create model instance based on algorithm.

        Args:
            algorithm: Algorithm name
            hyperparameters: Model hyperparameters

        Returns:
            Model instance
        """
        if algorithm == "linear_regression":
            return LinearRegression(**hyperparameters)

        elif algorithm == "logistic_regression":
            return LogisticRegression(**hyperparameters)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _cleanup_old_models(self, model_dir: Path) -> None:
        """
        Remove old model versions if exceeding max versions.

        Args:
            model_dir: Directory containing model files
        """
        try:
            # Get all model files
            model_files = sorted(
                model_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True
            )

            # Keep only max versions
            if len(model_files) > settings.MAX_MODEL_VERSIONS:
                for old_model in model_files[settings.MAX_MODEL_VERSIONS :]:
                    old_model.unlink()
                    logger.info(f"Removed old model version: {old_model}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old models: {str(e)}")
