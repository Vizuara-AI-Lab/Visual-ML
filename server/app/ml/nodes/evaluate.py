"""
Evaluation Node - Evaluates trained models and computes metrics.
Supports regression and classification metrics.
"""

from typing import Type, Optional, Dict, Any
import pandas as pd
from pydantic import Field
from pathlib import Path
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.ml.algorithms.regression.linear_regression import LinearRegression
from app.ml.algorithms.classification.logistic_regression import LogisticRegression
from app.core.exceptions import NodeExecutionError, ModelNotFoundError
from app.core.logging import logger


class EvaluateInput(NodeInput):
    """Input schema for Evaluate node."""

    model_path: str = Field(..., description="Path to trained model")
    test_dataset_path: str = Field(..., description="Path to test dataset")
    target_column: str = Field(..., description="Name of target column")
    task_type: str = Field(..., description="Task type: 'regression' or 'classification'")


class EvaluateOutput(NodeOutput):
    """Output schema for Evaluate node."""

    test_samples: int = Field(..., description="Number of test samples")
    metrics: Dict[str, Any] = Field(..., description="Evaluation metrics")

    # Regression metrics
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    r2: Optional[float] = Field(None, description="R-squared score")

    # Classification metrics
    accuracy: Optional[float] = Field(None, description="Accuracy score")
    precision: Optional[float] = Field(None, description="Precision score")
    recall: Optional[float] = Field(None, description="Recall score")
    f1: Optional[float] = Field(None, description="F1 score")
    confusion_matrix: Optional[list] = Field(None, description="Confusion matrix")

    per_class_metrics: Optional[Dict[str, Any]] = Field(None, description="Per-class metrics")


class EvaluateNode(BaseNode):
    """
    Evaluation Node - Compute model performance metrics.

    Responsibilities:
    - Load trained model
    - Load test dataset
    - Make predictions
    - Calculate metrics (task-specific)
    - Return structured evaluation results

    Regression metrics:
    - MAE, MSE, RMSE, R²

    Classification metrics:
    - Accuracy, Precision, Recall, F1
    - Confusion matrix
    - Per-class metrics (multi-class)

    Production features:
    - Comprehensive metrics
    - Task-specific metric selection
    - Confusion matrix for classification
    - Per-class breakdown for multi-class
    """

    node_type = "evaluate"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return EvaluateInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return EvaluateOutput

    async def _execute(self, input_data: EvaluateInput) -> EvaluateOutput:
        """
        Execute model evaluation.

        Args:
            input_data: Validated input data

        Returns:
            Evaluation result with metrics
        """
        try:
            logger.info(f"Evaluating model: {input_data.model_path}")

            # Load model
            model_path = Path(input_data.model_path)
            if not model_path.exists():
                raise ModelNotFoundError(str(model_path))

            model = self._load_model(model_path, input_data.task_type)

            # Load test data
            df_test = pd.read_csv(input_data.test_dataset_path)

            # Validate target column
            if input_data.target_column not in df_test.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X_test = df_test.drop(columns=[input_data.target_column])
            y_test = df_test[input_data.target_column]

            # Evaluate model
            metrics = model.evaluate(X_test, y_test)

            logger.info(f"Evaluation complete - Metrics: {metrics}")

            # Build output based on task type
            output_data = {
                "node_type": self.node_type,
                "execution_time_ms": 0,
                "test_samples": len(X_test),
                "metrics": metrics,
            }

            if input_data.task_type == "regression":
                output_data.update(
                    {
                        "mae": metrics.get("mae"),
                        "mse": metrics.get("mse"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                    }
                )

            elif input_data.task_type == "classification":
                output_data.update(
                    {
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "confusion_matrix": metrics.get("confusion_matrix"),
                        "per_class_metrics": metrics.get("per_class_metrics"),
                    }
                )

            return EvaluateOutput(**output_data)

        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    def _load_model(self, model_path: Path, task_type: str):
        """
        Load model based on task type.

        Args:
            model_path: Path to model file
            task_type: Task type

        Returns:
            Loaded model instance
        """
        if task_type == "regression":
            return LinearRegression.load(model_path)
        elif task_type == "classification":
            return LogisticRegression.load(model_path)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
