"""
MAE Score Node - Calculate Mean Absolute Error for regression models.
"""

from typing import Type, Dict, Any, List, Optional
from pydantic import Field
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError
from app.core.logging import logger


class MAEScoreInput(NodeInput):
    """Input schema for MAE Score node."""

    model_output_id: str = Field(..., description="Model output ID from ML algorithm")

    # Optional fields from ML model output (auto-filled by frontend)
    model_id: Optional[str] = Field(None, description="Model ID")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    training_samples: Optional[int] = Field(None, description="Number of training samples")
    n_features: Optional[int] = Field(None, description="Number of features")
    training_metrics: Optional[Dict[str, float]] = Field(None, description="Training metrics")
    training_time_seconds: Optional[float] = Field(None, description="Training duration")
    coefficients: Optional[List[float]] = Field(None, description="Model coefficients")
    intercept: Optional[float] = Field(None, description="Model intercept")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Training metadata")


class MAEScoreOutput(NodeOutput):
    """Output schema for MAE Score node."""

    mae_score: float = Field(..., description="MAE score value")
    display_value: str = Field(..., description="Formatted display value")
    interpretation: str = Field(..., description="Interpretation of the score")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class MAEScoreNode(BaseNode):
    """
    MAE Score Node - Extract and display Mean Absolute Error from ML model output.

    MAE measures the average absolute difference between predicted and actual values.
    Range: 0 to ∞ (lower is better)
    - 0 = Perfect predictions
    - Higher values = Larger prediction errors
    - Less sensitive to outliers than MSE/RMSE
    """

    node_type = "mae_score"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return MAEScoreInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return MAEScoreOutput

    async def _execute(self, input_data: MAEScoreInput) -> MAEScoreOutput:
        """
        Execute MAE score extraction/display.

        Args:
            input_data: Validated input data with model output

        Returns:
            MAE score result with formatted display
        """
        try:
            logger.info(f"Extracting MAE Score for model: {input_data.model_output_id}")

            # Extract MAE score from training metrics
            mae_score = 0.0
            if input_data.training_metrics and "mae" in input_data.training_metrics:
                mae_score = input_data.training_metrics["mae"]
            elif (
                input_data.training_metrics and "mean_absolute_error" in input_data.training_metrics
            ):
                mae_score = input_data.training_metrics["mean_absolute_error"]
            elif input_data.training_metrics and "MAE" in input_data.training_metrics:
                mae_score = input_data.training_metrics["MAE"]
            else:
                logger.debug(
                    f"MAE score not pre-calculated in training metrics, will use default value: {input_data.training_metrics}"
                )

            # Format display value
            display_value = f"{mae_score:.4f}"

            # Interpretation (relative to the scale of target variable)
            if mae_score < 0.5:
                interpretation = "Excellent - Very low prediction error"
            elif mae_score < 2.0:
                interpretation = "Good - Low prediction error"
            elif mae_score < 5.0:
                interpretation = "Moderate - Average prediction error"
            elif mae_score < 10.0:
                interpretation = "High - Significant prediction error"
            else:
                interpretation = "Very High - Large prediction error"

            # Build model info
            model_info = {
                "model_id": input_data.model_id,
                "model_path": input_data.model_path,
                "training_samples": input_data.training_samples,
                "n_features": input_data.n_features,
            }

            logger.info(f"MAE Score extracted: {mae_score:.4f} ({interpretation})")

            return MAEScoreOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                mae_score=mae_score,
                display_value=display_value,
                interpretation=interpretation,
                model_info=model_info,
            )

        except Exception as e:
            logger.error(f"MAE Score extraction failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
