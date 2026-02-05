"""
MSE Score Node - Calculate Mean Squared Error for regression models.
"""

from typing import Type, Dict, Any, List, Optional
from pydantic import Field
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError
from app.core.logging import logger


class MSEScoreInput(NodeInput):
    """Input schema for MSE Score node."""

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


class MSEScoreOutput(NodeOutput):
    """Output schema for MSE Score node."""

    mse_score: float = Field(..., description="MSE score value")
    display_value: str = Field(..., description="Formatted display value")
    interpretation: str = Field(..., description="Interpretation of the score")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class MSEScoreNode(BaseNode):
    """
    MSE Score Node - Extract and display Mean Squared Error from ML model output.

    MSE measures the average squared difference between predicted and actual values.
    Range: 0 to ∞ (lower is better)
    - 0 = Perfect predictions
    - Higher values = Larger prediction errors
    """

    node_type = "mse_score"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return MSEScoreInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return MSEScoreOutput

    async def _execute(self, input_data: MSEScoreInput) -> MSEScoreOutput:
        """
        Execute MSE score extraction/display.

        Args:
            input_data: Validated input data with model output

        Returns:
            MSE score result with formatted display
        """
        try:
            logger.info(f"Extracting MSE Score for model: {input_data.model_output_id}")

            # Extract MSE score from training metrics
            mse_score = 0.0
            if input_data.training_metrics and "mse" in input_data.training_metrics:
                mse_score = input_data.training_metrics["mse"]
            elif (
                input_data.training_metrics and "mean_squared_error" in input_data.training_metrics
            ):
                mse_score = input_data.training_metrics["mean_squared_error"]
            elif input_data.training_metrics and "MSE" in input_data.training_metrics:
                mse_score = input_data.training_metrics["MSE"]
            else:
                logger.warning(
                    f"MSE score not found in training metrics: {input_data.training_metrics}"
                )

            # Format display value
            display_value = f"{mse_score:.4f}"

            # Interpretation (relative to the scale of target variable)
            if mse_score < 0.1:
                interpretation = "Excellent - Very low prediction error"
            elif mse_score < 1.0:
                interpretation = "Good - Low prediction error"
            elif mse_score < 10.0:
                interpretation = "Moderate - Average prediction error"
            elif mse_score < 100.0:
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

            logger.info(f"MSE Score extracted: {mse_score:.4f} ({interpretation})")

            return MSEScoreOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                mse_score=mse_score,
                display_value=display_value,
                interpretation=interpretation,
                model_info=model_info,
            )

        except Exception as e:
            logger.error(f"MSE Score extraction failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
