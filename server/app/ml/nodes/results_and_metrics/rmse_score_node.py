"""
RMSE Score Node - Calculate Root Mean Squared Error for regression models.
"""

import math
from typing import Type, Dict, Any, List, Optional
from pydantic import Field
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError
from app.core.logging import logger


class RMSEScoreInput(NodeInput):
    """Input schema for RMSE Score node."""

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


class RMSEScoreOutput(NodeOutput):
    """Output schema for RMSE Score node."""

    rmse_score: float = Field(..., description="RMSE score value")
    display_value: str = Field(..., description="Formatted display value")
    interpretation: str = Field(..., description="Interpretation of the score")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class RMSEScoreNode(BaseNode):
    """
    RMSE Score Node - Extract and display Root Mean Squared Error from ML model output.

    RMSE is the square root of MSE, giving error in the same units as the target variable.
    Range: 0 to ∞ (lower is better)
    - 0 = Perfect predictions
    - Higher values = Larger prediction errors
    - More interpretable than MSE as it's in original units
    """

    node_type = "rmse_score"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return RMSEScoreInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return RMSEScoreOutput

    async def _execute(self, input_data: RMSEScoreInput) -> RMSEScoreOutput:
        """
        Execute RMSE score extraction/display.

        Args:
            input_data: Validated input data with model output

        Returns:
            RMSE score result with formatted display
        """
        try:
            logger.info(f"Extracting RMSE Score for model: {input_data.model_output_id}")

            # Extract RMSE score from training metrics
            rmse_score = 0.0
            if input_data.training_metrics and "rmse" in input_data.training_metrics:
                rmse_score = input_data.training_metrics["rmse"]
            elif (
                input_data.training_metrics
                and "root_mean_squared_error" in input_data.training_metrics
            ):
                rmse_score = input_data.training_metrics["root_mean_squared_error"]
            elif input_data.training_metrics and "RMSE" in input_data.training_metrics:
                rmse_score = input_data.training_metrics["RMSE"]
            elif input_data.training_metrics and "mse" in input_data.training_metrics:
                # Calculate RMSE from MSE if RMSE not directly available
                rmse_score = math.sqrt(input_data.training_metrics["mse"])
                logger.info(f"Calculated RMSE from MSE: {rmse_score:.4f}")
            else:
                logger.debug(
                    f"RMSE score not pre-calculated in training metrics, will use default value: {input_data.training_metrics}"
                )

            # Format display value
            display_value = f"{rmse_score:.4f}"

            # Interpretation (relative to the scale of target variable)
            if rmse_score < 0.5:
                interpretation = "Excellent - Very low prediction error"
            elif rmse_score < 2.0:
                interpretation = "Good - Low prediction error"
            elif rmse_score < 5.0:
                interpretation = "Moderate - Average prediction error"
            elif rmse_score < 10.0:
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

            logger.info(f"RMSE Score extracted: {rmse_score:.4f} ({interpretation})")

            return RMSEScoreOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                rmse_score=rmse_score,
                display_value=display_value,
                interpretation=interpretation,
                model_info=model_info,
            )

        except Exception as e:
            logger.error(f"RMSE Score extraction failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
