"""
R² Score Node - Calculate coefficient of determination for regression models.
"""

from typing import Type, Dict, Any, List, Optional
from pydantic import Field
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError
from app.core.logging import logger


class R2ScoreInput(NodeInput):
    """Input schema for R² Score node."""

    model_output_id: str = Field(..., description="Model output ID from ML algorithm")
    display_format: str = Field(
        "percentage", description="Display format: 'percentage' or 'decimal'"
    )

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


class R2ScoreOutput(NodeOutput):
    """Output schema for R² Score node."""

    r2_score: float = Field(..., description="R² score value")
    display_value: str = Field(..., description="Formatted display value")
    interpretation: str = Field(..., description="Interpretation of the score")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class R2ScoreNode(BaseNode):
    """
    R² Score Node - Extract and display R² score from ML model output.

    R² (coefficient of determination) measures how well predictions fit actual values.
    Range: 0 to 1 (higher is better)
    - 1.0 = Perfect fit
    - 0.0 = Model is no better than predicting the mean
    """

    node_type = "r2_score"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return R2ScoreInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return R2ScoreOutput

    async def _execute(self, input_data: R2ScoreInput) -> R2ScoreOutput:
        """
        Execute R² score extraction/display.

        Args:
            input_data: Validated input data with model output

        Returns:
            R² score result with formatted display
        """
        try:
            logger.info(f"Extracting R² Score for model: {input_data.model_output_id}")

            # Extract R² score from training metrics
            r2_score = 0.0
            if input_data.training_metrics and "r2" in input_data.training_metrics:
                r2_score = input_data.training_metrics["r2"]
            elif input_data.training_metrics and "r2_score" in input_data.training_metrics:
                r2_score = input_data.training_metrics["r2_score"]
            elif input_data.training_metrics and "R²" in input_data.training_metrics:
                r2_score = input_data.training_metrics["R²"]
            else:
                logger.warning(
                    f"R² score not found in training metrics: {input_data.training_metrics}"
                )

            # Format display value
            if input_data.display_format == "percentage":
                display_value = f"{r2_score * 100:.2f}%"
            else:
                display_value = f"{r2_score:.4f}"

            # Interpretation
            if r2_score >= 0.9:
                interpretation = "Excellent fit - Model explains 90%+ of variance"
            elif r2_score >= 0.7:
                interpretation = "Good fit - Model explains 70-90% of variance"
            elif r2_score >= 0.5:
                interpretation = "Moderate fit - Model explains 50-70% of variance"
            elif r2_score >= 0.3:
                interpretation = "Weak fit - Model explains 30-50% of variance"
            else:
                interpretation = "Poor fit - Model explains <30% of variance"

            # Build model info
            model_info = {
                "model_id": input_data.model_id,
                "model_path": input_data.model_path,
                "training_samples": input_data.training_samples,
                "n_features": input_data.n_features,
            }

            logger.info(f"R² Score extracted: {r2_score:.4f} ({interpretation})")

            return R2ScoreOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                r2_score=r2_score,
                display_value=display_value,
                interpretation=interpretation,
                model_info=model_info,
            )

        except Exception as e:
            logger.error(f"R² Score extraction failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
