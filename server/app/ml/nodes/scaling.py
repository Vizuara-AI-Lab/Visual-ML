"""
Scaling Node - Scale and normalize numerical features.
Supports Standard, MinMax, Robust, and Normalizer scaling methods.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import joblib
import io

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class ScalingInput(NodeInput):
    """Input schema for Scaling node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    method: str = Field("standard", description="Scaling method: standard, minmax, robust, normalize")
    columns: List[str] = Field(default_factory=list, description="Columns to scale (empty = all numeric)")


class ScalingOutput(NodeOutput):
    """Output schema for Scaling node."""

    scaled_dataset_id: str = Field(..., description="ID of scaled dataset")
    scaled_path: str = Field(..., description="Path to scaled dataset")
    artifacts_path: Optional[str] = Field(None, description="Path to scaler artifact")
    
    scaled_columns: List[str] = Field(..., description="Columns that were scaled")
    scaling_method: str = Field(..., description="Scaling method used")
    scaling_summary: Dict[str, Any] = Field(..., description="Summary of scaling operations")


class ScalingNode(BaseNode):
    """
    Scaling Node - Scale and normalize features.
    
    Supports:
    - Standard Scaler: (x - mean) / std
    - MinMax Scaler: (x - min) / (max - min)
    - Robust Scaler: Uses median and IQR
    - Normalizer: Scale samples to unit norm
    """

    node_type = "scaling"

    def get_input_schema(self) -> Type[NodeInput]:
        return ScalingInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return ScalingOutput

    async def _execute(self, input_data: ScalingInput) -> ScalingOutput:
        """Execute scaling."""
        try:
            logger.info(f"Starting scaling for dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID"
                )

            df_scaled = df.copy()
            
            # Determine columns to scale
            if input_data.columns:
                columns_to_scale = input_data.columns
            else:
                # Scale all numeric columns
                columns_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()

            # Validate columns exist and are numeric
            for col in columns_to_scale:
                if col not in df.columns:
                    raise ValueError(f"Column {col} not found in dataset")
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Column {col} is not numeric")

            # Select scaler
            if input_data.method == "standard":
                scaler = StandardScaler()
            elif input_data.method == "minmax":
                scaler = MinMaxScaler()
            elif input_data.method == "robust":
                scaler = RobustScaler()
            elif input_data.method == "normalize":
                scaler = Normalizer()
            else:
                raise ValueError(f"Unknown scaling method: {input_data.method}")

            # Apply scaling
            df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

            # Save scaler artifact
            artifacts_id = generate_id("scaler")
            artifacts_path = Path(settings.UPLOAD_DIR) / "artifacts" / f"{artifacts_id}.pkl"
            artifacts_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, artifacts_path)

            # Save scaled dataset
            scaled_id = generate_id("scaled")
            scaled_path = Path(settings.UPLOAD_DIR) / f"{scaled_id}.csv"
            scaled_path.parent.mkdir(parents=True, exist_ok=True)
            df_scaled.to_csv(scaled_path, index=False)

            # Create summary
            scaling_summary = {
                "method": input_data.method,
                "columns_scaled": len(columns_to_scale),
                "scaler_params": scaler.get_params() if hasattr(scaler, 'get_params') else {}
            }

            logger.info(
                f"Scaling complete - Method: {input_data.method}, "
                f"Columns scaled: {len(columns_to_scale)}, "
                f"Saved to: {scaled_path}"
            )

            return ScalingOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                scaled_dataset_id=scaled_id,
                scaled_path=str(scaled_path),
                artifacts_path=str(artifacts_path),
                scaled_columns=columns_to_scale,
                scaling_method=input_data.method,
                scaling_summary=scaling_summary,
            )

        except Exception as e:
            logger.error(f"Scaling failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type,
                reason=str(e),
                input_data=input_data.model_dump()
            )

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
