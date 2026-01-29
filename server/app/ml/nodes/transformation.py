"""
Transformation Node - Apply mathematical transformations to features.
Supports Log, Square Root, Box-Cox, and Polynomial transformations.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
import io

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class TransformationInput(NodeInput):
    """Input schema for Transformation node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    transformation_type: str = Field("log", description="Transformation type: log, sqrt, power, polynomial")
    columns: List[str] = Field(default_factory=list, description="Columns to transform")
    degree: int = Field(2, description="Polynomial degree (for polynomial transformation)")
    include_bias: bool = Field(False, description="Include bias term in polynomial features")


class TransformationOutput(NodeOutput):
    """Output schema for Transformation node."""

    transformed_dataset_id: str = Field(..., description="ID of transformed dataset")
    transformed_path: str = Field(..., description="Path to transformed dataset")
    
    original_columns: int = Field(..., description="Columns before transformation")
    final_columns: int = Field(..., description="Columns after transformation")
    transformed_columns: List[str] = Field(..., description="Columns that were transformed")
    new_columns: List[str] = Field(default_factory=list, description="New columns created")
    transformation_summary: Dict[str, Any] = Field(..., description="Summary of transformations")


class TransformationNode(BaseNode):
    """
    Transformation Node - Apply mathematical transformations.
    
    Supports:
    - Log Transform: log(x + 1) to handle zeros
    - Square Root: sqrt(x)
    - Power Transform: Box-Cox transformation
    - Polynomial Features: Create interaction terms
    """

    node_type = "transformation"

    def get_input_schema(self) -> Type[NodeInput]:
        return TransformationInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return TransformationOutput

    async def _execute(self, input_data: TransformationInput) -> TransformationOutput:
        """Execute transformation."""
        try:
            logger.info(f"Starting transformation for dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID"
                )

            original_columns = len(df.columns)
            df_transformed = df.copy()
            transformation_summary = {}
            new_columns_list = []

            # Apply transformation
            if input_data.transformation_type == "polynomial":
                # Polynomial features create new columns
                df_transformed, new_cols = self._polynomial_transform(
                    df_transformed, input_data.columns, input_data.degree, input_data.include_bias
                )
                new_columns_list = new_cols
                transformation_summary["polynomial"] = {
                    "degree": input_data.degree,
                    "input_columns": input_data.columns,
                    "output_columns": len(new_cols)
                }
            else:
                # Other transformations modify columns in-place
                for column in input_data.columns:
                    if column not in df.columns:
                        logger.warning(f"Column {column} not found in dataset")
                        continue

                    if input_data.transformation_type == "log":
                        df_transformed[column] = self._log_transform(df_transformed[column])
                        transformation_summary[column] = {"method": "log"}

                    elif input_data.transformation_type == "sqrt":
                        df_transformed[column] = self._sqrt_transform(df_transformed[column])
                        transformation_summary[column] = {"method": "sqrt"}

                    elif input_data.transformation_type == "power":
                        df_transformed[column] = self._power_transform(df_transformed[column])
                        transformation_summary[column] = {"method": "box-cox"}

            final_columns = len(df_transformed.columns)

            # Save transformed dataset
            transformed_id = generate_id("transformed")
            transformed_path = Path(settings.UPLOAD_DIR) / f"{transformed_id}.csv"
            transformed_path.parent.mkdir(parents=True, exist_ok=True)
            df_transformed.to_csv(transformed_path, index=False)

            logger.info(
                f"Transformation complete - Columns: {original_columns} → {final_columns}, "
                f"Saved to: {transformed_path}"
            )

            return TransformationOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                transformed_dataset_id=transformed_id,
                transformed_path=str(transformed_path),
                original_columns=original_columns,
                final_columns=final_columns,
                transformed_columns=input_data.columns,
                new_columns=new_columns_list,
                transformation_summary=transformation_summary,
            )

        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}", exc_info=True)
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

    def _log_transform(self, series: pd.Series) -> pd.Series:
        """Apply log transformation (log1p to handle zeros)."""
        if (series < 0).any():
            logger.warning("Negative values found, adding offset")
            series = series - series.min() + 1
        return np.log1p(series)

    def _sqrt_transform(self, series: pd.Series) -> pd.Series:
        """Apply square root transformation."""
        if (series < 0).any():
            logger.warning("Negative values found, taking absolute value")
            series = series.abs()
        return np.sqrt(series)

    def _power_transform(self, series: pd.Series) -> pd.Series:
        """Apply Box-Cox power transformation."""
        if (series <= 0).any():
            logger.warning("Non-positive values found, adding offset")
            series = series - series.min() + 1
        
        pt = PowerTransformer(method='box-cox')
        transformed = pt.fit_transform(series.values.reshape(-1, 1))
        return pd.Series(transformed.flatten(), index=series.index)

    def _polynomial_transform(
        self, df: pd.DataFrame, columns: List[str], degree: int, include_bias: bool
    ) -> tuple[pd.DataFrame, List[str]]:
        """Create polynomial features."""
        # Extract specified columns
        X = df[columns].values
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        # Generate column names
        feature_names = poly.get_feature_names_out(columns)
        
        # Drop original columns and add polynomial features
        df = df.drop(columns=columns)
        poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)
        
        return df, feature_names.tolist()
