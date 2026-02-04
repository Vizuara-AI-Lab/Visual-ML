"""
Transformation Node - Apply mathematical transformations to features.
Supports Log, Square Root, and Box-Cox transformations.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.preprocessing import PowerTransformer
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
    transformation_type: str = Field("log", description="Transformation type: log, sqrt, power")
    columns: List[str] = Field(default_factory=list, description="Columns to transform")


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
            logger.info(f"Transformation type: {input_data.transformation_type}")
            logger.info(f"Columns selected for transformation: {input_data.columns}")
            logger.info(f"Total columns to transform: {len(input_data.columns)}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            logger.info(f"Available columns in dataset: {df.columns.tolist()}")

            original_columns = len(df.columns)
            df_transformed = df.copy()
            transformation_summary = {}
            new_columns_list = []

            # Validate and convert columns to numeric
            non_numeric_cols = []
            numeric_cols = []
            conversion_warnings = []

            for column in input_data.columns:
                if column not in df.columns:
                    logger.warning(f"Column {column} not found in dataset")
                    continue

                # Check if already numeric
                if pd.api.types.is_numeric_dtype(df_transformed[column]):
                    numeric_cols.append(column)
                    continue

                # Try to convert to numeric
                logger.info(
                    f"Column '{column}' is type {df_transformed[column].dtype}, attempting conversion to numeric..."
                )

                # Show sample values for debugging
                sample_values = df_transformed[column].head(5).tolist()
                logger.info(f"Sample values from '{column}': {sample_values}")

                # Convert to numeric, coercing errors to NaN
                original_count = len(df_transformed)
                df_transformed[column] = pd.to_numeric(df_transformed[column], errors="coerce")

                # Check how many values were converted to NaN
                nan_count = df_transformed[column].isna().sum()

                if nan_count == original_count:
                    # All values failed conversion
                    non_numeric_values = df[column].dropna().unique()[:5].tolist()
                    non_numeric_cols.append(
                        f"{column} (contains non-numeric values: {non_numeric_values})"
                    )
                elif nan_count > 0:
                    # Some values failed conversion
                    conversion_warnings.append(
                        f"{column}: {nan_count} non-numeric values converted to NaN"
                    )
                    numeric_cols.append(column)
                    logger.warning(
                        f"Column '{column}' converted to numeric with {nan_count} NaN values"
                    )
                else:
                    # Successful conversion
                    numeric_cols.append(column)
                    logger.info(f"Column '{column}' successfully converted to numeric")

            if non_numeric_cols:
                raise NodeExecutionError(
                    node_type=self.node_type,
                    reason=f"Cannot apply {input_data.transformation_type} transformation to non-numeric columns: {', '.join(non_numeric_cols)}. These columns contain text values that cannot be converted to numbers. Please select only numeric columns or handle missing values first.",
                    input_data=input_data.model_dump(),
                )

            if not numeric_cols:
                raise NodeExecutionError(
                    node_type=self.node_type,
                    reason="No valid numeric columns found for transformation",
                    input_data=input_data.model_dump(),
                )

            # Apply transformation - all transformations modify columns in-place
            for column in numeric_cols:
                if input_data.transformation_type == "log":
                    df_transformed[column] = self._log_transform(df_transformed[column])
                    transformation_summary[column] = {"method": "log"}

                elif input_data.transformation_type == "sqrt":
                    df_transformed[column] = self._sqrt_transform(df_transformed[column])
                    transformation_summary[column] = {"method": "sqrt"}

                elif input_data.transformation_type == "power":
                    df_transformed[column] = self._power_transform(df_transformed[column])
                    transformation_summary[column] = {"method": "box-cox"}

            # Add conversion warnings to summary
            if conversion_warnings:
                transformation_summary["warnings"] = conversion_warnings

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
                transformed_columns=numeric_cols,
                new_columns=new_columns_list,
                transformation_summary=transformation_summary,
            )

        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            # FIRST: Try to load from uploads folder (for preprocessed datasets)
            upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
            if upload_path.exists():
                logger.info(f"Loading dataset from uploads folder: {upload_path}")
                df = pd.read_csv(upload_path)
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

        pt = PowerTransformer(method="box-cox")
        transformed = pt.fit_transform(series.values.reshape(-1, 1))
        return pd.Series(transformed.flatten(), index=series.index)
