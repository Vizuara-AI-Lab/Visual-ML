"""
Encoding Node - Encode categorical variables using various methods.
Supports One-Hot, Label, Ordinal, and Target encoding.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
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


class EncodingInput(NodeInput):
    """Input schema for Encoding node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    encoding_method: str = Field("onehot", description="Encoding method: onehot, label, ordinal, target")
    columns: List[str] = Field(default_factory=list, description="Columns to encode")
    target_column: Optional[str] = Field(None, description="Target column for target encoding")
    drop_first: bool = Field(False, description="Drop first category in one-hot encoding")
    handle_unknown: str = Field("error", description="How to handle unknown categories: error, ignore")


class EncodingOutput(NodeOutput):
    """Output schema for Encoding node."""

    encoded_dataset_id: str = Field(..., description="ID of encoded dataset")
    encoded_path: str = Field(..., description="Path to encoded dataset")
    artifacts_path: Optional[str] = Field(None, description="Path to encoding artifacts")
    
    original_columns: int = Field(..., description="Columns before encoding")
    final_columns: int = Field(..., description="Columns after encoding")
    encoded_columns: List[str] = Field(..., description="Columns that were encoded")
    new_columns: List[str] = Field(..., description="New columns created")
    encoding_summary: Dict[str, Any] = Field(..., description="Summary of encoding operations")


class EncodingNode(BaseNode):
    """
    Encoding Node - Encode categorical variables.
    
    Supports:
    - One-Hot Encoding: Create binary columns for each category
    - Label Encoding: Convert categories to integers
    - Ordinal Encoding: Convert categories to ordered integers
    - Target Encoding: Encode based on target variable mean
    """

    node_type = "encoding"

    def get_input_schema(self) -> Type[NodeInput]:
        return EncodingInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return EncodingOutput

    async def _execute(self, input_data: EncodingInput) -> EncodingOutput:
        """Execute encoding."""
        try:
            logger.info(f"Starting encoding for dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID"
                )

            original_columns = len(df.columns)
            df_encoded = df.copy()
            encoding_summary = {}
            new_columns_list = []
            
            # Apply encoding to specified columns
            for column in input_data.columns:
                if column not in df.columns:
                    logger.warning(f"Column {column} not found in dataset")
                    continue

                if input_data.encoding_method == "onehot":
                    df_encoded, new_cols = self._onehot_encode(
                        df_encoded, column, input_data.drop_first
                    )
                    new_columns_list.extend(new_cols)
                    encoding_summary[column] = {
                        "method": "onehot",
                        "new_columns": new_cols,
                        "unique_values": len(new_cols)
                    }

                elif input_data.encoding_method == "label":
                    df_encoded = self._label_encode(df_encoded, column)
                    encoding_summary[column] = {
                        "method": "label",
                        "unique_values": df[column].nunique()
                    }

                elif input_data.encoding_method == "ordinal":
                    df_encoded = self._ordinal_encode(df_encoded, column)
                    encoding_summary[column] = {
                        "method": "ordinal",
                        "unique_values": df[column].nunique()
                    }

                elif input_data.encoding_method == "target":
                    if not input_data.target_column:
                        raise ValueError("Target column required for target encoding")
                    df_encoded = self._target_encode(
                        df_encoded, column, input_data.target_column
                    )
                    encoding_summary[column] = {
                        "method": "target",
                        "target_column": input_data.target_column
                    }

            final_columns = len(df_encoded.columns)

            # Save encoded dataset
            encoded_id = generate_id("encoded")
            encoded_path = Path(settings.UPLOAD_DIR) / f"{encoded_id}.csv"
            encoded_path.parent.mkdir(parents=True, exist_ok=True)
            df_encoded.to_csv(encoded_path, index=False)

            logger.info(
                f"Encoding complete - Columns: {original_columns} → {final_columns}, "
                f"Saved to: {encoded_path}"
            )

            return EncodingOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                encoded_dataset_id=encoded_id,
                encoded_path=str(encoded_path),
                artifacts_path=None,
                original_columns=original_columns,
                final_columns=final_columns,
                encoded_columns=input_data.columns,
                new_columns=new_columns_list,
                encoding_summary=encoding_summary,
            )

        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}", exc_info=True)
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

    def _onehot_encode(
        self, df: pd.DataFrame, column: str, drop_first: bool = False
    ) -> tuple[pd.DataFrame, List[str]]:
        """Apply one-hot encoding."""
        # Get dummies
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=drop_first)
        new_columns = dummies.columns.tolist()
        
        # Drop original column and add dummies
        df = df.drop(columns=[column])
        df = pd.concat([df, dummies], axis=1)
        
        return df, new_columns

    def _label_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply label encoding."""
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        return df

    def _ordinal_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply ordinal encoding."""
        # Sort categories and assign integers
        categories = sorted(df[column].unique())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        df[column] = df[column].map(mapping)
        return df

    def _target_encode(
        self, df: pd.DataFrame, column: str, target_column: str
    ) -> pd.DataFrame:
        """Apply target encoding."""
        # Calculate mean target value for each category
        target_means = df.groupby(column)[target_column].mean()
        df[f"{column}_encoded"] = df[column].map(target_means)
        df = df.drop(columns=[column])
        return df
