"""
Select Dataset Node - Allows selecting from previously uploaded datasets.
"""

from typing import Optional, Dict, Any
from pydantic import Field
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.logging import logger
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service
import pandas as pd
import io


class SelectDatasetInput(NodeInput):
    """Input schema for SelectDataset node."""

    dataset_id: str = Field(..., description="Dataset ID to select")
    filename: Optional[str] = Field(None, description="Filename (metadata)")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[list[str]] = Field(None, description="Column names (metadata)")


class SelectDatasetOutput(NodeOutput):
    """Output schema for SelectDataset node."""

    dataset_id: str = Field(..., description="Selected dataset identifier")
    filename: str = Field(..., description="Dataset filename")
    file_path: str = Field(..., description="Path to stored file")
    storage_backend: str = Field(..., description="Storage backend (s3 or local)")
    n_rows: int = Field(..., description="Number of rows in dataset")
    n_columns: int = Field(..., description="Number of columns in dataset")
    columns: list[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types of columns")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")


class SelectDatasetNode(BaseNode):
    """
    Select Dataset Node - Selects from previously uploaded datasets.

    This node retrieves a dataset that was previously uploaded and stored
    in the database, making it available for the pipeline.
    """

    node_type = "select_dataset"

    def get_input_schema(self) -> type[NodeInput]:
        """Return input schema."""
        return SelectDatasetInput

    def get_output_schema(self) -> type[NodeOutput]:
        """Return output schema."""
        return SelectDatasetOutput

    async def _execute(self, input_data: SelectDatasetInput) -> SelectDatasetOutput:
        """
        Execute dataset selection.

        Args:
            input_data: Validated input containing dataset_id

        Returns:
            SelectDatasetOutput with dataset metadata
        """
        try:
            # Get dataset from database
            db = SessionLocal()
            try:
                dataset = (
                    db.query(Dataset).filter(Dataset.dataset_id == input_data.dataset_id).first()
                )

                if not dataset:
                    raise ValueError(f"Dataset not found: {input_data.dataset_id}")

                # Load data from storage to get accurate metadata
                if dataset.storage_backend == "s3" and dataset.s3_key:
                    logger.info(f"Loading dataset from S3: {dataset.s3_key}")
                    file_content = await s3_service.download_file(dataset.s3_key)
                    df = pd.read_csv(io.BytesIO(file_content))
                elif dataset.local_path:
                    logger.info(f"Loading dataset from local: {dataset.local_path}")
                    df = pd.read_csv(dataset.local_path)
                else:
                    raise ValueError(f"No storage path found for dataset: {input_data.dataset_id}")

                # Get metadata
                dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

                logger.info(
                    f"✅ Selected dataset: {dataset.filename} "
                    f"({len(df)} rows × {len(df.columns)} cols)"
                )

                return SelectDatasetOutput(
                    node_type=self.node_type,
                    execution_time_ms=0,
                    dataset_id=dataset.dataset_id,
                    filename=dataset.filename,
                    file_path=dataset.s3_key or dataset.local_path or "",
                    storage_backend=dataset.storage_backend,
                    n_rows=len(df),
                    n_columns=len(df.columns),
                    columns=df.columns.tolist(),
                    dtypes=dtypes,
                    memory_usage_mb=memory_mb,
                )

            finally:
                db.close()

        except Exception as e:
            logger.error(f"❌ Failed to select dataset: {str(e)}")
            raise ValueError(f"Failed to select dataset: {str(e)}")
