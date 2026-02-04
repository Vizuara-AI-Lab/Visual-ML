"""
Feature Selection Node - Select most important features.
Supports Variance Threshold and Correlation methods.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
import io

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class FeatureSelectionInput(NodeInput):
    """Input schema for Feature Selection node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    method: str = Field("variance", description="Selection method: variance, correlation")

    # Common parameters
    n_features: Optional[int] = Field(
        None,
        description="Number of features to select (K) - used by correlation (topk mode)",
    )

    # Variance Threshold parameters
    variance_threshold: float = Field(
        0.0, description="Variance threshold - features with variance below this are removed"
    )

    # Correlation parameters
    correlation_mode: Optional[str] = Field(
        None, description="Correlation mode: 'threshold' or 'topk'"
    )
    correlation_threshold: float = Field(
        0.95,
        description="Correlation threshold - remove features with absolute correlation above this",
    )


class FeatureSelectionOutput(NodeOutput):
    """Output schema for Feature Selection node."""

    selected_dataset_id: str = Field(..., description="ID of dataset with selected features")
    selected_path: str = Field(..., description="Path to selected dataset")

    original_features: int = Field(..., description="Number of features before selection")
    selected_features: int = Field(..., description="Number of features after selection")
    selected_feature_names: List[str] = Field(..., description="Names of selected features")
    removed_feature_names: List[str] = Field(
        default_factory=list, description="Names of removed features"
    )
    feature_scores: Dict[str, float] = Field(
        default_factory=dict, description="Feature importance scores"
    )
    selection_summary: Dict[str, Any] = Field(..., description="Summary of selection")


class FeatureSelectionNode(BaseNode):
    """
    Feature Selection Node - Select most important features.

    Supports:
    - Variance Threshold: Remove low-variance features
    - Correlation: Remove highly correlated features
    """

    node_type = "feature_selection"

    def get_input_schema(self) -> Type[NodeInput]:
        return FeatureSelectionInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return FeatureSelectionOutput

    async def _execute(self, input_data: FeatureSelectionInput) -> FeatureSelectionOutput:
        """Execute feature selection."""
        try:
            logger.info(f"Starting feature selection for dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            original_features = len(df.columns)
            feature_scores = {}
            removed_features = []

            # Validate method-specific parameters
            self._validate_parameters(input_data)

            # Apply feature selection
            if input_data.method == "variance":
                df_selected, selected_features, removed_features, feature_scores = (
                    self._variance_selection(df, input_data.variance_threshold)
                )

            elif input_data.method == "correlation":
                # correlation_mode is validated in _validate_parameters and defaults to "threshold"
                mode = input_data.correlation_mode or "threshold"
                df_selected, selected_features, removed_features, feature_scores = (
                    self._correlation_selection(
                        df,
                        mode,
                        input_data.correlation_threshold,
                        input_data.n_features,
                    )
                )
            else:
                raise ValueError(f"Unknown selection method: {input_data.method}")

            # Save selected dataset
            selected_id = generate_id("selected")
            selected_path = Path(settings.UPLOAD_DIR) / f"{selected_id}.csv"
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            df_selected.to_csv(selected_path, index=False)

            # Build selection summary with threshold info
            selection_summary = {
                "method": input_data.method,
                "original_features": original_features,
                "selected_features": len(selected_features),
                "removed_features": len(removed_features),
                "reduction_percentage": round(
                    (1 - len(selected_features) / original_features) * 100, 2
                ),
            }

            # Add method-specific parameters to summary
            if input_data.method == "variance":
                selection_summary["variance_threshold"] = input_data.variance_threshold
            elif input_data.method == "correlation":
                selection_summary["correlation_mode"] = input_data.correlation_mode
                selection_summary["correlation_threshold"] = input_data.correlation_threshold
                if input_data.n_features:
                    selection_summary["n_features"] = input_data.n_features

            logger.info(
                f"Feature selection complete - Method: {input_data.method}, "
                f"Features: {original_features} → {len(selected_features)}, "
                f"Saved to: {selected_path}"
            )

            return FeatureSelectionOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                selected_dataset_id=selected_id,
                selected_path=str(selected_path),
                original_features=original_features,
                selected_features=len(selected_features),
                selected_feature_names=selected_features,
                removed_feature_names=removed_features,
                feature_scores=feature_scores,
                selection_summary=selection_summary,
            )

        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    def _validate_parameters(self, input_data: FeatureSelectionInput) -> None:
        """Validate that parameters are appropriate for the selected method."""

        if input_data.method == "variance":
            # Variance Threshold: only uses threshold, not K
            # Just ignore n_features if provided, don't error
            pass

        elif input_data.method == "correlation":
            # Auto-default to "threshold" mode if not specified
            if not input_data.correlation_mode:
                logger.info("Correlation mode not specified, defaulting to 'threshold' mode")
                input_data.correlation_mode = "threshold"

            if input_data.correlation_mode not in ["threshold", "topk"]:
                raise ValueError(
                    f"Invalid correlation mode: {input_data.correlation_mode}. "
                    "Must be 'threshold' or 'topk'."
                )

            # Validate mode-specific parameters
            if input_data.correlation_mode == "topk" and not input_data.n_features:
                raise ValueError(
                    "Correlation 'topk' mode requires 'number_of_features' (K) parameter. "
                    "Specify how many features to keep."
                )
        else:
            raise ValueError(f"Unknown selection method: {input_data.method}")

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

    def _variance_selection(
        self, df: pd.DataFrame, threshold: float
    ) -> tuple[pd.DataFrame, List[str], List[str], Dict[str, float]]:
        """Select features based on variance threshold.

        Returns:
            - df_selected: DataFrame with selected features
            - selected_features: List of selected numeric feature names
            - removed_features: List of removed numeric feature names
            - variance_scores: Dict mapping all numeric features to their variance
        """
        numeric_df = df.select_dtypes(include=[np.number])
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric_df)

        # Calculate variance for all numeric features
        variance_scores = {col: float(numeric_df[col].var()) for col in numeric_df.columns}

        selected_features = numeric_df.columns[selector.get_support()].tolist()
        removed_features = numeric_df.columns[~selector.get_support()].tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Keep non-numeric columns + selected numeric columns
        all_selected = non_numeric_cols + selected_features
        return df[all_selected], selected_features, removed_features, variance_scores

    def _correlation_selection(
        self, df: pd.DataFrame, mode: str, threshold: float, n_features: Optional[int]
    ) -> tuple[pd.DataFrame, List[str], List[str], Dict[str, float]]:
        """Select features by removing highly correlated ones.

        Two modes:
        - threshold: Remove features with absolute correlation > threshold
        - topk: Remove highly correlated features, then keep top K

        Returns:
            - df_selected: DataFrame with selected features
            - selected_features: List of selected numeric feature names
            - removed_features: List of removed numeric feature names
            - correlation_scores: Dict mapping features to max correlation score
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

        # Calculate max correlation for each feature (excluding self-correlation)
        correlation_scores = {}
        for col in numeric_df.columns:
            # Get max correlation with other features (excluding diagonal)
            other_corrs = corr_matrix[col].drop(col)
            correlation_scores[col] = float(other_corrs.max()) if len(other_corrs) > 0 else 0.0

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        if mode == "threshold":
            # Threshold mode: Remove features with correlation > threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            selected_features = [col for col in numeric_df.columns if col not in to_drop]

        elif mode == "topk":
            # Top-K mode: Remove highly correlated features, then keep top K
            # First remove features with very high correlation (> threshold)
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            remaining_features = [col for col in numeric_df.columns if col not in to_drop]

            # Then select top K features
            selected_features = (
                remaining_features[:n_features] if n_features else remaining_features
            )
        else:
            raise ValueError(f"Invalid correlation mode: {mode}")

        removed_features = [col for col in numeric_df.columns if col not in selected_features]
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        all_selected = non_numeric_cols + selected_features

        return df[all_selected], selected_features, removed_features, correlation_scores
