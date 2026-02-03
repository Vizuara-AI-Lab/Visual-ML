"""
Feature Selection Node - Select most important features.
Supports Variance Threshold, Correlation, Mutual Information, and SelectKBest methods.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_regression,
    mutual_info_classif,
    f_classif,
    f_regression,
)
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
    method: str = Field(
        "variance", description="Selection method: variance, correlation, mutual_info, kbest"
    )

    # Common parameters
    n_features: Optional[int] = Field(
        None,
        description="Number of features to select (K) - used by correlation (topk mode), mutual_info, and kbest",
    )
    target_column: Optional[str] = Field(
        None, description="Target column (required for mutual_info and kbest methods)"
    )
    task_type: str = Field("regression", description="Task type: regression or classification")

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

    # Mutual Information parameters
    mi_threshold: Optional[float] = Field(
        None, description="Optional MI threshold for advanced filtering"
    )

    # SelectKBest parameters
    scoring_function: Optional[str] = Field(
        None, description="Scoring function for SelectKBest: 'f_test', 'mutual_info', 'chi2'"
    )


class FeatureSelectionOutput(NodeOutput):
    """Output schema for Feature Selection node."""

    selected_dataset_id: str = Field(..., description="ID of dataset with selected features")
    selected_path: str = Field(..., description="Path to selected dataset")

    original_features: int = Field(..., description="Number of features before selection")
    selected_features: int = Field(..., description="Number of features after selection")
    selected_feature_names: List[str] = Field(..., description="Names of selected features")
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
    - Mutual Information: Select based on mutual information with target
    - SelectKBest: Select K best features using statistical tests
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

            # Validate method-specific parameters
            self._validate_parameters(input_data)

            # Apply feature selection
            if input_data.method == "variance":
                df_selected, selected_features = self._variance_selection(
                    df, input_data.variance_threshold
                )

            elif input_data.method == "correlation":
                df_selected, selected_features = self._correlation_selection(
                    df,
                    input_data.correlation_mode,
                    input_data.correlation_threshold,
                    input_data.n_features,
                )

            elif input_data.method == "mutual_info":
                if not input_data.target_column:
                    raise ValueError("Target column required for mutual information selection")
                df_selected, selected_features, feature_scores = self._mutual_info_selection(
                    df, input_data.target_column, input_data.n_features, input_data.task_type
                )

            elif input_data.method == "kbest":
                if not input_data.target_column:
                    raise ValueError("Target column required for SelectKBest")
                df_selected, selected_features, feature_scores = self._kbest_selection(
                    df, input_data.target_column, input_data.n_features, input_data.task_type
                )
            else:
                raise ValueError(f"Unknown selection method: {input_data.method}")

            # Save selected dataset
            selected_id = generate_id("selected")
            selected_path = Path(settings.UPLOAD_DIR) / f"{selected_id}.csv"
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            df_selected.to_csv(selected_path, index=False)

            selection_summary = {
                "method": input_data.method,
                "original_features": original_features,
                "selected_features": len(selected_features),
                "reduction_percentage": round(
                    (1 - len(selected_features) / original_features) * 100, 2
                ),
            }

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

        elif input_data.method == "mutual_info":
            # Mutual Information: requires target column and K
            if not input_data.target_column:
                raise ValueError(
                    "Mutual Information method requires 'target_column'. "
                    "Features are ranked by their mutual information with the target."
                )

            if not input_data.n_features:
                raise ValueError(
                    "Mutual Information is a ranking method that requires 'number_of_features' (K). "
                    "Specify how many top-ranked features to select."
                )

        elif input_data.method == "kbest":
            # SelectKBest: requires target column, K
            # scoring_function is auto-determined from task_type
            if not input_data.target_column:
                raise ValueError(
                    "SelectKBest method requires 'target_column'. "
                    "Features are ranked using statistical tests against the target."
                )

            if not input_data.n_features:
                raise ValueError(
                    "SelectKBest is a ranking method that requires 'number_of_features' (K). "
                    "Specify how many top-ranked features to select."
                )

            # scoring_function is optional - auto-determined from task_type in _kbest_selection

        else:
            raise ValueError(f"Unknown selection method: {input_data.method}")

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

    def _variance_selection(
        self, df: pd.DataFrame, threshold: float
    ) -> tuple[pd.DataFrame, List[str]]:
        """Select features based on variance threshold."""
        numeric_df = df.select_dtypes(include=[np.number])
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric_df)

        selected_features = numeric_df.columns[selector.get_support()].tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Keep non-numeric columns + selected numeric columns
        all_selected = non_numeric_cols + selected_features
        return df[all_selected], selected_features

    def _correlation_selection(
        self, df: pd.DataFrame, mode: str, threshold: float, n_features: Optional[int]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Select features by removing highly correlated ones.

        Two modes:
        - threshold: Remove features with absolute correlation > threshold
        - topk: Remove highly correlated features, then keep top K
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

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

        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        all_selected = non_numeric_cols + selected_features

        return df[all_selected], selected_features

    def _mutual_info_selection(
        self, df: pd.DataFrame, target_column: str, n_features: int, task_type: str
    ) -> tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """Select features based on mutual information with target."""
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Select only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]

        # Calculate mutual information
        if task_type == "classification":
            mi_scores = mutual_info_classif(X_numeric, y)
        else:
            mi_scores = mutual_info_regression(X_numeric, y)

        # Create scores dictionary
        feature_scores = dict(zip(numeric_cols, mi_scores))

        # Select top N features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        selected_features = [f[0] for f in top_features]

        # Include target column
        all_selected = selected_features + [target_column]

        return df[all_selected], selected_features, feature_scores

    def _kbest_selection(
        self, df: pd.DataFrame, target_column: str, n_features: int, task_type: str
    ) -> tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """Select K best features using statistical tests."""
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Select only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]

        # Select score function based on task type
        if task_type == "classification":
            score_func = f_classif
        else:
            score_func = f_regression

        # Apply SelectKBest
        selector = SelectKBest(score_func=score_func, k=min(n_features, len(numeric_cols)))
        selector.fit(X_numeric, y)

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [col for col, selected in zip(numeric_cols, selected_mask) if selected]

        # Get scores
        scores = selector.scores_
        feature_scores = dict(zip(numeric_cols, scores))

        # Include target column
        all_selected = selected_features + [target_column]

        return df[all_selected], selected_features, feature_scores
