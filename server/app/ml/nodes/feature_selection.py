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

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
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

    # Learning activity data (all Optional for backward compatibility)
    feature_score_details: Optional[Dict[str, Any]] = Field(
        None, description="Enriched feature list for score visualization"
    )
    feature_correlation_matrix: Optional[Dict[str, Any]] = Field(
        None, description="Correlation matrix data for heatmap"
    )
    threshold_simulation_data: Optional[Dict[str, Any]] = Field(
        None, description="Pre-sorted features for client-side threshold slider"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about feature selection"
    )


class FeatureSelectionNode(BaseNode):
    """
    Feature Selection Node - Select most important features.

    Supports:
    - Variance Threshold: Remove low-variance features
    - Correlation: Remove highly correlated features
    """

    node_type = "feature_selection"

    @property
    def metadata(self) -> NodeMetadata:
        """Return node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.FEATURE_ENGINEERING,
            primary_output_field="selected_dataset_id",
            output_fields={
                "selected_dataset_id": "ID of dataset with selected features",
                "selected_path": "Path to selected dataset file",
                "selected_feature_names": "Names of selected features",
                "removed_feature_names": "Names of removed features",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            allowed_source_categories=[
                NodeCategory.DATA_SOURCE,
                NodeCategory.PREPROCESSING,
                NodeCategory.FEATURE_ENGINEERING,
            ],
        )

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

            # Generate learning activity data (non-blocking)
            feature_score_details = None
            try:
                feature_score_details = self._generate_feature_score_details(
                    df, feature_scores, selected_features, removed_features, input_data.method
                )
            except Exception as e:
                logger.warning(f"Feature score details generation failed: {e}")

            feature_correlation_matrix = None
            try:
                feature_correlation_matrix = self._generate_feature_correlation_matrix(
                    df, selected_features, removed_features
                )
            except Exception as e:
                logger.warning(f"Feature correlation matrix generation failed: {e}")

            threshold_simulation_data = None
            try:
                threshold_simulation_data = self._generate_threshold_simulation_data(
                    feature_scores, input_data
                )
            except Exception as e:
                logger.warning(f"Threshold simulation data generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_feature_selection_quiz(
                    df, feature_scores, selected_features, removed_features, selection_summary
                )
            except Exception as e:
                logger.warning(f"Feature selection quiz generation failed: {e}")

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
                feature_score_details=feature_score_details,
                feature_correlation_matrix=feature_correlation_matrix,
                threshold_simulation_data=threshold_simulation_data,
                quiz_questions=quiz_questions,
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
            # Recognize common missing value indicators: ?, NA, N/A, null, empty strings
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            # FIRST: Try to load from uploads folder (for preprocessed datasets)
            upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
            if upload_path.exists():
                logger.info(f"Loading dataset from uploads folder: {upload_path}")
                df = pd.read_csv(upload_path, na_values=missing_values, keep_default_na=True)
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
                df = pd.read_csv(
                    io.BytesIO(file_content), na_values=missing_values, keep_default_na=True
                )
            elif dataset.local_path:
                logger.info(f"Loading dataset from local: {dataset.local_path}")
                df = pd.read_csv(dataset.local_path, na_values=missing_values, keep_default_na=True)
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

    # --- Learning Activity Helpers ---

    def _generate_feature_score_details(
        self, df: pd.DataFrame, feature_scores: Dict[str, float],
        selected: List[str], removed: List[str], method: str
    ) -> Dict[str, Any]:
        """Enriched feature list for score visualization."""
        if not feature_scores:
            return {"features": [], "method": method}

        max_score = max(feature_scores.values()) if feature_scores else 1
        min_score = min(feature_scores.values()) if feature_scores else 0
        score_range = max_score - min_score if max_score != min_score else 1

        features_list = []
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=(method == "variance"),
        )

        for rank, (name, score) in enumerate(sorted_features, 1):
            bar_width = ((score - min_score) / score_range * 100) if score_range > 0 else 50

            feat_data: Dict[str, Any] = {
                "name": name,
                "score": round(score, 6),
                "rank": rank,
                "selected": name in selected,
                "bar_width_pct": round(bar_width, 1),
            }

            if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
                col_vals = df[name].dropna()
                if len(col_vals) > 0:
                    feat_data["stats"] = {
                        "mean": round(float(col_vals.mean()), 4),
                        "std": round(float(col_vals.std()), 4),
                        "min": round(float(col_vals.min()), 4),
                        "max": round(float(col_vals.max()), 4),
                    }

            features_list.append(feat_data)

        return {
            "features": features_list,
            "max_score": round(max_score, 6),
            "min_score": round(min_score, 6),
            "method": method,
            "total_features": len(feature_scores),
            "selected_count": len(selected),
            "removed_count": len(removed),
        }

    def _generate_feature_correlation_matrix(
        self, df: pd.DataFrame, selected: List[str], removed: List[str]
    ) -> Dict[str, Any]:
        """Generate correlation matrix data for heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        cols = numeric_df.columns.tolist()[:20]  # Cap at 20

        if len(cols) < 2:
            return {"columns": cols, "matrix": [], "highly_correlated_pairs": []}

        corr_matrix = numeric_df[cols].corr()
        matrix = [[round(float(corr_matrix.iloc[i, j]), 4) for j in range(len(cols))] for i in range(len(cols))]

        # Find highly correlated pairs
        highly_correlated = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.7:
                    which_removed = None
                    if cols[i] in removed:
                        which_removed = cols[i]
                    elif cols[j] in removed:
                        which_removed = cols[j]
                    highly_correlated.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "correlation": round(float(corr_val), 4),
                        "which_removed": which_removed,
                    })

        highly_correlated.sort(key=lambda x: x["correlation"], reverse=True)

        feature_status = {}
        for col in cols:
            if col in selected:
                feature_status[col] = "selected"
            elif col in removed:
                feature_status[col] = "removed"
            else:
                feature_status[col] = "other"

        return {
            "columns": cols,
            "matrix": matrix,
            "highly_correlated_pairs": highly_correlated[:10],
            "feature_status": feature_status,
        }

    def _generate_threshold_simulation_data(
        self, feature_scores: Dict[str, float], input_data: FeatureSelectionInput
    ) -> Dict[str, Any]:
        """Pre-sorted features for client-side threshold slider."""
        if not feature_scores:
            return {"features_sorted": [], "applied_threshold": 0, "total_features": 0}

        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1])
        score_range = {
            "min": round(min(feature_scores.values()), 6),
            "max": round(max(feature_scores.values()), 6),
        }

        applied_threshold = 0
        if input_data.method == "variance":
            applied_threshold = input_data.variance_threshold
        elif input_data.method == "correlation":
            applied_threshold = input_data.correlation_threshold

        return {
            "features_sorted": [{"name": n, "score": round(s, 6)} for n, s in sorted_features],
            "applied_threshold": applied_threshold,
            "method": input_data.method,
            "score_range": score_range,
            "total_features": len(feature_scores),
        }

    def _generate_feature_selection_quiz(
        self, df: pd.DataFrame, feature_scores: Dict[str, float],
        selected: List[str], removed: List[str], summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about feature selection from actual data."""
        import random as _random
        questions = []
        q_id = 0

        sorted_by_score = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Q1: Score interpretation
        if len(sorted_by_score) >= 2:
            high_feat, high_score = sorted_by_score[0]
            low_feat, low_score = sorted_by_score[-1]
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"Feature '{high_feat}' has a score of {round(high_score, 4)} and '{low_feat}' has {round(low_score, 4)}. Which has more information?",
                "options": [
                    f"{high_feat} (higher score)",
                    f"{low_feat} (lower score)",
                    "They have equal information",
                    "Cannot determine from scores alone",
                ],
                "correct_answer": 0,
                "explanation": f"In variance-based selection, higher variance means more spread in values, indicating more information. {high_feat} ({round(high_score, 4)}) has more variance than {low_feat} ({round(low_score, 4)}).",
                "difficulty": "easy",
            })

        # Q2: Threshold question
        if removed and feature_scores:
            removed_feat = removed[0]
            removed_score = feature_scores.get(removed_feat, 0)
            threshold = summary.get("variance_threshold", summary.get("correlation_threshold", 0))
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"The threshold is {threshold}. Feature '{removed_feat}' has a score of {round(removed_score, 4)}. Was it selected or removed?",
                "options": ["Removed", "Selected", "It depends on other features", "Score is not used"],
                "correct_answer": 0,
                "explanation": f"Feature '{removed_feat}' was removed because its score ({round(removed_score, 4)}) {'is below' if summary.get('method') == 'variance' else 'exceeds'} the threshold ({threshold}).",
                "difficulty": "easy",
            })

        # Q3: Correlation question
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr().abs()
            # Find highest correlation pair
            max_corr = 0
            pair = ("", "")
            for i, c1 in enumerate(numeric_df.columns):
                for j, c2 in enumerate(numeric_df.columns):
                    if i < j and corr_matrix.loc[c1, c2] > max_corr:
                        max_corr = corr_matrix.loc[c1, c2]
                        pair = (c1, c2)

            if max_corr > 0.5:
                q_id += 1
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"Features '{pair[0]}' and '{pair[1]}' have a correlation of {round(max_corr, 2)}. Why might we remove one of them?",
                    "options": [
                        "They carry similar information, so one is redundant",
                        "High correlation means they are both important",
                        "Correlation doesn't affect feature selection",
                        "Both should always be removed",
                    ],
                    "correct_answer": 0,
                    "explanation": f"When two features are highly correlated ({round(max_corr, 2)}), they provide nearly the same information. Keeping both adds redundancy without improving the model, and can slow training.",
                    "difficulty": "medium",
                })

        # Q4: Reduction math
        orig = summary.get("original_features", 0)
        sel = summary.get("selected_features", 0)
        if orig > 0 and sel > 0:
            reduction = round((1 - sel / orig) * 100, 1)
            wrong1 = round(reduction + 15, 1)
            wrong2 = round(reduction - 10, 1) if reduction > 10 else round(reduction + 25, 1)
            wrong3 = round(sel / orig * 100, 1)
            q_id += 1
            options4 = [f"{reduction}%", f"{wrong1}%", f"{wrong2}%", f"{wrong3}%"]
            _random.shuffle(options4)
            correct_idx4 = options4.index(f"{reduction}%")
            questions.append({
                "id": f"q{q_id}",
                "question": f"We started with {orig} features and kept {sel}. What percentage of features were removed?",
                "options": options4,
                "correct_answer": correct_idx4,
                "explanation": f"Removed {orig - sel} out of {orig} features = ({orig - sel}/{orig}) × 100 = {reduction}% reduction.",
                "difficulty": "easy",
            })

        # Q5: Why select features
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is the main benefit of removing low-variance features?",
            "options": [
                "They contain very little useful information for predictions",
                "They make the dataset larger",
                "They are always categorical columns",
                "They cause errors in ML models",
            ],
            "correct_answer": 0,
            "explanation": "Features with very low variance have nearly the same value for all samples. Since they barely change, they can't help the model distinguish between different outcomes. Removing them simplifies the model without losing predictive power.",
            "difficulty": "easy",
        })

        _random.shuffle(questions)
        return questions[:5]
