"""
Data Cleaning/Preprocessing Node - Handles missing values and feature extraction.
Supports text preprocessing with TF-IDF and numeric feature scaling.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id


class PreprocessInput(NodeInput):
    """Input schema for Preprocess node."""

    dataset_path: str = Field(..., description="Path to dataset file")
    target_column: str = Field(..., description="Name of target column")

    # Missing value handling
    handle_missing: bool = Field(True, description="Whether to handle missing values")
    missing_strategy: str = Field(
        "drop", description="Strategy: 'drop', 'mean', 'median', 'mode', 'fill'"
    )
    fill_value: Optional[float] = Field(None, description="Value to fill if strategy='fill'")

    # Feature extraction
    text_columns: List[str] = Field(default_factory=list, description="Columns to apply TF-IDF")
    numeric_columns: List[str] = Field(default_factory=list, description="Columns to scale")

    # Text preprocessing
    lowercase: bool = Field(True, description="Convert text to lowercase")
    remove_stopwords: bool = Field(False, description="Remove stopwords")
    max_features: int = Field(1000, description="Maximum TF-IDF features")

    # Numeric scaling
    scale_features: bool = Field(False, description="Apply StandardScaler to numeric features")

    @field_validator("missing_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate missing value strategy."""
        valid_strategies = ["drop", "mean", "median", "mode", "fill"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy. Choose from: {', '.join(valid_strategies)}")
        return v


class PreprocessOutput(NodeOutput):
    """Output schema for Preprocess node."""

    preprocessed_dataset_id: str = Field(..., description="ID of preprocessed dataset")
    preprocessed_path: str = Field(..., description="Path to preprocessed data")
    artifacts_path: Optional[str] = Field(None, description="Path to preprocessing artifacts")

    original_rows: int = Field(..., description="Rows before preprocessing")
    final_rows: int = Field(..., description="Rows after preprocessing")
    rows_dropped: int = Field(..., description="Number of rows dropped")

    original_columns: int = Field(..., description="Columns before preprocessing")
    final_columns: int = Field(..., description="Columns after preprocessing")

    feature_names: List[str] = Field(..., description="Final feature column names")
    preprocessing_summary: Dict[str, Any] = Field(..., description="Summary of operations")


class PreprocessNode(BaseNode):
    """
    Preprocessing Node - Data cleaning and feature extraction.

    Responsibilities:
    - Handle missing values (drop, impute, fill)
    - Text preprocessing (lowercase, trimming, stopwords)
    - Feature extraction (TF-IDF for text, scaling for numeric)
    - Save preprocessing artifacts (vectorizer, scaler)
    - Return preprocessed dataset reference

    Production features:
    - Reusable for other ML algorithms
    - Saves transformation artifacts for inference
    - Handles both text and numeric data
    - Comprehensive preprocessing summary
    """

    node_type = "preprocess"

    @property
    def metadata(self) -> NodeMetadata:
        """Define node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.PREPROCESSING,
            primary_output_field="preprocessed_dataset_id",
            output_fields={
                "preprocessed_dataset_id": "Identifier for the preprocessed dataset",
                "preprocessed_path": "Path to preprocessed dataset file",
                "original_rows": "Number of rows before preprocessing",
                "final_rows": "Number of rows after preprocessing",
                "feature_names": "List of feature column names after preprocessing",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            max_inputs=1,
            allowed_source_categories=[NodeCategory.DATA_SOURCE, NodeCategory.PREPROCESSING],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return PreprocessInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return PreprocessOutput

    async def _execute(self, input_data: PreprocessInput) -> PreprocessOutput:
        """
        Execute preprocessing pipeline.

        Args:
            input_data: Validated input data

        Returns:
            Preprocessing result with metadata
        """
        try:
            logger.info(f"Starting preprocessing for {input_data.dataset_path}")

            # Load dataset
            df = pd.read_csv(input_data.dataset_path)
            original_rows, original_cols = df.shape

            preprocessing_summary = {}
            artifacts = {}

            # Validate target column
            if input_data.target_column not in df.columns:
                raise InvalidDatasetError(
                    f"Target column '{input_data.target_column}' not found",
                    expected_format=f"CSV with column: {input_data.target_column}",
                )

            # Separate features and target
            X = df.drop(columns=[input_data.target_column])
            y = df[input_data.target_column]

            # Handle missing values
            if input_data.handle_missing:
                X, missing_summary = self._handle_missing_values(
                    X, strategy=input_data.missing_strategy, fill_value=input_data.fill_value
                )
                preprocessing_summary["missing_values"] = missing_summary

            # Text preprocessing and TF-IDF
            if input_data.text_columns:
                X, tfidf_vectorizer = self._apply_tfidf(
                    X,
                    text_columns=input_data.text_columns,
                    lowercase=input_data.lowercase,
                    max_features=input_data.max_features,
                    remove_stopwords=input_data.remove_stopwords,
                )
                artifacts["tfidf_vectorizer"] = tfidf_vectorizer
                preprocessing_summary["tfidf"] = {
                    "text_columns": input_data.text_columns,
                    "max_features": input_data.max_features,
                    "features_created": tfidf_vectorizer.get_feature_names_out().shape[0],
                }

            # Numeric feature scaling
            if input_data.scale_features and input_data.numeric_columns:
                X, scaler = self._scale_numeric_features(
                    X, numeric_columns=input_data.numeric_columns
                )
                artifacts["scaler"] = scaler
                preprocessing_summary["scaling"] = {
                    "numeric_columns": input_data.numeric_columns,
                    "scaler_type": "StandardScaler",
                }

            # Align target with features (in case rows were dropped)
            y = y.iloc[X.index]

            # Combine back for storage
            df_processed = X.copy()
            df_processed[input_data.target_column] = y

            # Generate IDs and paths
            preprocessed_id = generate_id("preprocessed")
            preprocessed_path = Path(settings.UPLOAD_DIR) / f"{preprocessed_id}.csv"

            # Save preprocessed data
            df_processed.to_csv(preprocessed_path, index=False)

            # Save preprocessing artifacts
            artifacts_path = None
            if artifacts:
                artifacts_path = (
                    Path(settings.MODEL_ARTIFACTS_DIR) / f"{preprocessed_id}_artifacts.joblib"
                )
                artifacts_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(artifacts, artifacts_path)
                logger.info(f"Preprocessing artifacts saved to {artifacts_path}")

            final_rows, final_cols = df_processed.shape

            logger.info(
                f"Preprocessing complete - Rows: {original_rows} → {final_rows}, "
                f"Columns: {original_cols} → {final_cols}"
            )

            return PreprocessOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                preprocessed_dataset_id=preprocessed_id,
                preprocessed_path=str(preprocessed_path),
                artifacts_path=str(artifacts_path) if artifacts_path else None,
                original_rows=original_rows,
                final_rows=final_rows,
                rows_dropped=original_rows - final_rows,
                original_columns=original_cols,
                final_columns=final_cols,
                feature_names=X.columns.tolist(),
                preprocessing_summary=preprocessing_summary,
            )

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    def _handle_missing_values(
        self, X: pd.DataFrame, strategy: str, fill_value: Optional[float] = None
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in DataFrame."""

        missing_before = X.isnull().sum().to_dict()
        rows_before = len(X)

        if strategy == "drop":
            X = X.dropna()

        elif strategy == "mean":
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        elif strategy == "median":
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        elif strategy == "mode":
            for col in X.columns:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

        elif strategy == "fill":
            X = X.fillna(fill_value if fill_value is not None else 0)

        rows_after = len(X)

        return X, {
            "strategy": strategy,
            "missing_before": {k: v for k, v in missing_before.items() if v > 0},
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_dropped": rows_before - rows_after,
        }

    def _apply_tfidf(
        self,
        X: pd.DataFrame,
        text_columns: List[str],
        lowercase: bool,
        max_features: int,
        remove_stopwords: bool,
    ) -> tuple[pd.DataFrame, TfidfVectorizer]:
        """Apply TF-IDF to text columns."""

        # Validate text columns exist
        missing_cols = set(text_columns) - set(X.columns)
        if missing_cols:
            raise InvalidDatasetError(
                f"Text columns not found: {missing_cols}",
                expected_format=f"CSV with columns: {', '.join(text_columns)}",
            )

        # Combine text columns
        combined_text = X[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

        # Apply TF-IDF
        vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            max_features=max_features,
            stop_words="english" if remove_stopwords else None,
        )

        tfidf_matrix = vectorizer.fit_transform(combined_text)

        # Create DataFrame from TF-IDF features
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()],
            index=X.index,
        )

        # Drop original text columns and add TF-IDF features
        X = X.drop(columns=text_columns)
        X = pd.concat([X, tfidf_df], axis=1)

        return X, vectorizer

    def _scale_numeric_features(
        self, X: pd.DataFrame, numeric_columns: List[str]
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """Scale numeric features using StandardScaler."""

        # Validate numeric columns exist
        missing_cols = set(numeric_columns) - set(X.columns)
        if missing_cols:
            raise InvalidDatasetError(
                f"Numeric columns not found: {missing_cols}",
                expected_format=f"CSV with columns: {', '.join(numeric_columns)}",
            )

        # Apply scaling
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

        return X, scaler
