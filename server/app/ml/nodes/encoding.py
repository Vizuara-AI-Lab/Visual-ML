"""
Encoding Node - Encode categorical variables using various methods.
Supports One-Hot and Label encoding with per-column configuration.
"""

from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
import io
import json

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class ColumnEncodingConfig(BaseModel):
    """Configuration for encoding a single column."""

    column_name: str = Field(..., description="Name of the column to encode")
    encoding_method: str = Field(..., description="Encoding method: onehot, label, none")
    handle_unknown: str = Field(
        "error", description="How to handle unknown categories: error, ignore, use_encoded_value"
    )
    handle_missing: str = Field(
        "error", description="How to handle missing values: error, most_frequent, create_category"
    )
    drop_first: bool = Field(False, description="Drop first category in one-hot encoding")


class EncodingInput(NodeInput):
    """Input schema for Encoding node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    column_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-column encoding configuration"
    )
    columns: Optional[List[str]] = Field(
        None, description="Available columns from upstream node (auto-filled by DAG)"
    )


class EncodingOutput(NodeOutput):
    """Output schema for Encoding node."""

    encoded_dataset_id: str = Field(..., description="ID of encoded dataset")
    encoded_path: str = Field(..., description="Path to encoded dataset")
    artifacts_path: Optional[str] = Field(None, description="Path to encoding artifacts")

    original_columns: int = Field(..., description="Columns before encoding")
    final_columns: int = Field(..., description="Columns after encoding")
    columns: List[str] = Field(..., description="All column names in encoded dataset")
    encoded_columns: List[str] = Field(..., description="Columns that were encoded")
    new_columns: List[str] = Field(..., description="New columns created")
    encoding_summary: Dict[str, Any] = Field(..., description="Summary of encoding operations")
    warnings: List[str] = Field(default_factory=list, description="Warnings about encoding choices")
    column_type_info: Dict[str, str] = Field(
        default_factory=dict, description="Detected column types"
    )

    # Learning activity data (all Optional for backward compatibility)
    encoding_before_after: Optional[Dict[str, Any]] = Field(
        None, description="Before/after sample rows for encoding visualization"
    )
    encoding_map: Optional[Dict[str, Any]] = Field(
        None, description="Complete category-to-number mapping for each encoded column"
    )
    encoding_method_comparison: Optional[Dict[str, Any]] = Field(
        None, description="Comparison of one-hot vs label encoding for each column"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about encoding"
    )


class EncodingNode(BaseNode):
    """
    Encoding Node - Encode categorical variables with per-column configuration.

    Supports:
    - One-Hot Encoding: Create binary columns for each category
    - Label Encoding: Convert categories to integers
    """

    node_type = "encoding"

    @property
    def metadata(self) -> NodeMetadata:
        """Return node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.PREPROCESSING,
            primary_output_field="encoded_dataset_id",
            output_fields={
                "encoded_dataset_id": "ID of the encoded dataset",
                "encoded_path": "Path to encoded dataset file",
                "artifacts_path": "Path to encoding artifacts (encoders)",
                "columns": "All columns in the encoded dataset (for downstream nodes)",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            allowed_source_categories=[
                NodeCategory.DATA_SOURCE,
                NodeCategory.PREPROCESSING,
                NodeCategory.DATA_TRANSFORM,
            ],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        return EncodingInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return EncodingOutput

    async def _execute(self, input_data: EncodingInput) -> EncodingOutput:
        """Execute encoding with per-column configuration."""
        try:
            # DEBUG: Log all input fields
            logger.info(
                f"[ENCODING DEBUG] Input data: dataset_id={input_data.dataset_id}, "
                f"columns={getattr(input_data, 'columns', 'NOT FOUND')}"
            )

            logger.info(f"Starting encoding for dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            original_columns = len(df.columns)
            df_encoded = df.copy()
            encoding_summary = {}
            new_columns_list = []
            warnings = []
            encoders = {}  # Store encoders for artifact saving

            # Detect column types
            column_types = self._detect_column_types(df)

            # Process each column configuration
            for column_name, config_dict in input_data.column_configs.items():
                if column_name not in df.columns:
                    warning_msg = f"Column '{column_name}' not found in dataset - skipping encoding"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)
                    continue

                # Parse configuration
                config = ColumnEncodingConfig(**config_dict)

                if config.encoding_method == "none":
                    continue

                # Validate encoding choice and collect warnings
                column_warnings = self._validate_encoding_config(
                    column_name,
                    column_types.get(column_name, "unknown"),
                    config.encoding_method,
                    df[column_name].nunique(),
                )
                warnings.extend(column_warnings)

                # Handle missing values first
                df_encoded = self._handle_missing_values(
                    df_encoded, column_name, config.handle_missing
                )

                # Apply encoding
                if config.encoding_method == "onehot":
                    df_encoded, new_cols, encoder = self._onehot_encode(
                        df_encoded, column_name, config.drop_first, config.handle_unknown
                    )
                    new_columns_list.extend(new_cols)
                    encoders[column_name] = {"type": "onehot", "categories": encoder}
                    encoding_summary[column_name] = {
                        "method": "onehot",
                        "new_columns": new_cols,
                        "unique_values": len(new_cols),
                        "handle_unknown": config.handle_unknown,
                        "handle_missing": config.handle_missing,
                    }

                elif config.encoding_method == "label":
                    df_encoded, encoder = self._label_encode(
                        df_encoded, column_name, config.handle_unknown
                    )
                    encoders[column_name] = {"type": "label", "classes": list(encoder.classes_)}
                    encoding_summary[column_name] = {
                        "method": "label",
                        "unique_values": len(encoder.classes_),
                        "handle_unknown": config.handle_unknown,
                        "handle_missing": config.handle_missing,
                    }

            final_columns = len(df_encoded.columns)

            # Save encoded dataset
            encoded_id = generate_id("encoded")
            encoded_path = Path(settings.UPLOAD_DIR) / f"{encoded_id}.csv"
            encoded_path.parent.mkdir(parents=True, exist_ok=True)
            df_encoded.to_csv(encoded_path, index=False)

            # Save encoding artifacts
            artifacts_path = None
            if encoders:
                artifacts_path = self._save_encoding_artifacts(encoders, encoded_id)

            logger.info(
                f"Encoding complete - Columns: {original_columns} → {final_columns}, "
                f"Warnings: {len(warnings)}, Saved to: {encoded_path}"
            )

            # Generate learning activity data (non-blocking)
            encoding_before_after = None
            try:
                encoding_before_after = self._generate_encoding_before_after(
                    df, df_encoded, encoding_summary
                )
            except Exception as e:
                logger.warning(f"Encoding before/after generation failed: {e}")

            encoding_map = None
            try:
                encoding_map = self._generate_encoding_map(df, encoding_summary)
            except Exception as e:
                logger.warning(f"Encoding map generation failed: {e}")

            encoding_method_comparison = None
            try:
                encoding_method_comparison = self._generate_encoding_method_comparison(
                    df, encoding_summary
                )
            except Exception as e:
                logger.warning(f"Encoding method comparison generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_encoding_quiz(
                    df, encoding_summary, column_types
                )
            except Exception as e:
                logger.warning(f"Encoding quiz generation failed: {e}")

            return EncodingOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                encoded_dataset_id=encoded_id,
                encoded_path=str(encoded_path),
                artifacts_path=artifacts_path,
                original_columns=original_columns,
                final_columns=final_columns,
                columns=df_encoded.columns.tolist(),
                encoded_columns=list(input_data.column_configs.keys()),
                new_columns=new_columns_list,
                encoding_summary=encoding_summary,
                warnings=warnings,
                column_type_info=column_types,
                encoding_before_after=encoding_before_after,
                encoding_map=encoding_map,
                encoding_method_comparison=encoding_method_comparison,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect column types for validation.
        Returns: dict mapping column names to types (categorical_nominal, numeric, high_cardinality)
        """
        column_types = {}

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                column_types[column] = "numeric"
            else:
                unique_count = df[column].nunique()
                if unique_count > 50:
                    column_types[column] = "high_cardinality"
                else:
                    # Default to nominal for categorical
                    column_types[column] = "categorical_nominal"

        return column_types

    def _validate_encoding_config(
        self, column: str, column_type: str, encoding_method: str, unique_count: int
    ) -> List[str]:
        """
        Validate encoding method for column type.
        Returns list of warnings.
        """
        warnings = []

        if encoding_method == "label" and column_type == "categorical_nominal":
            warnings.append(
                f"⚠️ Column '{column}': Label encoding on nominal data implies an order that may not exist. "
                "Consider using One-Hot encoding instead."
            )

        if encoding_method == "onehot" and unique_count > 50:
            warnings.append(
                f"⚠️ Column '{column}': One-Hot encoding with {unique_count} categories will create many columns. "
                "Consider using Label encoding for high-cardinality features."
            )

        return warnings

    def _handle_missing_values(self, df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
        """Handle missing values according to strategy."""
        if strategy == "error":
            if df[column].isna().any():
                raise ValueError(
                    f"Missing values found in column '{column}'. Choose a handling strategy."
                )

        elif strategy == "most_frequent":
            if df[column].isna().any():
                most_frequent = df[column].mode()[0] if not df[column].mode().empty else "UNKNOWN"
                df[column] = df[column].fillna(most_frequent)
                logger.info(
                    f"Filled missing values in '{column}' with most frequent: {most_frequent}"
                )

        elif strategy == "create_category":
            df[column] = df[column].fillna("__MISSING__")
            logger.info(f"Created '__MISSING__' category for missing values in '{column}'")

        return df

    def _save_encoding_artifacts(self, encoders: Dict[str, Any], artifact_id: str) -> str:
        """Save encoding artifacts to disk."""
        artifacts_path = Path(settings.UPLOAD_DIR) / f"{artifact_id}_encoders.json"

        # Convert to JSON-serializable format
        serializable_encoders = {}
        for column, encoder_info in encoders.items():
            serializable_encoders[column] = encoder_info

        with open(artifacts_path, "w") as f:
            json.dump(serializable_encoders, f, indent=2)

        logger.info(f"Saved encoding artifacts to: {artifacts_path}")
        return str(artifacts_path)

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

    def _onehot_encode(
        self, df: pd.DataFrame, column: str, drop_first: bool = False, handle_unknown: str = "error"
    ) -> tuple[pd.DataFrame, List[str], List[str]]:
        """Apply one-hot encoding."""
        # Store original categories
        original_categories = df[column].unique().tolist()

        # Get dummies
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=drop_first)
        new_columns = dummies.columns.tolist()

        # Drop original column and add dummies
        df = df.drop(columns=[column])
        df = pd.concat([df, dummies], axis=1)

        return df, new_columns, original_categories

    def _label_encode(
        self, df: pd.DataFrame, column: str, handle_unknown: str = "error"
    ) -> tuple[pd.DataFrame, LabelEncoder]:
        """Apply label encoding."""
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        return df, le

    # --- Learning Activity Helpers ---

    def _generate_encoding_before_after(
        self, df_original: pd.DataFrame, df_encoded: pd.DataFrame, encoding_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate before/after sample rows for each encoded column."""
        columns_data = {}

        for col_name, summary in encoding_summary.items():
            if col_name not in df_original.columns:
                continue

            method = summary.get("method", "unknown")
            unique_vals = df_original[col_name].dropna().unique()
            sample_categories = unique_vals[:5].tolist()

            samples = []
            for cat in sample_categories:
                row_idx = df_original[df_original[col_name] == cat].index
                if len(row_idx) == 0:
                    continue
                idx = row_idx[0]

                if method == "onehot":
                    new_cols = summary.get("new_columns", [])
                    encoded_vals = {}
                    for nc in new_cols:
                        if nc in df_encoded.columns and idx in df_encoded.index:
                            encoded_vals[nc] = int(df_encoded.at[idx, nc])
                    samples.append({"original": str(cat), "encoded": encoded_vals})
                elif method == "label":
                    if col_name in df_encoded.columns and idx in df_encoded.index:
                        val = df_encoded.at[idx, col_name]
                        samples.append({"original": str(cat), "encoded": int(val)})

            columns_data[col_name] = {
                "method": method,
                "samples": samples,
            }

        return {
            "columns": columns_data,
            "total_rows": len(df_original),
            "rows_sampled": 5,
        }

    def _generate_encoding_map(
        self, df: pd.DataFrame, encoding_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate complete category-to-number mapping for each encoded column."""
        result = {}

        for col_name, summary in encoding_summary.items():
            if col_name not in df.columns:
                continue

            method = summary.get("method", "unknown")
            value_counts = df[col_name].value_counts()

            if method == "onehot":
                new_cols = summary.get("new_columns", [])
                mapping = []
                for cat in value_counts.index:
                    cat_str = str(cat)
                    matched_col = None
                    for nc in new_cols:
                        if nc == f"{col_name}_{cat_str}" or nc.endswith(f"_{cat_str}"):
                            matched_col = nc
                            break
                    mapping.append({
                        "category": cat_str,
                        "encoded_column": matched_col or f"{col_name}_{cat_str}",
                        "count": int(value_counts[cat]),
                    })
                result[col_name] = {"method": "onehot", "mapping": mapping}

            elif method == "label":
                le = LabelEncoder()
                le.fit(df[col_name].dropna().astype(str))
                mapping = []
                for i, cls in enumerate(le.classes_):
                    count = int(value_counts.get(cls, 0))
                    mapping.append({
                        "category": str(cls),
                        "value": i,
                        "count": count,
                    })
                result[col_name] = {"method": "label", "mapping": mapping}

        return result

    def _generate_encoding_method_comparison(
        self, df: pd.DataFrame, encoding_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare one-hot vs label encoding for each encoded column."""
        result = {}

        for col_name, summary in encoding_summary.items():
            if col_name not in df.columns:
                continue

            unique_vals = df[col_name].dropna().unique()
            unique_count = len(unique_vals)
            applied = summary.get("method", "unknown")
            sample_cats = [str(v) for v in unique_vals[:5]]

            # One-hot result
            onehot_cols = [f"{col_name}_{c}" for c in sample_cats]
            sample_row = {c: (1 if c == onehot_cols[0] else 0) for c in onehot_cols}

            # Label result
            le = LabelEncoder()
            le.fit(df[col_name].dropna().astype(str))
            label_mapping = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
            sample_label = {k: v for k, v in list(label_mapping.items())[:5]}

            recommendation = (
                "One-hot is better for nominal categories (no natural order) with few unique values"
                if unique_count <= 10
                else "Label encoding is more practical for high-cardinality columns"
            )

            result[col_name] = {
                "unique_count": unique_count,
                "applied_method": applied,
                "onehot_result": {
                    "new_columns_count": unique_count,
                    "sample_row": sample_row,
                    "pros": "No false ordering between categories",
                    "cons": f"Creates {unique_count} new columns (increases dimensionality)",
                },
                "label_result": {
                    "new_columns_count": 0,
                    "sample_mapping": sample_label,
                    "pros": "Compact — only 1 column",
                    "cons": "Implies ordering (0 < 1 < 2) which may mislead the model",
                },
                "recommendation": recommendation,
            }

        return result

    def _generate_encoding_quiz(
        self, df: pd.DataFrame, encoding_summary: Dict[str, Any], column_types: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about encoding from actual data."""
        import random as _random
        questions = []
        q_id = 0

        encoded_cols = list(encoding_summary.keys())

        # Q1: Column count question (one-hot)
        for col in encoded_cols:
            summary = encoding_summary[col]
            if summary.get("method") == "onehot":
                n_cats = summary.get("unique_values", 0)
                if n_cats >= 2:
                    q_id += 1
                    wrong1 = max(1, n_cats - 1)
                    wrong2 = n_cats + 1
                    wrong3 = n_cats * 2
                    options = [str(n_cats), str(wrong1), str(wrong2), str(wrong3)]
                    _random.shuffle(options)
                    correct_idx = options.index(str(n_cats))
                    questions.append({
                        "id": f"q{q_id}",
                        "question": f"Column '{col}' had {n_cats} unique categories and was one-hot encoded. How many new columns were created?",
                        "options": options,
                        "correct_answer": correct_idx,
                        "explanation": f"One-hot encoding creates one binary column for each category. With {n_cats} categories, {n_cats} new columns are created.",
                        "difficulty": "easy",
                    })
                    break

        # Q2: Method choice question
        for col in encoded_cols:
            if col in df.columns and column_types.get(col) == "categorical_nominal":
                q_id += 1
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"Column '{col}' contains categories with no natural ordering (like colors or cities). Which encoding preserves this correctly?",
                    "options": ["One-Hot Encoding", "Label Encoding", "Both work equally well", "Neither works"],
                    "correct_answer": 0,
                    "explanation": "One-hot encoding creates separate binary columns, so no ordering is implied. Label encoding assigns numbers (0, 1, 2...) which implies an order that doesn't exist.",
                    "difficulty": "medium",
                })
                break

        # Q3: Label ordering trap
        for col in encoded_cols:
            summary = encoding_summary[col]
            if summary.get("method") == "label" and col in df.columns:
                unique_vals = df[col].dropna().unique()[:3]
                if len(unique_vals) >= 3:
                    vals_str = ", ".join([f"{v}={i}" for i, v in enumerate(unique_vals)])
                    q_id += 1
                    questions.append({
                        "id": f"q{q_id}",
                        "question": f"Column '{col}' is label-encoded as {vals_str}. What potential problem does this create?",
                        "options": [
                            "The model may assume a meaningful order between categories",
                            "The model will ignore this column",
                            "It creates too many columns",
                            "It removes missing values",
                        ],
                        "correct_answer": 0,
                        "explanation": f"Label encoding assigns numbers that imply an order. The model might think {unique_vals[2]}({2}) > {unique_vals[1]}({1}) > {unique_vals[0]}({0}), even if no such order exists.",
                        "difficulty": "medium",
                    })
                    break

        # Q4: One-hot interpretation
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "In one-hot encoding, if a row has Color_Blue=1, what values do Color_Red and Color_Green have?",
            "options": ["Both are 0", "Both are 1", "They can be any number", "They are removed"],
            "correct_answer": 0,
            "explanation": "In one-hot encoding, exactly one column is 1 and all others are 0 for each row. This is called 'mutually exclusive' — a row belongs to exactly one category.",
            "difficulty": "easy",
        })

        # Q5: Why encode
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why do we need to encode categorical columns before training an ML model?",
            "options": [
                "ML algorithms can only work with numbers",
                "It makes the dataset smaller",
                "It removes duplicate rows",
                "It fixes missing values",
            ],
            "correct_answer": 0,
            "explanation": "Most ML algorithms perform mathematical operations (addition, multiplication) and can only work with numerical inputs. Encoding converts text categories into numbers the model can process.",
            "difficulty": "easy",
        })

        _random.shuffle(questions)
        return questions[:5]
