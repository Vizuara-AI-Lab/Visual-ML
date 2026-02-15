"""
Missing Value Handler Node - Dedicated node for handling missing values with column-wise control.
Supports preview, reversible operations, and operation logging.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field, field_validator
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service
import io


class ColumnConfig(BaseNode):
    """Configuration for a single column's missing value handling."""

    strategy: str = Field(
        "drop",
        description="Strategy: 'drop', 'drop_column', 'mean', 'median', 'mode', 'fill', 'forward_fill', 'backward_fill', 'none'",
    )
    fill_value: Optional[Any] = Field(None, description="Value to fill if strategy='fill'")
    enabled: bool = Field(True, description="Whether to apply strategy to this column")


class MissingValueHandlerInput(NodeInput):
    """Input schema for Missing Value Handler node."""

    dataset_id: str = Field(..., description="Dataset ID to process")

    # Column-wise configuration
    column_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-column configuration: {column_name: {strategy, fill_value, enabled}}",
    )

    # Global fallback strategy
    default_strategy: str = Field(
        "none", description="Default strategy for columns not in column_configs"
    )

    # Preview mode
    preview_mode: bool = Field(False, description="If true, return preview without saving changes")

    preview_rows: int = Field(10, description="Number of rows to show in preview (default: 10)")

    # Operation tracking
    operation_id: Optional[str] = Field(
        None, description="Operation ID for tracking (auto-generated if not provided)"
    )

    # Metadata (passed from previous nodes)
    filename: Optional[str] = Field(None, description="Filename (metadata)")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[List[str]] = Field(None, description="Column names (metadata)")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Data types (metadata)")

    @field_validator("default_strategy")
    @classmethod
    def validate_default_strategy(cls, v: str) -> str:
        """Validate default strategy."""
        valid_strategies = [
            "drop",
            "drop_column",
            "mean",
            "median",
            "mode",
            "fill",
            "forward_fill",
            "backward_fill",
            "none",
        ]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy. Choose from: {', '.join(valid_strategies)}")
        return v


class MissingValueHandlerOutput(NodeOutput):
    """Output schema for Missing Value Handler node."""

    preprocessed_dataset_id: str = Field(..., description="ID of preprocessed dataset")
    preprocessed_path: Optional[str] = Field(None, description="Path to preprocessed data")

    # Column information (for frontend to detect available columns)
    columns: List[str] = Field(..., description="List of column names in preprocessed dataset")

    # Statistics
    original_rows: int = Field(..., description="Rows before preprocessing")
    final_rows: int = Field(..., description="Rows after preprocessing")
    rows_dropped: int = Field(..., description="Number of rows dropped")

    # Missing value analysis
    before_stats: Dict[str, Any] = Field(..., description="Missing value stats before processing")
    after_stats: Dict[str, Any] = Field(..., description="Missing value stats after processing")

    # Operation tracking
    operation_log: Dict[str, Any] = Field(..., description="Details of operations performed")
    reversible_operation_id: str = Field(..., description="ID for reversal tracking")

    # Preview data (only populated in preview mode)
    preview_data: Optional[Dict[str, Any]] = Field(None, description="Before/after preview data")

    # Learning activity data (all Optional for backward compatibility)
    strategy_comparison: Optional[Dict[str, Any]] = Field(
        None, description="Per-column comparison of all strategies for learning"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about the dataset"
    )
    missing_heatmap: Optional[Dict[str, Any]] = Field(
        None, description="Boolean grid of missing values for heatmap visualization"
    )


class MissingValueHandlerNode(BaseNode):
    """
    Missing Value Handler Node - Dedicated preprocessing for missing values.

    Features:
    - Column-wise configuration (different strategies per column)
    - Preview before/after changes
    - Reversible operations with operation logging
    - Multiple strategies: drop, mean, median, mode, fill, forward_fill, backward_fill
    - Comprehensive statistics and reporting
    """

    node_type = "missing_value_handler"

    @property
    def metadata(self) -> NodeMetadata:
        """Return node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.PREPROCESSING,
            primary_output_field="preprocessed_dataset_id",
            output_fields={
                "preprocessed_dataset_id": "ID of the preprocessed dataset",
                "preprocessed_path": "Path to preprocessed dataset file",
                "columns": "Available column names after preprocessing",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            allowed_source_categories=[NodeCategory.DATA_SOURCE, NodeCategory.PREPROCESSING],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return MissingValueHandlerInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return MissingValueHandlerOutput

    async def _execute(self, input_data: MissingValueHandlerInput) -> MissingValueHandlerOutput:
        """
        Execute missing value handling.

        Args:
            input_data: Validated input data

        Returns:
            Processing result with statistics and metadata
        """
        try:
            logger.info(f"Starting missing value handling for dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            original_rows = len(df)

            # Analyze missing values before processing
            before_stats = self._analyze_missing_values(df)

            # Create a copy for processing
            df_processed = df.copy()

            # Apply column-wise strategies
            operation_log = {}
            for column in df.columns:
                if column in input_data.column_configs:
                    config = input_data.column_configs[column]
                    strategy = config.get("strategy", "none")
                    fill_value = config.get("fill_value")
                    enabled = config.get("enabled", True)
                else:
                    strategy = input_data.default_strategy
                    fill_value = None
                    enabled = True

                if enabled and strategy != "none":
                    df_processed, col_log = self._apply_column_strategy(
                        df_processed, column, strategy, fill_value
                    )
                    operation_log[column] = col_log

            # Analyze missing values after processing
            after_stats = self._analyze_missing_values(df_processed)

            final_rows = len(df_processed)
            rows_dropped = original_rows - final_rows

            # Generate operation ID
            operation_id = input_data.operation_id or generate_id("operation")

            # Generate learning activity data (non-blocking)
            strategy_comparison = None
            try:
                strategy_comparison = self._generate_strategy_comparison(
                    df, input_data.column_configs
                )
            except Exception as e:
                logger.warning(f"Strategy comparison generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_quiz_questions(
                    df, before_stats, after_stats, operation_log
                )
            except Exception as e:
                logger.warning(f"Quiz generation failed: {e}")

            missing_heatmap = None
            try:
                missing_heatmap = self._generate_missing_heatmap(df)
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")

            # Preview mode - return preview without saving
            if input_data.preview_mode:
                preview_data = self._generate_preview(
                    df, df_processed, before_stats, after_stats, input_data.preview_rows
                )

                logger.info(f"Preview mode - returning preview data without saving")

                return MissingValueHandlerOutput(
                    node_type=self.node_type,
                    execution_time_ms=0,
                    preprocessed_dataset_id=input_data.dataset_id,  # Same as input in preview
                    preprocessed_path=None,
                    columns=df_processed.columns.tolist(),
                    original_rows=original_rows,
                    final_rows=final_rows,
                    rows_dropped=rows_dropped,
                    before_stats=before_stats,
                    after_stats=after_stats,
                    operation_log=operation_log,
                    reversible_operation_id=operation_id,
                    preview_data=preview_data,
                    strategy_comparison=strategy_comparison,
                    quiz_questions=quiz_questions,
                    missing_heatmap=missing_heatmap,
                )

            # Save processed dataset
            preprocessed_id = generate_id("preprocessed")
            preprocessed_path = Path(settings.UPLOAD_DIR) / f"{preprocessed_id}.csv"

            # Ensure directory exists
            preprocessed_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df_processed.to_csv(preprocessed_path, index=False)

            logger.info(
                f"Missing value handling complete - "
                f"Rows: {original_rows} → {final_rows} ({rows_dropped} dropped), "
                f"Saved to: {preprocessed_path}"
            )

            # Generate preview data for display
            preview_data = self._generate_preview(
                df, df_processed, before_stats, after_stats, input_data.preview_rows
            )

            return MissingValueHandlerOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                preprocessed_dataset_id=preprocessed_id,
                preprocessed_path=str(preprocessed_path),
                columns=df_processed.columns.tolist(),
                original_rows=original_rows,
                final_rows=final_rows,
                rows_dropped=rows_dropped,
                before_stats=before_stats,
                after_stats=after_stats,
                operation_log=operation_log,
                reversible_operation_id=operation_id,
                preview_data=preview_data,
                strategy_comparison=strategy_comparison,
                quiz_questions=quiz_questions,
                missing_heatmap=missing_heatmap,
            )

        except Exception as e:
            logger.error(f"Missing value handling failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Load dataset from storage (S3 or local).

        Args:
            dataset_id: Dataset identifier

        Returns:
            DataFrame or None if not found
        """
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

            # Load from S3 or local storage
            # Recognize common missing value indicators: ?, NA, N/A, null, empty strings
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

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

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values in DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing value statistics
        """
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns_with_missing": sum(1 for count in missing_counts.values() if count > 0),
            "total_missing_values": sum(missing_counts.values()),
            "missing_by_column": {
                col: {
                    "count": int(missing_counts[col]),
                    "percentage": round(missing_percentages[col], 2),
                }
                for col in df.columns
                if missing_counts[col] > 0
            },
        }

    def _apply_column_strategy(
        self, df: pd.DataFrame, column: str, strategy: str, fill_value: Optional[Any] = None
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply missing value strategy to a specific column.

        Args:
            df: DataFrame to process
            column: Column name
            strategy: Strategy to apply
            fill_value: Value to fill (if strategy='fill')

        Returns:
            Tuple of (processed DataFrame, operation log)
        """
        if column not in df.columns:
            return df, {"error": f"Column {column} not found"}

        missing_before = df[column].isnull().sum()

        if missing_before == 0 and strategy != "drop_column":
            return df, {
                "strategy": strategy,
                "missing_before": 0,
                "missing_after": 0,
                "action": "no_action_needed",
            }

        try:
            if strategy == "drop":
                # Drop rows with missing values in this column
                df = df.dropna(subset=[column])

            elif strategy == "drop_column":
                # Drop the entire column
                df = df.drop(columns=[column])
                return df, {
                    "strategy": "drop_column",
                    "missing_before": int(missing_before),
                    "action": "column_dropped",
                    "column_dropped": column,
                }

            elif strategy == "mean":
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                else:
                    logger.warning(f"Cannot apply mean to non-numeric column: {column}")
                    return df, {"error": "Cannot apply mean to non-numeric column"}

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                else:
                    logger.warning(f"Cannot apply median to non-numeric column: {column}")
                    return df, {"error": "Cannot apply median to non-numeric column"}

            elif strategy == "mode":
                mode_values = df[column].mode()
                if not mode_values.empty:
                    df[column] = df[column].fillna(mode_values[0])

            elif strategy == "fill":
                df[column] = df[column].fillna(fill_value if fill_value is not None else 0)

            elif strategy == "forward_fill":
                df[column] = df[column].fillna(method="ffill")

            elif strategy == "backward_fill":
                df[column] = df[column].fillna(method="bfill")

            missing_after = df[column].isnull().sum()

            return df, {
                "strategy": strategy,
                "missing_before": int(missing_before),
                "missing_after": int(missing_after),
                "filled_count": int(missing_before - missing_after),
                "fill_value": fill_value if strategy == "fill" else None,
            }

        except Exception as e:
            logger.error(f"Error applying strategy {strategy} to column {column}: {str(e)}")
            return df, {"error": str(e)}

    def _generate_preview(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        preview_rows: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate preview data showing before/after comparison.

        Args:
            df_before: Original DataFrame
            df_after: Processed DataFrame
            before_stats: Statistics before processing
            after_stats: Statistics after processing
            preview_rows: Number of rows to include in preview

        Returns:
            Preview data dictionary
        """
        # Get sample rows
        sample_size = min(preview_rows, len(df_before))

        # Find rows with missing values for better preview
        rows_with_missing = df_before[df_before.isnull().any(axis=1)].head(sample_size)

        if len(rows_with_missing) > 0:
            indices = rows_with_missing.index.tolist()
            # If we need more rows to fill sample_size
            if len(indices) < sample_size:
                remaining = sample_size - len(indices)
                other_indices = [idx for idx in df_before.index if idx not in indices][:remaining]
                indices.extend(other_indices)
        else:
            indices = df_before.index[:sample_size].tolist()

        # Prepare samples, replacing NaN with None for JSON serialization
        before_sample = df_before.loc[indices].replace({np.nan: None}).to_dict(orient="records")

        # Identify valid indices in after dataframe (some rows might have been dropped)
        valid_indices = [idx for idx in indices if idx in df_after.index]
        after_sample = df_after.loc[valid_indices].replace({np.nan: None}).to_dict(orient="records")

        # Track changes for highlighting
        changes = []
        for idx in valid_indices:
            row_changes = {}
            for col in df_before.columns:
                if col not in df_after.columns:
                    continue

                val_before = df_before.at[idx, col]
                val_after = df_after.at[idx, col]

                # Check for changes (handling NaN)
                is_nan_before = pd.isna(val_before)
                is_nan_after = pd.isna(val_after)

                # Change detected if:
                # 1. Was NaN and is now not NaN
                # 2. Was not NaN and is now NaN (unlikely for missing value handler but possible)
                # 3. Both not NaN and values are different
                if (is_nan_before != is_nan_after) or (
                    not is_nan_before and not is_nan_after and val_before != val_after
                ):
                    row_changes[col] = True

            changes.append(row_changes)

        return {
            "before_sample": before_sample,
            "after_sample": after_sample,
            "changes": changes,
            "before_stats": before_stats,
            "after_stats": after_stats,
            "rows_shown": len(before_sample),
            "total_rows_before": len(df_before),
            "total_rows_after": len(df_after),
        }

    def _generate_strategy_comparison(
        self, df: pd.DataFrame, column_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparison of all strategies for each column with missing values."""
        import random as _random
        comparison = {}

        for column in df.columns:
            missing_count = int(df[column].isnull().sum())
            if missing_count == 0:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df[column])

            # Get sample values (up to 8) showing at least one null
            null_indices = df[column][df[column].isnull()].head(3).index.tolist()
            non_null_indices = df[column][df[column].notna()].head(5).index.tolist()
            combined_indices = sorted(set(null_indices + non_null_indices))[:8]
            sample_values = [
                None if pd.isna(df[column].iloc[i]) else (
                    round(float(df[column].iloc[i]), 2) if is_numeric
                    else str(df[column].iloc[i])
                )
                for i in range(len(df)) if i in combined_indices
            ]

            applied_strategy = None
            if column in column_configs:
                applied_strategy = column_configs[column].get("strategy", "none")

            strategies = {}
            non_null_data = df[column].dropna()

            if is_numeric and len(non_null_data) > 0:
                mean_val = round(float(non_null_data.mean()), 2)
                median_val = round(float(non_null_data.median()), 2)
                mode_vals = non_null_data.mode()
                mode_val = round(float(mode_vals.iloc[0]), 2) if len(mode_vals) > 0 else None

                strategies["mean"] = {
                    "fill_value": mean_val,
                    "description": "Replace with the average of all values",
                    "when_to_use": "When data is evenly distributed without extreme outliers"
                }
                strategies["median"] = {
                    "fill_value": median_val,
                    "description": "Replace with the middle value when sorted",
                    "when_to_use": "When data has outliers or is skewed"
                }
                if mode_val is not None:
                    strategies["mode"] = {
                        "fill_value": mode_val,
                        "description": "Replace with the most frequent value",
                        "when_to_use": "When one value appears much more often than others"
                    }
            else:
                mode_vals = non_null_data.mode()
                if len(mode_vals) > 0:
                    strategies["mode"] = {
                        "fill_value": str(mode_vals.iloc[0]),
                        "description": "Replace with the most frequent category",
                        "when_to_use": "Best default for text/categorical columns"
                    }

            strategies["drop"] = {
                "rows_removed": missing_count,
                "description": "Remove all rows that have a missing value here",
                "when_to_use": "When very few rows are missing and dataset is large"
            }

            # Forward/backward fill examples
            if missing_count > 0:
                try:
                    ff_series = df[column].ffill()
                    bf_series = df[column].bfill()
                    first_null_idx = df[column][df[column].isnull()].index[0]

                    ff_val = ff_series.iloc[first_null_idx]
                    bf_val = bf_series.iloc[first_null_idx]
                    strategies["forward_fill"] = {
                        "fill_value": None if pd.isna(ff_val) else (
                            round(float(ff_val), 2) if is_numeric else str(ff_val)
                        ),
                        "description": "Copy the value from the row above",
                        "when_to_use": "When data has a time-based order (like daily readings)"
                    }
                    strategies["backward_fill"] = {
                        "fill_value": None if pd.isna(bf_val) else (
                            round(float(bf_val), 2) if is_numeric else str(bf_val)
                        ),
                        "description": "Copy the value from the row below",
                        "when_to_use": "When data has a time-based order and you prefer future values"
                    }
                except Exception:
                    pass

            comparison[column] = {
                "missing_count": missing_count,
                "is_numeric": is_numeric,
                "sample_values": sample_values,
                "strategies": strategies,
                "applied_strategy": applied_strategy,
            }

        return comparison

    def _generate_quiz_questions(
        self,
        df: pd.DataFrame,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        operation_log: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions from the actual dataset analysis."""
        import random as _random
        questions = []
        q_id = 0

        missing_by_col = before_stats.get("missing_by_column", {})

        # Q Type 1: For numeric columns, ask about mean vs median
        for col in df.columns:
            if col not in missing_by_col:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) < 5:
                continue

            mean_val = non_null.mean()
            median_val = non_null.median()
            std_val = non_null.std()
            if std_val > 0 and abs(mean_val - median_val) > 0.1 * std_val:
                q_id += 1
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"Column '{col}' has a mean of {mean_val:.1f} and median of {median_val:.1f}. The difference suggests possible outliers. Which strategy is more robust to outliers?",
                    "options": ["Mean", "Median", "Mode", "Drop Rows"],
                    "correct_answer": 1,
                    "explanation": f"Median ({median_val:.1f}) is more robust to outliers than mean ({mean_val:.1f}) because it uses the middle value, not affected by extreme values.",
                    "difficulty": "medium"
                })
                if len(questions) >= 6:
                    break

        # Q Type 2: Categorical column -- can you use mean?
        for col in df.columns:
            if col not in missing_by_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"Column '{col}' contains text/categories (not numbers). Which strategy can you NOT use?",
                "options": ["Mode (most frequent)", "Mean (average)", "Drop rows", "Forward fill"],
                "correct_answer": 1,
                "explanation": f"Mean calculates the average, which only works for numbers. For text columns like '{col}', use Mode (most frequent value) instead.",
                "difficulty": "easy"
            })
            if len(questions) >= 6:
                break

        # Q Type 3: Missing percentage question
        if missing_by_col:
            col_with_most = max(missing_by_col.items(), key=lambda x: x[1]["percentage"])
            col_name, col_stats = col_with_most
            pct = col_stats["percentage"]
            total_rows = before_stats.get("total_rows", 100)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"Column '{col_name}' has {pct:.1f}% missing values. If you use 'Drop Rows', approximately how many rows would be removed from {total_rows} total rows?",
                "options": [
                    f"About {max(1, int(total_rows * pct / 100 * 0.5))} rows",
                    f"About {max(1, int(total_rows * pct / 100))} rows",
                    f"About {max(1, int(total_rows * pct / 100 * 2))} rows",
                    "0 rows",
                ],
                "correct_answer": 1,
                "explanation": f"With {pct:.1f}% missing, roughly {int(total_rows * pct / 100)} out of {total_rows} rows would be dropped.",
                "difficulty": "medium"
            })

        # Q Type 4: General knowledge -- what does forward fill do?
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does 'Forward Fill' do when it finds a missing value?",
            "options": [
                "Deletes the row",
                "Copies the value from the previous row",
                "Calculates the average of all values",
                "Replaces it with zero"
            ],
            "correct_answer": 1,
            "explanation": "Forward fill copies the value from the row above (the previous row). This is useful when data is ordered in time, like daily temperature readings.",
            "difficulty": "easy"
        })

        # Q Type 5: Which strategy was used? (based on actual operation log)
        if operation_log:
            for col, op in list(operation_log.items())[:1]:
                strategy = op.get("strategy", "unknown")
                if strategy in ("unknown", "none") or "error" in op:
                    continue
                filled = op.get("filled_count", 0)
                strategy_label = strategy.replace("_", " ").title()
                q_id += 1
                all_options = ["Mean", "Median", "Mode", "Drop Rows", "Forward Fill", "Backward Fill"]
                correct_options = [strategy_label]
                other_options = [o for o in all_options if o.lower() != strategy_label.lower()]
                _random.shuffle(other_options)
                final_options = (correct_options + other_options[:3])
                _random.shuffle(final_options)
                correct_idx = final_options.index(strategy_label)
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"In this pipeline, column '{col}' had its {filled} missing values handled. What strategy was applied?",
                    "options": final_options,
                    "correct_answer": correct_idx,
                    "explanation": f"The strategy '{strategy_label}' was chosen for column '{col}', which filled {filled} missing values.",
                    "difficulty": "easy"
                })

        _random.shuffle(questions)
        return questions[:5]

    def _generate_missing_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a boolean grid showing where missing values are located."""
        max_rows = 50
        max_cols = 20

        columns = df.columns.tolist()[:max_cols]
        df_subset = df[columns]
        total_rows = len(df_subset)

        if total_rows <= max_rows:
            sampled_df = df_subset
            row_indices = list(range(total_rows))
        else:
            step = total_rows / max_rows
            row_indices = [int(i * step) for i in range(max_rows)]
            sampled_df = df_subset.iloc[row_indices]

        grid = sampled_df.isnull().values.tolist()

        col_missing_pct = {}
        for col in columns:
            pct = round(float(df[col].isnull().sum() / total_rows * 100), 1)
            col_missing_pct[col] = pct

        cols_with_missing = [c for c in columns if df[c].isnull().any()]

        if len(cols_with_missing) == 0:
            pattern = "No missing values detected in the dataset."
        elif len(cols_with_missing) == 1:
            pattern = f"Missing values are concentrated in column '{cols_with_missing[0]}'"
        else:
            rows_with_any_missing = df_subset.isnull().any(axis=1).sum()
            rows_with_all_cols_missing = df_subset[cols_with_missing].isnull().all(axis=1).sum()

            if rows_with_all_cols_missing > rows_with_any_missing * 0.5 and rows_with_any_missing > 0:
                pattern = "Missing values tend to appear together in the same rows (correlated pattern)"
            else:
                worst_col = max(col_missing_pct.items(), key=lambda x: x[1])
                pattern = f"Missing values appear scattered across columns. '{worst_col[0]}' has the most ({worst_col[1]}%)"

        return {
            "columns": columns,
            "rows_sampled": len(sampled_df),
            "total_rows": total_rows,
            "grid": grid,
            "column_missing_pct": col_missing_pct,
            "row_indices": row_indices,
            "pattern": pattern,
        }
