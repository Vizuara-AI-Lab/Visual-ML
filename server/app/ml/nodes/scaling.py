"""
Scaling Node - Scale and normalize numerical features.
Supports Standard, MinMax, Robust, and Normalizer scaling methods.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import joblib
import io

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class ScalingInput(NodeInput):
    """Input schema for Scaling node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    method: str = Field(
        "standard", description="Scaling method: standard, minmax, robust, normalize"
    )
    columns: List[str] = Field(
        default_factory=list, description="Columns to scale (empty = all numeric)"
    )


class ScalingOutput(NodeOutput):
    """Output schema for Scaling node."""

    scaled_dataset_id: str = Field(..., description="ID of scaled dataset")
    scaled_path: str = Field(..., description="Path to scaled dataset")
    artifacts_path: Optional[str] = Field(None, description="Path to scaler artifact")

    scaled_columns: List[str] = Field(..., description="Columns that were scaled")
    columns: List[str] = Field(..., description="All columns in the dataset (for downstream nodes)")
    scaling_method: str = Field(..., description="Scaling method used")
    scaling_summary: Dict[str, Any] = Field(..., description="Summary of scaling operations")
    warnings: List[str] = Field(default_factory=list, description="Warnings about skipped columns")

    # Learning activity data (all Optional for backward compatibility)
    scaling_before_after: Optional[Dict[str, Any]] = Field(
        None, description="Per-column statistics before and after scaling"
    )
    scaling_method_comparison: Optional[Dict[str, Any]] = Field(
        None, description="Comparison of all scaling methods on each column"
    )
    scaling_outlier_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Analysis of outlier impact on scaling for each column"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about scaling"
    )


class ScalingNode(BaseNode):
    """
    Scaling Node - Scale and normalize features.

    Supports:
    - Standard Scaler: (x - mean) / std
    - MinMax Scaler: (x - min) / (max - min)
    - Robust Scaler: Uses median and IQR
    - Normalizer: Scale samples to unit norm
    """

    node_type = "scaling"

    @property
    def metadata(self) -> NodeMetadata:
        """Return node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.PREPROCESSING,
            primary_output_field="scaled_dataset_id",
            output_fields={
                "scaled_dataset_id": "ID of the scaled dataset",
                "scaled_path": "Path to scaled dataset file",
                "artifacts_path": "Path to scaler artifacts",
                "scaled_columns": "Columns that were scaled",
                "columns": "All columns in the dataset (for downstream nodes)",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            allowed_source_categories=[NodeCategory.DATA_SOURCE, NodeCategory.PREPROCESSING],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        return ScalingInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return ScalingOutput

    async def _execute(self, input_data: ScalingInput) -> ScalingOutput:
        """Execute scaling."""
        try:
            logger.info(f"Starting scaling for dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            df_scaled = df.copy()

            # Determine columns to scale
            if input_data.columns is not None and len(input_data.columns) > 0:
                columns_to_scale = input_data.columns
                logger.info(f"Received columns for scaling: {columns_to_scale}")

                # WARN if all columns were sent - likely a frontend bug
                if len(columns_to_scale) == len(df.columns):
                    logger.warning(
                        f"WARNING: All {len(df.columns)} columns were sent for scaling. "
                        f"This might be a frontend bug. Will filter to numeric only."
                    )
            else:
                # Scale all numeric columns
                columns_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
                logger.info(
                    f"No columns specified, auto-detecting numeric columns: {columns_to_scale}"
                )

            # Filter to ONLY numeric columns and warn about skipped ones
            numeric_columns = []
            skipped_columns = []
            warnings = []

            for col in columns_to_scale:
                if col not in df.columns:
                    warning_msg = f"Column '{col}' not found in dataset - skipping scaling"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)
                    skipped_columns.append(col)
                    continue

                # Check if column is numeric or can be converted
                if pd.api.types.is_numeric_dtype(df_scaled[col]):
                    numeric_columns.append(col)
                else:
                    # Try to convert to numeric
                    try:
                        test_conversion = pd.to_numeric(df_scaled[col], errors="coerce")
                        if not test_conversion.isna().all():
                            # Has some valid numeric values
                            numeric_columns.append(col)
                        else:
                            warning_msg = f"Column '{col}' is not numeric and has no valid numeric values - skipping scaling"
                            logger.warning(warning_msg)
                            warnings.append(warning_msg)
                            skipped_columns.append(col)
                    except Exception:
                        warning_msg = f"Column '{col}' is not numeric - skipping scaling"
                        logger.warning(warning_msg)
                        warnings.append(warning_msg)
                        skipped_columns.append(col)

            if len(skipped_columns) > 0:
                logger.info(f"Skipped non-numeric columns: {skipped_columns}")

            if len(numeric_columns) == 0:
                raise ValueError(
                    "No valid numeric columns to scale. Please select at least one numeric column."
                )

            columns_to_scale = numeric_columns
            logger.info(f"Final columns to scale (numeric only): {columns_to_scale}")

            # Convert non-numeric columns to numeric where needed
            rows_before = len(df_scaled)
            for col in columns_to_scale:
                if not pd.api.types.is_numeric_dtype(df_scaled[col]):
                    df_scaled[col] = pd.to_numeric(df_scaled[col], errors="coerce")
                    # Drop rows with NaN in this column
                    df_scaled = df_scaled.dropna(subset=[col])

            rows_dropped = rows_before - len(df_scaled)
            if rows_dropped > 0:
                logger.info(
                    f"Dropped {rows_dropped} rows with non-numeric values in scaling columns"
                )

            # Select scaler
            if input_data.method == "standard":
                scaler = StandardScaler()
            elif input_data.method == "minmax":
                scaler = MinMaxScaler()
            elif input_data.method == "robust":
                scaler = RobustScaler()
            elif input_data.method == "normalize":
                scaler = Normalizer()
            else:
                raise ValueError(f"Unknown scaling method: {input_data.method}")

            # Apply scaling ONLY to the selected columns from the cleaned dataframe
            logger.info(f"Applying {input_data.method} scaling to columns: {columns_to_scale}")
            df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

            # Save scaler artifact
            artifacts_id = generate_id("scaler")
            artifacts_path = Path(settings.UPLOAD_DIR) / "artifacts" / f"{artifacts_id}.pkl"
            artifacts_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, artifacts_path)

            # Save scaled dataset
            scaled_id = generate_id("scaled")
            scaled_path = Path(settings.UPLOAD_DIR) / f"{scaled_id}.csv"
            scaled_path.parent.mkdir(parents=True, exist_ok=True)
            df_scaled.to_csv(scaled_path, index=False)

            # Create summary
            scaling_summary = {
                "method": input_data.method,
                "columns_scaled": len(columns_to_scale),
                "columns_list": columns_to_scale,
                "original_rows": len(df),
                "final_rows": len(df_scaled),
                "scaler_params": scaler.get_params() if hasattr(scaler, "get_params") else {},
            }

            logger.info(
                f"Scaling complete - Method: {input_data.method}, "
                f"Columns scaled: {len(columns_to_scale)}, "
                f"Saved to: {scaled_path}"
            )

            # Generate learning activity data (non-blocking)
            scaling_before_after = None
            try:
                scaling_before_after = self._generate_scaling_before_after(
                    df, df_scaled, columns_to_scale, input_data.method
                )
            except Exception as e:
                logger.warning(f"Scaling before/after generation failed: {e}")

            scaling_method_comparison = None
            try:
                scaling_method_comparison = self._generate_scaling_method_comparison(
                    df, columns_to_scale
                )
            except Exception as e:
                logger.warning(f"Scaling method comparison generation failed: {e}")

            scaling_outlier_analysis = None
            try:
                scaling_outlier_analysis = self._generate_scaling_outlier_analysis(
                    df, columns_to_scale
                )
            except Exception as e:
                logger.warning(f"Scaling outlier analysis generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_scaling_quiz(
                    df, df_scaled, columns_to_scale, input_data.method
                )
            except Exception as e:
                logger.warning(f"Scaling quiz generation failed: {e}")

            return ScalingOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                scaled_dataset_id=scaled_id,
                scaled_path=str(scaled_path),
                artifacts_path=str(artifacts_path),
                scaled_columns=columns_to_scale,
                columns=df_scaled.columns.tolist(),
                scaling_method=input_data.method,
                scaling_summary=scaling_summary,
                warnings=warnings,
                scaling_before_after=scaling_before_after,
                scaling_method_comparison=scaling_method_comparison,
                scaling_outlier_analysis=scaling_outlier_analysis,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"Scaling failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

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

    # --- Learning Activity Helpers ---

    def _generate_scaling_before_after(
        self, df: pd.DataFrame, df_scaled: pd.DataFrame,
        columns_to_scale: List[str], method: str
    ) -> Dict[str, Any]:
        """Generate per-column statistics before and after scaling."""
        formulas = {
            "standard": "(x - mean) / std",
            "minmax": "(x - min) / (max - min)",
            "robust": "(x - median) / IQR",
            "normalize": "x / ||x|| (unit norm per sample)",
        }

        columns_data = {}
        for col in columns_to_scale[:5]:
            before_vals = df[col].dropna()
            after_vals = df_scaled[col].dropna() if col in df_scaled.columns else pd.Series()

            sample_before = before_vals.head(5).tolist()
            sample_after = after_vals.head(5).tolist()

            columns_data[col] = {
                "before": {
                    "mean": round(float(before_vals.mean()), 4),
                    "std": round(float(before_vals.std()), 4),
                    "min": round(float(before_vals.min()), 4),
                    "max": round(float(before_vals.max()), 4),
                    "median": round(float(before_vals.median()), 4),
                    "sample_values": [round(float(v), 4) for v in sample_before],
                },
                "after": {
                    "mean": round(float(after_vals.mean()), 4) if len(after_vals) > 0 else 0,
                    "std": round(float(after_vals.std()), 4) if len(after_vals) > 0 else 0,
                    "min": round(float(after_vals.min()), 4) if len(after_vals) > 0 else 0,
                    "max": round(float(after_vals.max()), 4) if len(after_vals) > 0 else 0,
                    "median": round(float(after_vals.median()), 4) if len(after_vals) > 0 else 0,
                    "sample_values": [round(float(v), 4) for v in sample_after],
                },
                "formula": formulas.get(method, "unknown"),
                "method": method,
            }

        return {"columns": columns_data, "total_columns": len(columns_to_scale)}

    def _generate_scaling_method_comparison(
        self, df: pd.DataFrame, columns_to_scale: List[str]
    ) -> Dict[str, Any]:
        """Apply all 4 scaling methods to each column for comparison."""
        result = {}

        for col in columns_to_scale[:5]:
            col_data = df[[col]].dropna()
            if len(col_data) == 0:
                continue

            methods_result = {}
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }

            for m_name, scaler in scalers.items():
                scaled_vals = scaler.fit_transform(col_data)
                flat = scaled_vals.flatten()
                sample = [round(float(v), 4) for v in flat[:5]]

                formulas = {
                    "standard": "(x - mean) / std",
                    "minmax": "(x - min) / (max - min)",
                    "robust": "(x - median) / IQR",
                }
                descriptions = {
                    "standard": "Centers data around 0 with standard deviation 1",
                    "minmax": "Scales data to a fixed range [0, 1]",
                    "robust": "Uses median and IQR, resistant to outliers",
                }
                best_for = {
                    "standard": "Normally distributed data without many outliers",
                    "minmax": "Data that needs to be in a bounded range (e.g. neural networks)",
                    "robust": "Data with outliers that shouldn't dominate the scaling",
                }

                methods_result[m_name] = {
                    "mean": round(float(flat.mean()), 4),
                    "std": round(float(flat.std()), 4),
                    "min": round(float(flat.min()), 4),
                    "max": round(float(flat.max()), 4),
                    "formula": formulas[m_name],
                    "description": descriptions[m_name],
                    "best_for": best_for[m_name],
                    "sample_values": sample,
                }

            # Normalizer operates on rows, so just add description
            methods_result["normalize"] = {
                "description": "Scales each sample (row) to unit norm",
                "best_for": "Text data or when direction matters more than magnitude",
                "formula": "x / ||x|| (L2 norm)",
                "note": "Operates on rows, not columns — comparison not shown",
            }

            result[col] = methods_result

        return result

    def _generate_scaling_outlier_analysis(
        self, df: pd.DataFrame, columns_to_scale: List[str]
    ) -> Dict[str, Any]:
        """Analyze outlier impact on scaling for each column using IQR."""
        result = {}

        for col in columns_to_scale[:5]:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue

            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (vals < lower_bound) | (vals > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outlier_values = vals[outlier_mask].head(5).tolist()

            # Stats with and without outliers
            clean_vals = vals[~outlier_mask]

            col_result: Dict[str, Any] = {
                "has_outliers": outlier_count > 0,
                "outlier_count": outlier_count,
                "total_values": len(vals),
                "iqr_bounds": {
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "lower": round(lower_bound, 4),
                    "upper": round(upper_bound, 4),
                },
                "outlier_values": [round(float(v), 4) for v in outlier_values],
            }

            if outlier_count > 0 and len(clean_vals) > 0:
                col_result["stats_with_outliers"] = {
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std()), 4),
                }
                col_result["stats_without_outliers"] = {
                    "mean": round(float(clean_vals.mean()), 4),
                    "std": round(float(clean_vals.std()), 4),
                }

                # Compare standard vs robust scaling
                col_data = df[[col]].dropna()
                std_scaled = StandardScaler().fit_transform(col_data).flatten()
                rob_scaled = RobustScaler().fit_transform(col_data).flatten()

                col_result["standard_vs_robust"] = {
                    "standard": {
                        "min": round(float(std_scaled.min()), 4),
                        "max": round(float(std_scaled.max()), 4),
                        "range": round(float(std_scaled.max() - std_scaled.min()), 4),
                    },
                    "robust": {
                        "min": round(float(rob_scaled.min()), 4),
                        "max": round(float(rob_scaled.max()), 4),
                        "range": round(float(rob_scaled.max() - rob_scaled.min()), 4),
                    },
                    "message": "Robust scaling produces a smaller range, meaning outliers have less impact.",
                }
            else:
                col_result["message"] = "No outliers detected using IQR method."

            result[col] = col_result

        return result

    def _generate_scaling_quiz(
        self, df: pd.DataFrame, df_scaled: pd.DataFrame,
        columns_to_scale: List[str], method: str
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about scaling from actual data."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: Range question
        if columns_to_scale:
            col = columns_to_scale[0]
            orig_mean = round(float(df[col].mean()), 2)
            q_id += 1
            if method == "standard":
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"After standard scaling, column '{col}' has a mean of approximately 0. What was the original mean?",
                    "options": [str(orig_mean), "0", "1", str(round(orig_mean * 2, 2))],
                    "correct_answer": 0,
                    "explanation": f"Standard scaling shifts data so the mean becomes 0, but the original mean was {orig_mean}. The formula subtracts the mean from every value.",
                    "difficulty": "medium",
                })
            elif method == "minmax":
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"After MinMax scaling, column '{col}' has values between 0 and 1. What does a scaled value of 0.5 represent?",
                    "options": [
                        "The midpoint between the original min and max",
                        "The mean of the original data",
                        "The median of the original data",
                        "Half the standard deviation",
                    ],
                    "correct_answer": 0,
                    "explanation": "MinMax scaling maps the minimum to 0 and maximum to 1. A value of 0.5 means it was exactly halfway between the original min and max.",
                    "difficulty": "medium",
                })

        # Q2: Method recognition
        q_id += 1
        options = ["MinMax Scaling", "Standard Scaling", "Robust Scaling", "Normalizer"]
        _random.shuffle(options)
        correct_idx = options.index("MinMax Scaling")
        questions.append({
            "id": f"q{q_id}",
            "question": "Which scaling method guarantees that all values will be between 0 and 1?",
            "options": options,
            "correct_answer": correct_idx,
            "explanation": "MinMax Scaling uses the formula (x - min) / (max - min), which always produces values in the range [0, 1].",
            "difficulty": "easy",
        })

        # Q3: Outlier awareness
        outlier_col = None
        for col in columns_to_scale[:5]:
            vals = df[col].dropna()
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if ((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum() > 0:
                outlier_col = col
                break

        if outlier_col:
            q_id += 1
            options3 = ["Robust Scaling", "Standard Scaling", "MinMax Scaling", "It doesn't matter"]
            _random.shuffle(options3)
            correct_idx3 = options3.index("Robust Scaling")
            questions.append({
                "id": f"q{q_id}",
                "question": f"Column '{outlier_col}' has outliers. Which scaling method is most resistant to their influence?",
                "options": options3,
                "correct_answer": correct_idx3,
                "explanation": "Robust Scaling uses the median and IQR instead of mean and standard deviation, so extreme outliers have much less effect on the result.",
                "difficulty": "medium",
            })

        # Q4: Formula question
        if columns_to_scale:
            col = columns_to_scale[0]
            mean_val = round(float(df[col].mean()), 1)
            std_val = round(float(df[col].std()), 1)
            if std_val > 0:
                test_val = round(mean_val + 2 * std_val, 1)
                expected = round((test_val - mean_val) / std_val, 2)
                wrong1 = round(expected + 1, 2)
                wrong2 = round(expected - 0.5, 2)
                wrong3 = round(test_val / mean_val, 2) if mean_val != 0 else 0.5
                q_id += 1
                options4 = [str(expected), str(wrong1), str(wrong2), str(wrong3)]
                _random.shuffle(options4)
                correct_idx4 = options4.index(str(expected))
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"Column '{col}' has mean={mean_val} and std={std_val}. Using standard scaling, what is the scaled value for {test_val}?",
                    "options": options4,
                    "correct_answer": correct_idx4,
                    "explanation": f"Standard scaling: (x - mean) / std = ({test_val} - {mean_val}) / {std_val} = {expected}",
                    "difficulty": "hard",
                })

        # Q5: Why scale
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What problem occurs if features have very different scales (e.g., Salary: 20,000-100,000 vs Age: 18-65)?",
            "options": [
                "Features with larger values dominate the model's learning",
                "The model will crash with an error",
                "It makes training faster",
                "Nothing — models handle different scales automatically",
            ],
            "correct_answer": 0,
            "explanation": "When features have very different scales, algorithms like gradient descent, KNN, and SVM give more weight to larger-valued features. Scaling ensures all features contribute equally.",
            "difficulty": "easy",
        })

        _random.shuffle(questions)
        return questions[:5]
