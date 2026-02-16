"""
Dataset Analyzer Service

Analyzes uploaded datasets to provide intelligent insights:
- Data quality assessment
- Column type detection
- Preprocessing recommendations
- Target column suggestions
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from app.mentor.schemas import DatasetInsight
from app.core.logging import logger


class DatasetAnalyzer:
    """Intelligent dataset analysis for mentor system."""

    # Thresholds for detection
    HIGH_CARDINALITY_THRESHOLD = 50
    HIGH_MISSING_THRESHOLD = 0.2  # 20%
    HIGH_VARIANCE_THRESHOLD = 100  # For scaling detection

    def analyze_dataset(
        self,
        dataset_id: str,
        columns: List[str],
        dtypes: Dict[str, str],
        n_rows: int,
        n_columns: int,
        missing_values: Dict[str, int], 
        statistics: Optional[Dict[str, Any]] = None,
        preview_data: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetInsight:
        """
        Perform comprehensive dataset analysis.
 
        Args:
            dataset_id: Dataset identifier
            columns: List of column names
            dtypes: Column data types
            n_rows: Number of rows
            n_columns: Number of columns
            missing_values: Missing value counts per column
            statistics: Optional column statistics
            preview_data: Optional preview rows

        Returns:
            DatasetInsight with complete analysis
        """
        logger.info(f"Analyzing dataset: {dataset_id}")

        # Classify columns
        numeric_cols, categorical_cols = self._classify_columns(dtypes, columns, preview_data)

        # Detect data quality issues
        warnings = self._generate_warnings(
            columns, missing_values, n_rows, preview_data, statistics
        )

        # Generate preprocessing recommendations
        recommendations = self._generate_recommendations(
            numeric_cols, categorical_cols, missing_values, n_rows, statistics
        )

        # Detect high cardinality columns
        high_cardinality = self._detect_high_cardinality(categorical_cols, preview_data, statistics)

        # Detect columns needing scaling 
        scaling_needed = self._detect_scaling_needs(numeric_cols, statistics)

        # Suggest target column
        recommended_target = self._suggest_target_column(columns, dtypes, preview_data)

        # Generate summary
        summary = self._generate_summary(
            n_rows, n_columns, numeric_cols, categorical_cols, missing_values
        )

        return DatasetInsight(
            summary=summary,
            n_rows=n_rows,
            n_columns=n_columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            missing_values=missing_values,
            high_cardinality_columns=high_cardinality,
            scaling_needed=scaling_needed,
            recommended_target=recommended_target,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _classify_columns(
        self,
        dtypes: Dict[str, str],
        columns: Optional[List[str]] = None,
        preview_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Classify columns as numeric or categorical."""
        numeric_types = ["int64", "float64", "int32", "float32", "number", "integer", "float"]

        numeric_cols = []
        categorical_cols = []

        for col, dtype in dtypes.items():
            dtype_lower = str(dtype).lower()
            if any(nt in dtype_lower for nt in numeric_types):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # Fallback: if dtypes is empty but we have columns and preview data, infer types
        if not dtypes and columns and preview_data:
            logger.info("dtypes empty, inferring column types from preview data")
            for col in columns:
                values = [row.get(col) for row in preview_data if row.get(col) is not None]
                if values and all(isinstance(v, (int, float)) for v in values):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

        return numeric_cols, categorical_cols

    def _generate_warnings(
        self,
        columns: List[str],
        missing_values: Dict[str, int],
        n_rows: int,
        preview_data: Optional[List[Dict]] = None,
        statistics: Optional[Dict] = None,
    ) -> List[str]:
        """Generate data quality warnings."""
        warnings = []

        # Check for high missing values
        for col, missing_count in missing_values.items():
            if missing_count > 0:
                missing_pct = (missing_count / n_rows) * 100
                if missing_pct > self.HIGH_MISSING_THRESHOLD * 100:
                    warnings.append(
                        f"Column '{col}' has {missing_pct:.1f}% missing values - "
                        f"consider using Missing Value Handler"
                    )

        # Check for potential duplicates (if preview available)
        if preview_data and len(preview_data) > 1:
            unique_rows = len({str(sorted(row.items())) for row in preview_data})
            if unique_rows < len(preview_data) * 0.9:
                warnings.append("Potential duplicate rows detected - consider deduplication")

        # Check for very small dataset
        if n_rows < 100:
            warnings.append(
                f"Small dataset ({n_rows} rows) - results may not be statistically significant"
            )

        return warnings

    def _generate_recommendations(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        missing_values: Dict[str, int],
        n_rows: int,
        statistics: Optional[Dict] = None,
    ) -> List[str]:
        """Generate preprocessing recommendations."""
        recommendations = []

        # Encoding recommendations
        if categorical_cols:
            cat_with_missing = [col for col in categorical_cols if missing_values.get(col, 0) > 0]

            if cat_with_missing:
                recommendations.append(
                    f"Handle missing values in categorical columns: {', '.join(cat_with_missing[:3])}"
                    + (
                        f" and {len(cat_with_missing) - 3} more"
                        if len(cat_with_missing) > 3
                        else ""
                    )
                )

            recommendations.append(
                f"Encode categorical columns ({len(categorical_cols)} found) using "
                "Label Encoding or One-Hot Encoding"
            )

        # Scaling recommendations
        if numeric_cols and statistics:
            high_variance_cols = []
            for col in numeric_cols:
                col_stats = statistics.get(col, {})
                if isinstance(col_stats, dict):
                    std = col_stats.get("std", 0)
                    if std > self.HIGH_VARIANCE_THRESHOLD:
                        high_variance_cols.append(col)

            if high_variance_cols:
                recommendations.append(
                    f"Consider scaling/normalization for high-variance features: "
                    f"{', '.join(high_variance_cols[:3])}"
                )

        # Missing value handling
        total_missing = sum(missing_values.values())
        if total_missing > 0:
            recommendations.append(
                f"Total {total_missing} missing values across {len([v for v in missing_values.values() if v > 0])} "
                "columns - use Missing Value Handler node"
            )

        return recommendations

    def _detect_high_cardinality(
        self,
        categorical_cols: List[str],
        preview_data: Optional[List[Dict]] = None,
        statistics: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Detect high cardinality categorical columns."""
        high_cardinality = []

        if not categorical_cols:
            return high_cardinality

        # Use statistics if available
        if statistics:
            for col in categorical_cols:
                col_stats = statistics.get(col, {})
                if isinstance(col_stats, dict):
                    unique_count = col_stats.get("unique", 0)
                    if unique_count > self.HIGH_CARDINALITY_THRESHOLD:
                        high_cardinality.append(
                            {
                                "column": col,
                                "unique_values": unique_count,
                                "warning": f"High cardinality ({unique_count} unique values) may cause "
                                "memory issues with one-hot encoding",
                            }
                        )

        # Fallback to preview data
        elif preview_data:
            for col in categorical_cols:
                unique_values = set()
                for row in preview_data:
                    if col in row and row[col] is not None:
                        unique_values.add(str(row[col]))

                if len(unique_values) > min(
                    self.HIGH_CARDINALITY_THRESHOLD, len(preview_data) * 0.8
                ):
                    high_cardinality.append(
                        {
                            "column": col,
                            "unique_values": len(unique_values),
                            "warning": f"High cardinality detected in preview - consider label encoding",
                        }
                    )

        return high_cardinality

    def _detect_scaling_needs(
        self, numeric_cols: List[str], statistics: Optional[Dict] = None
    ) -> List[str]:
        """Detect numeric columns that need scaling."""
        if not statistics or not numeric_cols:
            return []

        scaling_needed = []

        for col in numeric_cols:
            col_stats = statistics.get(col, {})
            if isinstance(col_stats, dict):
                std = col_stats.get("std", 0)
                mean = col_stats.get("mean", 0)

                # Detect if variance is large
                if std > self.HIGH_VARIANCE_THRESHOLD:
                    scaling_needed.append(col)
                # Or if min/max range is large
                elif "min" in col_stats and "max" in col_stats:
                    range_val = col_stats["max"] - col_stats["min"]
                    if range_val > 1000:
                        scaling_needed.append(col)

        return scaling_needed

    def _suggest_target_column(
        self, columns: List[str], dtypes: Dict[str, str], preview_data: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """Suggest a likely target column based on patterns."""
        # Common target column name patterns
        target_patterns = ["target", "label", "class", "y", "output", "result", "price", "value"]

        # Check for exact or partial matches
        for pattern in target_patterns:
            for col in columns:
                if pattern in col.lower():
                    return col

        # Default to last column (common convention)
        if columns:
            return columns[-1]

        return None

    def _generate_summary(
        self,
        n_rows: int,
        n_columns: int,
        numeric_cols: List[str],
        categorical_cols: List[str],
        missing_values: Dict[str, int],
    ) -> str:
        """Generate human-readable dataset summary."""
        total_missing = sum(missing_values.values())
        missing_pct = (total_missing / (n_rows * n_columns) * 100) if n_rows * n_columns > 0 else 0

        summary_parts = [
            f"Your dataset has {n_rows:,} rows and {n_columns} columns.",
            f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features.",
        ]

        if total_missing > 0:
            summary_parts.append(
                f"Detected {total_missing:,} missing values ({missing_pct:.1f}% of data)."
            )
        else:
            summary_parts.append("No missing values detected - excellent data quality!")

        return " ".join(summary_parts)


# Singleton instance
dataset_analyzer = DatasetAnalyzer()
