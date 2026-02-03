"""
View Nodes - Display and visualize dataset content
"""

from typing import Type, Optional, Dict, Any, List, Union
from pydantic import Field, field_validator
import pandas as pd
import io
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.logging import logger
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service
from pathlib import Path


class TableViewInput(NodeInput):
    """Input schema for TableView node."""

    dataset_id: str = Field(..., description="Dataset ID to view")
    max_rows: int = Field(100, description="Maximum rows to display", ge=10, le=1000)
    columns_to_show: Optional[List[str]] = Field(None, description="Specific columns to show")

    # Optional metadata fields (passed from upload nodes)
    filename: Optional[str] = Field(None, description="Filename (metadata)")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[List[str]] = Field(None, description="Column names (metadata)")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Data types (metadata)")


class TableViewOutput(NodeOutput):
    """Output schema for TableView node."""

    dataset_id: str = Field(..., description="Dataset ID")
    view_type: str = Field("table", description="Type of view")
    data: List[Dict[str, Any]] = Field(..., description="Table data")
    total_rows: int = Field(..., description="Total rows in dataset")
    displayed_rows: int = Field(..., description="Number of rows displayed")
    columns: List[str] = Field(..., description="Column names")


class DataPreviewInput(NodeInput):
    """Input schema for DataPreview node."""

    dataset_id: str = Field(..., description="Dataset ID to preview")
    head_rows: int = Field(5, description="Number of first rows", ge=1, le=50)
    tail_rows: int = Field(5, description="Number of last rows", ge=1, le=50)

    # Optional metadata fields (passed from upload nodes)
    filename: Optional[str] = Field(None, description="Filename (metadata)")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[List[str]] = Field(None, description="Column names (metadata)")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Data types (metadata)")


class DataPreviewOutput(NodeOutput):
    """Output schema for DataPreview node."""

    dataset_id: str = Field(..., description="Dataset ID")
    view_type: str = Field("preview", description="Type of view")
    head_data: List[Dict[str, Any]] = Field(..., description="First rows")
    tail_data: List[Dict[str, Any]] = Field(..., description="Last rows")
    total_rows: int = Field(..., description="Total rows in dataset")


class StatisticsViewInput(NodeInput):
    """Input schema for StatisticsView node."""

    dataset_id: str = Field(..., description="Dataset ID to analyze")
    include_all: bool = Field(True, description="Include all columns")

    # Optional metadata fields (passed from upload nodes)
    filename: Optional[str] = Field(None, description="Filename (metadata)")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[List[str]] = Field(None, description="Column names (metadata)")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Data types (metadata)")


class StatisticsViewOutput(NodeOutput):
    """Output schema for StatisticsView node."""

    dataset_id: str = Field(..., description="Dataset ID")
    view_type: str = Field("statistics", description="Type of view")
    statistics: Dict[str, Dict[str, Any]] = Field(..., description="Statistical summary per column")


class ColumnInfoInput(NodeInput):
    """Input schema for ColumnInfo node."""

    dataset_id: str = Field(..., description="Dataset ID to analyze")
    show_dtypes: bool = Field(True, description="Show data types")
    show_missing: bool = Field(True, description="Show missing values count")
    show_unique: bool = Field(True, description="Show unique value counts")

    # Optional metadata fields (passed from upload nodes)
    filename: Optional[str] = Field(None, description="Filename (metadata)")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[List[str]] = Field(None, description="Column names (metadata)")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Data types (metadata)")


class ColumnInfoOutput(NodeOutput):
    """Output schema for ColumnInfo node."""

    dataset_id: str = Field(..., description="Dataset ID")
    view_type: str = Field("column_info", description="Type of view")
    column_info: List[Dict[str, Any]] = Field(..., description="Information per column")


class ChartViewInput(NodeInput):
    """Input schema for ChartView node."""

    dataset_id: str = Field(..., description="Dataset ID to visualize")
    chart_type: str = Field("bar", description="Type of chart")
    x_column: Optional[str] = Field(None, description="X-axis column")
    y_column: Optional[str] = Field(None, description="Y-axis column")
    y_columns: Optional[Union[List[str], str]] = Field(None, description="Multiple Y-axis columns")

    @field_validator("y_columns", mode="before")
    @classmethod
    def parse_y_columns(cls, v):
        """Convert string or comma-separated string to list if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            # If it contains comma, split by comma
            if "," in v:
                return [col.strip() for col in v.split(",") if col.strip()]
            # Otherwise, treat as single column
            return [v.strip()] if v.strip() else None
        # Already a list
        return v

    # Optional metadata fields (passed from upload nodes)
    filename: Optional[str] = Field(None, description="Filename (metadata)")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[List[str]] = Field(None, description="Column names (metadata)")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Data types (metadata)")


class ChartViewOutput(NodeOutput):
    """Output schema for ChartView node."""

    dataset_id: str = Field(..., description="Dataset ID")
    view_type: str = Field("chart", description="Type of view")
    chart_type: str = Field(..., description="Chart type")
    chart_data: Dict[str, Any] = Field(..., description="Chart configuration and data")


# Base View Node class
class BaseViewNode(BaseNode):
    """Base class for all view nodes."""

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
                return None

            # Load from S3 or local storage
            if dataset.storage_backend == "s3" and dataset.s3_key:
                # Download from S3
                file_content = await s3_service.download_file(dataset.s3_key)
                df = pd.read_csv(io.BytesIO(file_content))
            elif dataset.local_path:
                # Load from local path
                df = pd.read_csv(dataset.local_path)
            else:
                logger.error(f"No storage path found for dataset: {dataset_id}")
                return None

            db.close()
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None


class TableViewNode(BaseViewNode):
    """Table View Node - Display dataset in table format."""

    node_type = "table_view"

    def get_input_schema(self) -> Type[NodeInput]:
        return TableViewInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return TableViewOutput

    async def _execute(self, input_data: TableViewInput) -> TableViewOutput:
        """Return table view data."""
        logger.info(f"TableView node - dataset_id: {input_data.dataset_id}")

        # Load dataset
        df = await self._load_dataset(input_data.dataset_id)

        if df is None or df.empty:
            logger.warning(f"Dataset is None or empty for dataset_id: {input_data.dataset_id}")
            return TableViewOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                dataset_id=input_data.dataset_id,
                view_type="table",
                data=[],
                total_rows=0,
                displayed_rows=0,
                columns=[],
            )

        logger.info(f"Dataset loaded - shape: {df.shape}, columns: {df.columns.tolist()}")

        # Get column names - if columns_to_show is None or empty, use all columns
        columns = (
            df.columns.tolist()
            if input_data.columns_to_show is None or len(input_data.columns_to_show) == 0
            else input_data.columns_to_show
        )

        logger.info(f"Columns to display: {columns}")

        # Limit rows
        max_rows = min(input_data.max_rows, len(df))
        df_limited = df.head(max_rows)

        logger.info(f"Limited dataframe - shape: {df_limited.shape}")

        # Convert to list of dicts
        try:
            data = df_limited[columns].to_dict(orient="records")
            logger.info(f"Converted {len(data)} rows to dict format")
        except Exception as e:
            logger.error(f"Error converting dataframe to dict: {str(e)}")
            data = []

        return TableViewOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=input_data.dataset_id,
            view_type="table",
            data=data,
            total_rows=len(df),
            displayed_rows=len(df_limited),
            columns=columns,
        )


class DataPreviewNode(BaseViewNode):
    """Data Preview Node - Show head and tail of dataset."""

    node_type = "data_preview"

    def get_input_schema(self) -> Type[NodeInput]:
        return DataPreviewInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return DataPreviewOutput

    async def _execute(self, input_data: DataPreviewInput) -> DataPreviewOutput:
        """Return preview data."""
        logger.info(f"DataPreview node - dataset_id: {input_data.dataset_id}")

        df = await self._load_dataset(input_data.dataset_id)

        if df is None or df.empty:
            return DataPreviewOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                dataset_id=input_data.dataset_id,
                view_type="preview",
                head_data=[],
                tail_data=[],
                total_rows=0,
            )

        # Get head and tail
        head_data = df.head(input_data.head_rows).to_dict(orient="records")
        tail_data = df.tail(input_data.tail_rows).to_dict(orient="records")

        return DataPreviewOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=input_data.dataset_id,
            view_type="preview",
            head_data=head_data,
            tail_data=tail_data,
            total_rows=len(df),
        )


class StatisticsViewNode(BaseViewNode):
    """Statistics View Node - Show statistical summary."""

    node_type = "statistics_view"

    def get_input_schema(self) -> Type[NodeInput]:
        return StatisticsViewInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return StatisticsViewOutput

    async def _execute(self, input_data: StatisticsViewInput) -> StatisticsViewOutput:
        """Return statistics."""
        logger.info(f"StatisticsView node - dataset_id: {input_data.dataset_id}")

        df = await self._load_dataset(input_data.dataset_id)

        if df is None or df.empty:
            return StatisticsViewOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                dataset_id=input_data.dataset_id,
                view_type="statistics",
                statistics={},
            )

        # Calculate statistics for numeric columns
        stats_dict = {}
        numeric_df = df.select_dtypes(include=["number"])

        for col in numeric_df.columns:
            stats_dict[col] = {
                "mean": float(numeric_df[col].mean()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max()),
                "median": float(numeric_df[col].median()),
            }

        return StatisticsViewOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=input_data.dataset_id,
            view_type="statistics",
            statistics=stats_dict,
        )


class ColumnInfoNode(BaseViewNode):
    """Column Info Node - Show column information."""

    node_type = "column_info"

    def get_input_schema(self) -> Type[NodeInput]:
        return ColumnInfoInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return ColumnInfoOutput

    async def _execute(self, input_data: ColumnInfoInput) -> ColumnInfoOutput:
        """Return column information."""
        logger.info(f"ColumnInfo node - dataset_id: {input_data.dataset_id}")

        df = await self._load_dataset(input_data.dataset_id)

        if df is None or df.empty:
            return ColumnInfoOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                dataset_id=input_data.dataset_id,
                view_type="column_info",
                column_info=[],
            )

        # Gather column information
        column_info = []
        for col in df.columns:
            info = {
                "column": col,
                "dtype": str(df[col].dtype),
            }

            if input_data.show_missing:
                info["missing"] = int(df[col].isna().sum())

            if input_data.show_unique:
                info["unique"] = int(df[col].nunique())

            column_info.append(info)

        return ColumnInfoOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=input_data.dataset_id,
            view_type="column_info",
            column_info=column_info,
        )


class ChartViewNode(BaseViewNode):
    """Chart View Node - Visualize data with charts."""

    node_type = "chart_view"

    def get_input_schema(self) -> Type[NodeInput]:
        return ChartViewInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return ChartViewOutput

    async def _execute(self, input_data: ChartViewInput) -> ChartViewOutput:
        """Return chart data."""
        logger.info(
            f"ChartView node - dataset_id: {input_data.dataset_id}, type: {input_data.chart_type}"
        )

        # Load dataset
        df = await self._load_dataset(input_data.dataset_id)

        if df is None or df.empty:
            return ChartViewOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                dataset_id=input_data.dataset_id,
                view_type="chart",
                chart_type=input_data.chart_type,
                chart_data={"error": "Dataset is empty or not found"},
            )

        # Generate chart data based on chart type
        chart_data = {}

        try:
            if input_data.chart_type == "bar":
                chart_data = self._generate_bar_chart_data(df, input_data)
            elif input_data.chart_type == "line":
                chart_data = self._generate_line_chart_data(df, input_data)
            elif input_data.chart_type == "scatter":
                chart_data = self._generate_scatter_chart_data(df, input_data)
            elif input_data.chart_type == "histogram":
                chart_data = self._generate_histogram_data(df, input_data)
            elif input_data.chart_type == "pie":
                chart_data = self._generate_pie_chart_data(df, input_data)
            else:
                chart_data = {"error": f"Unsupported chart type: {input_data.chart_type}"}
        except Exception as e:
            logger.error(f"Error generating chart data: {str(e)}")
            chart_data = {"error": str(e)}

        return ChartViewOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=input_data.dataset_id,
            view_type="chart",
            chart_type=input_data.chart_type,
            chart_data=chart_data,
        )

    def _get_y_columns(self, input_data: ChartViewInput, df: pd.DataFrame) -> List[str]:
        """Get list of y columns from input data."""
        y_cols = []

        # Check if multiple columns specified
        if input_data.y_columns:
            # Already converted to list by validator, but handle both cases
            if isinstance(input_data.y_columns, list):
                y_cols = input_data.y_columns
            else:
                # Fallback for string format
                y_cols = [col.strip() for col in input_data.y_columns.split(",") if col.strip()]
        elif input_data.y_column:
            y_cols = [input_data.y_column]
        else:
            # Default: use numeric columns (excluding first which is usually x)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_cols) > 0:
                y_cols = numeric_cols[:3]  # Limit to first 3 numeric columns

        return y_cols

    def _generate_bar_chart_data(
        self, df: pd.DataFrame, input_data: ChartViewInput
    ) -> Dict[str, Any]:
        """Generate bar chart data."""
        x_col = input_data.x_column
        y_cols = self._get_y_columns(input_data, df)

        # If no x column specified, use first column
        if not x_col and len(df.columns) > 0:
            x_col = df.columns[0]

        if not x_col:
            return {"error": "Please specify x column"}

        if not y_cols:
            return {"error": "Please specify y column(s)"}

        # Limit to first 50 rows for performance
        df_limited = df.head(50)

        datasets = []
        for col in y_cols:
            if col in df_limited.columns:
                datasets.append(
                    {
                        "label": col,
                        "data": df_limited[col].tolist(),
                    }
                )

        return {
            "labels": df_limited[x_col].astype(str).tolist(),
            "datasets": datasets,
            "x_column": x_col,
            "y_columns": y_cols,
        }

    def _generate_line_chart_data(
        self, df: pd.DataFrame, input_data: ChartViewInput
    ) -> Dict[str, Any]:
        """Generate line chart data."""
        x_col = input_data.x_column
        y_cols = self._get_y_columns(input_data, df)

        # If no x column specified, use first column
        if not x_col and len(df.columns) > 0:
            x_col = df.columns[0]

        if not x_col:
            return {"error": "Please specify x column"}

        if not y_cols:
            return {"error": "Please specify y column(s)"}

        # Limit to first 100 rows for performance
        df_limited = df.head(100)

        datasets = []
        for col in y_cols:
            if col in df_limited.columns:
                datasets.append(
                    {
                        "label": col,
                        "data": df_limited[col].tolist(),
                    }
                )

        return {
            "labels": df_limited[x_col].astype(str).tolist(),
            "datasets": datasets,
            "x_column": x_col,
            "y_columns": y_cols,
        }

    def _generate_scatter_chart_data(
        self, df: pd.DataFrame, input_data: ChartViewInput
    ) -> Dict[str, Any]:
        """Generate scatter plot data."""
        x_col = input_data.x_column
        y_cols = self._get_y_columns(input_data, df)

        # If no x column specified, use first numeric column
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not x_col and len(numeric_cols) > 0:
            x_col = numeric_cols[0]

        if not x_col:
            return {"error": "Please specify x column"}

        if not y_cols:
            return {"error": "Please specify y column(s)"}

        # Limit to first 200 rows for performance
        df_limited = df.head(200)

        datasets = []
        for col in y_cols:
            if col in df_limited.columns:
                # Create scatter data points for this column
                data_points = [
                    {"x": float(x), "y": float(y)}
                    for x, y in zip(df_limited[x_col], df_limited[col])
                    if pd.notna(x) and pd.notna(y)
                ]
                datasets.append(
                    {
                        "label": f"{x_col} vs {col}",
                        "data": data_points,
                    }
                )

        return {
            "datasets": datasets,
            "x_column": x_col,
            "y_columns": y_cols,
        }

    def _generate_histogram_data(
        self, df: pd.DataFrame, input_data: ChartViewInput
    ) -> Dict[str, Any]:
        """Generate histogram data."""
        # Use y_columns or x_column for histogram
        y_cols = self._get_y_columns(input_data, df)

        if input_data.x_column and not y_cols:
            cols = [input_data.x_column]
        elif y_cols:
            cols = y_cols
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_cols) > 0:
                cols = [numeric_cols[0]]
            else:
                cols = []

        if not cols:
            return {"error": "Please specify a numeric column"}

        # For histogram, only use first column to avoid confusion
        col = cols[0]

        # Create histogram bins
        data = df[col].dropna()
        if len(data) == 0:
            return {"error": "No data available"}

        # Use pandas to create histogram
        counts, bins = pd.cut(data, bins=20, retbins=True, duplicates="drop")
        hist_data = counts.value_counts().sort_index()

        # Create labels from bins
        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]

        return {
            "labels": labels[: len(hist_data)],
            "datasets": [
                {
                    "label": col,
                    "data": hist_data.tolist(),
                }
            ],
            "x_column": "Bins",
            "y_columns": [col],
        }

    def _generate_pie_chart_data(
        self, df: pd.DataFrame, input_data: ChartViewInput
    ) -> Dict[str, Any]:
        """Generate pie chart data."""
        # Use x_column for pie chart categories
        col = input_data.x_column

        if not col and len(df.columns) > 0:
            # Use first categorical column
            col = df.columns[0]

        if not col:
            return {"error": "Please specify a column for pie chart"}

        # Count values and get top 10 categories
        value_counts = df[col].value_counts().head(10)
        total = value_counts.sum()

        # Calculate percentages
        percentages = [(count / total * 100) for count in value_counts]

        # Create enhanced labels with counts and percentages
        enhanced_labels = [
            f"{label} ({count} - {pct:.1f}%)"
            for label, count, pct in zip(value_counts.index.astype(str), value_counts, percentages)
        ]

        return {
            "labels": enhanced_labels,
            "datasets": [
                {
                    "label": col,
                    "data": value_counts.tolist(),
                }
            ],
            "column_name": col,
            "total_count": int(total),
            "categories_shown": len(value_counts),
        }
