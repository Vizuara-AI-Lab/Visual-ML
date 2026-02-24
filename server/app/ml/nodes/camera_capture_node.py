"""
Camera Capture Node — reads a pre-built camera dataset (created by the frontend
camera-capture UI) and exposes it as an image dataset for the rest of the pipeline.

The frontend collects pixel arrays per class, POSTs them to /ml/camera/dataset,
and stores the resulting dataset_id in this node's config.  On execution the node
simply validates the dataset exists and returns the same rich metadata as
ImageDatasetNode so downstream nodes (preprocessing, split, CNN) work identically.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path

from app.ml.nodes.base import (
    BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory,
)
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id


class CameraCaptureInput(NodeInput):
    """Input schema for Camera Capture node."""
    dataset_id: str = Field(..., description="Pre-built camera dataset ID (from /ml/camera/dataset)")
    class_names: Optional[List[str]] = Field(None, description="Class names for the dataset")
    target_size: str = Field("28x28", description="WxH pixel size used when capturing")


class CameraCaptureOutput(NodeOutput):
    """Output schema for Camera Capture node — mirrors ImageDatasetOutput."""
    dataset_id: str = Field(..., description="Dataset identifier")
    filename: str = Field(..., description="CSV filename")
    file_path: str = Field(..., description="Path to stored CSV file")
    storage_backend: str = Field(default="local")
    n_rows: int = Field(..., description="Number of images")
    n_columns: int = Field(..., description="Number of columns (pixels + label)")
    columns: list = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types")
    memory_usage_mb: float = Field(...)

    # Image metadata (compatible with downstream image nodes)
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    n_channels: int = Field(default=1, description="Number of channels (1 = grayscale)")
    n_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="Class names")
    dataset_label: str = Field(default="Camera Capture Dataset")

    # Explorer data (optional — reuse structure from image_dataset)
    sample_images: Optional[List[Dict[str, Any]]] = Field(None)
    class_distribution: Optional[Dict[str, Any]] = Field(None)


class CameraCaptureNode(BaseNode):
    """
    Camera Capture Node.

    This is a *source* node for the image pipeline.  The dataset CSV was already
    created by the /ml/camera/dataset REST endpoint when the user clicked
    "Build Dataset" in the UI.  This node validates the CSV, reads metadata,
    and produces an output compatible with Image Preprocessing / Split / CNN nodes.
    """

    node_type = "camera_capture"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.DATA_SOURCE,
            primary_output_field="dataset_id",
            output_fields={
                "dataset_id": "Unique identifier for the camera dataset",
                "n_rows": "Number of images",
                "n_columns": "Number of columns",
                "columns": "Column names",
                "image_width": "Image width",
                "image_height": "Image height",
                "n_channels": "Number of channels",
                "n_classes": "Number of classes",
                "class_names": "Class label names",
            },
            requires_input=False,
            can_branch=True,
            produces_dataset=True,
            max_inputs=0,
            allowed_source_categories=[],
        )

    def get_input_schema(self):
        return CameraCaptureInput

    def get_output_schema(self):
        return CameraCaptureOutput

    async def _execute(self, input_data: CameraCaptureInput) -> CameraCaptureOutput:
        dataset_id = input_data.dataset_id
        upload_dir = Path(settings.UPLOAD_DIR)
        file_path = upload_dir / f"{dataset_id}.csv"

        if not file_path.exists():
            raise ValueError(
                f"Camera dataset '{dataset_id}' not found at {file_path}. "
                "Please re-capture images and rebuild the dataset."
            )

        logger.info(f"Camera Capture node: loading dataset {dataset_id}")
        df = pd.read_csv(file_path)

        # Parse image dimensions from target_size ("WxH")
        size_parts = (input_data.target_size or "28x28").lower().split("x")
        width = int(size_parts[0]) if len(size_parts) > 0 else 28
        height = int(size_parts[1]) if len(size_parts) > 1 else 28

        # Derive class info from the label column
        label_col = "label"
        if label_col not in df.columns:
            raise ValueError(f"Dataset CSV is missing a 'label' column. Columns: {df.columns.tolist()}")

        unique_labels = sorted(df[label_col].unique())
        n_classes = len(unique_labels)

        # Class names: prefer input_data.class_names, fall back to label values
        if input_data.class_names and len(input_data.class_names) == n_classes:
            class_names = input_data.class_names
        else:
            class_names = [str(lbl) for lbl in unique_labels]

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Build class distribution
        total = len(df)
        classes_dist = []
        for idx, cls_name in enumerate(class_names):
            label_val = unique_labels[idx] if idx < len(unique_labels) else idx
            count = int((df[label_col] == label_val).sum())
            pct = round(count / total * 100, 1) if total > 0 else 0
            classes_dist.append({"name": cls_name, "index": idx, "count": count, "percentage": pct})

        class_distribution = {
            "total_samples": total,
            "n_classes": n_classes,
            "classes": classes_dist,
            "is_balanced": True,
            "balance_message": "Camera-captured dataset",
        }

        # Build sample images for the explorer gallery
        pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
        X = df[pixel_cols].values
        y = df[label_col].values
        sample_images = self._build_sample_images(X, y, class_names, width, height, unique_labels)

        logger.info(
            f"Camera dataset loaded: {len(df)} images, {width}×{height}, "
            f"{n_classes} classes, {len(pixel_cols)} pixel features"
        )

        return CameraCaptureOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=dataset_id,
            filename=f"{dataset_id}.csv",
            file_path=str(file_path),
            storage_backend="local",
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=df.columns.tolist(),
            dtypes=dtypes,
            memory_usage_mb=round(memory_mb, 2),
            image_width=width,
            image_height=height,
            n_channels=1,
            n_classes=n_classes,
            class_names=class_names,
            dataset_label="Camera Capture Dataset",
            sample_images=sample_images,
            class_distribution=class_distribution,
        )

    def _build_sample_images(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: list,
        width: int,
        height: int,
        unique_labels: list,
        n_samples: int = 6,
    ) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(42)
        samples = []
        for cls_idx, cls_name in enumerate(class_names):
            label_val = unique_labels[cls_idx] if cls_idx < len(unique_labels) else cls_idx
            mask = y == label_val
            cls_images = X[mask]
            if len(cls_images) == 0:
                continue
            chosen = rng.choice(len(cls_images), size=min(n_samples, len(cls_images)), replace=False)
            class_samples = [
                {
                    "pixels": cls_images[i].tolist(),
                    "min": float(cls_images[i].min()),
                    "max": float(cls_images[i].max()),
                    "mean": round(float(cls_images[i].mean()), 2),
                }
                for i in chosen
            ]
            samples.append({
                "class_index": cls_idx,
                "class_name": str(cls_name),
                "count": int(mask.sum()),
                "samples": class_samples,
                "width": width,
                "height": height,
                "channels": 1,
            })
        return samples
