"""
Image Dataset Node — Loads built-in image datasets for the image classification pipeline.
Stores images as flattened pixel DataFrames (pixel_0...pixel_N + label columns).
Returns rich explorer data for interactive learning.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pydantic import Field
from pathlib import Path
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id


# ─── Built-in Image Dataset Loaders ──────────────────────────────

def _load_digits_8x8() -> tuple:
    """sklearn digits: 8×8 grayscale, 10 classes (0-9), 1797 samples."""
    from sklearn.datasets import load_digits
    data = load_digits()
    return data.data, data.target, data.target_names.tolist(), 8, 8, 1, "Handwritten Digits (8×8)"


def _load_fashion_mnist_small() -> tuple:
    """Fashion-MNIST subset: 28×28 grayscale, 10 classes, ~2000 samples."""
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto")
        # Take a small stratified subset for performance
        rng = np.random.default_rng(42)
        indices = []
        for label in np.unique(y):
            label_indices = np.where(y == label)[0]
            chosen = rng.choice(label_indices, size=min(200, len(label_indices)), replace=False)
            indices.extend(chosen)
        indices = sorted(indices)
        X = X[indices].astype(float)
        y = y[indices].astype(int)
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        return X, y, class_names, 28, 28, 1, "Fashion-MNIST (28×28)"
    except Exception as e:
        logger.warning(f"Fashion-MNIST fetch failed ({e}), falling back to Digits")
        return _load_digits_8x8()


def _load_mnist_digits_small() -> tuple:
    """MNIST handwritten digits subset: 28×28 grayscale, 10 classes, ~2000 samples."""
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
        rng = np.random.default_rng(42)
        indices = []
        for label in np.unique(y):
            label_indices = np.where(y == label)[0]
            chosen = rng.choice(label_indices, size=min(200, len(label_indices)), replace=False)
            indices.extend(chosen)
        indices = sorted(indices)
        X = X[indices].astype(float)
        y = y[indices].astype(int)
        class_names = [str(i) for i in range(10)]
        return X, y, class_names, 28, 28, 1, "MNIST Digits (28×28)"
    except Exception as e:
        logger.warning(f"MNIST fetch failed ({e}), falling back to Digits")
        return _load_digits_8x8()


def _load_synthetic_shapes() -> tuple:
    """Synthetic shapes dataset: 16×16 grayscale, 4 classes, 800 samples."""
    rng = np.random.default_rng(42)
    size = 16
    n_per_class = 200
    images = []
    labels = []
    class_names = ["Circle", "Square", "Triangle", "Cross"]

    for _ in range(n_per_class):
        # Circle
        img = np.zeros((size, size))
        cx, cy = rng.integers(4, 12, 2)
        r = rng.integers(2, 5)
        for y in range(size):
            for x in range(size):
                if (x - cx)**2 + (y - cy)**2 <= r**2:
                    img[y, x] = rng.uniform(180, 255)
        img += rng.normal(0, 10, (size, size))
        images.append(img.clip(0, 255).flatten())
        labels.append(0)

        # Square
        img = np.zeros((size, size))
        x1, y1 = rng.integers(1, 8, 2)
        w = rng.integers(4, 8)
        img[y1:y1+w, x1:x1+w] = rng.uniform(180, 255)
        img += rng.normal(0, 10, (size, size))
        images.append(img.clip(0, 255).flatten())
        labels.append(1)

        # Triangle
        img = np.zeros((size, size))
        cx = rng.integers(4, 12)
        base_y = rng.integers(10, 14)
        apex_y = rng.integers(2, 6)
        for y in range(apex_y, base_y + 1):
            t = (y - apex_y) / max(1, base_y - apex_y)
            half_w = int(t * rng.integers(3, 6))
            x_start = max(0, cx - half_w)
            x_end = min(size, cx + half_w + 1)
            img[y, x_start:x_end] = rng.uniform(180, 255)
        img += rng.normal(0, 10, (size, size))
        images.append(img.clip(0, 255).flatten())
        labels.append(2)

        # Cross
        img = np.zeros((size, size))
        cx, cy = rng.integers(4, 12, 2)
        arm = rng.integers(2, 5)
        thickness = 1
        img[max(0, cy-arm):min(size, cy+arm+1), max(0, cx-thickness):min(size, cx+thickness+1)] = rng.uniform(180, 255)
        img[max(0, cy-thickness):min(size, cy+thickness+1), max(0, cx-arm):min(size, cx+arm+1)] = rng.uniform(180, 255)
        img += rng.normal(0, 10, (size, size))
        images.append(img.clip(0, 255).flatten())
        labels.append(3)

    X = np.array(images)
    y = np.array(labels)
    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm], class_names, 16, 16, 1, "Synthetic Shapes (16×16)"


IMAGE_DATASETS = {
    "digits_8x8":       ("Digits (8×8)", _load_digits_8x8),
    "mnist_28x28":      ("MNIST (28×28)", _load_mnist_digits_small),
    "fashion_mnist":    ("Fashion-MNIST", _load_fashion_mnist_small),
    "shapes":           ("Shapes (16×16)", _load_synthetic_shapes),
}

# Backward-compatible aliases used by older clients/pipelines.
IMAGE_DATASET_ALIASES = {
    "digits": "digits_8x8",
    "mnist": "mnist_28x28",
}


# ─── Node Implementation ──────────────────────────────────────────

class ImageDatasetInput(NodeInput):
    """Input schema for Image Dataset node."""
    source: str = Field("builtin", description="'builtin' or 'camera'")
    dataset_name: str = Field(
        "digits_8x8", description="Built-in image dataset to load"
    )
    # Camera-source fields (populated by the frontend camera capture UI)
    dataset_id: Optional[str] = Field(None, description="Pre-built camera dataset ID")
    target_size: str = Field("28x28", description="WxH used when capturing")
    class_names: Optional[str] = Field(None, description="Comma-separated class names")


class ImageDatasetOutput(NodeOutput):
    """Output schema for Image Dataset node."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Generated filename")
    file_path: str = Field(..., description="Path to stored CSV file")
    storage_backend: str = Field(default="local", description="Storage backend")
    n_rows: int = Field(..., description="Number of images")
    n_columns: int = Field(..., description="Number of columns (pixels + label)")
    columns: list = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")

    # Image-specific metadata
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    n_channels: int = Field(..., description="Number of channels (1=grayscale)")
    n_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="Class label names")
    dataset_label: str = Field(..., description="Human-readable dataset name")

    # Explorer activity data
    sample_images: Optional[List[Dict[str, Any]]] = Field(
        None, description="Sample images per class for gallery view"
    )
    class_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Class distribution chart data"
    )
    pixel_statistics: Optional[Dict[str, Any]] = Field(
        None, description="Pixel value statistics for histogram"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions"
    )


class ImageDatasetNode(BaseNode):
    """
    Image Dataset Node — Load built-in image datasets for the image classification pipeline.

    Loads the chosen image dataset, saves it as a CSV in the uploads directory
    (flattened pixel columns + label), and returns rich explorer data for
    interactive learning.
    """

    node_type = "image_dataset"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.DATA_SOURCE,
            primary_output_field="dataset_id",
            output_fields={
                "dataset_id": "Unique identifier for the image dataset",
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
        return ImageDatasetInput

    def get_output_schema(self):
        return ImageDatasetOutput

    async def _execute(self, input_data: ImageDatasetInput) -> ImageDatasetOutput:
        if input_data.source == "camera":
            return await self._execute_camera(input_data)
        return await self._execute_builtin(input_data)

    async def _execute_camera(self, input_data: ImageDatasetInput) -> ImageDatasetOutput:
        """Load a pre-built camera dataset (created by /ml/camera/dataset)."""
        dataset_id = input_data.dataset_id
        if not dataset_id:
            raise ValueError(
                "Camera source requires a dataset_id. "
                "Please capture images first using the camera panel."
            )

        upload_dir = Path(settings.UPLOAD_DIR)
        file_path = upload_dir / f"{dataset_id}.csv"
        if not file_path.exists():
            raise ValueError(
                f"Camera dataset '{dataset_id}' not found. "
                "Please re-capture images and rebuild the dataset."
            )

        logger.info(f"Image Dataset node (camera): loading {dataset_id}")
        df = pd.read_csv(file_path)

        size_parts = (input_data.target_size or "28x28").lower().split("x")
        width = int(size_parts[0]) if len(size_parts) > 0 else 28
        height = int(size_parts[1]) if len(size_parts) > 1 else 28

        label_col = "label"
        if label_col not in df.columns:
            raise ValueError(f"Camera dataset CSV is missing a 'label' column.")

        unique_labels = sorted(df[label_col].unique())
        n_classes = len(unique_labels)
        if input_data.class_names:
            class_names = [c.strip() for c in input_data.class_names.split(",") if c.strip()]
            if len(class_names) != n_classes:
                class_names = [str(lbl) for lbl in unique_labels]
        else:
            class_names = [str(lbl) for lbl in unique_labels]

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
        X = df[pixel_cols].values
        y_arr = df[label_col].values
        # remap labels to 0-based indices
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        y = np.array([label_to_idx[lbl] for lbl in y_arr])

        sample_images = self._generate_sample_images(X, y, class_names, width, height, 1)
        class_distribution = self._generate_class_distribution(y, class_names)

        return ImageDatasetOutput(
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

    async def _execute_builtin(self, input_data: ImageDatasetInput) -> ImageDatasetOutput:
        """Load a built-in image dataset (MNIST, Digits, etc.)."""
        requested_name = input_data.dataset_name
        dataset_name = IMAGE_DATASET_ALIASES.get(requested_name, requested_name)

        if dataset_name not in IMAGE_DATASETS:
            available = ", ".join(sorted(IMAGE_DATASETS.keys()))
            raise ValueError(
                f"Unknown image dataset: '{requested_name}'. Available: {available}"
            )

        label, loader_fn = IMAGE_DATASETS[dataset_name]
        logger.info(f"Loading image dataset: {label} ({dataset_name})")

        X, y, class_names, width, height, channels, dataset_label = loader_fn()
        n_pixels = width * height * channels

        # Build DataFrame: pixel_0 ... pixel_N + label
        pixel_cols = [f"pixel_{i}" for i in range(n_pixels)]
        df = pd.DataFrame(X, columns=pixel_cols)
        df["label"] = y

        # Save to uploads directory
        dataset_id = generate_id("img")
        filename = f"{dataset_id}.csv"
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        df.to_csv(file_path, index=False)

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        logger.info(
            f"Image dataset loaded: {dataset_label} — "
            f"{len(df)} images, {width}×{height}×{channels}, "
            f"{len(class_names)} classes, saved to {file_path}"
        )

        # --- Generate explorer activity data ---
        sample_images = self._generate_sample_images(X, y, class_names, width, height, channels)
        class_distribution = self._generate_class_distribution(y, class_names)
        pixel_statistics = self._generate_pixel_statistics(X)
        quiz_questions = self._generate_quiz(
            dataset_label, len(df), width, height, channels,
            len(class_names), class_names, n_pixels
        )

        return ImageDatasetOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=dataset_id,
            filename=filename,
            file_path=str(file_path),
            storage_backend="local",
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=df.columns.tolist(),
            dtypes=dtypes,
            memory_usage_mb=round(memory_mb, 2),
            image_width=width,
            image_height=height,
            n_channels=channels,
            n_classes=len(class_names),
            class_names=[str(c) for c in class_names],
            dataset_label=dataset_label,
            sample_images=sample_images,
            class_distribution=class_distribution,
            pixel_statistics=pixel_statistics,
            quiz_questions=quiz_questions,
        )

    # --- Explorer Data Generators ---

    def _generate_sample_images(
        self, X: np.ndarray, y: np.ndarray,
        class_names: list, width: int, height: int, channels: int,
        n_samples: int = 8
    ) -> List[Dict[str, Any]]:
        """Generate sample image pixel data per class for the gallery."""
        rng = np.random.default_rng(42)
        samples = []

        for cls_idx, cls_name in enumerate(class_names):
            cls_mask = y == cls_idx
            cls_images = X[cls_mask]
            if len(cls_images) == 0:
                continue

            chosen = rng.choice(
                len(cls_images),
                size=min(n_samples, len(cls_images)),
                replace=False
            )

            class_samples = []
            for idx in chosen:
                pixels = cls_images[idx].tolist()
                # Compute per-image stats
                img = cls_images[idx]
                class_samples.append({
                    "pixels": pixels,
                    "min": float(np.min(img)),
                    "max": float(np.max(img)),
                    "mean": round(float(np.mean(img)), 2),
                })

            samples.append({
                "class_index": int(cls_idx),
                "class_name": str(cls_name),
                "count": int(cls_mask.sum()),
                "samples": class_samples,
                "width": width,
                "height": height,
                "channels": channels,
            })

        return samples

    def _generate_class_distribution(
        self, y: np.ndarray, class_names: list
    ) -> Dict[str, Any]:
        """Generate class distribution data for bar chart."""
        total = len(y)
        classes = []
        counts = []

        for cls_idx, cls_name in enumerate(class_names):
            count = int(np.sum(y == cls_idx))
            counts.append(count)
            pct = round(count / total * 100, 1) if total > 0 else 0
            classes.append({
                "name": str(cls_name),
                "index": int(cls_idx),
                "count": count,
                "percentage": pct,
            })

        max_count = max(counts) if counts else 1
        min_count = min(counts) if counts else 1
        imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else float("inf")

        return {
            "total_samples": total,
            "n_classes": len(class_names),
            "classes": classes,
            "is_balanced": imbalance_ratio <= 1.5,
            "imbalance_ratio": imbalance_ratio,
            "balance_message": (
                "Classes are roughly balanced — great for training!"
                if imbalance_ratio <= 1.5
                else f"Classes are imbalanced ({imbalance_ratio}:1). Consider augmentation for minority classes."
            ),
        }

    def _generate_pixel_statistics(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate pixel value statistics for histograms."""
        flat = X.flatten()
        # Compute histogram bins
        hist_counts, hist_edges = np.histogram(flat, bins=32, range=(0, max(flat.max(), 255)))

        return {
            "global_min": float(np.min(flat)),
            "global_max": float(np.max(flat)),
            "global_mean": round(float(np.mean(flat)), 2),
            "global_std": round(float(np.std(flat)), 2),
            "histogram": {
                "counts": hist_counts.tolist(),
                "bin_edges": [round(float(e), 2) for e in hist_edges.tolist()],
            },
            "zero_pixel_pct": round(float(np.sum(flat == 0) / len(flat) * 100), 1),
            "nonzero_mean": round(float(np.mean(flat[flat > 0])), 2) if np.any(flat > 0) else 0,
        }

    def _generate_quiz(
        self, dataset_label: str, n_images: int,
        width: int, height: int, channels: int,
        n_classes: int, class_names: list, n_pixels: int
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about the image dataset."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: Image representation
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"Each image in this dataset is {width}×{height} pixels. How many numbers are needed to represent one image?",
            "options": [
                f"{n_pixels} numbers (one per pixel)",
                f"{width + height} numbers",
                f"{width} numbers",
                f"Just 1 number",
            ],
            "correct_answer": 0,
            "explanation": f"A {width}×{height} image has {width}×{height} = {n_pixels} pixels. Each pixel is a number (0-255 for grayscale), so we need {n_pixels} numbers to represent the image. This is why we 'flatten' the image into a row of {n_pixels} values.",
            "difficulty": "easy",
        })

        # Q2: What does pixel value 0 mean?
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "In a grayscale image, what does a pixel value of 0 represent?",
            "options": [
                "Black (no brightness)",
                "White (full brightness)",
                "The color red",
                "A missing value",
            ],
            "correct_answer": 0,
            "explanation": "In grayscale images, 0 = black (no brightness) and 255 = white (full brightness). Values in between represent shades of gray. This is why dark backgrounds show up as many 0-valued pixels in the dataset.",
            "difficulty": "easy",
        })

        # Q3: Dataset size
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This dataset has {n_images} images across {n_classes} classes. On average, how many images per class?",
            "options": [
                f"About {n_images // n_classes} images per class",
                f"{n_images} images per class",
                f"{n_classes} images per class",
                f"1 image per class",
            ],
            "correct_answer": 0,
            "explanation": f"With {n_images} total images and {n_classes} classes, the average is {n_images} ÷ {n_classes} = ~{n_images // n_classes} images per class. Having enough samples per class is crucial for the model to learn each class's visual patterns.",
            "difficulty": "easy",
        })

        # Q4: Why flatten?
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"Why do we 'flatten' a {width}×{height} image into a single row of {n_pixels} values?",
            "options": [
                "Traditional ML models (like MLP) expect a flat feature vector as input",
                "It makes the image look better",
                "It reduces the file size",
                "Images can't be stored as 2D arrays",
            ],
            "correct_answer": 0,
            "explanation": f"Traditional ML models like MLP expect each sample as a 1D feature vector. So we flatten the {width}×{height} grid into a row of {n_pixels} numbers. Each pixel becomes a 'feature'. CNNs can work with 2D images directly, but MLPs cannot.",
            "difficulty": "medium",
        })

        # Q5: Class names
        sample_classes = class_names[:3]
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This is a {n_classes}-class classification problem with classes like {', '.join(str(c) for c in sample_classes)}. What type of ML task is this?",
            "options": [
                "Multi-class image classification",
                "Binary classification",
                "Regression (predicting a continuous value)",
                "Clustering (unsupervised grouping)",
            ],
            "correct_answer": 0,
            "explanation": f"With {n_classes} distinct classes, this is multi-class classification. The model must learn to assign each image to one of {n_classes} categories. This is different from binary classification (2 classes) or regression (continuous output).",
            "difficulty": "medium",
        })

        # Q6: Data quality
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why is class balance important in image classification?",
            "options": [
                "Imbalanced classes cause the model to favor the majority class, reducing accuracy on minority classes",
                "Class balance doesn't matter in image classification",
                "Balanced classes make images look better",
                "The model trains faster with imbalanced classes",
            ],
            "correct_answer": 0,
            "explanation": "If one class has 1000 images and another has 10, the model can achieve 99% accuracy by always predicting the majority class. This is misleading! Solutions include: data augmentation, oversampling minority classes, or using weighted loss functions.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
