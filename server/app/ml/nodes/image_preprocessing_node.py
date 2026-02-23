"""
Image Preprocessing Node — Resize, normalize, and convert image data.
Operates on flattened pixel DataFrames produced by ImageDatasetNode.
Returns rich explorer data: before/after views, pixel histograms, normalization comparison.
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


class ImagePreprocessingInput(NodeInput):
    """Input schema for Image Preprocessing node."""
    dataset_id: str = Field(..., description="Dataset ID from image dataset node")

    # Preprocessing options
    normalize_method: str = Field(
        "min_max", description="Normalization: min_max (0-1), z_score (mean=0,std=1), divide_255, none"
    )
    target_size: Optional[str] = Field(
        None, description="Resize to WxH (e.g. '16x16'). Leave empty to keep original size."
    )
    invert: bool = Field(
        False, description="Invert pixel values (255-x). Useful for white-on-black datasets."
    )

    # Pass-through image metadata
    image_width: Optional[int] = Field(None, description="Original image width")
    image_height: Optional[int] = Field(None, description="Original image height")
    n_channels: Optional[int] = Field(None, description="Number of channels")
    n_classes: Optional[int] = Field(None, description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Class names")


class ImagePreprocessingOutput(NodeOutput):
    """Output schema for Image Preprocessing node."""
    preprocessed_dataset_id: str = Field(..., description="ID of preprocessed dataset")
    preprocessed_path: str = Field(..., description="Path to preprocessed CSV")
    n_rows: int = Field(..., description="Number of images")
    n_columns: int = Field(..., description="Number of columns")
    columns: list = Field(..., description="Column names")

    # Image metadata (pass-through or updated)
    image_width: int = Field(..., description="Image width after preprocessing")
    image_height: int = Field(..., description="Image height after preprocessing")
    n_channels: int = Field(1, description="Number of channels")
    n_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="Class names")

    # Preprocessing summary
    normalize_method: str = Field(..., description="Normalization method used")
    pixel_range_before: List[float] = Field(..., description="[min, max] before")
    pixel_range_after: List[float] = Field(..., description="[min, max] after")

    # Explorer activity data
    before_after_samples: Optional[List[Dict[str, Any]]] = Field(
        None, description="Before/after image comparison data"
    )
    normalization_comparison: Optional[Dict[str, Any]] = Field(
        None, description="Comparison of different normalization methods"
    )
    pixel_histograms: Optional[Dict[str, Any]] = Field(
        None, description="Before and after pixel histograms"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions"
    )


class ImagePreprocessingNode(BaseNode):
    """
    Image Preprocessing Node — Resize, normalize, and transform image data.

    Processes flattened pixel DataFrames and returns preprocessed data
    with rich explorer activity data for interactive learning.
    """

    node_type = "image_preprocessing"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.PREPROCESSING,
            primary_output_field="preprocessed_dataset_id",
            output_fields={
                "preprocessed_dataset_id": "Preprocessed image dataset ID",
                "image_width": "Image width",
                "image_height": "Image height",
                "n_classes": "Number of classes",
                "class_names": "Class label names",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            max_inputs=1,
            allowed_source_categories=[NodeCategory.DATA_SOURCE],
        )

    def get_input_schema(self):
        return ImagePreprocessingInput

    def get_output_schema(self):
        return ImagePreprocessingOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from uploads folder."""
        upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
        if upload_path.exists():
            return pd.read_csv(upload_path)
        logger.error(f"Dataset not found: {upload_path}")
        return None

    def _parse_target_size(self, target_size_str: Optional[str]) -> Optional[tuple]:
        """Parse 'WxH' string into (width, height) tuple."""
        if not target_size_str or not target_size_str.strip():
            return None
        try:
            parts = target_size_str.lower().split("x")
            return (int(parts[0].strip()), int(parts[1].strip()))
        except (ValueError, IndexError):
            logger.warning(f"Invalid target_size: '{target_size_str}', skipping resize")
            return None

    def _resize_images(
        self, X: np.ndarray, old_w: int, old_h: int,
        new_w: int, new_h: int, channels: int
    ) -> np.ndarray:
        """Resize images using simple nearest-neighbor interpolation."""
        n = len(X)
        resized = np.zeros((n, new_h * new_w * channels))

        for i in range(n):
            img = X[i].reshape(old_h, old_w)
            # Nearest-neighbor resize
            row_indices = (np.arange(new_h) * old_h / new_h).astype(int).clip(0, old_h - 1)
            col_indices = (np.arange(new_w) * old_w / new_w).astype(int).clip(0, old_w - 1)
            new_img = img[np.ix_(row_indices, col_indices)]
            resized[i] = new_img.flatten()

        return resized

    def _normalize(self, X: np.ndarray, method: str) -> np.ndarray:
        """Apply normalization to pixel values."""
        X = X.astype(np.float64)
        if method == "min_max":
            x_min, x_max = X.min(), X.max()
            if x_max > x_min:
                X = (X - x_min) / (x_max - x_min)
        elif method == "z_score":
            mean, std = X.mean(), X.std()
            if std > 0:
                X = (X - mean) / std
        elif method == "divide_255":
            X = X / 255.0
        # else: "none" — leave as-is
        return X

    async def _execute(self, input_data: ImagePreprocessingInput) -> ImagePreprocessingOutput:
        df = await self._load_dataset(input_data.dataset_id)
        if df is None or df.empty:
            raise ValueError(f"Dataset {input_data.dataset_id} not found or empty")

        # Separate pixels and labels
        label_col = "label" if "label" in df.columns else df.columns[-1]
        pixel_cols = [c for c in df.columns if c != label_col]
        X_before = df[pixel_cols].values.astype(np.float64)
        y = df[label_col].values

        # Determine image dimensions
        width = input_data.image_width or int(np.sqrt(len(pixel_cols)))
        height = input_data.image_height or int(np.sqrt(len(pixel_cols)))
        channels = input_data.n_channels or 1
        n_classes = input_data.n_classes or len(np.unique(y))
        class_names = input_data.class_names or [str(i) for i in range(n_classes)]

        logger.info(f"Preprocessing {len(df)} images ({width}×{height}×{channels})")

        # Store before stats
        before_min = float(X_before.min())
        before_max = float(X_before.max())

        X = X_before.copy()

        # 1. Invert
        if input_data.invert:
            actual_max = X.max()
            X = actual_max - X
            logger.info("Applied pixel inversion")

        # 2. Resize
        new_width, new_height = width, height
        target_size = self._parse_target_size(input_data.target_size)
        if target_size and (target_size[0] != width or target_size[1] != height):
            new_width, new_height = target_size
            X = self._resize_images(X, width, height, new_width, new_height, channels)
            logger.info(f"Resized: {width}×{height} → {new_width}×{new_height}")

        # 3. Normalize
        X = self._normalize(X, input_data.normalize_method)

        after_min = float(X.min())
        after_max = float(X.max())

        # Build output DataFrame
        n_pixels = new_width * new_height * channels
        new_pixel_cols = [f"pixel_{i}" for i in range(n_pixels)]
        df_out = pd.DataFrame(X, columns=new_pixel_cols)
        df_out["label"] = y

        # Save
        dataset_id = generate_id("imgpp")
        filename = f"{dataset_id}.csv"
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        df_out.to_csv(file_path, index=False)

        logger.info(
            f"Preprocessed dataset saved: {len(df_out)} images, "
            f"pixel range [{after_min:.4f}, {after_max:.4f}], method={input_data.normalize_method}"
        )

        # --- Generate explorer data ---
        before_after_samples = self._generate_before_after(
            X_before, X, y, class_names, width, height, new_width, new_height
        )
        normalization_comparison = self._generate_normalization_comparison(X_before)
        pixel_histograms = self._generate_histograms(X_before, X, input_data.normalize_method)
        quiz_questions = self._generate_quiz(
            input_data.normalize_method, before_min, before_max,
            after_min, after_max, width, height, new_width, new_height, n_pixels
        )

        return ImagePreprocessingOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            preprocessed_dataset_id=dataset_id,
            preprocessed_path=str(file_path),
            n_rows=len(df_out),
            n_columns=len(df_out.columns),
            columns=df_out.columns.tolist(),
            image_width=new_width,
            image_height=new_height,
            n_channels=channels,
            n_classes=n_classes,
            class_names=class_names,
            normalize_method=input_data.normalize_method,
            pixel_range_before=[before_min, before_max],
            pixel_range_after=[after_min, after_max],
            before_after_samples=before_after_samples,
            normalization_comparison=normalization_comparison,
            pixel_histograms=pixel_histograms,
            quiz_questions=quiz_questions,
        )

    # --- Explorer Data Generators ---

    def _generate_before_after(
        self, X_before: np.ndarray, X_after: np.ndarray,
        y: np.ndarray, class_names: list,
        old_w: int, old_h: int, new_w: int, new_h: int,
        n_samples: int = 6
    ) -> List[Dict[str, Any]]:
        """Before/after image pairs for visual comparison."""
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(X_before), size=min(n_samples, len(X_before)), replace=False)

        samples = []
        for idx in chosen:
            cls_idx = int(y[idx])
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
            samples.append({
                "before_pixels": X_before[idx].tolist(),
                "after_pixels": X_after[idx].tolist(),
                "before_width": old_w,
                "before_height": old_h,
                "after_width": new_w,
                "after_height": new_h,
                "class_name": cls_name,
                "before_min": round(float(np.min(X_before[idx])), 4),
                "before_max": round(float(np.max(X_before[idx])), 4),
                "after_min": round(float(np.min(X_after[idx])), 4),
                "after_max": round(float(np.max(X_after[idx])), 4),
            })

        return samples

    def _generate_normalization_comparison(self, X: np.ndarray) -> Dict[str, Any]:
        """Compare different normalization methods on a sample image."""
        sample = X[0].copy()

        methods = {}
        # Raw
        methods["raw"] = {
            "label": "Raw (Original)",
            "min": round(float(sample.min()), 4),
            "max": round(float(sample.max()), 4),
            "mean": round(float(sample.mean()), 4),
            "std": round(float(sample.std()), 4),
            "description": "Original pixel values as loaded from the dataset",
        }

        # Min-Max [0, 1]
        mn, mx = sample.min(), sample.max()
        mm = (sample - mn) / (mx - mn) if mx > mn else sample * 0
        methods["min_max"] = {
            "label": "Min-Max (0 to 1)",
            "min": round(float(mm.min()), 4),
            "max": round(float(mm.max()), 4),
            "mean": round(float(mm.mean()), 4),
            "std": round(float(mm.std()), 4),
            "description": "Scales all values to [0, 1] range. Best for most neural networks.",
        }

        # Divide by 255
        d255 = sample / 255.0
        methods["divide_255"] = {
            "label": "Divide by 255",
            "min": round(float(d255.min()), 4),
            "max": round(float(d255.max()), 4),
            "mean": round(float(d255.mean()), 4),
            "std": round(float(d255.std()), 4),
            "description": "Simple division by 255. Assumes pixel range 0-255. Fast and common.",
        }

        # Z-score
        mean, std = sample.mean(), sample.std()
        zscore = (sample - mean) / std if std > 0 else sample * 0
        methods["z_score"] = {
            "label": "Z-Score (StandardScaler)",
            "min": round(float(zscore.min()), 4),
            "max": round(float(zscore.max()), 4),
            "mean": round(float(zscore.mean()), 4),
            "std": round(float(zscore.std()), 4),
            "description": "Centers data at mean=0, std=1. Good when features have different distributions.",
        }

        return {
            "methods": methods,
            "recommendation": "min_max or divide_255 is best for image data. Z-score is more common for tabular data.",
        }

    def _generate_histograms(
        self, X_before: np.ndarray, X_after: np.ndarray, method: str
    ) -> Dict[str, Any]:
        """Generate before/after pixel value histograms."""
        flat_before = X_before.flatten()
        flat_after = X_after.flatten()

        before_min = float(flat_before.min())
        before_max = float(flat_before.max())
        after_min = float(flat_after.min())
        after_max = float(flat_after.max())

        h_before, e_before = np.histogram(flat_before, bins=32, range=(before_min, before_max + 0.01))
        h_after, e_after = np.histogram(flat_after, bins=32, range=(after_min, after_max + 0.01))

        return {
            "before": {
                "counts": h_before.tolist(),
                "bin_edges": [round(float(e), 4) for e in e_before.tolist()],
                "min": round(before_min, 4),
                "max": round(before_max, 4),
                "label": "Before Normalization",
            },
            "after": {
                "counts": h_after.tolist(),
                "bin_edges": [round(float(e), 4) for e in e_after.tolist()],
                "min": round(after_min, 4),
                "max": round(after_max, 4),
                "label": f"After {method.replace('_', ' ').title()}",
            },
            "method": method,
        }

    def _generate_quiz(
        self, method: str, before_min: float, before_max: float,
        after_min: float, after_max: float,
        old_w: int, old_h: int, new_w: int, new_h: int, n_pixels: int
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about image preprocessing."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: Why normalize?
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why do we normalize pixel values before training a neural network?",
            "options": [
                "Large pixel values (0-255) cause unstable gradients; normalization helps training converge faster",
                "Normalization makes images look better to humans",
                "It reduces the number of pixels in the image",
                "Neural networks can only process integers",
            ],
            "correct_answer": 0,
            "explanation": f"Raw pixel values (range [{before_min:.0f}-{before_max:.0f}]) are large numbers that cause large gradient updates, leading to unstable training. After normalization (range [{after_min:.4f}-{after_max:.4f}]), the values are small and uniform, making gradient descent converge much faster and more reliably.",
            "difficulty": "medium",
        })

        # Q2: Min-Max vs Z-Score
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is the difference between Min-Max normalization and Z-Score normalization?",
            "options": [
                "Min-Max scales to [0,1]; Z-Score centers at mean=0- std=1 — Min-Max is more common for images",
                "They are exactly the same",
                "Z-Score only works for color images",
                "Min-Max removes outliers; Z-Score doesn't",
            ],
            "correct_answer": 0,
            "explanation": "Min-Max normalization scales values to [0, 1] using (x - min) / (max - min). Z-Score normalization centers at mean=0 with std=1 using (x - mean) / std. For images, Min-Max or divide-by-255 is preferred because pixel values have a natural [0, 255] range.",
            "difficulty": "medium",
        })

        # Q3: Pixel range
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"After preprocessing, pixel values changed from [{before_min:.0f}, {before_max:.0f}] to [{after_min:.4f}, {after_max:.4f}]. What happened?",
            "options": [
                f"Normalization was applied ({method.replace('_', ' ')}), scaling values to a smaller, uniform range",
                "Some pixels were deleted",
                "The images were converted to color",
                "Nothing changed, these are the same values",
            ],
            "correct_answer": 0,
            "explanation": f"The '{method.replace('_', ' ')}' normalization transformed raw pixel values from [{before_min:.0f}, {before_max:.0f}] to [{after_min:.4f}, {after_max:.4f}]. This smaller range makes neural network training more stable and efficient.",
            "difficulty": "easy",
        })

        # Q4: Resize implications
        if new_w != old_w or new_h != old_h:
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"Images were resized from {old_w}×{old_h} to {new_w}×{new_h}. What is the trade-off?",
                "options": [
                    "Smaller images train faster but lose some detail; larger images preserve detail but are slower to train",
                    "Resizing always improves accuracy",
                    "Bigger images are always better",
                    "Resizing has no effect on training",
                ],
                "correct_answer": 0,
                "explanation": f"Resizing from {old_w}×{old_h} ({old_w*old_h} pixels) to {new_w}×{new_h} ({new_w*new_h} pixels) {'reduces' if new_w*new_h < old_w*old_h else 'increases'} the number of features. Fewer features mean faster training but potentially lost detail. It's a trade-off between speed and accuracy.",
                "difficulty": "hard",
            })

        # Q5: When NOT to normalize
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Is normalization always necessary for image data?",
            "options": [
                "Not always — tree-based models (Decision Trees, Random Forest) are scale-invariant, but neural networks strongly benefit from it",
                "Yes, all ML models require normalization",
                "No, normalization always hurts performance",
                "Only color images need normalization",
            ],
            "correct_answer": 0,
            "explanation": "Tree-based models split on thresholds, so they don't care about scale. But gradient-based models (neural networks, logistic regression) are very sensitive to feature scales. For image classification with neural networks, normalization is practically mandatory.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
