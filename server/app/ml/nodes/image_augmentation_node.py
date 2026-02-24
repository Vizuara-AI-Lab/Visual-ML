"""
Image Augmentation Node — Apply data augmentation transforms to image datasets.
Supports flip, rotate, brightness, noise, and crop operations.
Returns rich explorer data: augmented galleries, transform labs, and quizzes.
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


class ImageAugmentationInput(NodeInput):
    """Input schema for Image Augmentation node."""
    dataset_id: str = Field(..., description="Dataset ID from preprocessing node")

    # Augmentation options
    horizontal_flip: bool = Field(True, description="Randomly flip images horizontally")
    vertical_flip: bool = Field(False, description="Randomly flip images vertically")
    rotate_max: int = Field(15, description="Max rotation angle in degrees (0 to disable)")
    brightness_range: float = Field(0.2, description="Brightness variation factor (0 to disable)")
    noise_level: float = Field(0.05, description="Gaussian noise std (0 to disable)")
    augment_factor: int = Field(2, description="How many augmented copies per original image (1-5)")

    # Pass-through metadata
    image_width: Optional[int] = Field(None, description="Image width")
    image_height: Optional[int] = Field(None, description="Image height")
    n_channels: Optional[int] = Field(None, description="Number of channels")
    n_classes: Optional[int] = Field(None, description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Class names")


class ImageAugmentationOutput(NodeOutput):
    """Output schema for Image Augmentation node."""
    augmented_dataset_id: str = Field(..., description="ID of augmented dataset")
    augmented_path: str = Field(..., description="Path to augmented CSV")
    n_rows: int = Field(..., description="Total images after augmentation")
    n_columns: int = Field(..., description="Number of columns")
    columns: list = Field(..., description="Column names")
    original_count: int = Field(..., description="Original image count")
    augmented_count: int = Field(..., description="New augmented image count")

    # Pass-through
    image_width: int = Field(..., description="Image width")
    image_height: int = Field(..., description="Image height")
    n_channels: int = Field(1, description="Channels")
    n_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="Class names")

    # Explorer data
    augmentation_gallery: Optional[List[Dict[str, Any]]] = Field(
        None, description="Original → augmented side-by-side samples"
    )
    transform_effects: Optional[Dict[str, Any]] = Field(
        None, description="Visual effect of each transform type"
    )
    dataset_growth: Optional[Dict[str, Any]] = Field(
        None, description="Dataset size before/after augmentation"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions"
    )


class ImageAugmentationNode(BaseNode):
    """
    Image Augmentation Node — Expand training data through augmentation transforms.
    """

    node_type = "image_augmentation"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.PREPROCESSING,
            primary_output_field="augmented_dataset_id",
            output_fields={
                "augmented_dataset_id": "Augmented image dataset ID",
                "image_width": "Image width",
                "image_height": "Image height",
                "n_classes": "Number of classes",
                "class_names": "Class names",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.PREPROCESSING,
                NodeCategory.DATA_SOURCE,
            ],
        )

    def get_input_schema(self):
        return ImageAugmentationInput

    def get_output_schema(self):
        return ImageAugmentationOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
        if upload_path.exists():
            return pd.read_csv(upload_path)
        return None

    def _flip_horizontal(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Flip image horizontally."""
        return img.reshape(h, w)[:, ::-1].flatten()

    def _flip_vertical(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Flip image vertically."""
        return img.reshape(h, w)[::-1, :].flatten()

    def _rotate(self, img: np.ndarray, w: int, h: int, angle_deg: float) -> np.ndarray:
        """Rotate image by angle (simple nearest-neighbor)."""
        grid = img.reshape(h, w)
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        cy, cx = h / 2, w / 2
        result = np.zeros_like(grid)

        for y in range(h):
            for x in range(w):
                src_x = cos_a * (x - cx) + sin_a * (y - cy) + cx
                src_y = -sin_a * (x - cx) + cos_a * (y - cy) + cy
                sx, sy = int(round(src_x)), int(round(src_y))
                if 0 <= sx < w and 0 <= sy < h:
                    result[y, x] = grid[sy, sx]

        return result.flatten()

    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness by factor."""
        return np.clip(img * (1 + factor), img.min(), img.max())

    def _add_noise(self, img: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, std, img.shape)
        return np.clip(img + noise, img.min(), img.max())

    def _augment_image(
        self, img: np.ndarray, w: int, h: int, rng: np.random.Generator,
        h_flip: bool, v_flip: bool, rotate_max: int,
        brightness_range: float, noise_level: float
    ) -> np.ndarray:
        """Apply random augmentations to a single image."""
        aug = img.copy()

        if h_flip and rng.random() > 0.5:
            aug = self._flip_horizontal(aug, w, h)

        if v_flip and rng.random() > 0.5:
            aug = self._flip_vertical(aug, w, h)

        if rotate_max > 0:
            angle = rng.uniform(-rotate_max, rotate_max)
            if abs(angle) > 1:
                aug = self._rotate(aug, w, h, angle)

        if brightness_range > 0:
            factor = rng.uniform(-brightness_range, brightness_range)
            aug = self._adjust_brightness(aug, factor)

        if noise_level > 0:
            aug = self._add_noise(aug, noise_level)

        return aug

    async def _execute(self, input_data: ImageAugmentationInput) -> ImageAugmentationOutput:
        df = await self._load_dataset(input_data.dataset_id)
        if df is None or df.empty:
            raise ValueError(f"Dataset {input_data.dataset_id} not found or empty")

        label_col = "label" if "label" in df.columns else df.columns[-1]
        pixel_cols = [c for c in df.columns if c != label_col]
        X = df[pixel_cols].values
        y = df[label_col].values

        width = input_data.image_width or int(np.sqrt(len(pixel_cols)))
        height = input_data.image_height or int(np.sqrt(len(pixel_cols)))
        channels = input_data.n_channels or 1
        n_classes = input_data.n_classes or len(np.unique(y))
        class_names = input_data.class_names or [str(i) for i in range(n_classes)]

        original_count = len(X)
        augment_factor = max(1, min(5, input_data.augment_factor))

        logger.info(
            f"Augmenting {original_count} images with factor {augment_factor} "
            f"(h_flip={input_data.horizontal_flip}, rotate={input_data.rotate_max}°, "
            f"brightness={input_data.brightness_range}, noise={input_data.noise_level})"
        )

        rng = np.random.default_rng(42)
        augmented_X = []
        augmented_y = []

        # Store augmentation examples for explorer
        gallery_examples = []
        n_gallery = min(8, original_count)
        gallery_indices = rng.choice(original_count, size=n_gallery, replace=False)

        for i in range(original_count):
            for _ in range(augment_factor):
                aug = self._augment_image(
                    X[i], width, height, rng,
                    input_data.horizontal_flip,
                    input_data.vertical_flip,
                    input_data.rotate_max,
                    input_data.brightness_range,
                    input_data.noise_level,
                )
                augmented_X.append(aug)
                augmented_y.append(y[i])

                # Capture first augmentation for gallery
                if i in gallery_indices and len(gallery_examples) < n_gallery * 2:
                    cls_idx = int(y[i])
                    cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                    gallery_examples.append({
                        "original_pixels": X[i].tolist(),
                        "augmented_pixels": aug.tolist(),
                        "class_name": cls_name,
                        "width": width,
                        "height": height,
                    })

        # Combine original + augmented
        all_X = np.vstack([X, np.array(augmented_X)])
        all_y = np.concatenate([y, np.array(augmented_y)])

        # Shuffle
        perm = rng.permutation(len(all_X))
        all_X = all_X[perm]
        all_y = all_y[perm]

        # Build DataFrame
        df_out = pd.DataFrame(all_X, columns=pixel_cols)
        df_out["label"] = all_y

        # Save
        dataset_id = generate_id("imgaug")
        filename = f"{dataset_id}.csv"
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        df_out.to_csv(file_path, index=False)

        augmented_count = len(augmented_X)
        total_count = len(all_X)

        logger.info(
            f"Augmentation complete: {original_count} → {total_count} images "
            f"(+{augmented_count} augmented)"
        )

        # --- Generate explorer data ---
        transform_effects = self._generate_transform_effects(
            X[0], width, height, input_data
        )
        dataset_growth = self._generate_dataset_growth(
            original_count, augmented_count, total_count, y, all_y, class_names
        )
        quiz_questions = self._generate_quiz(
            original_count, total_count, augment_factor, input_data
        )

        return ImageAugmentationOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            augmented_dataset_id=dataset_id,
            augmented_path=str(file_path),
            n_rows=total_count,
            n_columns=len(df_out.columns),
            columns=df_out.columns.tolist(),
            original_count=original_count,
            augmented_count=augmented_count,
            image_width=width,
            image_height=height,
            n_channels=channels,
            n_classes=n_classes,
            class_names=class_names,
            augmentation_gallery=gallery_examples,
            transform_effects=transform_effects,
            dataset_growth=dataset_growth,
            quiz_questions=quiz_questions,
        )

    # --- Explorer Data Generators ---

    def _generate_transform_effects(
        self, sample: np.ndarray, w: int, h: int,
        config: ImageAugmentationInput
    ) -> Dict[str, Any]:
        """Show each transform applied individually to a sample image."""
        effects = {}

        effects["original"] = {
            "label": "Original",
            "pixels": sample.tolist(),
            "description": "Unmodified original image",
        }

        if config.horizontal_flip:
            flipped = self._flip_horizontal(sample, w, h)
            effects["horizontal_flip"] = {
                "label": "Horizontal Flip",
                "pixels": flipped.tolist(),
                "description": "Mirror the image left-to-right. Helps the model generalize to different orientations.",
            }

        if config.vertical_flip:
            flipped = self._flip_vertical(sample, w, h)
            effects["vertical_flip"] = {
                "label": "Vertical Flip",
                "pixels": flipped.tolist(),
                "description": "Mirror the image top-to-bottom. Useful for aerial/satellite imagery.",
            }

        if config.rotate_max > 0:
            rotated = self._rotate(sample, w, h, config.rotate_max * 0.7)
            effects["rotation"] = {
                "label": f"Rotation ({config.rotate_max}°)",
                "pixels": rotated.tolist(),
                "description": f"Rotate by up to ±{config.rotate_max}°. Teaches the model to handle tilted images.",
            }

        if config.brightness_range > 0:
            brighter = self._adjust_brightness(sample, config.brightness_range)
            darker = self._adjust_brightness(sample, -config.brightness_range)
            effects["brightness_up"] = {
                "label": f"Brighter (+{config.brightness_range:.0%})",
                "pixels": brighter.tolist(),
                "description": "Increase brightness. Simulates different lighting conditions.",
            }
            effects["brightness_down"] = {
                "label": f"Darker (-{config.brightness_range:.0%})",
                "pixels": darker.tolist(),
                "description": "Decrease brightness. Simulates shadows or dim lighting.",
            }

        if config.noise_level > 0:
            noisy = self._add_noise(sample, config.noise_level)
            effects["noise"] = {
                "label": f"Gaussian Noise (σ={config.noise_level})",
                "pixels": noisy.tolist(),
                "description": "Add random noise. Helps prevent overfitting to exact pixel patterns.",
            }

        return {
            "width": w,
            "height": h,
            "effects": effects,
        }

    def _generate_dataset_growth(
        self, original: int, augmented: int, total: int,
        y_original: np.ndarray, y_all: np.ndarray, class_names: list
    ) -> Dict[str, Any]:
        """Show how augmentation changed dataset size per class."""
        per_class = []
        for cls_idx, cls_name in enumerate(class_names):
            orig_count = int(np.sum(y_original == cls_idx))
            total_count = int(np.sum(y_all == cls_idx))
            per_class.append({
                "class_name": str(cls_name),
                "original": orig_count,
                "total": total_count,
                "added": total_count - orig_count,
            })

        return {
            "original_total": original,
            "augmented_added": augmented,
            "final_total": total,
            "growth_factor": round(total / original, 1) if original > 0 else 0,
            "per_class": per_class,
        }

    def _generate_quiz(
        self, original: int, total: int, factor: int,
        config: ImageAugmentationInput
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about data augmentation."""
        import random as _random
        questions = []
        q_id = 0

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why is data augmentation important for image classification?",
            "options": [
                "It artificially increases training data variety, reducing overfitting and improving generalization",
                "It makes images look prettier",
                "It reduces the dataset size for faster training",
                "It's required by all ML algorithms",
            ],
            "correct_answer": 0,
            "explanation": f"Data augmentation creates modified copies of existing images (we went from {original} → {total} images). This teaches the model to be invariant to transformations like rotation and brightness, reducing overfitting.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Which augmentation would NOT make sense for digit recognition (0-9)?",
            "options": [
                "Vertical flip — flipping '6' makes it look like '9', confusing the model",
                "Small rotation — digits are sometimes slightly tilted in real life",
                "Brightness adjustment — lighting varies in real scanned documents",
                "Adding noise — real images often have some noise",
            ],
            "correct_answer": 0,
            "explanation": "Vertical flipping changes the meaning of some digits (6↔9, 2→looks odd). Augmentations must preserve the label! Small rotations, brightness changes, and noise are safe because they don't change the digit's identity.",
            "difficulty": "hard",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"We used augmentation factor {factor}, turning {original} images into {total}. Why not use factor 100?",
            "options": [
                "Diminishing returns — too many similar copies add noise without new information, and training becomes very slow",
                "Higher factor always gives better results",
                "The computer would run out of colors",
                "Factor 100 would make images too small",
            ],
            "correct_answer": 0,
            "explanation": "While augmentation helps, there's a sweet spot. Factor 2-5 typically works well. Factor 100 would create 100 slightly different copies of each image, slowing training dramatically with minimal accuracy gain. The key is variety, not volume.",
            "difficulty": "hard",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Should augmentation be applied to the test set?",
            "options": [
                "No — the test set should represent real-world data without artificial modifications",
                "Yes — always augment everything",
                "Only for color images",
                "Only if the model is performing poorly",
            ],
            "correct_answer": 0,
            "explanation": "Augmentation is ONLY for training data. The test set must represent real, untouched data so we get an honest measure of how the model performs in the real world. Augmenting test data would give misleading evaluation results.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is 'overfitting' and how does augmentation help prevent it?",
            "options": [
                "Overfitting = model memorizes training data instead of learning patterns. Augmentation adds variety so the model can't just memorize.",
                "Overfitting means the model is too accurate",
                "Overfitting happens when images are too large",
                "Augmentation causes overfitting by adding too much data",
            ],
            "correct_answer": 0,
            "explanation": "Overfitting occurs when a model memorizes specific training images (e.g., 'digit 3 always has this exact shape') instead of learning general patterns. Augmentation fights this by showing the model many variations of each image, forcing it to learn the essential visual features.",
            "difficulty": "medium",
        })

        _random.shuffle(questions)
        return questions[:5]
