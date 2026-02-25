"""
Image Split Node — Stratified train/test split for image datasets.
Ensures balanced class distribution in both sets.
Returns rich explorer data: split visualization, class balance charts, and quizzes.
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


class ImageSplitInput(NodeInput):
    """Input schema for Image Split node."""
    dataset_id: str = Field(..., description="Dataset ID from augmentation/preprocessing node")
    test_size: float = Field(0.2, description="Test set fraction (0.1-0.5)")
    random_state: int = Field(42, description="Random seed for reproducibility")
    stratify: bool = Field(True, description="Ensure class balance in both sets")

    # Pass-through
    image_width: Optional[int] = Field(None, description="Image width")
    image_height: Optional[int] = Field(None, description="Image height")
    n_channels: Optional[int] = Field(None, description="Number of channels")
    n_classes: Optional[int] = Field(None, description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Class names")


class ImageSplitOutput(NodeOutput):
    """Output schema for Image Split node."""
    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: str = Field(..., description="Test dataset ID")
    train_path: str = Field(..., description="Path to training CSV")
    test_path: str = Field(..., description="Path to test CSV")
    train_size: int = Field(..., description="Number of training images")
    test_size: int = Field(..., description="Number of test images")
    target_column: str = Field(default="label", description="Target column name")

    # Pass-through
    image_width: int = Field(..., description="Image width")
    image_height: int = Field(..., description="Image height")
    n_channels: int = Field(1, description="Channels")
    n_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="Class names")

    # Explorer data
    split_visualization: Optional[Dict[str, Any]] = Field(
        None, description="Sample images from train and test sets"
    )
    class_balance: Optional[Dict[str, Any]] = Field(
        None, description="Per-class distribution in train vs test"
    )
    split_ratios: Optional[Dict[str, Any]] = Field(
        None, description="Split ratio details"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions"
    )


class ImageSplitNode(BaseNode):
    """
    Image Split Node — Stratified train/test split for image classification.
    """

    node_type = "image_split"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.DATA_TRANSFORM,
            primary_output_field="train_dataset_id",
            output_fields={
                "train_dataset_id": "Training dataset ID",
                "test_dataset_id": "Test dataset ID",
                "train_size": "Number of training images",
                "test_size": "Number of test images",
                "target_column": "Target column name",
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
        return ImageSplitInput

    def get_output_schema(self):
        return ImageSplitOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
        if upload_path.exists():
            return pd.read_csv(upload_path)
        return None

    async def _execute(self, input_data: ImageSplitInput) -> ImageSplitOutput:
        df = await self._load_dataset(input_data.dataset_id)
        if df is None or df.empty:
            raise ValueError(f"Dataset {input_data.dataset_id} not found or empty")

        label_col = "label" if "label" in df.columns else df.columns[-1]
        pixel_cols = [c for c in df.columns if c != label_col]
        y = df[label_col].values

        width = input_data.image_width or int(np.sqrt(len(pixel_cols)))
        height = input_data.image_height or int(np.sqrt(len(pixel_cols)))
        channels = input_data.n_channels or 1
        n_classes = input_data.n_classes or len(np.unique(y))
        class_names = input_data.class_names or [str(i) for i in range(n_classes)]

        test_frac = max(0.05, min(0.5, input_data.test_size))
        rng = np.random.default_rng(input_data.random_state)

        logger.info(f"Splitting {len(df)} images: {(1-test_frac)*100:.0f}% train / {test_frac*100:.0f}% test")

        # Stratified split
        if input_data.stratify:
            train_indices = []
            test_indices = []
            for cls_val in np.unique(y):
                cls_mask = np.where(y == cls_val)[0]
                n_test = max(1, int(len(cls_mask) * test_frac))
                perm = rng.permutation(cls_mask)
                test_indices.extend(perm[:n_test].tolist())
                train_indices.extend(perm[n_test:].tolist())
        else:
            perm = rng.permutation(len(df))
            n_test = max(1, int(len(df) * test_frac))
            test_indices = perm[:n_test].tolist()
            train_indices = perm[n_test:].tolist()

        df_train = df.iloc[train_indices].reset_index(drop=True)
        df_test = df.iloc[test_indices].reset_index(drop=True)

        # Save
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        train_id = generate_id("imgtrain")
        test_id = generate_id("imgtest")
        train_path = upload_dir / f"{train_id}.csv"
        test_path = upload_dir / f"{test_id}.csv"
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        logger.info(
            f"Image split complete: {len(df_train)} train + {len(df_test)} test "
            f"(stratify={input_data.stratify})"
        )

        # --- Explorer data ---
        split_visualization = self._generate_split_visualization(
            df_train, df_test, y[train_indices], y[test_indices],
            class_names, width, height
        )
        class_balance = self._generate_class_balance(
            y[train_indices], y[test_indices], class_names
        )
        split_ratios = self._generate_split_ratios(
            len(df_train), len(df_test), test_frac, input_data.stratify
        )
        quiz_questions = self._generate_quiz(
            len(df_train), len(df_test), test_frac, input_data.stratify, n_classes
        )

        return ImageSplitOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            train_dataset_id=train_id,
            test_dataset_id=test_id,
            train_path=str(train_path),
            test_path=str(test_path),
            train_size=len(df_train),
            test_size=len(df_test),
            target_column=label_col,
            image_width=width,
            image_height=height,
            n_channels=channels,
            n_classes=n_classes,
            class_names=class_names,
            split_visualization=split_visualization,
            class_balance=class_balance,
            split_ratios=split_ratios,
            quiz_questions=quiz_questions,
        )

    # --- Explorer Data Generators ---

    def _generate_split_visualization(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame,
        y_train: np.ndarray, y_test: np.ndarray,
        class_names: list, width: int, height: int,
        n_samples: int = 4
    ) -> Dict[str, Any]:
        """Show sample images from train and test sets per class."""
        rng = np.random.default_rng(42)
        label_col = "label" if "label" in df_train.columns else df_train.columns[-1]
        pixel_cols = [c for c in df_train.columns if c != label_col]

        classes = []
        for cls_idx, cls_name in enumerate(class_names):
            train_mask = y_train == cls_idx
            test_mask = y_test == cls_idx

            train_samples = []
            if np.any(train_mask):
                train_images = df_train.loc[train_mask, pixel_cols].values
                chosen = rng.choice(len(train_images), size=min(n_samples, len(train_images)), replace=False)
                for i in chosen:
                    train_samples.append(train_images[i].tolist())

            test_samples = []
            if np.any(test_mask):
                test_images = df_test.loc[test_mask, pixel_cols].values
                chosen = rng.choice(len(test_images), size=min(n_samples, len(test_images)), replace=False)
                for i in chosen:
                    test_samples.append(test_images[i].tolist())

            classes.append({
                "class_name": str(cls_name),
                "train_samples": train_samples,
                "test_samples": test_samples,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
            })

        return {
            "classes": classes,
            "width": width,
            "height": height,
        }

    def _generate_class_balance(
        self, y_train: np.ndarray, y_test: np.ndarray, class_names: list
    ) -> Dict[str, Any]:
        """Compare class distribution in train vs test."""
        total_train = len(y_train)
        total_test = len(y_test)

        classes = []
        for cls_idx, cls_name in enumerate(class_names):
            train_count = int(np.sum(y_train == cls_idx))
            test_count = int(np.sum(y_test == cls_idx))
            classes.append({
                "class_name": str(cls_name),
                "train_count": train_count,
                "train_pct": round(train_count / total_train * 100, 1) if total_train > 0 else 0,
                "test_count": test_count,
                "test_pct": round(test_count / total_test * 100, 1) if total_test > 0 else 0,
            })

        return {
            "total_train": total_train,
            "total_test": total_test,
            "classes": classes,
        }

    def _generate_split_ratios(
        self, train_size: int, test_size: int, test_frac: float, stratified: bool
    ) -> Dict[str, Any]:
        """Split ratio summary."""
        total = train_size + test_size
        return {
            "train_count": train_size,
            "test_count": test_size,
            "train_ratio": round(train_size / total, 3),
            "test_ratio": round(test_size / total, 3),
            "train_pct": round(train_size / total * 100, 1),
            "test_pct": round(test_size / total * 100, 1),
            "requested_test_frac": test_frac,
            "actual_test_frac": round(test_size / total, 3),
            "stratified": stratified,
        }

    def _generate_quiz(
        self, train_size: int, test_size: int, test_frac: float,
        stratified: bool, n_classes: int
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about train/test splitting."""
        import random as _random
        questions = []
        q_id = 0

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why do we split data into training and test sets?",
            "options": [
                "To evaluate how well the model generalizes to unseen data — not just memorize training examples",
                "To make the dataset smaller for faster training",
                "Because models can't process all data at once",
                "It's just a convention with no real purpose",
            ],
            "correct_answer": 0,
            "explanation": f"We used {train_size} images for training and held out {test_size} images for testing. The test set simulates 'real-world' data the model hasn't seen. If the model performs well on the test set, it has truly learned the patterns — not just memorized.",
            "difficulty": "easy",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"We used an {test_frac*100:.0f}/{(1-test_frac)*100:.0f} test/train split. What happens if we use too much data for testing?",
            "options": [
                "Too little training data — the model can't learn enough patterns, leading to underfitting",
                "The model becomes more accurate",
                "Nothing changes — the split ratio doesn't matter",
                "The test set becomes unreliable",
            ],
            "correct_answer": 0,
            "explanation": f"More test data means less training data. With {n_classes} classes, the model needs enough examples of each class to learn distinguishing features. A 80/20 or 70/30 split is common. Going beyond 50/50 leaves too little for learning.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is 'stratified' splitting and why is it important?",
            "options": [
                "It ensures each class has the same proportion in train and test sets — preventing biased evaluation",
                "It shuffles the data randomly",
                "It removes duplicate images",
                "It splits by image size",
            ],
            "correct_answer": 0,
            "explanation": f"Stratified splitting ensures class balance is preserved. With {n_classes} classes, a random split might put all 'rare class' images in the test set, leaving none for training. Stratification guarantees each class is proportionally represented in both sets.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is 'data leakage' in the context of train/test splitting?",
            "options": [
                "When information from the test set influences training — like fitting preprocessing on the full dataset before splitting",
                "When data is lost during the split",
                "When the test set is too small",
                "When augmentation is applied after splitting",
            ],
            "correct_answer": 0,
            "explanation": "Data leakage occurs when the model indirectly 'sees' test data during training. For example, computing normalization statistics on the full dataset (including test) before splitting would leak test information. Always split FIRST, then preprocess each set independently.",
            "difficulty": "hard",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why should augmentation be done BEFORE splitting into train/test?",
            "options": [
                "Actually, augmentation should be done AFTER splitting — only on the training set!",
                "It should be done before splitting to maximize data",
                "The order doesn't matter",
                "Augmentation should be applied only to the test set",
            ],
            "correct_answer": 0,
            "explanation": "Best practice: Split first, then augment ONLY the training set. If you augment before splitting, an augmented copy of a training image might end up in the test set — this is data leakage! However, in educational pipelines, we sometimes augment before split for simplicity.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
