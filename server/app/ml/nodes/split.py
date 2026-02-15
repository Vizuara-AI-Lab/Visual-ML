"""
Train/Test Split Node - Splits dataset into training and test sets.
Supports target selection, configurable ratios, and reproducible splits.

"""

from typing import Type, Optional, Dict, Any
import pandas as pd
import io
from pydantic import Field, field_validator
from pathlib import Path
from sklearn.model_selection import train_test_split
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service
from typing import List, Any
import numpy as np


class SplitInput(NodeInput):
    """Input schema for Split node."""

    dataset_id: str = Field(..., description="Dataset ID to process")
    target_column: str = Field(..., description="Name of target column")

    # Split ratios
    train_ratio: float = Field(0.8, description="Training set ratio (0-1)")
    test_ratio: float = Field(0.2, description="Test set ratio (0-1)")

    # Split type
    split_type: str = Field("random", description="Split type: random or stratified")

    # Reproducibility
    random_seed: int = Field(42, description="Random seed for reproducibility")
    shuffle: bool = Field(True, description="Whether to shuffle before splitting")

    @field_validator("train_ratio", "test_ratio")
    @classmethod
    def validate_ratio(cls, v: Optional[float]) -> Optional[float]:
        """Validate split ratio."""
        if v is not None:
            if not 0 < v < 1:
                raise ValueError(f"Ratio must be between 0 and 1, got {v}")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate ratios sum to 1."""
        total = self.train_ratio + self.test_ratio
        if not abs(total - 1.0) < 0.01:
            raise ValueError(f"Train and test ratios must sum to 1.0, got {total}")


class SplitOutput(NodeOutput):
    """Output schema for Split node."""

    train_dataset_id: str = Field(..., description="ID of training dataset")
    test_dataset_id: str = Field(..., description="ID of test dataset")

    train_path: str = Field(..., description="Path to training set")
    test_path: str = Field(..., description="Path to test set")

    train_size: int = Field(..., description="Number of samples in training set")
    test_size: int = Field(..., description="Number of samples in test set")

    target_column: str = Field(..., description="Target column name")
    columns: List[str] = Field(..., description="All column names in the dataset")
    feature_columns: List[str] = Field(..., description="Feature column names (excluding target)")

    split_summary: Dict[str, Any] = Field(..., description="Summary of split operation")

    # Learning activity data (all Optional for backward compatibility)
    split_visualization: Optional[Dict[str, Any]] = Field(
        None, description="Train/test split visualization data"
    )
    class_balance: Optional[Dict[str, Any]] = Field(
        None, description="Class distribution in train vs test sets"
    )
    ratio_explorer: Optional[Dict[str, Any]] = Field(
        None, description="Pre-computed results for different split ratios"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about train/test split"
    )


class SplitNode(BaseNode):
    """
    Target & Split Node - Select target column and split dataset for ML training.

    Responsibilities:
    - Select target column (y)
    - Treat remaining columns as features (X)
    - Split dataset into train/test sets (default 80/20)
    - Support random and stratified splitting
    - Reproducible splits with random seed
    - Save split datasets separately

    Production features:
    - Reproducible with random seed
    - Stratified splitting for classification tasks
    - Comprehensive split summary
    """

    node_type = "split"

    @property
    def metadata(self) -> NodeMetadata:
        """Define node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.DATA_TRANSFORM,
            primary_output_field="train_dataset_id",
            output_fields={
                "train_dataset_id": "Identifier for the training dataset",
                "test_dataset_id": "Identifier for the test dataset",
                "train_size": "Number of samples in training set",
                "test_size": "Number of samples in test set",
                "target_column": "Name of the target column",
                "columns": "All column names in the dataset",
                "feature_columns": "Feature column names (excluding target)",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=True,
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.DATA_SOURCE,
                NodeCategory.PREPROCESSING,
                NodeCategory.DATA_TRANSFORM,
            ],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return SplitInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return SplitOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            # Recognize common missing value indicators: ?, NA, N/A, null, empty strings
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            # FIRST: Try to load from uploads folder (for preprocessed/engineered datasets)
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

    async def _execute(self, input_data: SplitInput) -> SplitOutput:
        """
        Execute dataset splitting.

        Args:
            input_data: Validated input data

        Returns:
            Split result with paths to split datasets
        """
        try:
            logger.info(f"Splitting dataset: {input_data.dataset_id}")

            # Load dataset
            df = await self._load_dataset(input_data.dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            # Validate target column
            if input_data.target_column not in df.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X = df.drop(columns=[input_data.target_column])
            y = df[input_data.target_column]

            # Prepare stratify parameter
            stratify_param = y if input_data.split_type == "stratified" else None

            # Perform train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=input_data.test_ratio,
                random_state=input_data.random_seed,
                shuffle=input_data.shuffle,
                stratify=stratify_param,
            )

            # Combine features and target
            train_df = X_train.copy()
            train_df[input_data.target_column] = y_train

            test_df = X_test.copy()
            test_df[input_data.target_column] = y_test

            # Save datasets
            split_id = generate_id("split")
            upload_dir = Path(settings.UPLOAD_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)

            train_id = f"{split_id}_train"
            test_id = f"{split_id}_test"

            train_path = upload_dir / f"{train_id}.csv"
            test_path = upload_dir / f"{test_id}.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info(f"Dataset split complete - Train: {len(train_df)}, Test: {len(test_df)}")

            split_summary = {
                "total_samples": len(df),
                "target_column": input_data.target_column,
                "feature_columns": len(X.columns),
                "train_ratio": input_data.train_ratio,
                "test_ratio": input_data.test_ratio,
                "split_type": input_data.split_type,
                "random_seed": input_data.random_seed,
                "shuffled": input_data.shuffle,
            }

            # Generate learning activity data (non-blocking)
            split_visualization = None
            try:
                split_visualization = self._generate_split_visualization(
                    df, train_df, test_df, input_data
                )
            except Exception as e:
                logger.warning(f"Split visualization generation failed: {e}")

            class_balance = None
            try:
                class_balance = self._generate_class_balance(
                    y, y_train, y_test, input_data.target_column
                )
            except Exception as e:
                logger.warning(f"Class balance generation failed: {e}")

            ratio_explorer = None
            try:
                ratio_explorer = self._generate_ratio_explorer(
                    df, input_data
                )
            except Exception as e:
                logger.warning(f"Ratio explorer generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_split_quiz(
                    df, train_df, test_df, input_data, y
                )
            except Exception as e:
                logger.warning(f"Split quiz generation failed: {e}")

            return SplitOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                train_dataset_id=train_id,
                test_dataset_id=test_id,
                train_path=str(train_path),
                test_path=str(test_path),
                train_size=len(train_df),
                test_size=len(test_df),
                target_column=input_data.target_column,
                columns=train_df.columns.tolist(),
                feature_columns=X.columns.tolist(),
                split_summary=split_summary,
                split_visualization=split_visualization,
                class_balance=class_balance,
                ratio_explorer=ratio_explorer,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"Dataset split failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

    def _generate_split_visualization(
        self, df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
        input_data: SplitInput
    ) -> Dict[str, Any]:
        """Generate train/test split visualization data."""
        train_sample = train_df.head(5).to_dict(orient="records")
        test_sample = test_df.head(5).to_dict(orient="records")

        # Simplify sample values for JSON
        for row in train_sample + test_sample:
            for k, v in row.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    row[k] = None
                elif isinstance(v, (np.integer,)):
                    row[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    row[k] = round(float(v), 4)

        return {
            "total_samples": len(df),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_pct": round(len(train_df) / len(df) * 100, 1),
            "test_pct": round(len(test_df) / len(df) * 100, 1),
            "n_features": len(df.columns) - 1,
            "target_column": input_data.target_column,
            "split_type": input_data.split_type,
            "train_sample": train_sample,
            "test_sample": test_sample,
            "columns": df.columns.tolist()[:10],
        }

    def _generate_class_balance(
        self, y_full: pd.Series, y_train: pd.Series, y_test: pd.Series,
        target_column: str
    ) -> Dict[str, Any]:
        """Compare class distribution in full dataset, train, and test sets."""
        def dist(series: pd.Series) -> List[Dict[str, Any]]:
            counts = series.value_counts()
            total = len(series)
            return [
                {"class": str(cls), "count": int(cnt), "pct": round(cnt / total * 100, 1)}
                for cls, cnt in counts.items()
            ]

        full_dist = dist(y_full)
        train_dist = dist(y_train)
        test_dist = dist(y_test)

        # Check if balanced
        is_classification = y_full.nunique() <= 20
        is_balanced = True
        if is_classification and len(full_dist) > 1:
            pcts = [d["pct"] for d in full_dist]
            is_balanced = max(pcts) - min(pcts) < 20

        return {
            "target_column": target_column,
            "is_classification": is_classification,
            "n_classes": int(y_full.nunique()),
            "full_distribution": full_dist,
            "train_distribution": train_dist,
            "test_distribution": test_dist,
            "is_balanced": is_balanced,
        }

    def _generate_ratio_explorer(
        self, df: pd.DataFrame, input_data: SplitInput
    ) -> Dict[str, Any]:
        """Pre-computed results for different split ratios."""
        total = len(df)
        ratios = [
            {"label": "60/40", "train_ratio": 0.6, "test_ratio": 0.4},
            {"label": "70/30", "train_ratio": 0.7, "test_ratio": 0.3},
            {"label": "80/20", "train_ratio": 0.8, "test_ratio": 0.2},
            {"label": "90/10", "train_ratio": 0.9, "test_ratio": 0.1},
        ]

        results = []
        for r in ratios:
            train_n = int(total * r["train_ratio"])
            test_n = total - train_n
            is_applied = abs(r["train_ratio"] - input_data.train_ratio) < 0.01
            results.append({
                "label": r["label"],
                "train_ratio": r["train_ratio"],
                "test_ratio": r["test_ratio"],
                "train_size": train_n,
                "test_size": test_n,
                "is_applied": is_applied,
                "pros": self._ratio_pros(r["train_ratio"]),
                "cons": self._ratio_cons(r["train_ratio"]),
            })

        return {"total_samples": total, "ratios": results}

    def _ratio_pros(self, ratio: float) -> str:
        if ratio >= 0.9:
            return "Model sees the most data for training"
        elif ratio >= 0.8:
            return "Good balance between training data and test reliability"
        elif ratio >= 0.7:
            return "More test data gives a more reliable performance estimate"
        else:
            return "Large test set — very reliable performance estimate"

    def _ratio_cons(self, ratio: float) -> str:
        if ratio >= 0.9:
            return "Very small test set — performance estimate may be unreliable"
        elif ratio >= 0.8:
            return "Standard choice — generally works well"
        elif ratio >= 0.7:
            return "Less training data may reduce model performance"
        else:
            return "Much less training data — model may underfit"

    def _generate_split_quiz(
        self, df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
        input_data: SplitInput, y: pd.Series
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about train/test split."""
        import random as _random
        questions = []
        q_id = 0
        total = len(df)

        # Q1: Size calculation
        q_id += 1
        correct = len(test_df)
        wrong1 = len(train_df)
        wrong2 = total
        wrong3 = int(total * 0.5)
        options = [str(correct), str(wrong1), str(wrong2), str(wrong3)]
        _random.shuffle(options)
        questions.append({
            "id": f"q{q_id}",
            "question": f"Your dataset has {total} rows with an {int(input_data.train_ratio*100)}/{int(input_data.test_ratio*100)} split. How many rows are in the test set?",
            "options": options,
            "correct_answer": options.index(str(correct)),
            "explanation": f"With a {int(input_data.test_ratio*100)}% test ratio: {total} x {input_data.test_ratio} = {correct} test rows.",
            "difficulty": "easy",
        })

        # Q2: Why split
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why do we split data into training and test sets?",
            "options": [
                "To check if the model works on data it hasn't seen before",
                "To make the dataset smaller and faster to process",
                "Because ML algorithms require exactly two datasets",
                "To remove duplicate rows from the dataset",
            ],
            "correct_answer": 0,
            "explanation": "The test set simulates 'new, unseen data'. If the model performs well on the test set, it's likely to work well on real-world data too. Training and testing on the same data would give a misleadingly high score.",
            "difficulty": "easy",
        })

        # Q3: Data leakage
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What happens if the same rows appear in both training AND test sets?",
            "options": [
                "The model memorizes answers and gets a falsely high score",
                "The model trains faster",
                "Nothing — it doesn't matter",
                "The model crashes with an error",
            ],
            "correct_answer": 0,
            "explanation": "This is called 'data leakage'. The model has already seen the test answers during training, so its test score looks great but it won't generalize to truly new data. Always keep train and test completely separate!",
            "difficulty": "medium",
        })

        # Q4: Stratified split
        if y.nunique() <= 20:
            class_counts = y.value_counts()
            minority_class = str(class_counts.idxmin())
            minority_pct = round(class_counts.min() / total * 100, 1)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"Class '{minority_class}' makes up only {minority_pct}% of the data. What does stratified splitting ensure?",
                "options": [
                    "Both train and test sets have the same class proportions",
                    "The minority class is removed",
                    "The minority class gets more samples",
                    "All classes get equal numbers of samples",
                ],
                "correct_answer": 0,
                "explanation": f"Stratified splitting preserves the class ratio. Without it, a random split might put very few '{minority_class}' samples in the test set, making evaluation unreliable.",
                "difficulty": "medium",
            })

        # Q5: Target column
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The target column is '{input_data.target_column}'. What role does it play?",
            "options": [
                "It's the value the model tries to predict",
                "It's the most important feature",
                "It's used to sort the data",
                "It's removed before training",
            ],
            "correct_answer": 0,
            "explanation": f"The target column ('{input_data.target_column}') is what the model learns to predict. All other columns are features — the inputs the model uses to make predictions.",
            "difficulty": "easy",
        })

        _random.shuffle(questions)
        return questions[:5]
