"""
Logistic Regression Node - Individual node wrapper for Logistic Regression algorithm.
Supports binary and multi-class classification with configurable hyperparameters.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.ml.algorithms.classification.logistic_regression import LogisticRegression
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class LogisticRegressionInput(NodeInput):
    """Input schema for Logistic Regression node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: Optional[str] = Field(
        None, description="Name of target column (auto-filled from split node)"
    )

    # UI control
    show_advanced_options: bool = Field(
        False, description="Toggle for advanced options visibility in UI"
    )

    # Hyperparameters (all optional with sensible defaults)
    fit_intercept: bool = Field(True, description="Calculate intercept for the model")
    C: float = Field(1.0, description="Inverse of regularization strength")
    penalty: str = Field("l2", description="Regularization penalty type")
    solver: str = Field("lbfgs", description="Optimization algorithm")
    max_iter: int = Field(1000, description="Maximum iterations for convergence")
    random_state: int = Field(42, description="Random seed for reproducibility")


class LogisticRegressionOutput(NodeOutput):
    """Output schema for Logistic Regression node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    n_classes: int = Field(..., description="Number of classes")

    training_metrics: Dict[str, Any] = Field(
        ..., description="Training metrics (accuracy, precision, recall, F1)"
    )
    training_time_seconds: float = Field(..., description="Training duration")

    class_names: list = Field(..., description="Class names/labels")
    metadata: Dict[str, Any] = Field(..., description="Training metadata")

    # Pass-through fields for downstream nodes (e.g., confusion_matrix)
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (passed from split node)"
    )
    target_column: Optional[str] = Field(None, description="Target column (passed from split node)")

    # Learning activity data (all Optional for backward compatibility)
    class_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Training data class distribution"
    )
    metric_explainer: Optional[Dict[str, Any]] = Field(
        None, description="Metric explanations with real data examples"
    )
    sigmoid_data: Optional[Dict[str, Any]] = Field(
        None, description="Sigmoid curve data for visualization"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about logistic regression"
    )


class LogisticRegressionNode(BaseNode):
    """
    Logistic Regression Node - Train logistic regression models for classification.

    Responsibilities:
    - Load training dataset from database/S3
    - Train logistic regression model
    - Calculate classification metrics
    - Save model artifacts
    - Return model metadata

    Production features:
    - Binary and multi-class support
    - Configurable regularization
    - Multiple solver options
    - Model versioning
    """

    node_type = "logistic_regression"

    @property
    def metadata(self) -> NodeMetadata:
        """Return node metadata for DAG execution."""
        return NodeMetadata(
            category=NodeCategory.ML_ALGORITHM,
            primary_output_field="model_id",
            output_fields={
                "model_id": "Unique model identifier",
                "model_path": "Path to saved model file",
                "training_metrics": "Training performance metrics",
                "test_dataset_id": "Test dataset ID for evaluation",
                "target_column": "Target column name",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=False,
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.DATA_TRANSFORM,
                NodeCategory.PREPROCESSING,
                NodeCategory.FEATURE_ENGINEERING,
            ],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return LogisticRegressionInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return LogisticRegressionOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            # Recognize common missing value indicators: ?, NA, N/A, null, empty strings
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            # FIRST: Try to load from uploads folder (for train/test datasets from split node)
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

    async def _execute(self, input_data: LogisticRegressionInput) -> LogisticRegressionOutput:
        """Execute logistic regression training."""
        try:
            logger.info(f"Training Logistic Regression model")

            # Load training data
            df_train = await self._load_dataset(input_data.train_dataset_id)
            if df_train is None or df_train.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.train_dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            # Validate target column
            if not input_data.target_column:
                raise ValueError("Target column must be provided (auto-filled from split node)")

            if input_data.target_column not in df_train.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X_train = df_train.drop(columns=[input_data.target_column])
            y_train = df_train[input_data.target_column]

            # Validate that target is suitable for classification
            n_unique = y_train.nunique()
            unique_values_sample = y_train.unique()[:10].tolist()
            
            logger.info(f"Target column '{input_data.target_column}' has {n_unique} unique values")
            
            # Check if target has too many unique values (likely continuous/regression problem)
            if n_unique > 20:
                raise NodeExecutionError(
                    node_type=self.node_type,
                    reason=f"Target column '{input_data.target_column}' has {n_unique} unique values, which is too many for classification. Logistic Regression is for categorical targets (typically 2-20 classes). For continuous targets, use Linear Regression instead. Sample values: {unique_values_sample}",
                    input_data=input_data.model_dump()
                )
            
            # Warn if target has only one class
            if n_unique == 1:
                raise NodeExecutionError(
                    node_type=self.node_type,
                    reason=f"Target column '{input_data.target_column}' has only 1 unique value. Cannot train a classification model with a single class. Please check your data split or target column.",
                    input_data=input_data.model_dump()
                )
            
            # Log classification type
            if n_unique == 2:
                logger.info(f"Binary classification detected (classes: {unique_values_sample})")
            else:
                logger.info(f"Multi-class classification detected ({n_unique} classes)")

            # Initialize and train model
            model = LogisticRegression(
                C=input_data.C,
                penalty=input_data.penalty,
                solver=input_data.solver,
                max_iter=input_data.max_iter,
                random_state=input_data.random_state,
                fit_intercept=input_data.fit_intercept,
            )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id("model_logistic_regression")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "logistic_regression"
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename

            model.save(model_path)

            logger.info(f"Model saved to {model_path}")

            metrics = training_metadata.get("training_metrics", {})
            class_names_list = model.class_names

            # Generate learning activity data (non-blocking)
            class_distribution = None
            try:
                class_distribution = self._generate_class_distribution(
                    y_train, class_names_list, input_data.target_column
                )
            except Exception as e:
                logger.warning(f"Class distribution generation failed: {e}")

            metric_explainer = None
            try:
                metric_explainer = self._generate_metric_explainer(
                    metrics, class_names_list, len(X_train)
                )
            except Exception as e:
                logger.warning(f"Metric explainer generation failed: {e}")

            sigmoid_data = None
            try:
                sigmoid_data = self._generate_sigmoid_data()
            except Exception as e:
                logger.warning(f"Sigmoid data generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_logistic_regression_quiz(
                    metrics, class_names_list, input_data.target_column, len(X_train)
                )
            except Exception as e:
                logger.warning(f"Logistic regression quiz generation failed: {e}")

            return LogisticRegressionOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=X_train.shape[1],
                n_classes=len(class_names_list),
                training_metrics=metrics,
                training_time_seconds=training_time,
                class_names=class_names_list,
                metadata=model.training_metadata,
                test_dataset_id=input_data.test_dataset_id,
                target_column=input_data.target_column,
                class_distribution=class_distribution,
                metric_explainer=metric_explainer,
                sigmoid_data=sigmoid_data,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"Logistic Regression training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

    def _generate_class_distribution(
        self, y_train: pd.Series, class_names: list, target_column: str
    ) -> Dict[str, Any]:
        """Class distribution bar chart data with balance analysis."""
        total = len(y_train)
        counts = y_train.value_counts()

        classes = []
        for cls in class_names:
            count = int(counts.get(cls, 0))
            pct = round(count / total * 100, 1) if total > 0 else 0
            classes.append({
                "name": str(cls),
                "count": count,
                "percentage": pct,
                "bar_width_pct": round(count / counts.max() * 100, 1) if counts.max() > 0 else 0,
            })

        # Balance analysis
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = round(float(max_count / min_count), 2) if min_count > 0 else float("inf")
        is_balanced = imbalance_ratio <= 1.5

        return {
            "target_column": target_column,
            "total_samples": total,
            "n_classes": len(class_names),
            "classes": classes,
            "is_balanced": is_balanced,
            "imbalance_ratio": imbalance_ratio,
            "balance_message": (
                "Classes are roughly balanced - good for training!"
                if is_balanced
                else f"Classes are imbalanced (ratio {imbalance_ratio}:1). The model may favor the majority class."
            ),
        }

    def _generate_metric_explainer(
        self, metrics: Dict[str, Any], class_names: list, n_samples: int
    ) -> Dict[str, Any]:
        """Metric explanation cards with real numbers and analogies."""
        explanations = []

        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            acc_pct = round(float(accuracy) * 100, 1)
            correct = int(round(float(accuracy) * n_samples))
            wrong = n_samples - correct
            explanations.append({
                "metric": "Accuracy",
                "value": round(float(accuracy), 4),
                "value_pct": acc_pct,
                "analogy": f"Out of {n_samples} predictions, {correct} were correct and {wrong} were wrong.",
                "when_useful": "Good when classes are balanced. Misleading when one class dominates.",
                "color": "blue",
            })

        precision = metrics.get("precision") or metrics.get("weighted_precision")
        if precision is not None:
            prec_pct = round(float(precision) * 100, 1)
            explanations.append({
                "metric": "Precision",
                "value": round(float(precision), 4),
                "value_pct": prec_pct,
                "analogy": f"When the model says 'positive', it is right {prec_pct}% of the time. Like a careful doctor who only diagnoses a disease when very sure.",
                "when_useful": "Important when false positives are costly (e.g., spam filter marking real emails as spam).",
                "color": "green",
            })

        recall = metrics.get("recall") or metrics.get("weighted_recall")
        if recall is not None:
            rec_pct = round(float(recall) * 100, 1)
            explanations.append({
                "metric": "Recall",
                "value": round(float(recall), 4),
                "value_pct": rec_pct,
                "analogy": f"Of all actual positives, the model catches {rec_pct}%. Like a security guard who catches {rec_pct}% of intruders.",
                "when_useful": "Important when missing real cases is costly (e.g., disease screening, fraud detection).",
                "color": "orange",
            })

        f1 = metrics.get("f1_score") or metrics.get("weighted_f1")
        if f1 is not None:
            f1_pct = round(float(f1) * 100, 1)
            explanations.append({
                "metric": "F1 Score",
                "value": round(float(f1), 4),
                "value_pct": f1_pct,
                "analogy": f"The balance between precision and recall is {f1_pct}%. A high F1 means the model is both careful and thorough.",
                "when_useful": "Best single metric when you need both precision and recall to be good.",
                "color": "purple",
            })

        return {
            "metrics": explanations,
            "n_classes": len(class_names),
            "class_names": [str(c) for c in class_names],
            "n_samples": n_samples,
        }

    def _generate_sigmoid_data(self) -> Dict[str, Any]:
        """Pre-computed sigmoid curve points for visualization."""
        x_vals = np.linspace(-8, 8, 81).tolist()
        y_vals = [round(1.0 / (1.0 + np.exp(-x)), 4) for x in x_vals]

        # Key annotation points
        annotations = [
            {"x": -8, "y": round(1.0 / (1.0 + np.exp(8)), 4), "label": "Very unlikely (class 0)"},
            {"x": -2, "y": round(1.0 / (1.0 + np.exp(2)), 4), "label": "Leaning class 0"},
            {"x": 0, "y": 0.5, "label": "Decision boundary (50/50)"},
            {"x": 2, "y": round(1.0 / (1.0 + np.exp(-2)), 4), "label": "Leaning class 1"},
            {"x": 8, "y": round(1.0 / (1.0 + np.exp(-8)), 4), "label": "Very likely (class 1)"},
        ]

        return {
            "x_values": x_vals,
            "y_values": y_vals,
            "annotations": annotations,
            "x_label": "Model Output (z = w*x + b)",
            "y_label": "Probability of Class 1",
            "description": "The sigmoid function squashes any number into a probability between 0 and 1. This is how logistic regression converts raw scores into class predictions.",
            "formula": "P(class=1) = 1 / (1 + e^(-z))",
            "threshold_explanation": "If probability >= 0.5, predict class 1. Otherwise, predict class 0.",
        }

    def _generate_logistic_regression_quiz(
        self, metrics: Dict[str, Any], class_names: list,
        target_column: str, n_samples: int
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about logistic regression concepts."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: Classification vs regression
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What type of problem does logistic regression solve?",
            "options": [
                "Classification (predicting categories)",
                "Regression (predicting continuous numbers)",
                "Clustering (grouping similar items)",
                "Dimensionality reduction",
            ],
            "correct_answer": 0,
            "explanation": f"Despite its name, logistic regression is a classification algorithm. It predicts which category (class) a sample belongs to. Here it predicts '{target_column}' which has {len(class_names)} classes: {', '.join(str(c) for c in class_names[:5])}.",
            "difficulty": "easy",
        })

        # Q2: Sigmoid interpretation
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "The sigmoid function outputs 0.85. What does this mean?",
            "options": [
                "85% probability of belonging to the positive class",
                "The model is 85% accurate",
                "85% of the data was used for training",
                "The feature has 85% importance",
            ],
            "correct_answer": 0,
            "explanation": "The sigmoid function converts the model's raw score into a probability between 0 and 1. An output of 0.85 means the model is 85% confident the sample belongs to class 1 (positive class).",
            "difficulty": "medium",
        })

        # Q3: Based on actual metrics - accuracy interpretation
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            acc_pct = round(float(accuracy) * 100, 1)
            correct = int(round(float(accuracy) * n_samples))
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"The model accuracy is {round(float(accuracy), 4)} on {n_samples} training samples. How many did it classify correctly?",
                "options": [
                    f"About {correct} samples",
                    f"About {n_samples} samples",
                    f"About {n_samples - correct} samples",
                    f"About {int(n_samples * 0.5)} samples",
                ],
                "correct_answer": 0,
                "explanation": f"Accuracy = correct predictions / total predictions. With accuracy {round(float(accuracy), 4)} on {n_samples} samples: {round(float(accuracy), 4)} x {n_samples} = ~{correct} correct predictions.",
                "difficulty": "medium",
            })

        # Q4: Precision vs recall
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "A hospital wants to screen patients for a rare disease. Which metric matters most?",
            "options": [
                "Recall - catch as many sick patients as possible",
                "Precision - only flag patients who are definitely sick",
                "Accuracy - get the most overall correct predictions",
                "F1 Score - balance between precision and recall",
            ],
            "correct_answer": 0,
            "explanation": "For disease screening, missing a sick patient (false negative) is dangerous. High recall ensures we catch most actual cases, even if some healthy people get flagged for follow-up tests. Better safe than sorry!",
            "difficulty": "hard",
        })

        # Q5: Decision boundary
        q_id += 1
        n_cls = len(class_names)
        questions.append({
            "id": f"q{q_id}",
            "question": f"This model has {n_cls} classes. What does the 'decision boundary' do?",
            "options": [
                f"Separates the {n_cls} classes in the feature space",
                "Limits how many features the model can use",
                "Sets the maximum number of training iterations",
                "Defines the minimum sample size per class",
            ],
            "correct_answer": 0,
            "explanation": f"The decision boundary is an imaginary line (or surface) that separates different classes. Points on one side are predicted as one class, points on the other side as another. With {n_cls} classes, logistic regression creates boundaries between them.",
            "difficulty": "medium",
        })

        _random.shuffle(questions)
        return questions[:5]
