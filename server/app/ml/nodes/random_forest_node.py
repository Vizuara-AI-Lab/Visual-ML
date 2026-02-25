"""
Random Forest Node - Individual node wrapper for Random Forest algorithm.
Supports both classification and regression tasks with ensemble learning.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.ml.algorithms.classification.random_forest import RandomForestClassifier
from app.ml.algorithms.regression.random_forest_regressor import RandomForestRegressor
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class RandomForestInput(NodeInput):
    """Input schema for Random Forest node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: str = Field(..., description="Name of target column")
    task_type: str = Field(..., description="Task type: 'classification' or 'regression'")

    # Hyperparameters
    n_estimators: int = Field(100, description="Number of trees in the forest")
    max_depth: Optional[int] = Field(None, description="Maximum tree depth (None = unlimited)")
    min_samples_split: int = Field(2, description="Minimum samples required to split node")
    min_samples_leaf: int = Field(1, description="Minimum samples required in leaf node")
    random_state: int = Field(42, description="Random seed for reproducibility")


class RandomForestOutput(NodeOutput):
    """Output schema for Random Forest node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    task_type: str = Field(..., description="Task type used")
    n_estimators: int = Field(..., description="Number of trees in forest")

    training_metrics: Dict[str, Any] = Field(..., description="Training metrics")
    training_time_seconds: float = Field(..., description="Training duration")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")

    # Learning activity data (all Optional)
    feature_importance_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Feature importance analysis for visualization"
    )
    tree_comparison: Optional[Dict[str, Any]] = Field(
        None, description="Comparison of individual trees in the forest"
    )
    metric_explainer: Optional[Dict[str, Any]] = Field(
        None, description="Metric explanation cards with real values"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about random forests"
    )

    # Pass-through fields for downstream nodes (e.g., confusion_matrix)
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (passed from split node)"
    )
    target_column: Optional[str] = Field(None, description="Target column (passed from split node)")


class RandomForestNode(BaseNode):
    """
    Random Forest Node - Train random forest models for classification or regression.

    Responsibilities:
    - Load training dataset from database/S3
    - Train random forest model (classifier or regressor)
    - Calculate task-specific metrics
    - Save model artifacts
    - Return model metadata

    Production features:
    - Supports both classification and regression
    - Ensemble of decision trees
    - Configurable forest size and tree parameters
    - Feature importance from ensemble
    - Parallel training support
    - Model versioning
    """

    node_type = "random_forest"

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
                "n_estimators": "Number of trees in forest",
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
        return RandomForestInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return RandomForestOutput

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

    async def _execute(self, input_data: RandomForestInput) -> RandomForestOutput:
        """Execute random forest training."""
        try:
            logger.info(f"Training Random Forest model ({input_data.task_type})")

            # Load training data
            df_train = await self._load_dataset(input_data.train_dataset_id)
            if df_train is None or df_train.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.train_dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            # Validate target column
            if input_data.target_column not in df_train.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X_train = df_train.drop(columns=[input_data.target_column])
            y_train = df_train[input_data.target_column]

            # Initialize model based on task type
            if input_data.task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=input_data.n_estimators,
                    max_depth=input_data.max_depth,
                    min_samples_split=input_data.min_samples_split,
                    min_samples_leaf=input_data.min_samples_leaf,
                    random_state=input_data.random_state,
                )
            else:  # regression
                model = RandomForestRegressor(
                    n_estimators=input_data.n_estimators,
                    max_depth=input_data.max_depth,
                    min_samples_split=input_data.min_samples_split,
                    min_samples_leaf=input_data.min_samples_leaf,
                    random_state=input_data.random_state,
                )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id(f"model_random_forest_{input_data.task_type}")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "random_forest" / input_data.task_type
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename

            model.save(model_path)

            logger.info(f"Model saved to {model_path}")

            # Generate learning activity data (non-blocking)
            feature_names = X_train.columns.tolist()
            metrics = training_metadata.get("training_metrics", {})

            feature_importance_analysis = None
            try:
                feature_importance_analysis = self._generate_feature_importance_analysis(
                    training_metadata.get("feature_importances", [])
                )
            except Exception as e:
                logger.warning(f"Feature importance analysis generation failed: {e}")

            tree_comparison = None
            try:
                tree_comparison = self._generate_tree_comparison(
                    training_metadata.get("individual_trees", []),
                    input_data.n_estimators,
                )
            except Exception as e:
                logger.warning(f"Tree comparison generation failed: {e}")

            metric_explainer = None
            try:
                metric_explainer = self._generate_metric_explainer(
                    metrics, input_data.task_type,
                    len(X_train), input_data.target_column,
                )
            except Exception as e:
                logger.warning(f"Metric explainer generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_random_forest_quiz(
                    feature_names, metrics, input_data.task_type,
                    input_data.n_estimators, input_data.target_column,
                    training_metadata.get("feature_importances", []),
                )
            except Exception as e:
                logger.warning(f"Random forest quiz generation failed: {e}")

            return RandomForestOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=len(X_train.columns),
                task_type=input_data.task_type,
                n_estimators=input_data.n_estimators,
                training_metrics=metrics,
                training_time_seconds=training_time,
                metadata={
                    "target_column": input_data.target_column,
                    "feature_names": feature_names,
                    "hyperparameters": {
                        "n_estimators": input_data.n_estimators,
                        "max_depth": input_data.max_depth,
                        "min_samples_split": input_data.min_samples_split,
                        "min_samples_leaf": input_data.min_samples_leaf,
                        "random_state": input_data.random_state,
                    },
                    "full_training_metadata": training_metadata,
                },
                feature_importance_analysis=feature_importance_analysis,
                tree_comparison=tree_comparison,
                metric_explainer=metric_explainer,
                quiz_questions=quiz_questions,
                # Pass through split node fields for downstream nodes
                test_dataset_id=input_data.test_dataset_id,
                target_column=input_data.target_column,
            )

        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

    def _generate_feature_importance_analysis(
        self, feature_importances: list
    ) -> Dict[str, Any]:
        """Feature importance bar chart data."""
        if not feature_importances:
            return {"features": []}

        fi_list = list(feature_importances)  # already sorted from algorithm
        max_imp = fi_list[0]["importance"] if fi_list else 1

        for item in fi_list:
            item["bar_width_pct"] = round(item["importance"] / max_imp * 100, 1) if max_imp > 0 else 0
            item["interpretation"] = (
                f"'{item['feature']}' contributes {round(item['importance'] * 100, 1)}% "
                f"to the ensemble's predictions"
            )

        return {"features": fi_list, "total_features": len(fi_list)}

    def _generate_tree_comparison(
        self, individual_trees: list, n_estimators: int
    ) -> Dict[str, Any]:
        """Compare individual trees in the forest."""
        if not individual_trees:
            return {"trees": [], "summary": {}}

        depths = [t["depth"] for t in individual_trees]
        leaves = [t["n_leaves"] for t in individual_trees]

        return {
            "trees": individual_trees,
            "n_estimators_total": n_estimators,
            "n_trees_shown": len(individual_trees),
            "summary": {
                "avg_depth": round(sum(depths) / len(depths), 1),
                "min_depth": min(depths),
                "max_depth": max(depths),
                "avg_leaves": round(sum(leaves) / len(leaves), 1),
                "min_leaves": min(leaves),
                "max_leaves": max(leaves),
            },
            "description": (
                f"Showing {len(individual_trees)} of {n_estimators} trees. "
                f"Tree depths range from {min(depths)} to {max(depths)} "
                f"(avg: {round(sum(depths) / len(depths), 1)}). "
                f"Each tree sees a random subset of the data."
            ),
        }

    def _generate_metric_explainer(
        self, metrics: Dict, task_type: str, n_samples: int, target_column: str
    ) -> Dict[str, Any]:
        """Metric explanation cards with real values and analogies."""
        explanations = []

        if task_type == "classification":
            acc = metrics.get("accuracy")
            if acc is not None:
                correct = int(acc * n_samples)
                wrong = n_samples - correct
                explanations.append({
                    "metric": "Accuracy",
                    "value": round(float(acc), 4),
                    "value_pct": round(float(acc) * 100, 1),
                    "color": "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red",
                    "analogy": f"Out of {n_samples} predictions, {correct} were correct and {wrong} were wrong.",
                    "when_useful": "Good for balanced datasets where all classes are equally important.",
                })
            prec = metrics.get("precision")
            if prec is not None:
                explanations.append({
                    "metric": "Precision",
                    "value": round(float(prec), 4),
                    "value_pct": round(float(prec) * 100, 1),
                    "color": "blue",
                    "analogy": f"When the forest says 'positive', it's right {round(float(prec) * 100, 1)}% of the time.",
                    "when_useful": "Important when false positives are costly.",
                })
            rec = metrics.get("recall")
            if rec is not None:
                explanations.append({
                    "metric": "Recall",
                    "value": round(float(rec), 4),
                    "value_pct": round(float(rec) * 100, 1),
                    "color": "orange",
                    "analogy": f"Of all actual positives, the forest catches {round(float(rec) * 100, 1)}%.",
                    "when_useful": "Critical when missing positives is dangerous.",
                })
            f1 = metrics.get("f1")
            if f1 is not None:
                explanations.append({
                    "metric": "F1 Score",
                    "value": round(float(f1), 4),
                    "value_pct": round(float(f1) * 100, 1),
                    "color": "purple",
                    "analogy": "The harmonic mean of precision and recall — a single balanced score.",
                    "when_useful": "Best when you need a balance between precision and recall.",
                })
        else:
            r2 = metrics.get("r2")
            if r2 is not None:
                explanations.append({
                    "metric": "R² Score",
                    "value": round(float(r2), 4),
                    "value_pct": round(float(r2) * 100, 1),
                    "color": "green" if r2 >= 0.8 else "yellow" if r2 >= 0.5 else "red",
                    "analogy": f"The forest explains {round(float(r2) * 100, 1)}% of the variation in '{target_column}'.",
                    "when_useful": "Shows how well the model captures the overall pattern.",
                })
            mae = metrics.get("mae")
            if mae is not None:
                explanations.append({
                    "metric": "MAE",
                    "value": round(float(mae), 4),
                    "value_pct": None,
                    "color": "blue",
                    "analogy": f"On average, predictions are off by {round(float(mae), 2)} units.",
                    "when_useful": "Easy to interpret — the average error in the same units as the target.",
                })
            rmse = metrics.get("rmse")
            if rmse is not None:
                explanations.append({
                    "metric": "RMSE",
                    "value": round(float(rmse), 4),
                    "value_pct": None,
                    "color": "orange",
                    "analogy": f"Typical prediction error is about {round(float(rmse), 2)} units.",
                    "when_useful": "Better than MAE when large errors are especially bad.",
                })

        return {
            "metrics": explanations,
            "task_type": task_type,
            "n_samples": n_samples,
            "target_column": target_column,
        }

    def _generate_random_forest_quiz(
        self, feature_names: list, metrics: Dict, task_type: str,
        n_estimators: int, target_column: str,
        feature_importances: list,
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about random forests."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: What makes RF different from a single DT
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is the main advantage of Random Forest over a single Decision Tree?",
            "options": [
                "It combines many trees to reduce overfitting and improve generalization",
                "It trains faster because it uses fewer features",
                "It always produces a simpler model",
                "It doesn't need any hyperparameters",
            ],
            "correct_answer": 0,
            "explanation": f"Random Forest trains {n_estimators} trees, each on a random subset of data and features. By averaging their predictions (or majority voting), individual tree errors cancel out, producing a more robust model.",
            "difficulty": "easy",
        })

        # Q2: Bootstrap sampling
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is 'bootstrap sampling' in Random Forest?",
            "options": [
                "Each tree trains on a random sample (with replacement) from the training data",
                "The model removes duplicate rows before training",
                "Features are sorted alphabetically before training",
                "The dataset is split into exactly equal parts for each tree",
            ],
            "correct_answer": 0,
            "explanation": "Bootstrap sampling means each tree gets a random sample of the training data with replacement. Some data points appear multiple times, others not at all. This creates diversity among trees.",
            "difficulty": "medium",
        })

        # Q3: Most important feature
        if feature_importances and len(feature_importances) >= 2:
            strongest = feature_importances[0]  # already sorted
            others = [f["feature"] for f in feature_importances[1:4]]
            options = [strongest["feature"]] + others
            _random.shuffle(options)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": "Which feature is most important for this Random Forest's predictions?",
                "options": options,
                "correct_answer": options.index(strongest["feature"]),
                "explanation": f"'{strongest['feature']}' has the highest ensemble importance ({round(strongest['importance'] * 100, 1)}%). In RF, feature importance is averaged across all {n_estimators} trees.",
                "difficulty": "easy",
            })

        # Q4: Number of estimators
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This forest has {n_estimators} trees. What happens if we add more trees?",
            "options": [
                "Performance usually improves then plateaus, but training takes longer",
                "The model will always overfit",
                "Each tree becomes deeper",
                "The feature importance scores won't change at all",
            ],
            "correct_answer": 0,
            "explanation": f"More trees generally improve predictions up to a point. After that, adding trees only increases computation cost without significant accuracy gains. {n_estimators} trees is a common starting point.",
            "difficulty": "medium",
        })

        # Q5: How prediction works
        q_id += 1
        if task_type == "classification":
            questions.append({
                "id": f"q{q_id}",
                "question": f"How does a Random Forest of {n_estimators} trees make a classification prediction?",
                "options": [
                    f"Each tree votes for a class, and the class with the most votes wins (majority voting)",
                    "Only the most accurate tree's prediction is used",
                    "All trees must agree on the same class",
                    "The prediction comes from the first tree only",
                ],
                "correct_answer": 0,
                "explanation": f"In classification, each of the {n_estimators} trees independently predicts a class. The final prediction is the class that received the most votes — this is called majority voting.",
                "difficulty": "medium",
            })
        else:
            questions.append({
                "id": f"q{q_id}",
                "question": f"How does a Random Forest of {n_estimators} trees make a regression prediction?",
                "options": [
                    f"It averages the predictions from all {n_estimators} trees",
                    "It uses the prediction from the deepest tree",
                    "It takes the median of all predictions",
                    "Only the last tree's prediction is used",
                ],
                "correct_answer": 0,
                "explanation": f"In regression, each of the {n_estimators} trees independently predicts a value. The final prediction is the average of all these predictions, which tends to be more stable than any single tree.",
                "difficulty": "medium",
            })

        # Q6: Feature randomness
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why does Random Forest select a random subset of features at each split?",
            "options": [
                "To reduce correlation between trees and increase diversity",
                "To make the algorithm faster by using fewer features",
                "Because not all features fit in memory",
                "To remove unimportant features",
            ],
            "correct_answer": 0,
            "explanation": "If all trees used all features, they'd tend to make the same splits and be highly correlated. By randomly limiting features at each split, trees become diverse — their errors are less correlated, so averaging works better.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
