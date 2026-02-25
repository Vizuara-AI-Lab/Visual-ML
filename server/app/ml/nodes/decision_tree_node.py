"""
Decision Tree Node - Individual node wrapper for Decision Tree algorithm.
Supports both classification and regression tasks with configurable hyperparameters.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.ml.algorithms.classification.decision_tree import DecisionTreeClassifier
from app.ml.algorithms.regression.decision_tree_regressor import DecisionTreeRegressor
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class DecisionTreeInput(NodeInput):
    """Input schema for Decision Tree node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: str = Field(..., description="Name of target column")
    task_type: str = Field(..., description="Task type: 'classification' or 'regression'")

    # Hyperparameters
    max_depth: Optional[int] = Field(None, description="Maximum tree depth (None = unlimited)")
    min_samples_split: int = Field(2, description="Minimum samples required to split node")
    min_samples_leaf: int = Field(1, description="Minimum samples required in leaf node")
    random_state: int = Field(42, description="Random seed for reproducibility")


class DecisionTreeOutput(NodeOutput):
    """Output schema for Decision Tree node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    task_type: str = Field(..., description="Task type used")

    training_metrics: Dict[str, Any] = Field(..., description="Training metrics")
    training_time_seconds: float = Field(..., description="Training duration")

    tree_depth: int = Field(..., description="Actual tree depth")
    n_leaves: int = Field(..., description="Number of leaf nodes")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")

    # Learning activity data (all Optional)
    feature_importance_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Feature importance analysis for visualization"
    )
    decision_path_example: Optional[Dict[str, Any]] = Field(
        None, description="Example decision path through the tree"
    )
    split_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Analysis of splitting criteria"
    )
    metric_explainer: Optional[Dict[str, Any]] = Field(
        None, description="Metric explanation cards with real values"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about decision trees"
    )

    # Pass-through fields for downstream nodes (e.g., confusion_matrix)
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (passed from split node)"
    )
    target_column: Optional[str] = Field(None, description="Target column (passed from split node)")


class DecisionTreeNode(BaseNode):
    """
    Decision Tree Node - Train decision tree models for classification or regression.

    Responsibilities:
    - Load training dataset from database/S3
    - Train decision tree model (classifier or regressor)
    - Calculate task-specific metrics
    - Save model artifacts
    - Return model metadata

    Production features:
    - Supports both classification and regression
    - Configurable tree depth and splitting
    - Feature importance extraction
    - Model versioning
    """

    node_type = "decision_tree"

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
                "tree_depth": "Actual tree depth",
                "n_leaves": "Number of leaf nodes",
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
        return DecisionTreeInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return DecisionTreeOutput

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

    async def _execute(self, input_data: DecisionTreeInput) -> DecisionTreeOutput:
        """Execute decision tree training."""
        try:
            logger.info(f"Training Decision Tree model ({input_data.task_type})")

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
                model = DecisionTreeClassifier(
                    max_depth=input_data.max_depth,
                    min_samples_split=input_data.min_samples_split,
                    min_samples_leaf=input_data.min_samples_leaf,
                    random_state=input_data.random_state,
                )
            else:  # regression
                model = DecisionTreeRegressor(
                    max_depth=input_data.max_depth,
                    min_samples_split=input_data.min_samples_split,
                    min_samples_leaf=input_data.min_samples_leaf,
                    random_state=input_data.random_state,
                )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id(f"model_decision_tree_{input_data.task_type}")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "decision_tree" / input_data.task_type
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
                    training_metadata.get("feature_importances", []),
                    feature_names,
                )
            except Exception as e:
                logger.warning(f"Feature importance analysis generation failed: {e}")

            decision_path_example = None
            try:
                decision_path_example = self._generate_decision_path_example(
                    training_metadata.get("tree_structure", []),
                    input_data.task_type,
                )
            except Exception as e:
                logger.warning(f"Decision path example generation failed: {e}")

            split_analysis = None
            try:
                split_analysis = self._generate_split_analysis(
                    training_metadata.get("tree_structure", []),
                    input_data.task_type,
                )
            except Exception as e:
                logger.warning(f"Split analysis generation failed: {e}")

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
                quiz_questions = self._generate_decision_tree_quiz(
                    feature_names, metrics, input_data.task_type,
                    training_metadata.get("tree_depth", 0),
                    training_metadata.get("n_leaves", 0),
                    input_data.target_column,
                    training_metadata.get("feature_importances", []),
                )
            except Exception as e:
                logger.warning(f"Decision tree quiz generation failed: {e}")

            return DecisionTreeOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=len(X_train.columns),
                task_type=input_data.task_type,
                training_metrics=metrics,
                training_time_seconds=training_time,
                tree_depth=training_metadata.get("tree_depth", 0),
                n_leaves=training_metadata.get("n_leaves", 0),
                metadata={
                    "target_column": input_data.target_column,
                    "feature_names": feature_names,
                    "hyperparameters": {
                        "max_depth": input_data.max_depth,
                        "min_samples_split": input_data.min_samples_split,
                        "min_samples_leaf": input_data.min_samples_leaf,
                        "random_state": input_data.random_state,
                    },
                    "full_training_metadata": training_metadata,
                },
                feature_importance_analysis=feature_importance_analysis,
                decision_path_example=decision_path_example,
                split_analysis=split_analysis,
                metric_explainer=metric_explainer,
                quiz_questions=quiz_questions,
                # Pass through split node fields for downstream nodes
                test_dataset_id=input_data.test_dataset_id,
                target_column=input_data.target_column,
            )

        except Exception as e:
            logger.error(f"Decision Tree training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

    def _generate_feature_importance_analysis(
        self, feature_importances: list, feature_names: list
    ) -> Dict[str, Any]:
        """Feature importance bar chart data."""
        if not feature_importances:
            return {"features": []}

        fi_list = []
        if isinstance(feature_importances[0], dict):
            # Already structured
            fi_list = feature_importances
        else:
            for name, imp in zip(feature_names, feature_importances):
                fi_list.append({"feature": name, "importance": round(float(imp), 4)})

        fi_list.sort(key=lambda x: x["importance"], reverse=True)
        max_imp = fi_list[0]["importance"] if fi_list else 1

        for item in fi_list:
            item["bar_width_pct"] = round(item["importance"] / max_imp * 100, 1) if max_imp > 0 else 0
            item["interpretation"] = (
                f"'{item['feature']}' contributes {round(item['importance'] * 100, 1)}% "
                f"to the tree's splitting decisions"
            )

        return {"features": fi_list, "total_features": len(fi_list)}

    def _generate_decision_path_example(
        self, tree_structure: list, task_type: str
    ) -> Dict[str, Any]:
        """Generate an example decision path through the tree."""
        if not tree_structure:
            return {"steps": []}

        # Build a path from root to the first leaf (leftmost path)
        by_id = {n["id"]: n for n in tree_structure}
        steps = []
        current_id = tree_structure[0]["id"] if tree_structure else None

        while current_id is not None and current_id in by_id:
            node = by_id[current_id]
            if node["type"] == "internal":
                steps.append({
                    "node_id": node["id"],
                    "type": "split",
                    "question": f"Is {node.get('feature', '?')} < {node.get('threshold', '?')}?",
                    "feature": node.get("feature", "?"),
                    "threshold": node.get("threshold", 0),
                    "impurity": node.get("impurity", 0),
                    "samples": node.get("n_samples", 0),
                    "answer": "Yes (go left)",
                })
                current_id = node.get("left_child")
            else:
                label = node.get("class_label", node.get("value", "?"))
                steps.append({
                    "node_id": node["id"],
                    "type": "prediction",
                    "prediction": str(label),
                    "samples": node.get("n_samples", 0),
                    "confidence": "Based on the training samples reaching this leaf",
                })
                break

        return {
            "steps": steps,
            "task_type": task_type,
            "description": "Follow the path from root to leaf to see how the tree makes a decision",
        }

    def _generate_split_analysis(
        self, tree_structure: list, task_type: str
    ) -> Dict[str, Any]:
        """Analyze splitting criteria at each level."""
        if not tree_structure:
            return {"levels": []}

        internal_nodes = [n for n in tree_structure if n["type"] == "internal"]
        by_depth = {}
        for n in internal_nodes:
            d = n["depth"]
            if d not in by_depth:
                by_depth[d] = []
            by_depth[d].append(n)

        levels = []
        for depth in sorted(by_depth.keys()):
            nodes_at_level = by_depth[depth]
            features_used = list(set(n.get("feature", "?") for n in nodes_at_level))
            avg_impurity = sum(n.get("impurity", 0) for n in nodes_at_level) / len(nodes_at_level)
            levels.append({
                "depth": depth,
                "n_splits": len(nodes_at_level),
                "features_used": features_used,
                "avg_impurity": round(avg_impurity, 3),
                "description": (
                    f"Level {depth}: {len(nodes_at_level)} split(s) using "
                    f"{', '.join(features_used)} "
                    f"(avg {'Gini' if task_type == 'classification' else 'MSE'} = {round(avg_impurity, 3)})"
                ),
            })

        leaf_nodes = [n for n in tree_structure if n["type"] == "leaf"]
        return {
            "levels": levels,
            "total_internal_nodes": len(internal_nodes),
            "total_leaf_nodes": len(leaf_nodes),
            "impurity_measure": "Gini" if task_type == "classification" else "MSE",
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
                    "analogy": f"When the tree says 'positive', it's right {round(float(prec) * 100, 1)}% of the time.",
                    "when_useful": "Important when false positives are costly (e.g., spam filtering).",
                })
            rec = metrics.get("recall")
            if rec is not None:
                explanations.append({
                    "metric": "Recall",
                    "value": round(float(rec), 4),
                    "value_pct": round(float(rec) * 100, 1),
                    "color": "orange",
                    "analogy": f"Of all actual positives, the tree catches {round(float(rec) * 100, 1)}%.",
                    "when_useful": "Critical when missing positives is dangerous (e.g., disease detection).",
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
                    "analogy": f"The tree explains {round(float(r2) * 100, 1)}% of the variation in '{target_column}'.",
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
                    "analogy": f"Typical prediction error is about {round(float(rmse), 2)} units (penalizes large errors more).",
                    "when_useful": "Better than MAE when large errors are especially bad.",
                })

        return {
            "metrics": explanations,
            "task_type": task_type,
            "n_samples": n_samples,
            "target_column": target_column,
        }

    def _generate_decision_tree_quiz(
        self, feature_names: list, metrics: Dict, task_type: str,
        tree_depth: int, n_leaves: int, target_column: str,
        feature_importances: list,
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about decision trees."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: What is a decision tree
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does a decision tree do at each internal node?",
            "options": [
                "Asks a yes/no question about a feature to split the data",
                "Randomly assigns data points to groups",
                "Calculates the average of all features",
                "Removes outliers from the dataset",
            ],
            "correct_answer": 0,
            "explanation": "Each internal node in a decision tree asks a question like 'Is feature X < threshold?' and splits the data into two groups based on the answer. This continues until the data is pure or a stopping condition is met.",
            "difficulty": "easy",
        })

        # Q2: Most important feature
        fi_sorted = []
        if feature_importances:
            if isinstance(feature_importances[0], dict):
                fi_sorted = sorted(feature_importances, key=lambda x: x.get("importance", 0), reverse=True)
            else:
                fi_sorted = [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, feature_importances)]
                fi_sorted.sort(key=lambda x: x["importance"], reverse=True)

        if len(fi_sorted) >= 2:
            strongest = fi_sorted[0]["feature"]
            others = [f["feature"] for f in fi_sorted[1:4]]
            options = [strongest] + others
            _random.shuffle(options)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": "Which feature is most important for this decision tree's predictions?",
                "options": options,
                "correct_answer": options.index(strongest),
                "explanation": f"'{strongest}' has the highest feature importance score ({round(fi_sorted[0]['importance'] * 100, 1)}%), meaning it contributes most to the tree's splitting decisions.",
                "difficulty": "easy",
            })

        # Q3: Gini impurity
        if task_type == "classification":
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": "What does a Gini impurity of 0 mean at a tree node?",
                "options": [
                    "All samples at that node belong to the same class (pure node)",
                    "The node has no data",
                    "The model has 0% accuracy",
                    "The feature has no importance",
                ],
                "correct_answer": 0,
                "explanation": "Gini impurity measures how mixed the classes are. A Gini of 0 means all samples belong to one class — a perfectly pure node. A Gini of 0.5 (for binary) means an equal mix of both classes.",
                "difficulty": "medium",
            })

        # Q4: Tree depth
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This tree has depth {tree_depth} and {n_leaves} leaves. What happens if we increase the maximum depth?",
            "options": [
                "The tree may become more complex and risk overfitting",
                "The tree will always become more accurate",
                "The tree will have fewer leaves",
                "The training will be faster",
            ],
            "correct_answer": 0,
            "explanation": f"A deeper tree can learn more specific patterns, but risks memorizing the training data (overfitting). With depth {tree_depth} and {n_leaves} leaves, increasing depth would add more splits and potentially more leaves.",
            "difficulty": "medium",
        })

        # Q5: Overfitting
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "How can you prevent a decision tree from overfitting?",
            "options": [
                "Limit max_depth, increase min_samples_split, or prune the tree",
                "Train on more features",
                "Increase the tree depth to maximum",
                "Remove the target column",
            ],
            "correct_answer": 0,
            "explanation": "Overfitting happens when the tree is too complex. Setting max_depth, min_samples_split, or min_samples_leaf limits the tree's complexity. Pruning removes branches that don't improve generalization.",
            "difficulty": "hard",
        })

        # Q6: Classification metric
        if task_type == "classification":
            acc = metrics.get("accuracy")
            if acc is not None:
                acc_pct = round(float(acc) * 100, 1)
                q_id += 1
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"The training accuracy is {acc_pct}%. What does this tell us?",
                    "options": [
                        f"The tree correctly classified {acc_pct}% of training samples",
                        f"The tree will have {acc_pct}% accuracy on new data",
                        f"Only {acc_pct}% of features were used",
                        f"The tree has {acc_pct}% confidence",
                    ],
                    "correct_answer": 0,
                    "explanation": f"Training accuracy ({acc_pct}%) shows how well the tree fits the training data. Test accuracy (on unseen data) may be lower due to overfitting. Always evaluate on a separate test set.",
                    "difficulty": "medium",
                })

        _random.shuffle(questions)
        return questions[:5]
