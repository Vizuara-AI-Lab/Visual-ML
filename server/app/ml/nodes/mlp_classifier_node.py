"""
MLP Classifier Node — Pipeline node for MLP neural network classification.
Supports configurable architecture, training, and model persistence.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.ml.algorithms.neural_network.mlp_classifier import MLPClassifier
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class MLPClassifierInput(NodeInput):
    """Input schema for MLP Classifier node."""

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

    # Architecture
    hidden_layer_sizes: str = Field(
        "100", description="Hidden layer sizes, comma-separated (e.g. '100, 50, 25')"
    )
    activation: str = Field("relu", description="Activation function: relu, tanh, logistic")

    # Training
    solver: Optional[str] = Field("adam", description="Optimizer: adam, sgd, lbfgs")
    max_iter: Optional[int] = Field(200, description="Maximum training iterations")
    learning_rate_init: Optional[float] = Field(0.001, description="Initial learning rate")
    alpha: Optional[float] = Field(0.0001, description="L2 regularization strength")
    batch_size: Optional[str] = Field("auto", description="Mini-batch size ('auto' or integer)")

    # Early stopping
    early_stopping: Optional[bool] = Field(True, description="Stop training when validation score plateaus")

    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")


class MLPClassifierOutput(NodeOutput):
    """Output schema for MLP Classifier node."""

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

    # Pass-through fields for downstream nodes (confusion_matrix, etc.)
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (passed from split node)"
    )
    target_column: Optional[str] = Field(None, description="Target column (passed from split node)")

    # Neural network specific
    loss_curve: Optional[List[float]] = Field(
        None, description="Training loss over iterations for visualization"
    )
    validation_scores: Optional[List[float]] = Field(
        None, description="Validation scores per iteration"
    )
    architecture: Optional[Dict[str, Any]] = Field(
        None, description="Network architecture details"
    )
    n_iterations: Optional[int] = Field(None, description="Actual number of training iterations")

    # Learning activity data
    class_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Training data class distribution"
    )
    loss_curve_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Loss curve analysis for learning"
    )
    metric_explainer: Optional[Dict[str, Any]] = Field(
        None, description="Metric explanation cards with analogies"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about neural networks"
    )


class MLPClassifierNode(BaseNode):
    """
    MLP Classifier Node — Train neural network models for classification.

    Responsibilities:
    - Load training dataset from database/S3
    - Parse architecture configuration
    - Train MLP neural network
    - Calculate classification metrics
    - Save model artifacts
    - Return model metadata with loss curve for visualization
    """

    node_type = "mlp_classifier"

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
                "loss_curve": "Training loss over iterations",
                "architecture": "Network architecture details",
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
        return MLPClassifierInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return MLPClassifierOutput

    def _parse_hidden_layers(self, hidden_str: str) -> tuple:
        """Parse hidden layer sizes from string like '100, 50, 25' → (100, 50, 25)."""
        try:
            parts = [int(x.strip()) for x in hidden_str.split(",") if x.strip()]
            if not parts:
                return (100,)
            for p in parts:
                if p <= 0:
                    raise ValueError(f"Layer size must be positive, got {p}")
            return tuple(parts)
        except ValueError as e:
            logger.warning(f"Invalid hidden_layer_sizes '{hidden_str}', using default (100,): {e}")
            return (100,)

    def _parse_batch_size(self, batch_str: str):
        """Parse batch_size — 'auto' or integer."""
        if batch_str.strip().lower() == "auto":
            return "auto"
        try:
            val = int(batch_str.strip())
            return max(1, val)
        except ValueError:
            return "auto"

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
            if upload_path.exists():
                logger.info(f"Loading dataset from uploads folder: {upload_path}")
                df = pd.read_csv(upload_path, na_values=missing_values, keep_default_na=True)
                return df

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

    async def _execute(self, input_data: MLPClassifierInput) -> MLPClassifierOutput:
        """Execute MLP classifier training."""
        try:
            logger.info(f"Training MLP Classifier model")

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

            # Validate classification target
            n_unique = y_train.nunique()
            unique_values_sample = y_train.unique()[:10].tolist()

            logger.info(f"Target column '{input_data.target_column}' has {n_unique} unique values")

            if n_unique > 20:
                raise NodeExecutionError(
                    node_type=self.node_type,
                    reason=f"Target column '{input_data.target_column}' has {n_unique} unique values, which is too many for classification. MLP Classifier is for categorical targets. For continuous targets, use MLP Regressor instead. Sample values: {unique_values_sample}",
                    input_data=input_data.model_dump()
                )

            if n_unique == 1:
                raise NodeExecutionError(
                    node_type=self.node_type,
                    reason=f"Target column '{input_data.target_column}' has only 1 unique value. Cannot train a classifier with a single class.",
                    input_data=input_data.model_dump()
                )

            # Ensure all features are numeric
            non_numeric = X_train.select_dtypes(exclude=["number"]).columns.tolist()
            if non_numeric:
                raise ValueError(
                    f"Non-numeric columns found: {non_numeric}. "
                    f"Please encode categorical variables before training."
                )

            # Handle NaN/None values — MLP requires clean numeric data
            nan_count = int(X_train.isna().sum().sum())
            if nan_count > 0:
                logger.warning(
                    f"Found {nan_count} NaN values in features. Dropping rows with NaN."
                )
                combined = pd.concat([X_train, y_train], axis=1)
                combined = combined.dropna()
                X_train = combined.drop(columns=[input_data.target_column])
                y_train = combined[input_data.target_column]
                logger.info(f"After dropping NaN: {len(X_train)} samples remain")

                if len(X_train) < 10:
                    raise ValueError(
                        f"Only {len(X_train)} samples remain after removing NaN values. "
                        f"Need at least 10. Please handle missing values before training."
                    )

            # Ensure numeric dtype (convert nullable Int64/Float64 to standard numpy types)
            X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(np.float64)

            # Re-check for NaN after numeric conversion (to_numeric can introduce new NaN)
            new_nan_count = int(X_train.isna().sum().sum())
            if new_nan_count > 0:
                logger.warning(
                    f"Found {new_nan_count} new NaN values after numeric conversion. Dropping rows."
                )
                combined = pd.concat([X_train, y_train], axis=1)
                combined = combined.dropna()
                X_train = combined.drop(columns=[input_data.target_column])
                y_train = combined[input_data.target_column]
                logger.info(f"After post-conversion cleanup: {len(X_train)} samples remain")

                if len(X_train) < 10:
                    raise ValueError(
                        f"Only {len(X_train)} samples remain after cleaning. Need at least 10."
                    )

            # Ensure y_train has no None/NaN values
            y_train = y_train.dropna()

            # Parse architecture config with None-safe defaults
            hidden_layer_sizes = self._parse_hidden_layers(input_data.hidden_layer_sizes or "100")
            batch_size = self._parse_batch_size(input_data.batch_size or "auto")
            activation = input_data.activation or "relu"
            solver = input_data.solver or "adam"
            max_iter = input_data.max_iter if input_data.max_iter is not None else 200
            learning_rate_init = input_data.learning_rate_init if input_data.learning_rate_init is not None else 0.001
            alpha = input_data.alpha if input_data.alpha is not None else 0.0001
            early_stopping = input_data.early_stopping if input_data.early_stopping is not None else True
            random_state = input_data.random_state if input_data.random_state is not None else 42

            logger.info(
                f"MLP Architecture: input={X_train.shape[1]} → "
                f"hidden={hidden_layer_sizes} → output={n_unique} classes"
            )

            # Initialize and train model
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                alpha=alpha,
                batch_size=batch_size,
                early_stopping=early_stopping,
                random_state=random_state,
            )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=False)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id("model_mlp_classifier")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "mlp_classifier"
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename
            model.save(model_path)

            logger.info(f"MLP Classifier model saved to {model_path}")

            metrics = training_metadata.get("training_metrics", {})
            class_names_list = model.class_names
            loss_curve = training_metadata.get("loss_curve", [])
            validation_scores = training_metadata.get("validation_scores", [])
            architecture = training_metadata.get("architecture", {})

            # Generate learning activity data (non-blocking)
            class_distribution = None
            try:
                class_distribution = self._generate_class_distribution(
                    y_train, class_names_list, input_data.target_column
                )
            except Exception as e:
                logger.warning(f"Class distribution generation failed: {e}")

            loss_curve_analysis = None
            try:
                loss_curve_analysis = self._generate_loss_curve_analysis(
                    loss_curve, validation_scores, training_metadata.get("n_iter", 0),
                    input_data.max_iter
                )
            except Exception as e:
                logger.warning(f"Loss curve analysis generation failed: {e}")

            metric_explainer = None
            try:
                metric_explainer = self._generate_metric_explainer(
                    metrics, "classification", len(X_train), input_data.target_column
                )
            except Exception as e:
                logger.warning(f"Metric explainer generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_mlp_quiz(
                    metrics, class_names_list, input_data.target_column,
                    hidden_layer_sizes, input_data.activation, len(X_train),
                    architecture
                )
            except Exception as e:
                logger.warning(f"MLP quiz generation failed: {e}")

            return MLPClassifierOutput(
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
                loss_curve=loss_curve,
                validation_scores=validation_scores,
                architecture=architecture,
                n_iterations=training_metadata.get("n_iter", 0),
                class_distribution=class_distribution,
                loss_curve_analysis=loss_curve_analysis,
                metric_explainer=metric_explainer,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"MLP Classifier training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

    def _generate_metric_explainer(
        self, metrics: Dict, task_type: str, n_samples: int, target_column: str
    ) -> Dict[str, Any]:
        """Metric explanation cards with real values and analogies."""
        explanations = []

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
                "analogy": f"When the neural network says 'positive', it's right {round(float(prec) * 100, 1)}% of the time.",
                "when_useful": "Important when false positives are costly (e.g., spam detection).",
            })
        rec = metrics.get("recall")
        if rec is not None:
            explanations.append({
                "metric": "Recall",
                "value": round(float(rec), 4),
                "value_pct": round(float(rec) * 100, 1),
                "color": "orange",
                "analogy": f"Of all actual positives, the neural network catches {round(float(rec) * 100, 1)}%.",
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

        return {
            "metrics": explanations,
            "task_type": task_type,
            "n_samples": n_samples,
            "target_column": target_column,
        }

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
                "Classes are roughly balanced — good for training!"
                if is_balanced
                else f"Classes are imbalanced (ratio {imbalance_ratio}:1). Consider class_weight='balanced' or oversampling."
            ),
        }

    def _generate_loss_curve_analysis(
        self, loss_curve: list, validation_scores: list, n_iter: int, max_iter: int
    ) -> Dict[str, Any]:
        """Analyze loss curve for learning insights."""
        if not loss_curve:
            return {"has_data": False}

        analysis = {
            "has_data": True,
            "loss_values": loss_curve,
            "validation_scores": validation_scores,
            "n_iterations": n_iter,
            "max_iterations": max_iter,
            "initial_loss": round(loss_curve[0], 6) if loss_curve else None,
            "final_loss": round(loss_curve[-1], 6) if loss_curve else None,
            "loss_reduction_pct": round(
                (1 - loss_curve[-1] / loss_curve[0]) * 100, 1
            ) if loss_curve and loss_curve[0] > 0 else 0,
        }

        # Convergence analysis
        if n_iter < max_iter:
            analysis["convergence"] = "converged"
            analysis["convergence_message"] = (
                f"Training converged after {n_iter} iterations (max was {max_iter}). "
                f"The model found a good solution before reaching the iteration limit."
            )
        else:
            analysis["convergence"] = "hit_max_iter"
            analysis["convergence_message"] = (
                f"Training reached the maximum of {max_iter} iterations. "
                f"Consider increasing max_iter or adjusting learning rate."
            )

        return analysis

    def _generate_mlp_quiz(
        self, metrics: Dict[str, Any], class_names: list, target_column: str,
        hidden_layer_sizes: tuple, activation: str, n_samples: int,
        architecture: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about neural networks."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: What is an MLP?
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does MLP stand for in machine learning?",
            "options": [
                "Multi-Layer Perceptron",
                "Maximum Likelihood Predictor",
                "Machine Learning Pipeline",
                "Multi-Linear Processor",
            ],
            "correct_answer": 0,
            "explanation": "MLP stands for Multi-Layer Perceptron — a type of neural network with one or more hidden layers of neurons between input and output.",
            "difficulty": "easy",
        })

        # Q2: Architecture interpretation
        n_layers = len(hidden_layer_sizes)
        total_params = architecture.get("total_params", 0)
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This model has {n_layers} hidden layer(s) with sizes {hidden_layer_sizes}. What does this mean?",
            "options": [
                f"The network has {n_layers} layer(s) of neurons between input and output",
                f"The network trains for {hidden_layer_sizes[0]} iterations",
                f"The network uses {n_layers} different datasets",
                f"The network has {hidden_layer_sizes[0]} output classes",
            ],
            "correct_answer": 0,
            "explanation": f"Hidden layers are layers of neurons between the input features and the output prediction. This network has {n_layers} hidden layer(s) with {hidden_layer_sizes} neurons, giving it {total_params} trainable parameters.",
            "difficulty": "medium",
        })

        # Q3: Activation function
        act_descriptions = {
            "relu": "outputs the input directly if positive, zero otherwise (max(0, x))",
            "tanh": "squashes values between -1 and 1",
            "logistic": "squashes values between 0 and 1 (sigmoid)",
        }
        act_desc = act_descriptions.get(activation, "transforms each neuron's output")
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This model uses '{activation}' activation. What does an activation function do?",
            "options": [
                "Introduces non-linearity so the network can learn complex patterns",
                "Determines how many neurons to use",
                "Controls the learning rate",
                "Splits the data into train and test sets",
            ],
            "correct_answer": 0,
            "explanation": f"Activation functions introduce non-linearity. Without them, stacking layers would be equivalent to a single linear layer. The '{activation}' function {act_desc}.",
            "difficulty": "medium",
        })

        # Q4: Why scaling matters
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why is feature scaling (normalization) important for neural networks?",
            "options": [
                "Different scales cause uneven gradient updates, making training unstable",
                "Neural networks can only process integers",
                "Scaling increases the dataset size",
                "It reduces the number of features",
            ],
            "correct_answer": 0,
            "explanation": "Neural networks learn through gradient-based optimization. If features have very different scales (e.g., age 0-100 vs salary 30000-200000), gradients become unbalanced and training converges slowly or not at all. Always scale features before neural network training!",
            "difficulty": "hard",
        })

        # Q5: Accuracy interpretation
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            acc_pct = round(float(accuracy) * 100, 1)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"The MLP achieved {acc_pct}% accuracy on {n_samples} training samples with {len(class_names)} classes. Is this a good result?",
                "options": [
                    f"Compare with a baseline: random guessing gives {round(100/len(class_names), 1)}% — the MLP must beat this",
                    f"Any accuracy above 50% is always good",
                    f"Accuracy doesn't matter for neural networks",
                    f"{acc_pct}% means the model memorized {acc_pct}% of the data",
                ],
                "correct_answer": 0,
                "explanation": f"With {len(class_names)} classes, random guessing gives {round(100/len(class_names), 1)}% accuracy. The MLP achieved {acc_pct}%, which is {'much better' if accuracy > 1/len(class_names) * 1.5 else 'slightly better'} than random. But be careful — high training accuracy might indicate overfitting. Always check on test data!",
                "difficulty": "hard",
            })

        _random.shuffle(questions)
        return questions[:5]
