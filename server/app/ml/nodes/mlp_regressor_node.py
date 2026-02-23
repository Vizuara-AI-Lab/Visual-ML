"""
MLP Regressor Node — Pipeline node for MLP neural network regression.
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
from app.ml.algorithms.neural_network.mlp_regressor import MLPRegressor
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class MLPRegressorInput(NodeInput):
    """Input schema for MLP Regressor node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
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
    solver: str = Field("adam", description="Optimizer: adam, sgd, lbfgs")
    max_iter: int = Field(200, description="Maximum training iterations")
    learning_rate_init: float = Field(0.001, description="Initial learning rate")
    alpha: float = Field(0.0001, description="L2 regularization strength")
    batch_size: str = Field("auto", description="Mini-batch size ('auto' or integer)")

    # Early stopping
    early_stopping: bool = Field(True, description="Stop training when validation score plateaus")

    random_state: int = Field(42, description="Random seed for reproducibility")


class MLPRegressorOutput(NodeOutput):
    """Output schema for MLP Regressor node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")

    training_metrics: Dict[str, float] = Field(
        ..., description="Training metrics (MAE, MSE, RMSE, R²)"
    )
    training_time_seconds: float = Field(..., description="Training duration")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")

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
    loss_curve_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Loss curve analysis for learning"
    )
    prediction_playground: Optional[Dict[str, Any]] = Field(
        None, description="Data for interactive prediction playground"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about neural networks"
    )


class MLPRegressorNode(BaseNode):
    """
    MLP Regressor Node — Train neural network models for regression.

    Responsibilities:
    - Load training dataset from database/S3
    - Parse architecture configuration
    - Train MLP neural network
    - Calculate regression metrics
    - Save model artifacts
    - Return model metadata with loss curve for visualization
    """

    node_type = "mlp_regressor"

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
        return MLPRegressorInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return MLPRegressorOutput

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

    async def _execute(self, input_data: MLPRegressorInput) -> MLPRegressorOutput:
        """Execute MLP regressor training."""
        try:
            logger.info(f"Training MLP Regressor model")

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

            # Validate dataset
            if X_train.empty:
                raise ValueError("Training dataset is empty")

            # Ensure all features are numeric
            non_numeric = X_train.select_dtypes(exclude=["number"]).columns.tolist()
            if non_numeric:
                raise ValueError(
                    f"Non-numeric columns found: {non_numeric}. "
                    f"Please encode categorical variables before training."
                )

            # Parse architecture config
            hidden_layer_sizes = self._parse_hidden_layers(input_data.hidden_layer_sizes)
            batch_size = self._parse_batch_size(input_data.batch_size)

            logger.info(
                f"MLP Architecture: input={X_train.shape[1]} → "
                f"hidden={hidden_layer_sizes} → output=1"
            )

            # Initialize and train model
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=input_data.activation,
                solver=input_data.solver,
                max_iter=input_data.max_iter,
                learning_rate_init=input_data.learning_rate_init,
                alpha=input_data.alpha,
                batch_size=batch_size,
                early_stopping=input_data.early_stopping,
                random_state=input_data.random_state,
            )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id("model_mlp_regressor")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "mlp_regressor"
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename
            model.save(model_path)

            logger.info(f"MLP Regressor model saved to {model_path}")

            metrics = training_metadata.get("training_metrics", {})
            loss_curve = training_metadata.get("loss_curve", [])
            validation_scores = training_metadata.get("validation_scores", [])
            architecture = training_metadata.get("architecture", {})
            feature_names = X_train.columns.tolist()

            # Generate learning activity data (non-blocking)
            loss_curve_analysis = None
            try:
                loss_curve_analysis = self._generate_loss_curve_analysis(
                    loss_curve, validation_scores, training_metadata.get("n_iter", 0),
                    input_data.max_iter
                )
            except Exception as e:
                logger.warning(f"Loss curve analysis generation failed: {e}")

            prediction_playground = None
            try:
                prediction_playground = self._generate_prediction_playground(
                    feature_names, X_train
                )
            except Exception as e:
                logger.warning(f"Prediction playground generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_mlp_regressor_quiz(
                    metrics, input_data.target_column,
                    hidden_layer_sizes, input_data.activation, len(X_train),
                    architecture
                )
            except Exception as e:
                logger.warning(f"MLP quiz generation failed: {e}")

            return MLPRegressorOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=X_train.shape[1],
                training_metrics=metrics,
                training_time_seconds=training_time,
                metadata=model.training_metadata,
                loss_curve=loss_curve,
                validation_scores=validation_scores,
                architecture=architecture,
                n_iterations=training_metadata.get("n_iter", 0),
                loss_curve_analysis=loss_curve_analysis,
                prediction_playground=prediction_playground,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"MLP Regressor training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

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

        if n_iter < max_iter:
            analysis["convergence"] = "converged"
            analysis["convergence_message"] = (
                f"Training converged after {n_iter} iterations (max was {max_iter}). "
                f"The model found a good solution."
            )
        else:
            analysis["convergence"] = "hit_max_iter"
            analysis["convergence_message"] = (
                f"Training reached the maximum of {max_iter} iterations. "
                f"Consider increasing max_iter or adjusting learning rate."
            )

        return analysis

    def _generate_prediction_playground(
        self, feature_names: list, X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        """Data for interactive prediction exploration."""
        features = []
        for name in feature_names:
            if name in X_train.columns:
                col = X_train[name].dropna()
                if len(col) > 0:
                    features.append({
                        "name": name,
                        "min": round(float(col.min()), 4),
                        "max": round(float(col.max()), 4),
                        "mean": round(float(col.mean()), 4),
                        "step": round(float((col.max() - col.min()) / 50), 4) if col.max() != col.min() else 1,
                    })

        return {
            "features": features[:10],
            "note": "Neural networks learn non-linear relationships, so predictions may change dramatically with small input changes.",
        }

    def _generate_mlp_regressor_quiz(
        self, metrics: Dict[str, float], target_column: str,
        hidden_layer_sizes: tuple, activation: str, n_samples: int,
        architecture: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about neural network regression."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: MLP vs Linear Regression
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "How does MLP Regressor differ from Linear Regression?",
            "options": [
                "MLP can learn non-linear relationships through hidden layers and activation functions",
                "MLP always gives better results than Linear Regression",
                "MLP does not need any training data",
                "MLP can only handle categorical targets",
            ],
            "correct_answer": 0,
            "explanation": "Linear Regression fits a straight line/plane. MLP Regressor has hidden layers with activation functions that allow it to learn curves, waves, and complex non-linear patterns. However, it needs more data and computation than Linear Regression.",
            "difficulty": "medium",
        })

        # Q2: R² interpretation
        r2 = metrics.get("r2")
        if r2 is not None:
            r2_pct = round(float(r2) * 100, 1)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"The MLP Regressor achieved R² = {round(float(r2), 4)} on training data. What does this mean?",
                "options": [
                    f"The model explains {r2_pct}% of the variation in {target_column}",
                    f"The model is {r2_pct}% faster than linear regression",
                    f"The model uses {r2_pct}% of the features",
                    f"The model has {r2_pct}% of parameters trained",
                ],
                "correct_answer": 0,
                "explanation": f"R² measures how well the model explains the variability in the target. R²={round(float(r2), 4)} means {r2_pct}% of changes in '{target_column}' are captured by the model. R²=1.0 is perfect, R²=0 is no better than predicting the mean.",
                "difficulty": "medium",
            })

        # Q3: Architecture question
        total_params = architecture.get("total_params", 0)
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This network has {total_params} trainable parameters. What are these parameters?",
            "options": [
                "Weights and biases that the network adjusts during training",
                "The number of data points used for training",
                "The number of features in the dataset",
                "The number of iterations the model ran",
            ],
            "correct_answer": 0,
            "explanation": f"Parameters are the weights (connections between neurons) and biases that the network learns. With architecture {hidden_layer_sizes}, there are {total_params} values that get updated every iteration to minimize the loss function.",
            "difficulty": "medium",
        })

        # Q4: Loss function for regression
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What loss function does MLP Regressor typically minimize?",
            "options": [
                "Mean Squared Error (MSE) — average of squared prediction errors",
                "Cross-Entropy — measures probability distribution differences",
                "Hinge Loss — used for SVM classification",
                "Accuracy — percentage of correct predictions",
            ],
            "correct_answer": 0,
            "explanation": "For regression, neural networks minimize Mean Squared Error (MSE): the average of (actual - predicted)² across all samples. Squaring ensures both positive and negative errors contribute equally and penalizes large errors more.",
            "difficulty": "hard",
        })

        # Q5: Overfitting warning
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "If the training loss is very low but test performance is poor, what's happening?",
            "options": [
                "Overfitting — the model memorized training data instead of learning patterns",
                "Underfitting — the model is too simple",
                "The data is too clean",
                "The learning rate is too low",
            ],
            "correct_answer": 0,
            "explanation": "Overfitting occurs when the model performs well on training data but poorly on unseen data. Solutions include: reducing network size, adding regularization (alpha), enabling early stopping, or getting more training data.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
