"""
MLP Classifier — Multi-Layer Perceptron neural network for classification.
Scikit-learn backend with production-ready features.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier as SKMLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib
from pathlib import Path
from datetime import datetime
from app.core.logging import logger, log_ml_operation
from app.core.exceptions import ModelTrainingError, PredictionError
from app.ml.node_validators import validate_dataframe, validate_target_column


class MLPClassifier:
    """
    MLP Classifier model wrapper with production features.

    Features:
    - Scikit-learn MLPClassifier backend
    - Configurable hidden layers, activation, solver
    - Loss curve tracking for visualization
    - Binary and multi-class support
    - Model persistence
    - Comprehensive classification metrics
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = "relu",
        solver: str = "adam",
        max_iter: int = 200,
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        batch_size: str = "auto",
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        random_state: Optional[int] = 42,
    ):
        """
        Initialize MLP Classifier.

        Args:
            hidden_layer_sizes: Tuple of hidden layer neuron counts, e.g. (100, 50, 25)
            activation: Activation function — 'relu', 'tanh', 'logistic'
            solver: Optimizer — 'adam', 'sgd', 'lbfgs'
            max_iter: Maximum training iterations
            learning_rate_init: Initial learning rate (for adam/sgd)
            alpha: L2 regularization strength
            batch_size: Mini-batch size ('auto' or int)
            early_stopping: Stop training when validation score stops improving
            validation_fraction: Fraction of training data for validation (if early_stopping)
            n_iter_no_change: Iterations with no improvement before stopping
            random_state: Random seed for reproducibility
        """
        self.model = SKMLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            batch_size=batch_size,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
        )
        self.feature_names: Optional[list] = None
        self.target_name: Optional[str] = None
        self.class_names: Optional[list] = None
        self.training_metadata: Dict[str, Any] = {}
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, Any]:
        """
        Train the MLP classification model.

        Args:
            X: Feature DataFrame
            y: Target Series
            validate: Whether to validate input data

        Returns:
            Training metadata and metrics

        Raises:
            ModelTrainingError: If training fails
        """
        try:
            logger.info(
                f"Training MLP Classifier with {len(X)} samples, {len(X.columns)} features"
            )

            if validate:
                validate_dataframe(X, min_rows=10)
                temp_df = X.copy()
                temp_df["__target__"] = y
                validate_target_column(temp_df, "__target__", task_type="classification")

            # Store feature names
            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, "name") else "target"
            self.class_names = sorted(y.unique().tolist())

            # Convert to numpy for training
            X_array = X.values
            y_array = y.values

            # Train model
            start_time = datetime.utcnow()
            self.model.fit(X_array, y_array)
            training_time = (datetime.utcnow() - start_time).total_seconds()

            # Calculate training metrics
            y_pred = self.model.predict(X_array)
            metrics = self._calculate_metrics(y_array, y_pred)

            # Extract loss curve for visualization
            loss_curve = []
            if hasattr(self.model, "loss_curve_") and self.model.loss_curve_:
                loss_curve = [float(l) for l in self.model.loss_curve_]

            # Extract validation scores if early_stopping was used
            validation_scores = []
            if hasattr(self.model, "validation_scores_") and self.model.validation_scores_:
                validation_scores = [float(s) for s in self.model.validation_scores_]

            # Build architecture description
            hidden_layers = list(self.model.hidden_layer_sizes) if hasattr(
                self.model.hidden_layer_sizes, '__iter__'
            ) else [self.model.hidden_layer_sizes]

            architecture = {
                "n_features": X.shape[1],
                "hidden_layers": hidden_layers,
                "n_classes": len(self.class_names),
                "activation": self.model.activation,
                "solver": self.model.out_activation_ if hasattr(self.model, "out_activation_") else "softmax",
                "total_params": self._count_parameters(),
            }

            # Store metadata
            self.training_metadata = {
                "algorithm": "mlp_classifier",
                "n_samples": len(X),
                "n_features": len(X.columns),
                "n_classes": len(self.class_names),
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "class_names": self.class_names,
                "training_time_seconds": training_time,
                "timestamp": datetime.utcnow().isoformat(),
                "n_iter": self.model.n_iter_,
                "best_loss": float(self.model.best_loss_) if hasattr(self.model, "best_loss_") else None,
                "hyperparameters": {
                    "hidden_layer_sizes": hidden_layers,
                    "activation": self.model.activation,
                    "solver": self.model.solver,
                    "max_iter": self.model.max_iter,
                    "learning_rate_init": self.model.learning_rate_init,
                    "alpha": self.model.alpha,
                    "early_stopping": self.model.early_stopping,
                },
                "training_metrics": metrics,
                "loss_curve": loss_curve,
                "validation_scores": validation_scores,
                "architecture": architecture,
            }

            self.is_trained = True

            log_ml_operation(
                operation="mlp_classifier_training",
                details={
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "n_classes": len(self.class_names),
                    "n_iter": self.model.n_iter_,
                    "training_time": training_time,
                    "accuracy": metrics["accuracy"],
                },
                level="info",
            )

            logger.info(
                f"MLP Classifier training completed — Accuracy: {metrics['accuracy']:.4f}, "
                f"Iterations: {self.model.n_iter_}"
            )

            return self.training_metadata

        except Exception as e:
            error_msg = str(e).replace("{", "{{").replace("}", "}}")
            logger.error(f"MLP Classifier training failed: {error_msg}", exc_info=True)
            raise ModelTrainingError(
                algorithm="mlp_classifier", reason=str(e), traceback=str(e.__traceback__)
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.is_trained:
            raise PredictionError("Model is not trained yet")

        try:
            if self.feature_names:
                if set(X.columns) != set(self.feature_names):
                    missing = set(self.feature_names) - set(X.columns)
                    extra = set(X.columns) - set(self.feature_names)
                    error_msg = []
                    if missing:
                        error_msg.append(f"Missing features: {missing}")
                    if extra:
                        error_msg.append(f"Extra features: {extra}")
                    raise PredictionError(
                        reason=" | ".join(error_msg), expected_features=self.feature_names
                    )
                X = X[self.feature_names]

            predictions = self.model.predict(X.values)

            log_ml_operation(
                operation="mlp_classifier_prediction",
                details={"n_samples": len(X)},
                level="info",
            )

            return predictions

        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise PredictionError(reason=str(e), expected_features=self.feature_names)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise PredictionError("Model is not trained yet")

        try:
            if self.feature_names:
                X = X[self.feature_names]
            return self.model.predict_proba(X.values)

        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}", exc_info=True)
            raise PredictionError(reason=str(e))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model on test data."""
        predictions = self.predict(X)
        return self._calculate_metrics(y.values, predictions)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        is_binary = len(np.unique(y_true)) == 2
        average = "binary" if is_binary else "weighted"

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "confusion_matrix": cm.tolist(),
        }

        if not is_binary and self.class_names:
            metrics["per_class_metrics"] = {}
            for i, class_name in enumerate(self.class_names):
                metrics["per_class_metrics"][str(class_name)] = {
                    "precision": float(
                        precision_score(y_true == class_name, y_pred == class_name, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_true == class_name, y_pred == class_name, zero_division=0)
                    ),
                    "f1": float(
                        f1_score(y_true == class_name, y_pred == class_name, zero_division=0)
                    ),
                }

        return metrics

    def _count_parameters(self) -> int:
        """Count total trainable parameters in the network."""
        total = 0
        if hasattr(self.model, "coefs_"):
            for coef in self.model.coefs_:
                total += coef.size
            for intercept in self.model.intercepts_:
                total += intercept.size
        return total

    def save(self, filepath: Path) -> None:
        """Save model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "class_names": self.class_names,
            "training_metadata": self.training_metadata,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "MLPClassifier":
        """Load model from disk."""
        model_data = joblib.load(filepath)

        instance = cls()
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.target_name = model_data["target_name"]
        instance.class_names = model_data["class_names"]
        instance.training_metadata = model_data["training_metadata"]
        instance.is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Approximate feature importance using first-layer weights.
        For MLP, uses magnitude of incoming weights to each feature.
        """
        if not self.is_trained or self.feature_names is None:
            raise PredictionError("Model must be trained first")

        if not hasattr(self.model, "coefs_") or len(self.model.coefs_) == 0:
            raise PredictionError("No coefficients available")

        # Use mean absolute weight from input→first hidden layer
        first_layer_weights = np.abs(self.model.coefs_[0])
        importance = first_layer_weights.mean(axis=1)

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
                "abs_importance": np.abs(importance),
            }
        ).sort_values("abs_importance", ascending=False)
