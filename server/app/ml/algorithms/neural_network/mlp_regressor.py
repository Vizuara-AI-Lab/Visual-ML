"""
MLP Regressor — Multi-Layer Perceptron neural network for regression.
Scikit-learn backend with production-ready features.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor as SKMLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from datetime import datetime
from app.core.logging import logger, log_ml_operation
from app.core.exceptions import ModelTrainingError, PredictionError
from app.ml.node_validators import validate_dataframe, validate_target_column


class MLPRegressor:
    """
    MLP Regressor model wrapper with production features.

    Features:
    - Scikit-learn MLPRegressor backend
    - Configurable hidden layers, activation, solver
    - Loss curve tracking for visualization
    - Model persistence
    - Comprehensive regression metrics
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
        Initialize MLP Regressor.

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
        self.model = SKMLPRegressor(
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
        self.training_metadata: Dict[str, Any] = {}
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, Any]:
        """
        Train the MLP regression model.

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
                f"Training MLP Regressor with {len(X)} samples, {len(X.columns)} features"
            )

            if validate:
                validate_dataframe(X, min_rows=10)
                temp_df = X.copy()
                temp_df["__target__"] = y
                validate_target_column(temp_df, "__target__", task_type="regression")

            # Store feature names
            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, "name") else "target"

            # Convert to numpy for training — ensure no Python None values remain
            X_array = np.asarray(X.values, dtype=np.float64)
            y_array = np.asarray(y.values, dtype=np.float64)

            # Safety check: ensure no NaN in features or target
            nan_mask = np.isnan(X_array).any(axis=1) | np.isnan(y_array)
            if nan_mask.any():
                X_array = X_array[~nan_mask]
                y_array = y_array[~nan_mask]
                logger.warning(
                    f"Dropped {nan_mask.sum()} rows with NaN during final cleanup. "
                    f"{len(X_array)} samples remain."
                )

            # Train model
            start_time = datetime.utcnow()
            self.model.fit(X_array, y_array)
            training_time = (datetime.utcnow() - start_time).total_seconds()

            # Calculate training metrics
            y_pred = self.model.predict(X_array)
            metrics = self._calculate_metrics(y_array, y_pred)

            # Extract loss curve for visualization (guard against None values)
            loss_curve = []
            if hasattr(self.model, "loss_curve_") and self.model.loss_curve_:
                loss_curve = [float(l) for l in self.model.loss_curve_ if l is not None]

            # Extract validation scores if early_stopping was used
            validation_scores = []
            if hasattr(self.model, "validation_scores_") and self.model.validation_scores_:
                validation_scores = [float(s) for s in self.model.validation_scores_ if s is not None]

            # Build architecture description
            hidden_layers = list(self.model.hidden_layer_sizes) if hasattr(
                self.model.hidden_layer_sizes, '__iter__'
            ) else [self.model.hidden_layer_sizes]

            architecture = {
                "n_features": X.shape[1],
                "hidden_layers": hidden_layers,
                "n_outputs": 1,
                "activation": self.model.activation,
                "total_params": self._count_parameters(),
            }

            # Store metadata
            self.training_metadata = {
                "algorithm": "mlp_regressor",
                "n_samples": len(X),
                "n_features": len(X.columns),
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "training_time_seconds": training_time,
                "timestamp": datetime.utcnow().isoformat(),
                "n_iter": self.model.n_iter_,
                "best_loss": float(self.model.best_loss_) if hasattr(self.model, "best_loss_") and self.model.best_loss_ is not None else None,
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
                operation="mlp_regressor_training",
                details={
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "n_iter": self.model.n_iter_,
                    "training_time": training_time,
                    "r2": metrics["r2"],
                },
                level="info",
            )

            logger.info(
                f"MLP Regressor training completed — R²: {metrics['r2']:.4f}, "
                f"Iterations: {self.model.n_iter_}"
            )

            return self.training_metadata

        except Exception as e:
            error_msg = str(e).replace("{", "{{").replace("}", "}}")
            logger.error(f"MLP Regressor training failed: {error_msg}", exc_info=True)
            raise ModelTrainingError(
                algorithm="mlp_regressor", reason=str(e), traceback=str(e.__traceback__)
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
                operation="mlp_regressor_prediction",
                details={"n_samples": len(X)},
                level="info",
            )

            return predictions

        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise PredictionError(reason=str(e), expected_features=self.feature_names)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on test data."""
        predictions = self.predict(X)
        return self._calculate_metrics(y.values, predictions)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

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
            "training_metadata": self.training_metadata,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "MLPRegressor":
        """Load model from disk."""
        model_data = joblib.load(filepath)

        instance = cls()
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.target_name = model_data["target_name"]
        instance.training_metadata = model_data["training_metadata"]
        instance.is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Approximate feature importance using first-layer weights.
        """
        if not self.is_trained or self.feature_names is None:
            raise PredictionError("Model must be trained first")

        if not hasattr(self.model, "coefs_") or len(self.model.coefs_) == 0:
            raise PredictionError("No coefficients available")

        first_layer_weights = np.abs(self.model.coefs_[0])
        importance = first_layer_weights.mean(axis=1)

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
                "abs_importance": np.abs(importance),
            }
        ).sort_values("abs_importance", ascending=False)
