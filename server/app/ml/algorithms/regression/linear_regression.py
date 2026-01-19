"""
Linear Regression implementation with production-ready features.
Supports training, evaluation, and persistence.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from datetime import datetime
from app.core.logging import logger, log_ml_operation
from app.core.exceptions import ModelTrainingError, PredictionError
from app.ml.node_validators import validate_dataframe, validate_target_column


class LinearRegression:
    """
    Linear Regression model wrapper with production features.

    Features:
    - Scikit-learn backend
    - Automatic validation
    - Model persistence
    - Comprehensive metrics
    - Feature tracking
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize Linear Regression model.

        Args:
            fit_intercept: Whether to calculate intercept
            copy_X: Whether to copy X (if False, may be overwritten)
            n_jobs: Number of jobs for computation (None = 1, -1 = all cores)
        """
        self.model = SKLinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
        )
        self.feature_names: Optional[list] = None
        self.target_name: Optional[str] = None
        self.training_metadata: Dict[str, Any] = {}
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, Any]:
        """
        Train the linear regression model.

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
                f"Training Linear Regression with {len(X)} samples, {len(X.columns)} features"
            )

            if validate:
                # Validate DataFrame
                validate_dataframe(X, min_rows=10)

                # Create temporary DataFrame for target validation
                temp_df = X.copy()
                temp_df["__target__"] = y
                validate_target_column(temp_df, "__target__", task_type="regression")

            # Store feature names
            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, "name") else "target"

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

            # Store metadata
            self.training_metadata = {
                "algorithm": "linear_regression",
                "n_samples": len(X),
                "n_features": len(X.columns),
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "training_time_seconds": training_time,
                "timestamp": datetime.utcnow().isoformat(),
                "coefficients": self.model.coef_.tolist(),
                "intercept": float(self.model.intercept_),
                "hyperparameters": {
                    "fit_intercept": self.model.fit_intercept,
                },
                "training_metrics": metrics,
            }

            self.is_trained = True

            log_ml_operation(
                operation="linear_regression_training",
                details={
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "training_time": training_time,
                    "r2_score": metrics["r2"],
                },
                level="info",
            )

            logger.info(f"Linear Regression training completed - R²: {metrics['r2']:.4f}")

            return self.training_metadata

        except Exception as e:
            logger.error(f"Linear Regression training failed: {str(e)}", exc_info=True)
            raise ModelTrainingError(
                algorithm="linear_regression", reason=str(e), traceback=str(e.__traceback__)
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Feature DataFrame

        Returns:
            Numpy array of predictions

        Raises:
            PredictionError: If prediction fails
        """
        if not self.is_trained:
            raise PredictionError("Model is not trained yet")

        try:
            # Validate features
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

                # Reorder columns to match training
                X = X[self.feature_names]

            # Make prediction
            predictions = self.model.predict(X.values)

            log_ml_operation(
                operation="linear_regression_prediction",
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
        """
        Evaluate model on test data.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        return self._calculate_metrics(y.values, predictions)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def save(self, filepath: Path) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
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
    def load(cls, filepath: Path) -> "LinearRegression":
        """
        Load model from disk.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded LinearRegression instance
        """
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
        Get feature importance (coefficients).

        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_trained or self.feature_names is None:
            raise PredictionError("Model must be trained first")

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": self.model.coef_,
                "abs_coefficient": np.abs(self.model.coef_),
            }
        ).sort_values("abs_coefficient", ascending=False)
