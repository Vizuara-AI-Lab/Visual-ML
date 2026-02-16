"""
Logistic Regression implementation with production-ready features.
Supports binary and multi-class classification.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
from pathlib import Path
from datetime import datetime
from app.core.logging import logger, log_ml_operation
from app.core.exceptions import ModelTrainingError, PredictionError
from app.ml.node_validators import validate_dataframe, validate_target_column


class LogisticRegression:
    """
    Logistic Regression model wrapper with production features.

    Features:
    - Scikit-learn backend
    - Binary and multi-class support
    - Automatic validation
    - Model persistence
    - Comprehensive metrics
    - Feature tracking
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        class_weight: Optional[str] = None,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize Logistic Regression model.

        Args:
            C: Inverse of regularization strength (smaller = stronger)
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            solver: Optimization algorithm
            max_iter: Maximum iterations for convergence
            tol: Tolerance for stopping criteria
            fit_intercept: Whether to calculate intercept
            class_weight: Weights for classes ('balanced' or None)
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs for computation
        """
        self.model = SKLogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names: Optional[list] = None
        self.target_name: Optional[str] = None
        self.class_names: Optional[list] = None
        self.training_metadata: Dict[str, Any] = {}
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, Any]:
        """
        Train the logistic regression model.

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
                f"Training Logistic Regression with {len(X)} samples, {len(X.columns)} features"
            )

            if validate:
                # Validate DataFrame
                validate_dataframe(X, min_rows=10)

                # Create temporary DataFrame for target validation
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

            # Store metadata
            self.training_metadata = {
                "algorithm": "logistic_regression",
                "n_samples": len(X),
                "n_features": len(X.columns),
                "n_classes": len(self.class_names),
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "class_names": self.class_names,
                "training_time_seconds": training_time,
                "timestamp": datetime.utcnow().isoformat(),
                "intercept": self.model.intercept_.tolist(),
                "n_iter": self.model.n_iter_.tolist() if hasattr(self.model, "n_iter_") else None,
                "hyperparameters": {
                    "C": self.model.C,
                    "penalty": self.model.penalty,
                    "solver": self.model.solver,
                    "max_iter": self.model.max_iter,
                    "fit_intercept": self.model.fit_intercept,
                    "class_weight": self.model.class_weight,
                },
                "training_metrics": metrics,
            }

            self.is_trained = True

            log_ml_operation(
                operation="logistic_regression_training",
                details={
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "n_classes": len(self.class_names),
                    "training_time": training_time,
                    "accuracy": metrics["accuracy"],
                },
                level="info",
            )

            logger.info(
                f"Logistic Regression training completed - Accuracy: {metrics['accuracy']:.4f}"
            )

            return self.training_metadata

        except Exception as e:
            error_msg = str(e).replace("{", "{{").replace("}", "}}")
            logger.error(f"Logistic Regression training failed: {error_msg}", exc_info=True)
            raise ModelTrainingError(
                algorithm="logistic_regression", reason=str(e), traceback=str(e.__traceback__)
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
                operation="logistic_regression_prediction",
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
        """
        Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Numpy array of class probabilities
        """
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

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        # Determine if binary or multi-class
        is_binary = len(np.unique(y_true)) == 2
        average = "binary" if is_binary else "weighted"

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "confusion_matrix": cm.tolist(),
        }

        # Add per-class metrics for multi-class
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
            "class_names": self.class_names,
            "training_metadata": self.training_metadata,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "LogisticRegression":
        """
        Load model from disk.

        Args:
            filepath: Path to load model from

        Returns:
            Loaded LogisticRegression instance
        """
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
        Get feature importance (coefficients).
        For multi-class, returns coefficients for each class.

        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_trained or self.feature_names is None:
            raise PredictionError("Model must be trained first")

        if len(self.model.coef_.shape) == 1:
            # Binary classification
            return pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "coefficient": self.model.coef_,
                    "abs_coefficient": np.abs(self.model.coef_),
                }
            ).sort_values("abs_coefficient", ascending=False)
        else:
            # Multi-class classification
            importance_data = []
            for i, class_name in enumerate(self.class_names):
                for j, feature_name in enumerate(self.feature_names):
                    importance_data.append(
                        {
                            "class": class_name,
                            "feature": feature_name,
                            "coefficient": self.model.coef_[i, j],
                            "abs_coefficient": abs(self.model.coef_[i, j]),
                        }
                    )

            return pd.DataFrame(importance_data).sort_values("abs_coefficient", ascending=False)
