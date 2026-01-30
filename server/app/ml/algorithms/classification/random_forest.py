"""
Random Forest Classifier implementation with production-ready features.
Supports binary and multi-class classification with ensemble learning.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
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


class RandomForestClassifier:
    """
    Random Forest Classifier wrapper with production features.

    Features:
    - Scikit-learn backend
    - Ensemble of decision trees
    - Binary and multi-class support
    - Automatic validation
    - Model persistence
    - Feature importance from ensemble
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
    ):
        """
        Initialize Random Forest Classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required in leaf node
            criterion: Split quality measure ('gini' or 'entropy')
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs for parallel processing (-1 = all cores)
        """
        self.model = SKRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names: Optional[list] = None
        self.target_name: Optional[str] = None
        self.class_names: Optional[list] = None
        self.training_metadata: Dict[str, Any] = {}
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, Any]:
        """Train the random forest classifier."""
        try:
            logger.info(
                f"Training Random Forest Classifier with {len(X)} samples, {len(X.columns)} features"
            )

            if validate:
                validate_dataframe(X, min_rows=10)
                temp_df = X.copy()
                temp_df["__target__"] = y
                validate_target_column(temp_df, "__target__", task_type="classification")

            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, "name") else "target"
            self.class_names = sorted(y.unique().tolist())

            X_array = X.values
            y_array = y.values

            start_time = datetime.utcnow()
            self.model.fit(X_array, y_array)
            training_time = (datetime.utcnow() - start_time).total_seconds()

            y_pred = self.model.predict(X_array)
            metrics = self._calculate_metrics(y_array, y_pred)

            self.training_metadata = {
                "algorithm": "random_forest_classifier",
                "n_samples": len(X),
                "n_features": len(X.columns),
                "n_classes": len(self.class_names),
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "class_names": self.class_names,
                "training_time_seconds": training_time,
                "timestamp": datetime.utcnow().isoformat(),
                "n_estimators": self.model.n_estimators,
                "hyperparameters": {
                    "n_estimators": self.model.n_estimators,
                    "max_depth": self.model.max_depth,
                    "min_samples_split": self.model.min_samples_split,
                    "min_samples_leaf": self.model.min_samples_leaf,
                    "criterion": self.model.criterion,
                },
                "training_metrics": metrics,
            }

            self.is_trained = True

            log_ml_operation(
                operation="random_forest_classifier_training",
                details={
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "n_classes": len(self.class_names),
                    "n_estimators": self.model.n_estimators,
                    "training_time": training_time,
                    "accuracy": metrics["accuracy"],
                },
                level="info",
            )

            logger.info(
                f"Random Forest Classifier training completed - Accuracy: {metrics['accuracy']:.4f}"
            )

            return self.training_metadata

        except Exception as e:
            logger.error(f"Random Forest Classifier training failed: {str(e)}", exc_info=True)
            raise ModelTrainingError(
                algorithm="random_forest_classifier", reason=str(e), traceback=str(e.__traceback__)
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
                operation="random_forest_classifier_prediction",
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

        return metrics

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
    def load(cls, filepath: Path) -> "RandomForestClassifier":
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
        """Get feature importance scores from ensemble."""
        if not self.is_trained or self.feature_names is None:
            raise PredictionError("Model must be trained first")

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
