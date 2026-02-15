"""
Random Forest Regressor implementation with production-ready features.
Supports regression tasks with ensemble learning.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from datetime import datetime
from app.core.logging import logger, log_ml_operation
from app.core.exceptions import ModelTrainingError, PredictionError
from app.ml.node_validators import validate_dataframe, validate_target_column


class RandomForestRegressor:
    """
    Random Forest Regressor wrapper with production features.

    Features:
    - Scikit-learn backend
    - Ensemble of decision trees
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
        criterion: str = "squared_error",
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
    ):
        """
        Initialize Random Forest Regressor.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required in leaf node
            criterion: Split quality measure ('squared_error', 'absolute_error', 'friedman_mse', 'poisson')
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs for parallel processing (-1 = all cores)
        """
        self.model = SKRandomForestRegressor(
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
        self.training_metadata: Dict[str, Any] = {}
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, Any]:
        """Train the random forest regressor."""
        try:
            logger.info(
                f"Training Random Forest Regressor with {len(X)} samples, {len(X.columns)} features"
            )

            if validate:
                validate_dataframe(X, min_rows=10)
                temp_df = X.copy()
                temp_df["__target__"] = y
                validate_target_column(temp_df, "__target__", task_type="regression")

            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, "name") else "target"

            X_array = X.values
            y_array = y.values

            start_time = datetime.utcnow()
            self.model.fit(X_array, y_array)
            training_time = (datetime.utcnow() - start_time).total_seconds()

            y_pred = self.model.predict(X_array)
            metrics = self._calculate_metrics(y_array, y_pred)

            self.training_metadata = {
                "algorithm": "random_forest_regressor",
                "n_samples": len(X),
                "n_features": len(X.columns),
                "feature_names": self.feature_names,
                "target_name": self.target_name,
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
                operation="random_forest_regressor_training",
                details={
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "n_estimators": self.model.n_estimators,
                    "training_time": training_time,
                    "r2_score": metrics["r2"],
                },
                level="info",
            )

            logger.info(f"Random Forest Regressor training completed - R²: {metrics['r2']:.4f}")

            return self.training_metadata

        except Exception as e:
            error_msg = str(e).replace("{", "{{").replace("}", "}}")
            logger.error(f"Random Forest Regressor training failed: {error_msg}", exc_info=True)
            raise ModelTrainingError(
                algorithm="random_forest_regressor", reason=str(e), traceback=str(e.__traceback__)
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
                operation="random_forest_regressor_prediction",
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
    def load(cls, filepath: Path) -> "RandomForestRegressor":
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
        """Get feature importance scores from ensemble."""
        if not self.is_trained or self.feature_names is None:
            raise PredictionError("Model must be trained first")

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
