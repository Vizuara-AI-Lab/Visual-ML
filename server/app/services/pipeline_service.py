"""
ML Pipeline Service - Business logic for ML operations.
Handles caching, async operations, and background tasks.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
from functools import lru_cache
from app.ml.engine import pipeline_engine
from app.ml.algorithms.regression.linear_regression import LinearRegression
from app.ml.algorithms.classification.logistic_regression import LogisticRegression
from app.core.exceptions import ModelNotFoundError, PredictionError
from app.core.logging import logger, log_ml_operation
from app.core.config import settings


class ModelCache:
    """
    LRU cache for loaded models to avoid repeated disk I/O.
    Keeps N most recently used models in memory.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize model cache.

        Args:
            max_size: Maximum number of models to keep in memory
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}

    def get(self, model_path: str) -> Optional[Any]:
        """Get model from cache."""
        if model_path in self.cache:
            self.access_times[model_path] = datetime.utcnow()
            logger.debug(f"Model cache hit: {model_path}")
            return self.cache[model_path]
        return None

    def put(self, model_path: str, model: Any) -> None:
        """Put model in cache."""
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest_path = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_path]
            del self.access_times[oldest_path]
            logger.debug(f"Evicted model from cache: {oldest_path}")

        self.cache[model_path] = model
        self.access_times[model_path] = datetime.utcnow()
        logger.debug(f"Model cached: {model_path}")

    def clear(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Model cache cleared")


class MLPipelineService:
    """
    ML Pipeline Service - High-level ML operations.

    Features:
    - Node execution
    - Pipeline orchestration
    - Model training
    - Prediction with caching
    - Model management
    """

    def __init__(self):
        """Initialize service."""
        self.model_cache = ModelCache(max_size=settings.MODEL_CACHE_SIZE)

    async def execute_node(
        self, node_type: str, input_data: Dict[str, Any], dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a single pipeline node.

        Args:
            node_type: Type of node to execute
            input_data: Node input data
            dry_run: Validate without executing

        Returns:
            Node execution result
        """
        try:
            result = await pipeline_engine.execute_node(
                node_type=node_type, input_data=input_data, dry_run=dry_run
            )
            return result
        except Exception as e:
            logger.error(f"Node execution failed: {str(e)}", exc_info=True)
            raise

    async def execute_pipeline(
        self,
        pipeline: List[Dict[str, Any]],
        dry_run: bool = False,
        current_user: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a complete pipeline.

        Args:
            pipeline: List of node configurations
            dry_run: Validate without executing
            current_user: Current user context for authentication

        Returns:
            List of node results
        """
        try:
            results = await pipeline_engine.execute_pipeline(
                pipeline=pipeline, dry_run=dry_run, current_user=current_user
            )
            return results
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise

    async def train_model(
        self,
        dataset_path: str,
        target_column: str,
        algorithm: str,
        task_type: str,
        hyperparameters: Dict[str, Any],
        test_ratio: float = 0.2,
        val_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Train a model end-to-end.

        Args:
            dataset_path: Path to dataset
            target_column: Target column name
            algorithm: Algorithm to use
            task_type: Task type (regression/classification)
            hyperparameters: Model hyperparameters
            test_ratio: Test set ratio
            val_ratio: Validation set ratio

        Returns:
            Training results with model info
        """
        try:
            logger.info(f"Starting model training: {algorithm}")

            # Build pipeline
            pipeline = []

            # Split node
            split_input = {
                "dataset_path": dataset_path,
                "target_column": target_column,
                "train_ratio": 1 - test_ratio - (val_ratio or 0),
                "test_ratio": test_ratio,
                "val_ratio": val_ratio,
                "random_seed": settings.DEFAULT_RANDOM_SEED,
                "shuffle": True,
            }
            pipeline.append({"node_type": "split", "input": split_input})

            # Execute split
            split_result = await self.execute_node("split", split_input)

            # Train node
            train_input = {
                "train_dataset_path": split_result["train_path"],
                "target_column": target_column,
                "algorithm": algorithm,
                "task_type": task_type,
                "hyperparameters": hyperparameters,
            }
            train_result = await self.execute_node("train", train_input)

            # Evaluate node
            evaluate_input = {
                "model_path": train_result["model_path"],
                "test_dataset_path": split_result["test_path"],
                "target_column": target_column,
                "task_type": task_type,
            }
            eval_result = await self.execute_node("evaluate", evaluate_input)

            log_ml_operation(
                operation="model_training_complete",
                details={
                    "algorithm": algorithm,
                    "model_id": train_result["model_id"],
                    "test_metrics": eval_result["metrics"],
                },
                level="info",
            )

            return {
                "success": True,
                "model_id": train_result["model_id"],
                "model_path": train_result["model_path"],
                "model_version": train_result["model_version"],
                "training_metrics": train_result["training_metrics"],
                "test_metrics": eval_result["metrics"],
                "metadata": train_result["metadata"],
            }

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise

    def load_model(self, model_path: str, task_type: str) -> Any:
        """
        Load model with caching.

        Args:
            model_path: Path to model file
            task_type: Task type (regression/classification)

        Returns:
            Loaded model instance
        """
        # Check cache first
        cached_model = self.model_cache.get(model_path)
        if cached_model:
            return cached_model

        # Load from disk
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(model_path)

        try:
            if task_type == "regression":
                model = LinearRegression.load(model_path_obj)
            elif task_type == "classification":
                model = LogisticRegression.load(model_path_obj)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Cache model
            self.model_cache.put(model_path, model)

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise ModelNotFoundError(model_path)

    async def predict(
        self, model_path: str, features: Dict[str, Any], task_type: str
    ) -> Dict[str, Any]:
        """
        Make prediction with a trained model.

        Args:
            model_path: Path to trained model
            features: Feature dictionary
            task_type: Task type

        Returns:
            Prediction result
        """
        try:
            # Load model
            model = self.load_model(model_path, task_type)

            # Convert features to DataFrame
            df = pd.DataFrame([features])

            # Make prediction
            prediction = model.predict(df)

            result = {
                "prediction": (
                    prediction[0].tolist() if hasattr(prediction[0], "tolist") else prediction[0]
                ),
                "model_version": model.training_metadata.get("timestamp", "unknown"),
            }

            # Add probabilities for classification
            if task_type == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)
                result["probability"] = proba[0].tolist()

            log_ml_operation(
                operation="prediction",
                details={"model_path": model_path, "task_type": task_type},
                level="info",
            )

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise PredictionError(reason=str(e))

    async def predict_batch(
        self, model_path: str, features_list: List[Dict[str, Any]], task_type: str
    ) -> Dict[str, Any]:
        """
        Make batch predictions.

        Args:
            model_path: Path to trained model
            features_list: List of feature dictionaries
            task_type: Task type

        Returns:
            Batch prediction results
        """
        try:
            # Load model
            model = self.load_model(model_path, task_type)

            # Convert to DataFrame
            df = pd.DataFrame(features_list)

            # Make predictions
            predictions = model.predict(df)

            result = {
                "predictions": [
                    pred.tolist() if hasattr(pred, "tolist") else pred for pred in predictions
                ],
                "model_version": model.training_metadata.get("timestamp", "unknown"),
                "count": len(predictions),
            }

            # Add probabilities for classification
            if task_type == "classification" and hasattr(model, "predict_proba"):
                probas = model.predict_proba(df)
                result["probabilities"] = [proba.tolist() for proba in probas]

            log_ml_operation(
                operation="batch_prediction",
                details={
                    "model_path": model_path,
                    "batch_size": len(features_list),
                    "task_type": task_type,
                },
                level="info",
            )

            return result

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
            raise PredictionError(reason=str(e))

    def list_models(self, algorithm: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available models.

        Args:
            algorithm: Filter by algorithm (optional)

        Returns:
            List of model metadata
        """
        models = []
        model_dir = Path(settings.MODEL_ARTIFACTS_DIR)

        if not model_dir.exists():
            return models

        # Search for model files
        pattern = f"{algorithm}/**/*.joblib" if algorithm else "**/*.joblib"

        for model_file in model_dir.glob(pattern):
            try:
                stat = model_file.stat()
                models.append(
                    {
                        "model_path": str(model_file),
                        "algorithm": model_file.parent.name,
                        "model_id": model_file.stem,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get model info for {model_file}: {str(e)}")

        return sorted(models, key=lambda x: x["created_at"], reverse=True)

    def reload_model(self, model_path: str) -> bool:
        """
        Reload a model (clear from cache).

        Args:
            model_path: Path to model

        Returns:
            Success status
        """
        try:
            if model_path in self.model_cache.cache:
                del self.model_cache.cache[model_path]
                del self.model_cache.access_times[model_path]
                logger.info(f"Model reloaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to reload model: {str(e)}")
            return False


# Global service instance
ml_service = MLPipelineService()
