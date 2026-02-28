"""
KNN Node - Individual node wrapper for K-Nearest Neighbors algorithm.
Supports both classification and regression tasks with instance-based learning.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
import random as _random
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class KNNInput(NodeInput):
    """Input schema for KNN node."""

    model_config = {"extra": "ignore"}

    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: str = Field(..., description="Name of target column")
    task_type: str = Field("classification", description="Task type: 'classification' or 'regression'")

    # Hyperparameters
    n_neighbors: int = Field(5, ge=1, le=50, description="Number of neighbors (K)")
    weights: str = Field("uniform", description="Weight function: 'uniform' or 'distance'")
    metric: str = Field("euclidean", description="Distance metric")
    p: int = Field(2, ge=1, le=5, description="Power parameter for Minkowski metric")
    show_advanced_options: bool = Field(False, description="Show advanced options in UI")


class KNNOutput(NodeOutput):
    """Output schema for KNN node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    task_type: str = Field(..., description="Task type used")
    n_neighbors: int = Field(..., description="Number of neighbors (K)")
    weights: str = Field(..., description="Weight function used")
    distance_metric: str = Field(..., description="Distance metric used")

    training_metrics: Dict[str, Any] = Field(..., description="Training metrics")
    training_time_seconds: float = Field(..., description="Training duration")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")

    # Learning activity data (all Optional)
    k_sweep_data: Optional[List[Dict[str, Any]]] = Field(
        None, description="K sweep analysis showing accuracy vs K"
    )
    decision_boundary_grid: Optional[Dict[str, Any]] = Field(
        None, description="Decision boundary visualization data"
    )
    sample_neighbors: Optional[List[Dict[str, Any]]] = Field(
        None, description="Sample neighbor analysis for test points"
    )
    metric_explainer: Optional[Dict[str, Any]] = Field(
        None, description="Metric explanation cards with real values"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about KNN"
    )

    # Pass-through fields for downstream nodes
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (passed from split node)"
    )
    target_column: Optional[str] = Field(None, description="Target column (passed from split node)")


class KNNNode(BaseNode):
    """
    KNN Node - Train K-Nearest Neighbors models for classification or regression.

    Responsibilities:
    - Load training dataset from database/S3
    - Train KNN model (classifier or regressor)
    - Calculate task-specific metrics
    - Save model artifacts
    - Generate educational visualizations (K sweep, decision boundary, neighbor analysis)
    - Return model metadata

    Production features:
    - Supports both classification and regression
    - Configurable K, weights, and distance metric
    - K sweep analysis for hyperparameter insight
    - Decision boundary visualization (PCA-projected)
    - Sample neighbor analysis
    - Model versioning
    """

    node_type = "knn"

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
                "n_neighbors": "Number of neighbors (K)",
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
        return KNNInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return KNNOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            # FIRST: Try to load from uploads folder (for train/test datasets from split node)
            upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
            if upload_path.exists():
                logger.info(f"Loading dataset from uploads folder: {upload_path}")
                return pd.read_csv(upload_path, na_values=missing_values, keep_default_na=True)

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

    async def _execute(self, input_data: KNNInput) -> KNNOutput:
        """Execute KNN training."""
        try:
            import joblib
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                r2_score, mean_absolute_error, mean_squared_error,
            )

            logger.info(f"Training KNN model ({input_data.task_type}, K={input_data.n_neighbors})")

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

            # Select only numeric columns
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                raise ValueError("No numeric feature columns found in the dataset")
            X_train = X_train[numeric_cols]

            # Handle missing values - drop rows where X or y is NaN
            valid_mask = X_train.notna().all(axis=1) & y_train.notna()
            X_train = X_train[valid_mask].reset_index(drop=True)
            y_train = y_train[valid_mask].reset_index(drop=True)

            if len(X_train) == 0:
                raise ValueError("No valid training samples after removing missing values")

            # Load test data if available
            X_test = None
            y_test = None
            if input_data.test_dataset_id:
                try:
                    df_test = await self._load_dataset(input_data.test_dataset_id)
                    if df_test is not None and not df_test.empty and input_data.target_column in df_test.columns:
                        X_test_raw = df_test.drop(columns=[input_data.target_column])
                        y_test_raw = df_test[input_data.target_column]
                        # Use same numeric columns
                        available_cols = [c for c in numeric_cols if c in X_test_raw.columns]
                        X_test_raw = X_test_raw[available_cols]
                        valid_test = X_test_raw.notna().all(axis=1) & y_test_raw.notna()
                        X_test = X_test_raw[valid_test].reset_index(drop=True)
                        y_test = y_test_raw[valid_test].reset_index(drop=True)
                        # Fill any missing test cols with 0 (in case train had cols test doesn't)
                        for col in numeric_cols:
                            if col not in X_test.columns:
                                X_test[col] = 0
                        X_test = X_test[numeric_cols]
                        logger.info(f"Loaded test dataset with {len(X_test)} samples")
                except Exception as e:
                    logger.warning(f"Could not load test data: {e}")
                    X_test = None
                    y_test = None

            # Create KNN model
            if input_data.task_type == "classification":
                model = KNeighborsClassifier(
                    n_neighbors=min(input_data.n_neighbors, len(X_train)),
                    weights=input_data.weights,
                    metric=input_data.metric,
                    p=input_data.p,
                    n_jobs=-1,
                )
            else:
                model = KNeighborsRegressor(
                    n_neighbors=min(input_data.n_neighbors, len(X_train)),
                    weights=input_data.weights,
                    metric=input_data.metric,
                    p=input_data.p,
                    n_jobs=-1,
                )

            # Train (fit) the model
            training_start = datetime.utcnow()
            model.fit(X_train, y_train)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Evaluate metrics
            # Use test data if available, otherwise use train data
            if X_test is not None and y_test is not None and len(X_test) > 0:
                X_eval, y_eval = X_test, y_test
                evaluated_on = "test"
            else:
                X_eval, y_eval = X_train, y_train
                evaluated_on = "train"

            y_pred = model.predict(X_eval)
            metrics: Dict[str, Any] = {}

            if input_data.task_type == "classification":
                metrics["accuracy"] = float(accuracy_score(y_eval, y_pred))
                avg_method = "weighted" if len(np.unique(y_eval)) > 2 else "binary"
                try:
                    metrics["precision"] = float(precision_score(y_eval, y_pred, average=avg_method, zero_division=0))
                    metrics["recall"] = float(recall_score(y_eval, y_pred, average=avg_method, zero_division=0))
                    metrics["f1"] = float(f1_score(y_eval, y_pred, average=avg_method, zero_division=0))
                except Exception:
                    metrics["precision"] = float(precision_score(y_eval, y_pred, average="weighted", zero_division=0))
                    metrics["recall"] = float(recall_score(y_eval, y_pred, average="weighted", zero_division=0))
                    metrics["f1"] = float(f1_score(y_eval, y_pred, average="weighted", zero_division=0))
            else:
                metrics["r2"] = float(r2_score(y_eval, y_pred))
                metrics["mae"] = float(mean_absolute_error(y_eval, y_pred))
                metrics["rmse"] = float(np.sqrt(mean_squared_error(y_eval, y_pred)))

            # Generate model ID and save
            model_id = generate_id(f"model_knn_{input_data.task_type}")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "knn" / input_data.task_type
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename

            joblib.dump(model, model_path)
            logger.info(f"KNN model saved to {model_path}")

            # Feature names for learning activities
            feature_names = numeric_cols

            # --- Generate learning activity data (each in try/except) ---

            k_sweep_data = None
            try:
                k_sweep_data = self._generate_k_sweep(
                    X_train, y_train, X_test, y_test,
                    input_data.task_type, input_data.n_neighbors,
                    input_data.weights, input_data.metric, input_data.p,
                )
            except Exception as e:
                logger.warning(f"K sweep generation failed: {e}")

            decision_boundary_grid = None
            try:
                if input_data.task_type == "classification":
                    decision_boundary_grid = self._generate_decision_boundary(
                        X_train, y_train, input_data.n_neighbors,
                        input_data.weights, input_data.metric, input_data.p,
                    )
            except Exception as e:
                logger.warning(f"Decision boundary generation failed: {e}")

            sample_neighbors = None
            try:
                sample_neighbors = self._generate_sample_neighbors(
                    model, X_train, y_train, X_eval, y_eval,
                    input_data.task_type, feature_names,
                )
            except Exception as e:
                logger.warning(f"Sample neighbors generation failed: {e}")

            metric_explainer = None
            try:
                eval_sample_count = len(X_eval)
                metric_explainer = self._generate_metric_explainer(
                    metrics, input_data.task_type,
                    eval_sample_count, input_data.target_column,
                    evaluated_on=evaluated_on,
                )
            except Exception as e:
                logger.warning(f"Metric explainer generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_knn_quiz(
                    feature_names, metrics, input_data.task_type,
                    input_data.n_neighbors, input_data.weights,
                    input_data.metric, len(X_train.columns),
                )
            except Exception as e:
                logger.warning(f"KNN quiz generation failed: {e}")

            return KNNOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=len(numeric_cols),
                task_type=input_data.task_type,
                n_neighbors=min(input_data.n_neighbors, len(X_train)),
                weights=input_data.weights,
                distance_metric=input_data.metric,
                training_metrics=metrics,
                training_time_seconds=training_time,
                metadata={
                    "target_column": input_data.target_column,
                    "feature_names": feature_names,
                    "hyperparameters": {
                        "n_neighbors": input_data.n_neighbors,
                        "weights": input_data.weights,
                        "metric": input_data.metric,
                        "p": input_data.p,
                    },
                    "evaluated_on": evaluated_on,
                    "numeric_columns_used": numeric_cols,
                },
                k_sweep_data=k_sweep_data,
                decision_boundary_grid=decision_boundary_grid,
                sample_neighbors=sample_neighbors,
                metric_explainer=metric_explainer,
                quiz_questions=quiz_questions,
                test_dataset_id=input_data.test_dataset_id,
                target_column=input_data.target_column,
            )

        except Exception as e:
            logger.error(f"KNN training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

    def _generate_k_sweep(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        task_type: str,
        current_k: int,
        weights: str,
        metric: str,
        p: int,
    ) -> List[Dict[str, Any]]:
        """
        Run K=1..min(30, n_samples//2) with the same weights/metric/p.
        Record train_score + test_score for each K. Mark is_current_k and is_best_k.
        If n_train > 2000, subsample for speed.
        """
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.metrics import accuracy_score, r2_score

        n_train = len(X_train)
        max_k = min(30, max(1, n_train // 2))

        # Subsample if dataset is large
        if n_train > 2000:
            idx = np.random.RandomState(42).choice(n_train, 2000, replace=False)
            X_tr_sub = X_train.iloc[idx].reset_index(drop=True)
            y_tr_sub = y_train.iloc[idx].reset_index(drop=True)
        else:
            X_tr_sub = X_train
            y_tr_sub = y_train

        # Prepare test data
        has_test = X_test is not None and y_test is not None and len(X_test) > 0
        if has_test and len(X_test) > 2000:
            idx_t = np.random.RandomState(42).choice(len(X_test), 2000, replace=False)
            X_te_sub = X_test.iloc[idx_t].reset_index(drop=True)
            y_te_sub = y_test.iloc[idx_t].reset_index(drop=True)
        elif has_test:
            X_te_sub = X_test
            y_te_sub = y_test
        else:
            X_te_sub = None
            y_te_sub = None

        results = []
        best_test_score = -np.inf
        best_k = current_k

        for k in range(1, max_k + 1):
            if k > len(X_tr_sub):
                break

            if task_type == "classification":
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=p, n_jobs=-1)
            else:
                knn = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric, p=p, n_jobs=-1)

            knn.fit(X_tr_sub, y_tr_sub)

            # Train score
            y_train_pred = knn.predict(X_tr_sub)
            if task_type == "classification":
                train_score = float(accuracy_score(y_tr_sub, y_train_pred))
            else:
                train_score = float(r2_score(y_tr_sub, y_train_pred))

            # Test score
            test_score = None
            if X_te_sub is not None and y_te_sub is not None:
                y_test_pred = knn.predict(X_te_sub)
                if task_type == "classification":
                    test_score = float(accuracy_score(y_te_sub, y_test_pred))
                else:
                    test_score = float(r2_score(y_te_sub, y_test_pred))

                if test_score > best_test_score:
                    best_test_score = test_score
                    best_k = k
            else:
                # Without test data, use train score to find best (less reliable)
                if train_score > best_test_score:
                    best_test_score = train_score
                    best_k = k

            results.append({
                "k": k,
                "train_score": round(train_score, 4),
                "test_score": round(test_score, 4) if test_score is not None else None,
                "is_current_k": k == current_k,
                "is_best_k": False,  # will be set below
            })

        # Mark best K
        for entry in results:
            entry["is_best_k"] = entry["k"] == best_k

        return results

    def _generate_decision_boundary(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_neighbors: int,
        weights: str,
        metric: str,
        p: int,
    ) -> Dict[str, Any]:
        """
        PCA to 2D, train quick KNN on PCA'd data, create 50x50 meshgrid,
        predict + predict_proba for each cell. Cap data points to 500.
        Classification only.
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import LabelEncoder

        # Encode labels to integers
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        classes = le.classes_.tolist()

        # PCA to 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_train.values)
        explained_variance = pca.explained_variance_ratio_.tolist()

        # Train KNN on PCA'd data
        k = min(n_neighbors, len(X_pca))
        knn_pca = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=p)
        knn_pca.fit(X_pca, y_encoded)

        # Create 50x50 meshgrid
        nx, ny = 50, 50
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

        xx = np.linspace(x_min, x_max, nx)
        yy = np.linspace(y_min, y_max, ny)
        grid_x, grid_y = np.meshgrid(xx, yy)
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

        # Predict for each cell
        grid_predictions = knn_pca.predict(grid_points).tolist()
        grid_proba = knn_pca.predict_proba(grid_points)
        grid_confidence = np.max(grid_proba, axis=1).tolist()

        # Cap data points to 500
        n_points = len(X_pca)
        if n_points > 500:
            idx = np.random.RandomState(42).choice(n_points, 500, replace=False)
        else:
            idx = np.arange(n_points)

        points = []
        for i in idx:
            points.append({
                "x": float(X_pca[i, 0]),
                "y": float(X_pca[i, 1]),
                "class_idx": int(y_encoded[i]),
                "label": str(classes[y_encoded[i]]),
            })

        return {
            "grid": {
                "x_range": [float(x_min), float(x_max)],
                "y_range": [float(y_min), float(y_max)],
                "nx": nx,
                "ny": ny,
                "predictions": grid_predictions,
                "confidence": grid_confidence,
            },
            "points": points,
            "classes": [str(c) for c in classes],
            "pca_explained_variance": [round(float(v), 4) for v in explained_variance],
        }

    def _generate_sample_neighbors(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        task_type: str,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        For min(10, n_test) random test points, use model.kneighbors() to get K nearest neighbors.
        PCA project all to 2D. For each query point return: query position, label, prediction,
        correct flag, neighbors list (position, distance, label), vote_counts dict.
        """
        from sklearn.decomposition import PCA

        n_eval = len(X_eval)
        n_samples = min(10, n_eval)
        if n_samples == 0:
            return []

        # Select random test points
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(n_eval, n_samples, replace=False)

        # PCA project all training + eval points to 2D for visualization
        X_all = pd.concat([X_train, X_eval], ignore_index=True)
        n_components = min(2, X_all.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_all_pca = pca.fit_transform(X_all.values)

        X_train_pca = X_all_pca[:len(X_train)]
        X_eval_pca = X_all_pca[len(X_train):]

        # If only 1 component, pad with zeros
        if n_components == 1:
            X_train_pca = np.column_stack([X_train_pca, np.zeros(len(X_train_pca))])
            X_eval_pca = np.column_stack([X_eval_pca, np.zeros(len(X_eval_pca))])

        results = []
        for idx in sample_indices:
            query_point = X_eval.iloc[[idx]]
            true_label = y_eval.iloc[idx]
            prediction = model.predict(query_point)[0]

            # Get K nearest neighbors
            distances, neighbor_indices = model.kneighbors(query_point)
            distances = distances[0]
            neighbor_indices = neighbor_indices[0]

            # Build neighbor list
            neighbors = []
            vote_counts: Dict[str, int] = {}
            for dist, n_idx in zip(distances, neighbor_indices):
                n_label = str(y_train.iloc[n_idx])
                neighbors.append({
                    "position": {
                        "x": float(X_train_pca[n_idx, 0]),
                        "y": float(X_train_pca[n_idx, 1]),
                    },
                    "distance": round(float(dist), 4),
                    "label": n_label,
                })
                vote_counts[n_label] = vote_counts.get(n_label, 0) + 1

            is_correct = str(true_label) == str(prediction)

            results.append({
                "query_position": {
                    "x": float(X_eval_pca[idx, 0]),
                    "y": float(X_eval_pca[idx, 1]),
                },
                "label": str(true_label),
                "prediction": str(prediction),
                "correct": is_correct,
                "neighbors": neighbors,
                "vote_counts": vote_counts,
            })

        return results

    def _generate_metric_explainer(
        self, metrics: Dict, task_type: str, n_samples: int, target_column: str,
        evaluated_on: str = "train",
    ) -> Dict[str, Any]:
        """Metric explanation cards with real values and KNN-specific analogies."""
        explanations = []
        data_label = "test" if evaluated_on == "test" else "training"

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
                    "analogy": f"Out of {n_samples} {data_label} predictions, {correct} were correct and {wrong} were wrong.",
                    "when_useful": "Good for balanced datasets where all classes are equally important.",
                })
            prec = metrics.get("precision")
            if prec is not None:
                explanations.append({
                    "metric": "Precision",
                    "value": round(float(prec), 4),
                    "value_pct": round(float(prec) * 100, 1),
                    "color": "blue",
                    "analogy": f"When KNN says 'positive', it's right {round(float(prec) * 100, 1)}% of the time.",
                    "when_useful": "Important when false positives are costly.",
                })
            rec = metrics.get("recall")
            if rec is not None:
                explanations.append({
                    "metric": "Recall",
                    "value": round(float(rec), 4),
                    "value_pct": round(float(rec) * 100, 1),
                    "color": "orange",
                    "analogy": f"Of all actual positives, KNN catches {round(float(rec) * 100, 1)}%.",
                    "when_useful": "Critical when missing positives is dangerous.",
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
                    "metric": "R\u00b2 Score",
                    "value": round(float(r2), 4),
                    "value_pct": round(float(r2) * 100, 1),
                    "color": "green" if r2 >= 0.8 else "yellow" if r2 >= 0.5 else "red",
                    "analogy": f"KNN explains {round(float(r2) * 100, 1)}% of the variation in '{target_column}'.",
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
                    "analogy": f"Typical prediction error is about {round(float(rmse), 2)} units.",
                    "when_useful": "Better than MAE when large errors are especially bad.",
                })

        return {
            "metrics": explanations,
            "task_type": task_type,
            "n_samples": n_samples,
            "target_column": target_column,
            "evaluated_on": evaluated_on,
        }

    def _generate_knn_quiz(
        self,
        feature_names: List[str],
        metrics: Dict,
        task_type: str,
        n_neighbors: int,
        weights: str,
        metric: str,
        n_features: int,
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about KNN with real model values."""
        questions = []
        q_id = 0

        # Q1: What K represents
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "In K-Nearest Neighbors, what does 'K' represent?",
            "options": [
                "The number of closest training points used to make a prediction",
                "The number of features in the dataset",
                "The number of clusters to create",
                "The maximum depth of the decision boundary",
            ],
            "correct_answer": 0,
            "explanation": f"K is the number of nearest training points the algorithm looks at when making a prediction. In this model, K={n_neighbors}, meaning each prediction is based on the {n_neighbors} closest neighbors.",
            "difficulty": "easy",
        })

        # Q2: K=1 overfitting behavior
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What happens when K=1 in KNN?",
            "options": [
                "The model is likely to overfit because it relies on a single neighbor",
                "The model becomes more robust to noise",
                "The model always predicts the majority class",
                "The model ignores the training data entirely",
            ],
            "correct_answer": 0,
            "explanation": "With K=1, each prediction is based on the single closest training point. This makes the model very sensitive to noise and outliers in the training data — a classic sign of overfitting. The decision boundary becomes very jagged.",
            "difficulty": "medium",
        })

        # Q3: Effect of increasing K (uses real K value)
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This model uses K={n_neighbors}. What would likely happen if you increased K significantly?",
            "options": [
                "The decision boundary becomes smoother, potentially underfitting",
                "The model always becomes more accurate",
                "Training time increases dramatically",
                "The model memorizes the training data perfectly",
            ],
            "correct_answer": 0,
            "explanation": f"Increasing K from {n_neighbors} smooths the decision boundary because more neighbors are averaged. Very large K values can cause underfitting because the model considers too many distant points, losing local patterns.",
            "difficulty": "medium",
        })

        # Q4: Uniform vs distance weights
        q_id += 1
        current_weight_desc = "all neighbors equally (uniform)" if weights == "uniform" else "closer neighbors more heavily (distance)"
        other_weight = "distance" if weights == "uniform" else "uniform"
        questions.append({
            "id": f"q{q_id}",
            "question": f"This model uses '{weights}' weighting. What is the difference between 'uniform' and 'distance' weights?",
            "options": [
                "Uniform treats all K neighbors equally; distance gives more weight to closer neighbors",
                "Uniform uses Euclidean distance; distance uses Manhattan distance",
                "Uniform is faster to compute; distance is more accurate",
                "Uniform works for classification only; distance works for regression only",
            ],
            "correct_answer": 0,
            "explanation": f"With 'uniform' weights, all K neighbors get an equal vote. With 'distance' weights, closer neighbors have more influence — their votes are weighted by 1/distance. This model uses '{weights}', meaning it weights {current_weight_desc}.",
            "difficulty": "medium",
        })

        # Q5: Distance metric question (uses real feature names)
        q_id += 1
        sample_features = feature_names[:3] if len(feature_names) >= 3 else feature_names
        features_str = ", ".join(f"'{f}'" for f in sample_features)
        questions.append({
            "id": f"q{q_id}",
            "question": f"This model measures distance using the '{metric}' metric across features like {features_str}. Why is feature scaling important for KNN?",
            "options": [
                "Features with larger ranges would dominate the distance calculation",
                "KNN cannot handle unscaled features at all",
                "Scaling makes the algorithm run faster",
                "Scaling is only needed for regression, not classification",
            ],
            "correct_answer": 0,
            "explanation": f"KNN uses distance between points to find neighbors. If features have very different scales (e.g., age 0-100 vs. salary 30,000-200,000), the larger-scale feature dominates the distance. Scaling ensures all {len(feature_names)} features contribute fairly.",
            "difficulty": "hard",
        })

        # Q6: Curse of dimensionality (uses real n_features)
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This dataset has {n_features} features. What is the 'curse of dimensionality' and how does it affect KNN?",
            "options": [
                "In high dimensions, all points become roughly equidistant, making neighbors less meaningful",
                "KNN cannot work with more than 10 features",
                "More features always improve KNN accuracy",
                "High dimensions only affect regression, not classification",
            ],
            "correct_answer": 0,
            "explanation": f"With {n_features} features, each data point lives in {n_features}-dimensional space. As dimensions increase, the distance between the nearest and farthest neighbors converges, making it harder for KNN to distinguish truly 'close' neighbors. This is why feature selection or dimensionality reduction can help.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
