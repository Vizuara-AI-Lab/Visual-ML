"""
SVM Node - Support Vector Machine pipeline node.
Supports both classification (SVC) and regression (SVR) tasks with configurable
kernel, regularization, and margin parameters. Produces rich learning-activity
data including decision-boundary grids, support-vector visualizations, kernel
comparisons, and C-sensitivity analysis.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
import time
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


# ---------------------------------------------------------------------------
# Input / Output schemas
# ---------------------------------------------------------------------------

class SVMInput(NodeInput):
    """Input schema for SVM node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: str = Field(..., description="Name of target column")
    task_type: str = Field("classification", description="Task type: 'classification' or 'regression'")

    # Hyperparameters
    kernel: str = Field("rbf", description="Kernel type: 'linear', 'rbf', 'poly', 'sigmoid'")
    C: float = Field(1.0, ge=0.01, le=100, description="Regularization parameter")
    gamma: str = Field("scale", description="Kernel coefficient: 'scale', 'auto', or float")
    degree: int = Field(3, ge=2, le=5, description="Degree for polynomial kernel")
    random_state: int = Field(42, description="Random seed for reproducibility")

    # UI control
    show_advanced_options: bool = Field(
        False, description="Toggle for advanced options visibility in UI"
    )

    class Config:
        extra = "ignore"


class SVMOutput(NodeOutput):
    """Output schema for SVM node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    task_type: str = Field(..., description="Task type used")
    kernel: str = Field(..., description="Kernel used")

    n_support_vectors: int = Field(..., description="Total number of support vectors")
    support_vector_ratio: float = Field(..., description="Ratio of support vectors to training samples")

    training_metrics: Dict[str, Any] = Field(..., description="Training/test metrics")
    training_time_seconds: float = Field(..., description="Training duration")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")

    # Learning activity data (all Optional)
    decision_boundary_grid: Optional[Dict[str, Any]] = Field(
        None, description="Decision boundary grid for 2D visualization"
    )
    support_vectors_2d: Optional[Dict[str, Any]] = Field(
        None, description="Support vectors projected to 2D via PCA"
    )
    margin_info: Optional[Dict[str, Any]] = Field(
        None, description="Margin width and support vector distribution"
    )
    kernel_comparison: Optional[List[Dict[str, Any]]] = Field(
        None, description="Accuracy/score comparison across kernels"
    )
    c_sensitivity_data: Optional[Dict[str, Any]] = Field(
        None, description="Model performance across different C values"
    )
    metric_explainer: Optional[Dict[str, Any]] = Field(
        None, description="Metric explanation cards with real values"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about SVMs"
    )

    # Pass-through fields for downstream nodes (e.g., confusion_matrix)
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (passed from split node)"
    )
    target_column: Optional[str] = Field(None, description="Target column (passed from split node)")


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------

class SVMNode(BaseNode):
    """
    SVM Node - Train Support Vector Machine models for classification or regression.

    Responsibilities:
    - Load training dataset from database/S3
    - Train SVM model (SVC with probability=True or SVR)
    - Calculate task-specific metrics
    - Save model artifacts via joblib
    - Generate rich learning-activity data (decision boundary, support vectors,
      kernel comparison, C sensitivity, metric explainer, quiz)

    Production features:
    - Supports classification (SVC) and regression (SVR)
    - Configurable kernel, C, gamma, degree
    - Support vector analysis
    - Decision boundary visualization (classification)
    - Model versioning
    """

    node_type = "svm"

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
                "kernel": "Kernel function used",
                "n_support_vectors": "Number of support vectors",
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
        return SVMInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return SVMOutput

    # ------------------------------------------------------------------
    # Dataset loading (exact copy of random_forest pattern)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def _execute(self, input_data: SVMInput) -> SVMOutput:
        """Execute SVM training."""
        import joblib
        from sklearn.svm import SVC, SVR
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            r2_score, mean_absolute_error, mean_squared_error,
        )
        from sklearn.preprocessing import StandardScaler

        try:
            logger.info(f"Training SVM model ({input_data.task_type}, kernel={input_data.kernel})")

            # --- Load training data ---
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
            y_train = df_train[input_data.target_column]
            X_train = df_train.drop(columns=[input_data.target_column])

            # Keep only numeric columns
            non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                logger.info(f"Dropping non-numeric columns: {non_numeric}")
                X_train = X_train.select_dtypes(include=[np.number])

            if X_train.empty or len(X_train.columns) == 0:
                raise InvalidDatasetError(
                    reason="No numeric feature columns remaining after filtering",
                    expected_format="Dataset with numeric features",
                )

            # Handle missing values
            X_train = X_train.fillna(X_train.mean())
            y_train = y_train.loc[X_train.index]

            feature_names = X_train.columns.tolist()

            # Scale features (important for SVM)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # --- Load test data if available ---
            X_test_scaled = None
            y_test = None
            if input_data.test_dataset_id:
                try:
                    df_test = await self._load_dataset(input_data.test_dataset_id)
                    if df_test is not None and not df_test.empty and input_data.target_column in df_test.columns:
                        y_test = df_test[input_data.target_column]
                        X_test = df_test.drop(columns=[input_data.target_column])
                        X_test = X_test[[c for c in feature_names if c in X_test.columns]]
                        X_test = X_test.fillna(X_test.mean())
                        X_test_scaled = scaler.transform(X_test)
                except Exception as e:
                    logger.warning(f"Could not load test data: {e}")

            # --- Parse gamma ---
            gamma_value: Any = input_data.gamma
            if input_data.gamma not in ("scale", "auto"):
                try:
                    gamma_value = float(input_data.gamma)
                except ValueError:
                    gamma_value = "scale"

            # --- Build model ---
            if input_data.task_type == "classification":
                model = SVC(
                    kernel=input_data.kernel,
                    C=input_data.C,
                    gamma=gamma_value,
                    degree=input_data.degree,
                    random_state=input_data.random_state,
                    probability=True,
                )
            else:
                model = SVR(
                    kernel=input_data.kernel,
                    C=input_data.C,
                    gamma=gamma_value,
                    degree=input_data.degree,
                )

            # --- Train ---
            t_start = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - t_start

            # --- Evaluate ---
            # Prefer test metrics when available
            if X_test_scaled is not None and y_test is not None:
                y_pred = model.predict(X_test_scaled)
                eval_samples = len(y_test)
                evaluated_on = "test"
            else:
                y_pred = model.predict(X_train_scaled)
                eval_samples = len(y_train)
                evaluated_on = "train"

            if input_data.task_type == "classification":
                avg = "weighted" if len(np.unique(y_train)) > 2 else "binary"
                metrics: Dict[str, Any] = {
                    "accuracy": round(float(accuracy_score(
                        y_test if evaluated_on == "test" else y_train, y_pred
                    )), 4),
                    "precision": round(float(precision_score(
                        y_test if evaluated_on == "test" else y_train, y_pred, average=avg, zero_division=0
                    )), 4),
                    "recall": round(float(recall_score(
                        y_test if evaluated_on == "test" else y_train, y_pred, average=avg, zero_division=0
                    )), 4),
                    "f1": round(float(f1_score(
                        y_test if evaluated_on == "test" else y_train, y_pred, average=avg, zero_division=0
                    )), 4),
                }
            else:
                y_true = y_test if evaluated_on == "test" else y_train
                metrics = {
                    "r2": round(float(r2_score(y_true, y_pred)), 4),
                    "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
                }

            # Train metrics separately (for metadata)
            if evaluated_on == "test":
                y_train_pred = model.predict(X_train_scaled)
                if input_data.task_type == "classification":
                    avg_t = "weighted" if len(np.unique(y_train)) > 2 else "binary"
                    train_metrics = {
                        "accuracy": round(float(accuracy_score(y_train, y_train_pred)), 4),
                        "precision": round(float(precision_score(y_train, y_train_pred, average=avg_t, zero_division=0)), 4),
                        "recall": round(float(recall_score(y_train, y_train_pred, average=avg_t, zero_division=0)), 4),
                        "f1": round(float(f1_score(y_train, y_train_pred, average=avg_t, zero_division=0)), 4),
                    }
                else:
                    train_metrics = {
                        "r2": round(float(r2_score(y_train, y_train_pred)), 4),
                        "mae": round(float(mean_absolute_error(y_train, y_train_pred)), 4),
                        "rmse": round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 4),
                    }
            else:
                train_metrics = metrics

            # --- Support vector info ---
            if input_data.task_type == "classification":
                n_sv = int(sum(model.n_support_))
            else:
                n_sv = len(model.support_)
            sv_ratio = round(n_sv / len(X_train_scaled), 4) if len(X_train_scaled) > 0 else 0.0

            # --- Save model ---
            model_id = generate_id(f"model_svm_{input_data.task_type}")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "svm" / input_data.task_type
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename
            joblib.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, model_path)
            logger.info(f"Model saved to {model_path}")

            # ----------------------------------------------------------
            # Generate learning activity data (each in try/except)
            # ----------------------------------------------------------

            decision_boundary_grid = None
            if input_data.task_type == "classification":
                try:
                    decision_boundary_grid = self._generate_decision_boundary(
                        X_train_scaled, y_train, input_data.kernel, input_data.C,
                        gamma_value, input_data.degree, input_data.random_state,
                    )
                except Exception as e:
                    logger.warning(f"Decision boundary generation failed: {e}")

            support_vectors_2d = None
            try:
                support_vectors_2d = self._generate_support_vectors_2d(
                    X_train_scaled, y_train, model,
                )
            except Exception as e:
                logger.warning(f"Support vectors 2D generation failed: {e}")

            margin_info = None
            try:
                margin_info = self._generate_margin_info(
                    model, input_data.task_type, len(X_train_scaled), y_train,
                )
            except Exception as e:
                logger.warning(f"Margin info generation failed: {e}")

            kernel_comparison = None
            try:
                kernel_comparison = self._generate_kernel_comparison(
                    X_train_scaled, y_train, input_data.task_type,
                    input_data.C, gamma_value, input_data.kernel, input_data.random_state,
                )
            except Exception as e:
                logger.warning(f"Kernel comparison generation failed: {e}")

            c_sensitivity_data = None
            try:
                c_sensitivity_data = self._generate_c_sensitivity(
                    X_train_scaled, y_train, input_data.task_type,
                    input_data.kernel, gamma_value, input_data.degree,
                    input_data.C, input_data.random_state,
                )
            except Exception as e:
                logger.warning(f"C sensitivity generation failed: {e}")

            metric_explainer = None
            try:
                metric_explainer = self._generate_metric_explainer(
                    metrics, input_data.task_type, eval_samples,
                    input_data.target_column, evaluated_on,
                )
            except Exception as e:
                logger.warning(f"Metric explainer generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_svm_quiz(
                    metrics, input_data.task_type, input_data.kernel,
                    input_data.C, n_sv, len(X_train_scaled),
                    input_data.target_column,
                )
            except Exception as e:
                logger.warning(f"SVM quiz generation failed: {e}")

            # ----------------------------------------------------------
            # Return output
            # ----------------------------------------------------------

            return SVMOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train_scaled),
                n_features=len(feature_names),
                task_type=input_data.task_type,
                kernel=input_data.kernel,
                n_support_vectors=n_sv,
                support_vector_ratio=sv_ratio,
                training_metrics=metrics,
                training_time_seconds=round(training_time, 4),
                metadata={
                    "target_column": input_data.target_column,
                    "feature_names": feature_names,
                    "dropped_non_numeric": non_numeric,
                    "hyperparameters": {
                        "kernel": input_data.kernel,
                        "C": input_data.C,
                        "gamma": str(gamma_value),
                        "degree": input_data.degree,
                        "random_state": input_data.random_state,
                    },
                    "train_metrics": train_metrics,
                    "test_metrics": metrics if evaluated_on == "test" else None,
                    "evaluated_on": evaluated_on,
                },
                decision_boundary_grid=decision_boundary_grid,
                support_vectors_2d=support_vectors_2d,
                margin_info=margin_info,
                kernel_comparison=kernel_comparison,
                c_sensitivity_data=c_sensitivity_data,
                metric_explainer=metric_explainer,
                quiz_questions=quiz_questions,
                # Pass through for downstream nodes
                test_dataset_id=input_data.test_dataset_id,
                target_column=input_data.target_column,
            )

        except Exception as e:
            logger.error(f"SVM training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # ==================================================================
    # Learning Activity Helpers
    # ==================================================================

    def _generate_decision_boundary(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        kernel: str,
        C: float,
        gamma: Any,
        degree: int,
        random_state: int,
    ) -> Dict[str, Any]:
        """
        PCA to 2D, train a quick SVM on the 2D data with the same hyperparameters,
        create a 50x50 meshgrid and predict each cell.  For SVC use decision_function
        for confidence/margin visualization.  Classification only.
        Cap data points to 500.
        """
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC

        y_arr = np.array(y_train)
        n_samples = X_train.shape[0]

        # Cap data points
        max_pts = 500
        if n_samples > max_pts:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n_samples, max_pts, replace=False)
            X_sub = X_train[idx]
            y_sub = y_arr[idx]
        else:
            X_sub = X_train
            y_sub = y_arr

        # PCA to 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_sub)

        # Train quick SVM on PCA'd data
        svm_2d = SVC(
            kernel=kernel, C=C, gamma=gamma, degree=degree,
            random_state=random_state, probability=False,
        )
        svm_2d.fit(X_2d, y_sub)

        # 50x50 meshgrid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 50),
            np.linspace(y_min, y_max, 50),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_preds = svm_2d.predict(grid_points)

        # Decision function for confidence/margin
        decision_values = None
        try:
            df_vals = svm_2d.decision_function(grid_points)
            if df_vals.ndim == 1:
                decision_values = np.round(df_vals, 4).tolist()
            else:
                # Multi-class: take max absolute value across OvR columns
                decision_values = np.round(np.max(np.abs(df_vals), axis=1), 4).tolist()
        except Exception:
            pass

        # Unique class labels (serialize safely)
        classes = sorted(set(y_sub.tolist()))
        class_to_int = {c: i for i, c in enumerate(classes)}

        # Build points list
        points = []
        for i in range(len(X_2d)):
            points.append({
                "x": round(float(X_2d[i, 0]), 4),
                "y": round(float(X_2d[i, 1]), 4),
                "label": int(class_to_int[y_sub[i]]),
            })

        # Explained variance
        evr = pca.explained_variance_ratio_.tolist()

        return {
            "grid": {
                "x_min": round(float(x_min), 4),
                "x_max": round(float(x_max), 4),
                "y_min": round(float(y_min), 4),
                "y_max": round(float(y_max), 4),
                "nx": 50,
                "ny": 50,
                "predictions": [int(class_to_int.get(p, p)) for p in grid_preds.tolist()],
                "decision_values": decision_values,
            },
            "points": points,
            "classes": [str(c) for c in classes],
            "n_classes": len(classes),
            "pca_explained_variance": [round(float(v), 4) for v in evr],
            "pc1_label": f"PC1 ({evr[0]*100:.1f}%)" if len(evr) > 0 else "PC1",
            "pc2_label": f"PC2 ({evr[1]*100:.1f}%)" if len(evr) > 1 else "PC2",
        }

    # ------------------------------------------------------------------

    def _generate_support_vectors_2d(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        model: Any,
    ) -> Dict[str, Any]:
        """
        PCA project all training points to 2D.  Identify support vector indices
        from model.support_.  Return all_points, sv_points, counts, class labels.
        """
        from sklearn.decomposition import PCA

        y_arr = np.array(y_train)

        pca = PCA(n_components=min(2, X_train.shape[1]))
        X_2d = pca.fit_transform(X_train)

        sv_indices = set(model.support_.tolist())

        # Unique class labels
        classes = sorted(set(y_arr.tolist()))
        class_to_int = {c: i for i, c in enumerate(classes)}

        all_points = []
        sv_points = []
        for i in range(len(X_2d)):
            is_sv = i in sv_indices
            pt = {
                "x": round(float(X_2d[i, 0]), 4),
                "y": round(float(X_2d[i, 1]), 4) if X_2d.shape[1] >= 2 else 0.0,
                "label": int(class_to_int[y_arr[i]]),
                "is_sv": is_sv,
            }
            all_points.append(pt)
            if is_sv:
                sv_pt = dict(pt)
                sv_pt["original_index"] = i
                sv_pt["class_name"] = str(y_arr[i])
                sv_points.append(sv_pt)

        n_total = len(X_train)
        n_sv = len(sv_indices)
        sv_ratio = round(n_sv / n_total, 4) if n_total > 0 else 0.0

        return {
            "all_points": all_points,
            "sv_points": sv_points,
            "n_total": n_total,
            "n_sv": n_sv,
            "sv_ratio": sv_ratio,
            "class_labels": [str(c) for c in classes],
            "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_.tolist()],
        }

    # ------------------------------------------------------------------

    def _generate_margin_info(
        self,
        model: Any,
        task_type: str,
        n_train: int,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """
        Margin width (linear kernel, binary classification only) and
        support vector distribution across classes.
        """
        y_arr = np.array(y_train)
        classes = sorted(set(y_arr.tolist()))

        # Margin width: only computable for linear kernel, binary classification
        margin_width = None
        if task_type == "classification" and model.kernel == "linear" and len(classes) == 2:
            try:
                w = model.coef_[0]
                margin_width = round(float(2.0 / np.linalg.norm(w)), 6)
            except Exception:
                pass

        # Support vector counts
        if task_type == "classification":
            n_sv_total = int(sum(model.n_support_))
            n_support_per_class = {}
            for cls, count in zip(model.classes_, model.n_support_):
                n_support_per_class[str(cls)] = int(count)
        else:
            n_sv_total = len(model.support_)
            n_support_per_class = {"all": n_sv_total}

        sv_ratio = round(n_sv_total / n_train, 4) if n_train > 0 else 0.0

        return {
            "margin_width": margin_width,
            "kernel": model.kernel,
            "n_support_vectors": n_sv_total,
            "support_vector_ratio": sv_ratio,
            "n_training_samples": n_train,
            "n_support_per_class": n_support_per_class,
            "is_linear": model.kernel == "linear",
            "is_binary": len(classes) == 2,
            "description": (
                f"Margin width = {margin_width:.4f} (wider margin = better generalization)"
                if margin_width is not None
                else (
                    "Margin width is only computable for linear kernel with binary classification. "
                    f"Current kernel: {model.kernel}, classes: {len(classes)}."
                )
            ),
        }

    # ------------------------------------------------------------------

    def _generate_kernel_comparison(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        task_type: str,
        C: float,
        gamma: Any,
        current_kernel: str,
        random_state: int,
    ) -> List[Dict[str, Any]]:
        """
        For each kernel in [linear, rbf, poly]: train a quick SVM (C=1.0,
        gamma='scale', max_iter=2000), record accuracy/r2 + n_support_vectors
        + training time.  Cap training data to 2000.
        """
        from sklearn.svm import SVC, SVR
        from sklearn.metrics import accuracy_score, r2_score

        y_arr = np.array(y_train)
        n = len(X_train)

        # Cap training data
        max_n = 2000
        if n > max_n:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n, max_n, replace=False)
            X_sub = X_train[idx]
            y_sub = y_arr[idx]
        else:
            X_sub = X_train
            y_sub = y_arr

        results = []
        for kern in ["linear", "rbf", "poly"]:
            try:
                t0 = time.time()
                if task_type == "classification":
                    m = SVC(kernel=kern, C=1.0, gamma="scale", max_iter=2000,
                            random_state=random_state)
                    m.fit(X_sub, y_sub)
                    score = round(float(accuracy_score(y_sub, m.predict(X_sub))), 4)
                    n_sv = int(sum(m.n_support_))
                else:
                    m = SVR(kernel=kern, C=1.0, gamma="scale", max_iter=2000)
                    m.fit(X_sub, y_sub)
                    score = round(float(r2_score(y_sub, m.predict(X_sub))), 4)
                    n_sv = len(m.support_)
                elapsed = round(time.time() - t0, 4)

                results.append({
                    "kernel": kern,
                    "score": score,
                    "score_metric": "accuracy" if task_type == "classification" else "r2",
                    "n_support_vectors": n_sv,
                    "training_time": elapsed,
                    "is_current": kern == current_kernel,
                })
            except Exception as e:
                logger.warning(f"Kernel comparison for '{kern}' failed: {e}")
                continue

        return results

    # ------------------------------------------------------------------

    def _generate_c_sensitivity(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        task_type: str,
        kernel: str,
        gamma: Any,
        degree: int,
        current_C: float,
        random_state: int,
    ) -> Dict[str, Any]:
        """
        For C in [0.01, 0.1, 1.0, 10.0, 100.0]: train quick SVM with the same
        kernel/gamma, record accuracy/r2 + n_support_vectors.
        """
        from sklearn.svm import SVC, SVR
        from sklearn.metrics import accuracy_score, r2_score

        y_arr = np.array(y_train)
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        scores = []
        n_svs = []

        for c_val in c_values:
            try:
                if task_type == "classification":
                    m = SVC(kernel=kernel, C=c_val, gamma=gamma, degree=degree,
                            random_state=random_state, max_iter=2000)
                    m.fit(X_train, y_arr)
                    sc = round(float(accuracy_score(y_arr, m.predict(X_train))), 4)
                    nsv = int(sum(m.n_support_))
                else:
                    m = SVR(kernel=kernel, C=c_val, gamma=gamma, degree=degree, max_iter=2000)
                    m.fit(X_train, y_arr)
                    sc = round(float(r2_score(y_arr, m.predict(X_train))), 4)
                    nsv = len(m.support_)
                scores.append(sc)
                n_svs.append(nsv)
            except Exception as e:
                logger.warning(f"C sensitivity for C={c_val} failed: {e}")
                scores.append(None)
                n_svs.append(None)

        return {
            "c_values": c_values,
            "scores": scores,
            "score_metric": "accuracy" if task_type == "classification" else "r2",
            "n_support_vectors": n_svs,
            "current_c": current_C,
            "description": (
                "Lower C = wider margin, more support vectors (underfitting risk). "
                "Higher C = narrower margin, fewer support vectors (overfitting risk)."
            ),
        }

    # ------------------------------------------------------------------

    def _generate_metric_explainer(
        self,
        metrics: Dict[str, Any],
        task_type: str,
        n_samples: int,
        target_column: str,
        evaluated_on: str = "train",
    ) -> Dict[str, Any]:
        """Metric explanation cards with real values and SVM-specific analogies."""
        explanations: List[Dict[str, Any]] = []
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
                    "analogy": (
                        f"Out of {n_samples} {data_label} predictions, {correct} were correct "
                        f"and {wrong} were wrong."
                    ),
                    "when_useful": "Good for balanced datasets where all classes are equally important.",
                })
            prec = metrics.get("precision")
            if prec is not None:
                explanations.append({
                    "metric": "Precision",
                    "value": round(float(prec), 4),
                    "value_pct": round(float(prec) * 100, 1),
                    "color": "blue",
                    "analogy": (
                        f"When the SVM says 'positive', it's right "
                        f"{round(float(prec) * 100, 1)}% of the time."
                    ),
                    "when_useful": "Important when false positives are costly.",
                })
            rec = metrics.get("recall")
            if rec is not None:
                explanations.append({
                    "metric": "Recall",
                    "value": round(float(rec), 4),
                    "value_pct": round(float(rec) * 100, 1),
                    "color": "orange",
                    "analogy": (
                        f"Of all actual positives, the SVM catches "
                        f"{round(float(rec) * 100, 1)}%."
                    ),
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
                    "analogy": (
                        f"The SVM explains {round(float(r2) * 100, 1)}% of the variation "
                        f"in '{target_column}'."
                    ),
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

    # ------------------------------------------------------------------

    def _generate_svm_quiz(
        self,
        metrics: Dict[str, Any],
        task_type: str,
        kernel: str,
        C: float,
        n_support_vectors: int,
        n_training_samples: int,
        target_column: str,
    ) -> List[Dict[str, Any]]:
        """Auto-generate 6 SVM-specific quiz questions; return 5 (shuffled)."""
        questions: List[Dict[str, Any]] = []
        q_id = 0

        # Q1: What is a support vector
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is a 'support vector' in SVM?",
            "options": [
                "A training point that lies on or inside the margin boundary — the points that define the decision boundary",
                "The average of all data points in each class",
                "A feature that the model ignores during training",
                "A vector that points from one class centroid to another",
            ],
            "correct_answer": 0,
            "explanation": (
                "Support vectors are the training points closest to the decision boundary. "
                "They 'support' (define) the margin. If you removed all other points and kept "
                "only the support vectors, the decision boundary would stay the same."
            ),
            "difficulty": "easy",
        })

        # Q2: What C controls (uses real C value)
        q_id += 1
        c_desc = "low" if C <= 0.1 else "moderate" if C <= 10 else "high"
        questions.append({
            "id": f"q{q_id}",
            "question": f"The regularization parameter C is set to {C}. What does C control?",
            "options": [
                "The trade-off between a wider margin and fewer misclassifications — higher C means less tolerance for errors",
                "The number of support vectors directly",
                "The dimensionality of the feature space",
                "The learning rate of the training algorithm",
            ],
            "correct_answer": 0,
            "explanation": (
                f"C = {C} is a {c_desc} value. A small C prioritizes a wide margin (may allow some "
                f"misclassifications). A large C tries harder to classify every training point correctly, "
                f"risking overfitting. Think of C as how much the model 'cares' about each individual training point."
            ),
            "difficulty": "medium",
        })

        # Q3: What kernel was used and what it does (uses real kernel)
        q_id += 1
        kernel_descriptions = {
            "linear": "finds a straight hyperplane to separate classes",
            "rbf": "maps data into a higher-dimensional space using radial basis functions to find non-linear boundaries",
            "poly": "uses polynomial combinations of features to model curved decision boundaries",
            "sigmoid": "uses a sigmoid (tanh-like) function, similar to a neural network activation",
        }
        kernel_desc = kernel_descriptions.get(kernel, "transforms the feature space")
        wrong_kernels = [k for k in ["linear", "rbf", "poly", "sigmoid"] if k != kernel]
        options_q3 = [
            f"'{kernel}' — it {kernel_desc}",
            f"'{wrong_kernels[0]}' — it {kernel_descriptions[wrong_kernels[0]]}",
            "There is no kernel — SVM always works in the original feature space",
            f"'{wrong_kernels[1]}' — it {kernel_descriptions[wrong_kernels[1]]}",
        ]
        questions.append({
            "id": f"q{q_id}",
            "question": "Which kernel function did this SVM use, and what does it do?",
            "options": options_q3,
            "correct_answer": 0,
            "explanation": (
                f"This model uses the '{kernel}' kernel, which {kernel_desc}. "
                f"The kernel trick allows SVM to find complex decision boundaries without "
                f"explicitly computing the high-dimensional feature space."
            ),
            "difficulty": "medium",
        })

        # Q4: Why X out of Y points are support vectors (uses real counts)
        q_id += 1
        sv_pct = round(n_support_vectors / n_training_samples * 100, 1) if n_training_samples > 0 else 0
        questions.append({
            "id": f"q{q_id}",
            "question": (
                f"{n_support_vectors} out of {n_training_samples} training points "
                f"({sv_pct}%) are support vectors. What does this tell us?"
            ),
            "options": [
                "These are the critical points near the decision boundary — the model only 'remembers' these for predictions",
                "The rest of the points were removed as outliers",
                "The model failed to learn from most of the data",
                "Support vectors are randomly selected during training",
            ],
            "correct_answer": 0,
            "explanation": (
                f"Only {n_support_vectors} of {n_training_samples} points matter for the decision boundary. "
                f"A low ratio means the classes are well-separated (only a few points near the boundary). "
                f"A high ratio may indicate overlapping classes or a complex decision boundary."
            ),
            "difficulty": "medium",
        })

        # Q5: Effect of very large C
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What happens if you set the C parameter to a very large value (e.g., C = 1000)?",
            "options": [
                "The model tries very hard to classify every training point correctly, likely overfitting",
                "The margin becomes wider and the model becomes more robust",
                "The number of support vectors always increases",
                "The kernel function changes automatically",
            ],
            "correct_answer": 0,
            "explanation": (
                "A very large C makes the model prioritize zero training errors over margin width. "
                "This can lead to a very narrow margin that perfectly separates training data but "
                "generalizes poorly to new data — classic overfitting. A smaller C allows some "
                "misclassifications in exchange for a wider, more generalizable margin."
            ),
            "difficulty": "hard",
        })

        # Q6: SVM vs logistic regression
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "How does SVM differ from Logistic Regression for classification?",
            "options": [
                "SVM maximizes the margin between classes, while logistic regression maximizes the likelihood of the data",
                "SVM can only handle two classes, logistic regression handles many",
                "Logistic regression always outperforms SVM on large datasets",
                "SVM doesn't use a decision boundary",
            ],
            "correct_answer": 0,
            "explanation": (
                "SVM finds the decision boundary that maximizes the margin (distance to the nearest points). "
                "Logistic regression finds the boundary that maximizes the probability of correct classification. "
                "SVM focuses on the hardest-to-classify points (support vectors), while logistic regression "
                "considers all points. Both can handle multi-class problems."
            ),
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
