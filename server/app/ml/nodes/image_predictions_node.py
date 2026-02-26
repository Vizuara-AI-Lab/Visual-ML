"""
Image Predictions Node — Train a model on training images, then evaluate on test set.
Supports two algorithms: MLP (sklearn) and CNN (TensorFlow/Keras).
Produces: training metrics, architecture info, prediction gallery, confusion matrix,
confidence distribution, feature importance, and quizzes.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
import json
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class ImagePredictionsInput(NodeInput):
    """Input schema for Image Predictions node."""
    train_dataset_id: str = Field(..., description="Training dataset ID from image_split")
    test_dataset_id: str = Field(..., description="Test dataset ID from image_split")
    target_column: Optional[str] = Field("label", description="Target column name")

    # Algorithm selector
    algorithm: str = Field("mlp", description="Algorithm: 'mlp' or 'cnn'")

    # MLP-specific config
    hidden_layers: str = Field(
        "64,32", description="Hidden layer sizes (comma-separated)"
    )
    activation: str = Field("relu", description="Activation: relu, tanh, logistic")
    max_iter: int = Field(300, description="Maximum training iterations")
    learning_rate_init: float = Field(0.001, description="Initial learning rate")
    early_stopping: bool = Field(True, description="Stop when validation score plateaus")

    # CNN-specific config
    epochs: int = Field(20, description="CNN training epochs")
    batch_size: int = Field(32, description="CNN mini-batch size")
    cnn_learning_rate: float = Field(0.001, description="CNN learning rate")
    use_data_augmentation: bool = Field(True, description="Apply real-time data augmentation during CNN training")

    # Pass-through
    image_width: Optional[int] = Field(None, description="Image width")
    image_height: Optional[int] = Field(None, description="Image height")
    n_classes: Optional[int] = Field(None, description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Class names")


class ImagePredictionsOutput(NodeOutput):
    """Output schema for Image Predictions node."""
    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")
    algorithm: str = Field("mlp", description="Which algorithm was used: mlp or cnn")

    # Training info
    training_samples: int = Field(..., description="Training sample count")
    training_time_seconds: float = Field(..., description="Training duration")
    training_metrics: Dict[str, Any] = Field(..., description="Training metrics")

    # Test evaluation
    test_samples: int = Field(..., description="Number of test images evaluated")
    overall_accuracy: float = Field(..., description="Overall test accuracy [0-1]")
    per_class_metrics: Dict[str, Any] = Field(..., description="Per-class precision/recall/f1")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    class_names: List[str] = Field(..., description="Class names")

    # Explorer data
    prediction_gallery: Optional[List[Dict[str, Any]]] = Field(
        None, description="Sample predictions with images (correct/incorrect)"
    )
    confusion_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Confusion matrix analysis with most-confused pairs"
    )
    confidence_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Prediction confidence distribution"
    )
    metrics_summary: Optional[Dict[str, Any]] = Field(
        None, description="Visual metrics summary for dashboard"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions"
    )

    # Training details (for architecture/training curve tabs)
    loss_curve: Optional[List[float]] = Field(None, description="Training loss curve")
    validation_scores: Optional[List[float]] = Field(None, description="Validation scores")
    architecture: Optional[Dict[str, Any]] = Field(None, description="Architecture details")
    n_iterations: Optional[int] = Field(None, description="Actual training iterations")
    architecture_diagram: Optional[Dict[str, Any]] = Field(
        None, description="Layer-by-layer architecture visualization data"
    )
    training_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Training curve analysis"
    )
    feature_importance: Optional[Dict[str, Any]] = Field(
        None, description="Which pixels matter most (weight magnitude heatmap)"
    )

    # Pass-through for live testing tabs
    image_width: Optional[int] = Field(None, description="Image width for model input")
    image_height: Optional[int] = Field(None, description="Image height for model input")


class ImagePredictionsNode(BaseNode):
    """
    Image Predictions Node — Train + Evaluate in one step.
    Supports MLP (sklearn) and CNN (TensorFlow/Keras).
    """

    node_type = "image_predictions"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.VIEW,
            primary_output_field="overall_accuracy",
            output_fields={
                "overall_accuracy": "Overall test accuracy",
                "model_id": "Model identifier",
                "model_path": "Saved model path",
                "per_class_metrics": "Per-class precision/recall/f1",
                "confusion_matrix": "Confusion matrix",
            },
            requires_input=True,
            can_branch=False,
            produces_dataset=False,
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.DATA_TRANSFORM,
                NodeCategory.PREPROCESSING,
            ],
        )

    def get_input_schema(self):
        return ImagePredictionsInput

    def get_output_schema(self):
        return ImagePredictionsOutput

    # ── Dataset loading ──

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
        if upload_path.exists():
            return pd.read_csv(upload_path)

        try:
            db = SessionLocal()
            dataset = (
                db.query(Dataset)
                .filter(Dataset.dataset_id == dataset_id, Dataset.is_deleted == False)
                .first()
            )
            if not dataset:
                db.close()
                return None
            if dataset.storage_backend == "s3" and dataset.s3_key:
                file_content = await s3_service.download_file(dataset.s3_key)
                df = pd.read_csv(io.BytesIO(file_content))
            elif dataset.local_path:
                df = pd.read_csv(dataset.local_path)
            else:
                db.close()
                return None
            db.close()
            return df
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            return None

    # ── Main execution — dispatches to MLP, CNN, or Pose MLP ──

    async def _execute(self, input_data: ImagePredictionsInput) -> ImagePredictionsOutput:
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                confusion_matrix as sklearn_cm
            )

            # ── Load datasets (shared) ──
            df_train = await self._load_dataset(input_data.train_dataset_id)
            if df_train is None or df_train.empty:
                raise ValueError(f"Training dataset {input_data.train_dataset_id} not found or empty")

            df_test = await self._load_dataset(input_data.test_dataset_id)
            if df_test is None or df_test.empty:
                raise ValueError(f"Test dataset {input_data.test_dataset_id} not found or empty")

            target_col = input_data.target_column or "label"
            if target_col not in df_train.columns:
                target_col = df_train.columns[-1]

            feature_cols = [c for c in df_train.columns if c != target_col]

            # Detect pose data by checking for lm_* columns
            is_pose = any(c.startswith("lm_") for c in feature_cols)

            X_train = df_train[feature_cols].values.astype(np.float64)
            y_train = df_train[target_col].values
            X_test = df_test[feature_cols].values.astype(np.float64)
            y_true = df_test[target_col].values.astype(int)

            if is_pose:
                # Pose landmarks are already normalized (0-1 range from MediaPipe)
                logger.info("Detected pose landmark data (lm_* columns)")
            else:
                # Normalize pixel values to [0, 1]
                if X_train.max() > 1.0:
                    X_train = X_train / 255.0
                    X_test = X_test / 255.0
                    logger.info("Auto-normalized pixel values from [0-255] to [0-1]")

            n_classes = len(np.unique(np.concatenate([y_train, y_true])))
            class_names = input_data.class_names or [str(i) for i in range(n_classes)]

            if is_pose:
                width = len(feature_cols)  # 132
                height = 1
                return await self._train_pose_mlp(
                    input_data, X_train, y_train, X_test, y_true,
                    n_classes, class_names, feature_cols
                )
            else:
                width = input_data.image_width or int(np.sqrt(len(feature_cols)))
                height = input_data.image_height or int(np.sqrt(len(feature_cols)))

                # Dispatch to algorithm
                algorithm = input_data.algorithm.lower().strip()
                if algorithm == "cnn":
                    return await self._train_cnn(
                        input_data, X_train, y_train, X_test, y_true,
                        n_classes, class_names, width, height, feature_cols
                    )
                else:
                    return await self._train_mlp(
                        input_data, X_train, y_train, X_test, y_true,
                        n_classes, class_names, width, height, feature_cols
                    )

        except Exception as e:
            logger.error(f"Image predictions failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # ══════════════════════════════════════════════════════════════════
    #  MLP Training Path (existing, moved from _execute)
    # ══════════════════════════════════════════════════════════════════

    async def _train_mlp(
        self, input_data, X_train, y_train, X_test, y_true,
        n_classes, class_names, width, height, pixel_cols
    ) -> ImagePredictionsOutput:
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix as sklearn_cm
        )
        import joblib

        hidden_layers = self._parse_hidden_layers(input_data.hidden_layers)

        # Data augmentation for small datasets
        original_train_size = len(X_train)
        if len(X_train) < 200:
            X_train, y_train = self._augment_training_data(
                X_train, y_train, width, height, augment_factor=10
            )
            logger.info(f"Augmented training data: {original_train_size} → {len(X_train)} images")

        logger.info(
            f"MLP Training: {len(pixel_cols)} features, "
            f"{hidden_layers} hidden, {len(X_train)} train / {len(X_test)} test, "
            f"{n_classes} classes"
        )

        # Train
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=input_data.activation,
            solver="adam",
            max_iter=input_data.max_iter,
            learning_rate_init=input_data.learning_rate_init,
            early_stopping=input_data.early_stopping,
            random_state=42,
            verbose=False,
        )

        training_start = datetime.utcnow()
        model.fit(X_train, y_train)
        training_time = (datetime.utcnow() - training_start).total_seconds()

        # Training metrics
        y_train_pred = model.predict(X_train)
        train_accuracy = float(accuracy_score(y_train, y_train_pred))
        training_metrics = {
            "accuracy": round(train_accuracy, 4),
            "precision_macro": round(float(precision_score(y_train, y_train_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_train, y_train_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)), 4),
        }

        # Save model
        model_id = generate_id("model_img")
        model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "image_predictions"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_v{model_version}.joblib"
        model_path = model_dir / model_filename
        joblib.dump(model, model_path)

        # Save meta
        meta_path = model_path.with_suffix(".meta.json")
        meta = {
            "algorithm": "mlp",
            "train_pixel_min": float(X_train.min()),
            "train_pixel_max": float(X_train.max()),
            "train_pixel_mean": float(X_train.mean()),
            "train_pixel_std": float(X_train.std()),
            "n_features": len(pixel_cols),
            "n_classes": n_classes,
            "class_names": class_names,
            "image_width": width,
            "image_height": height,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        loss_curve = list(model.loss_curve_) if hasattr(model, "loss_curve_") else []
        val_scores = list(model.validation_scores_) if hasattr(model, "validation_scores_") and model.validation_scores_ else []
        n_iter = model.n_iter_ if hasattr(model, "n_iter_") else 0

        layers = [len(pixel_cols)] + list(hidden_layers) + [n_classes]
        architecture = {
            "layers": layers,
            "layer_names": (
                ["Input (pixels)"]
                + [f"Hidden ({s} neurons)" for s in hidden_layers]
                + [f"Output ({n_classes} classes)"]
            ),
            "activation": input_data.activation,
            "total_params": sum(
                layers[i] * layers[i + 1] + layers[i + 1]
                for i in range(len(layers) - 1)
            ),
        }

        logger.info(f"MLP done: train_acc={train_accuracy:.4f}, {n_iter} iters, {training_time:.1f}s")

        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        accuracy, per_class, cm_list = self._compute_metrics(
            y_true, y_pred, n_classes, class_names
        )

        cm = np.array(cm_list)
        logger.info(f"MLP eval: test_acc={accuracy:.4f}, {len(X_test)} test, {n_classes} classes")

        # Explorer data
        prediction_gallery = self._generate_prediction_gallery(
            X_test, y_true, y_pred, y_proba, class_names, width, height
        )
        confusion_analysis = self._generate_confusion_analysis(
            cm, class_names, X_test, y_true, y_pred, width, height
        )
        confidence_distribution = self._generate_confidence_distribution(y_proba, y_true, y_pred)
        metrics_summary = self._generate_metrics_summary(accuracy, per_class, class_names, len(X_test))
        quiz_questions = self._generate_quiz(
            accuracy, n_classes, class_names, per_class, cm_list, len(X_test),
            hidden_layers, input_data.activation, len(X_train),
            len(pixel_cols), training_time, n_iter
        )
        architecture_diagram = self._generate_architecture_diagram(
            architecture, hidden_layers, input_data.activation, width, height
        )
        training_analysis = self._generate_training_analysis(
            loss_curve, val_scores, n_iter, input_data.max_iter, training_metrics
        )
        feature_importance = self._generate_feature_importance(model, width, height, len(pixel_cols))

        return ImagePredictionsOutput(
            node_type=self.node_type,
            execution_time_ms=int(training_time * 1000),
            model_id=model_id,
            model_path=str(model_path),
            algorithm="mlp",
            training_samples=len(X_train),
            training_time_seconds=training_time,
            training_metrics=training_metrics,
            test_samples=len(X_test),
            overall_accuracy=round(accuracy, 4),
            per_class_metrics=per_class,
            confusion_matrix=cm_list,
            class_names=class_names,
            prediction_gallery=prediction_gallery,
            confusion_analysis=confusion_analysis,
            confidence_distribution=confidence_distribution,
            metrics_summary=metrics_summary,
            quiz_questions=quiz_questions,
            loss_curve=loss_curve,
            validation_scores=val_scores,
            architecture=architecture,
            n_iterations=n_iter,
            architecture_diagram=architecture_diagram,
            training_analysis=training_analysis,
            feature_importance=feature_importance,
            image_width=width,
            image_height=height,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Pose MLP Training Path (landmark-based classification)
    # ══════════════════════════════════════════════════════════════════

    async def _train_pose_mlp(
        self, input_data, X_train, y_train, X_test, y_true,
        n_classes, class_names, feature_cols
    ) -> ImagePredictionsOutput:
        """Train MLP on 132 pose landmark features."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix as sklearn_cm
        )
        import joblib

        hidden_layers = self._parse_hidden_layers(input_data.hidden_layers)
        n_features = len(feature_cols)
        n_landmarks = n_features // 4  # 33 landmarks × 4 values each

        logger.info(
            f"Pose MLP Training: {n_features} landmark features ({n_landmarks} landmarks), "
            f"{hidden_layers} hidden, {len(X_train)} train / {len(X_test)} test, "
            f"{n_classes} classes"
        )

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=input_data.activation,
            solver="adam",
            max_iter=input_data.max_iter,
            learning_rate_init=input_data.learning_rate_init,
            early_stopping=input_data.early_stopping,
            random_state=42,
            verbose=False,
        )

        training_start = datetime.utcnow()
        model.fit(X_train, y_train)
        training_time = (datetime.utcnow() - training_start).total_seconds()

        # Training metrics
        y_train_pred = model.predict(X_train)
        train_accuracy = float(accuracy_score(y_train, y_train_pred))
        training_metrics = {
            "accuracy": round(train_accuracy, 4),
            "precision_macro": round(float(precision_score(y_train, y_train_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_train, y_train_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)), 4),
        }

        # Save model
        model_id = generate_id("model_pose")
        model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "image_predictions"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_v{model_version}.joblib"
        model_path = model_dir / model_filename
        joblib.dump(model, model_path)

        # Save meta with pose-specific info
        meta_path = model_path.with_suffix(".meta.json")
        meta = {
            "algorithm": "pose_mlp",
            "data_type": "pose",
            "n_features": n_features,
            "n_landmarks": n_landmarks,
            "n_classes": n_classes,
            "class_names": class_names,
            "image_width": n_features,
            "image_height": 1,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        loss_curve = list(model.loss_curve_) if hasattr(model, "loss_curve_") else []
        val_scores = list(model.validation_scores_) if hasattr(model, "validation_scores_") and model.validation_scores_ else []
        n_iter = model.n_iter_ if hasattr(model, "n_iter_") else 0

        layers = [n_features] + list(hidden_layers) + [n_classes]
        architecture = {
            "layers": layers,
            "layer_names": (
                [f"Input ({n_landmarks} landmarks × 4)"]
                + [f"Hidden ({s} neurons)" for s in hidden_layers]
                + [f"Output ({n_classes} poses)"]
            ),
            "activation": input_data.activation,
            "total_params": sum(
                layers[i] * layers[i + 1] + layers[i + 1]
                for i in range(len(layers) - 1)
            ),
            "model_type": "pose_mlp",
        }

        logger.info(f"Pose MLP done: train_acc={train_accuracy:.4f}, {n_iter} iters, {training_time:.1f}s")

        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        accuracy, per_class, cm_list = self._compute_metrics(
            y_true, y_pred, n_classes, class_names
        )

        cm = np.array(cm_list)
        logger.info(f"Pose MLP eval: test_acc={accuracy:.4f}, {len(X_test)} test, {n_classes} classes")

        # Explorer data — pose-specific
        confusion_analysis = self._generate_pose_confusion_analysis(cm, class_names)
        confidence_distribution = self._generate_confidence_distribution(y_proba, y_true, y_pred)
        metrics_summary = self._generate_metrics_summary(accuracy, per_class, class_names, len(X_test))
        architecture_diagram = self._generate_pose_architecture_diagram(
            architecture, hidden_layers, input_data.activation, n_landmarks, n_features
        )
        training_analysis = self._generate_training_analysis(
            loss_curve, val_scores, n_iter, input_data.max_iter, training_metrics
        )
        feature_importance = self._generate_landmark_importance(model, n_landmarks, n_features)
        quiz_questions = self._generate_pose_quiz(
            accuracy, n_classes, class_names, per_class, cm_list, len(X_test),
            hidden_layers, input_data.activation, len(X_train),
            n_features, training_time, n_iter, n_landmarks
        )

        return ImagePredictionsOutput(
            node_type=self.node_type,
            execution_time_ms=int(training_time * 1000),
            model_id=model_id,
            model_path=str(model_path),
            algorithm="pose_mlp",
            training_samples=len(X_train),
            training_time_seconds=training_time,
            training_metrics=training_metrics,
            test_samples=len(X_test),
            overall_accuracy=round(accuracy, 4),
            per_class_metrics=per_class,
            confusion_matrix=cm_list,
            class_names=class_names,
            prediction_gallery=None,  # No pixel gallery for pose data
            confusion_analysis=confusion_analysis,
            confidence_distribution=confidence_distribution,
            metrics_summary=metrics_summary,
            quiz_questions=quiz_questions,
            loss_curve=loss_curve,
            validation_scores=val_scores,
            architecture=architecture,
            n_iterations=n_iter,
            architecture_diagram=architecture_diagram,
            training_analysis=training_analysis,
            feature_importance=feature_importance,
            image_width=n_features,
            image_height=1,
        )

    # ── Pose-specific helpers ──

    def _generate_pose_confusion_analysis(
        self, cm: np.ndarray, class_names: list
    ) -> Dict[str, Any]:
        """Confusion analysis for pose data (no example images)."""
        n = len(class_names)
        confused_pairs = []
        for i in range(n):
            for j in range(n):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append({
                        "true_class": str(class_names[i]),
                        "predicted_class": str(class_names[j]),
                        "count": int(cm[i][j]),
                    })
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)

        per_class_accuracy = []
        for i, name in enumerate(class_names):
            total = int(cm[i].sum())
            correct = int(cm[i][i])
            acc = round(correct / total, 4) if total > 0 else 0
            per_class_accuracy.append({
                "class_name": str(name), "accuracy": acc,
                "correct": correct, "total": total,
            })

        return {
            "confusion_matrix": cm.tolist(),
            "class_names": [str(c) for c in class_names],
            "most_confused_pairs": confused_pairs[:10],
            "per_class_accuracy": per_class_accuracy,
            "data_type": "pose",
        }

    def _generate_pose_architecture_diagram(
        self, architecture: Dict[str, Any], hidden_layers: tuple,
        activation: str, n_landmarks: int, n_features: int
    ) -> Dict[str, Any]:
        """Architecture diagram for pose MLP."""
        layers_info = []
        sizes = architecture["layers"]
        names = architecture["layer_names"]

        for i, (size, name) in enumerate(zip(sizes, names)):
            layer = {
                "index": i, "name": name, "size": size,
                "type": "input" if i == 0 else "output" if i == len(sizes) - 1 else "hidden",
            }
            if i > 0:
                params = sizes[i - 1] * size + size
                layer["params"] = params
                layer["activation"] = activation if i < len(sizes) - 1 else "softmax"

            if i == 0:
                layer["description"] = f"{n_landmarks} body landmarks × 4 values (x, y, z, visibility) = {n_features} features"
            elif i == len(sizes) - 1:
                layer["description"] = f"One neuron per pose class, softmax gives probability"
            else:
                layer["description"] = f"{size} neurons with {activation} activation"

            layers_info.append(layer)

        return {
            "layers": layers_info,
            "total_params": architecture["total_params"],
            "input_shape": f"{n_landmarks} landmarks × 4 = {n_features} features",
            "output_shape": f"{sizes[-1]} pose classes",
            "model_type": "pose_mlp",
        }

    def _generate_landmark_importance(
        self, model, n_landmarks: int, n_features: int
    ) -> Dict[str, Any]:
        """Compute per-landmark importance by grouping MLP first-layer weights."""
        LANDMARK_NAMES = [
            "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer",
            "Right Eye Inner", "Right Eye", "Right Eye Outer",
            "Left Ear", "Right Ear", "Mouth Left", "Mouth Right",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky",
            "Left Index", "Right Index", "Left Thumb", "Right Thumb",
            "Left Hip", "Right Hip", "Left Knee", "Right Knee",
            "Left Ankle", "Right Ankle", "Left Heel", "Right Heel",
            "Left Foot Index", "Right Foot Index",
        ]
        try:
            first_layer_weights = model.coefs_[0]  # shape: (n_features, hidden_size)
            # Sum absolute weights per feature
            feature_importance = np.sum(np.abs(first_layer_weights), axis=1)

            # Group by landmark (4 features per landmark: x, y, z, vis)
            landmark_importance = []
            for i in range(n_landmarks):
                base = i * 4
                lm_imp = float(np.sum(feature_importance[base:base + 4]))
                landmark_importance.append(lm_imp)

            landmark_importance = np.array(landmark_importance)
            # Normalize to [0, 1]
            lm_min, lm_max = landmark_importance.min(), landmark_importance.max()
            if lm_max - lm_min > 1e-8:
                landmark_importance = (landmark_importance - lm_min) / (lm_max - lm_min)

            # Build ranked list
            ranked = sorted(
                enumerate(landmark_importance),
                key=lambda x: -x[1]
            )
            landmarks_ranked = [
                {
                    "index": idx,
                    "name": LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"Landmark {idx}",
                    "importance": round(float(imp), 4),
                }
                for idx, imp in ranked
            ]

            return {
                "type": "landmark",
                "landmarks": landmarks_ranked,
                "landmark_importance": landmark_importance.tolist(),
                "n_landmarks": n_landmarks,
                "description": (
                    "Which body parts matter most for classifying poses. "
                    "Red = high importance, blue = low importance. "
                    "The model learns which joints best distinguish your poses."
                ),
            }
        except Exception as e:
            logger.warning(f"Landmark importance generation failed: {e}")
            return {
                "type": "landmark",
                "landmarks": [],
                "landmark_importance": [],
                "n_landmarks": n_landmarks,
                "description": "Not available",
            }

    def _generate_pose_quiz(
        self, accuracy: float, n_classes: int, class_names: list,
        per_class: Dict, cm: list, n_test: int,
        hidden_layers: tuple, activation: str, n_train: int,
        n_features: int, training_time: float, n_iter: int, n_landmarks: int
    ) -> List[Dict[str, Any]]:
        """Pose-specific quiz questions."""
        import random as _random
        questions = []
        q_id = 0

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The model uses {n_features} input features from {n_landmarks} body landmarks. Why is this better than using raw camera pixels?",
            "options": [
                f"Landmarks are a compact representation ({n_features} vs thousands of pixels) that captures body shape regardless of background, clothing, or lighting",
                "Pixels are always better than landmarks",
                "There is no difference",
                "Landmarks only work for faces, not full body",
            ],
            "correct_answer": 0,
            "explanation": f"MediaPipe extracts {n_landmarks} body joint positions, giving us a clean {n_features}-feature representation. This is invariant to background, lighting, and appearance — the model only sees body geometry.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The model got {accuracy*100:.1f}% accuracy on {n_test} test poses. What does the confusion matrix reveal?",
            "options": [
                "Which poses are most often confused with each other — e.g., similar poses might be misclassified",
                "The overall accuracy number",
                "Which body parts are most important",
                "The training speed",
            ],
            "correct_answer": 0,
            "explanation": f"The {n_classes}×{n_classes} confusion matrix shows exactly which poses get mixed up. Poses with similar body positions (like 'standing' vs 'idle') will show higher confusion rates.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why does an MLP work well for pose classification but not for raw image classification?",
            "options": [
                "Pose landmarks are tabular data (132 numbers) — perfect for MLP. Raw images have spatial structure that MLPs ignore but CNNs preserve.",
                "MLPs always work better than CNNs",
                "The choice of model doesn't matter",
                "MLP can't handle pose data either",
            ],
            "correct_answer": 0,
            "explanation": f"Pose landmarks from MediaPipe are already a high-quality feature extraction. The {n_features} numbers (joint positions) are tabular data, and MLPs excel at learning patterns from tabular features.",
            "difficulty": "hard",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does the 'landmark importance' visualization show?",
            "options": [
                "Which body joints the model relies on most — joints that differ between poses have high importance",
                "Which landmarks are detected most reliably",
                "The speed of each landmark detection",
                "The visibility of each body part",
            ],
            "correct_answer": 0,
            "explanation": "By analyzing the MLP's first-layer weights, we can see which landmarks (grouped by joint) carry the most discriminative power. For example, hand/arm landmarks might be most important for gesture-based poses.",
            "difficulty": "hard",
        })

        if per_class:
            classes_sorted = sorted(per_class.items(), key=lambda x: x[1]["f1"])
            worst_name, worst_data = classes_sorted[0]
            best_name, best_data = classes_sorted[-1]
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"'{best_name}' has F1={best_data['f1']:.2f} but '{worst_name}' has F1={worst_data['f1']:.2f}. Why might some poses be harder to classify?",
                "options": [
                    f"Poses like '{worst_name}' might look similar to other poses in body joint positions, making them harder to distinguish",
                    "The model always performs equally on all poses",
                    "F1 scores are random",
                    "Some poses use fewer landmarks",
                ],
                "correct_answer": 0,
                "explanation": f"Some poses share similar body positions. '{worst_name}' might overlap with other poses in landmark space. Solutions: capture more distinct poses, add more training samples, or use more expressive hidden layers.",
                "difficulty": "hard",
            })

        _random.shuffle(questions)
        return questions[:5]

    # ══════════════════════════════════════════════════════════════════
    #  CNN Training Path (new)
    # ══════════════════════════════════════════════════════════════════

    def _build_cnn_model(self, height: int, width: int, n_classes: int, learning_rate: float):
        """Build a simple CNN suitable for small grayscale image datasets."""
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(height, width, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation='softmax'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    async def _train_cnn(
        self, input_data, X_train_flat, y_train, X_test_flat, y_true,
        n_classes, class_names, width, height, pixel_cols
    ) -> ImagePredictionsOutput:
        """CNN training path using TensorFlow/Keras."""
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Reshape flat pixels to (batch, H, W, 1) for CNN
        X_train = X_train_flat.reshape(-1, height, width, 1).astype(np.float32)
        X_test = X_test_flat.reshape(-1, height, width, 1).astype(np.float32)

        # One-hot encode labels
        y_train_int = y_train.astype(int)
        y_train_cat = tf.keras.utils.to_categorical(y_train_int, n_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_true, n_classes)

        # Build model
        model = self._build_cnn_model(height, width, n_classes, input_data.cnn_learning_rate)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
        ]

        logger.info(
            f"CNN Training: {width}x{height} images, "
            f"{len(X_train)} train / {len(X_test)} test, "
            f"{n_classes} classes, {input_data.epochs} epochs"
        )

        training_start = datetime.utcnow()

        if input_data.use_data_augmentation and len(X_train) < 500:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
            )
            datagen.fit(X_train)
            history = model.fit(
                datagen.flow(X_train, y_train_cat, batch_size=input_data.batch_size),
                epochs=input_data.epochs,
                validation_data=(X_test, y_test_cat),
                callbacks=callbacks,
                verbose=0,
            )
        else:
            history = model.fit(
                X_train, y_train_cat,
                batch_size=input_data.batch_size,
                epochs=input_data.epochs,
                validation_data=(X_test, y_test_cat),
                callbacks=callbacks,
                verbose=0,
            )

        training_time = (datetime.utcnow() - training_start).total_seconds()

        # Training curves from history
        loss_curve = [float(v) for v in history.history.get('loss', [])]
        val_acc_curve = [float(v) for v in history.history.get('val_accuracy', [])]
        n_iter = len(loss_curve)

        # Training metrics
        y_train_pred_proba = model.predict(X_train, verbose=0)
        y_train_pred = np.argmax(y_train_pred_proba, axis=1)
        train_accuracy = float(accuracy_score(y_train_int, y_train_pred))
        training_metrics = {
            "accuracy": round(train_accuracy, 4),
            "precision_macro": round(float(precision_score(y_train_int, y_train_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_train_int, y_train_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_train_int, y_train_pred, average="macro", zero_division=0)), 4),
            "final_loss": round(loss_curve[-1], 6) if loss_curve else 0,
        }

        # Save CNN model as .keras
        model_id = generate_id("model_cnn")
        model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "image_predictions"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_v{model_version}.keras"
        model_path = model_dir / model_filename
        model.save(str(model_path))

        # Save meta
        meta_path = model_path.with_suffix(".meta.json")
        meta = {
            "algorithm": "cnn",
            "train_pixel_min": float(X_train.min()),
            "train_pixel_max": float(X_train.max()),
            "train_pixel_mean": float(X_train.mean()),
            "train_pixel_std": float(X_train.std()),
            "n_features": len(pixel_cols),
            "n_classes": n_classes,
            "class_names": class_names,
            "image_width": width,
            "image_height": height,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        logger.info(f"CNN done: train_acc={train_accuracy:.4f}, {n_iter} epochs, {training_time:.1f}s")

        # Evaluate on test set
        y_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        accuracy, per_class, cm_list = self._compute_metrics(
            y_true, y_pred, n_classes, class_names
        )
        cm = np.array(cm_list)

        logger.info(f"CNN eval: test_acc={accuracy:.4f}, {len(X_test)} test, {n_classes} classes")

        # Architecture info
        architecture = self._build_cnn_architecture_info(model, width, height, n_classes)
        architecture_diagram = self._generate_cnn_architecture_diagram(model, width, height, n_classes)

        # Explorer data (use flat arrays for gallery rendering)
        prediction_gallery = self._generate_prediction_gallery(
            X_test_flat, y_true, y_pred, y_proba, class_names, width, height
        )
        confusion_analysis = self._generate_confusion_analysis(
            cm, class_names, X_test_flat, y_true, y_pred, width, height
        )
        confidence_distribution = self._generate_confidence_distribution(y_proba, y_true, y_pred)
        metrics_summary = self._generate_metrics_summary(accuracy, per_class, class_names, len(X_test))
        training_analysis = self._generate_training_analysis(
            loss_curve, val_acc_curve, n_iter, input_data.epochs, training_metrics
        )
        feature_importance = self._generate_cnn_feature_importance(model, X_test[:20], width, height)
        quiz_questions = self._generate_cnn_quiz(
            accuracy, n_classes, class_names, per_class, cm_list, len(X_test),
            width, height, len(X_train), training_time, n_iter
        )

        return ImagePredictionsOutput(
            node_type=self.node_type,
            execution_time_ms=int(training_time * 1000),
            model_id=model_id,
            model_path=str(model_path),
            algorithm="cnn",
            training_samples=len(X_train),
            training_time_seconds=training_time,
            training_metrics=training_metrics,
            test_samples=len(X_test),
            overall_accuracy=round(accuracy, 4),
            per_class_metrics=per_class,
            confusion_matrix=cm_list,
            class_names=class_names,
            prediction_gallery=prediction_gallery,
            confusion_analysis=confusion_analysis,
            confidence_distribution=confidence_distribution,
            metrics_summary=metrics_summary,
            quiz_questions=quiz_questions,
            loss_curve=loss_curve,
            validation_scores=val_acc_curve,
            architecture=architecture,
            n_iterations=n_iter,
            architecture_diagram=architecture_diagram,
            training_analysis=training_analysis,
            feature_importance=feature_importance,
            image_width=width,
            image_height=height,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Shared Helpers
    # ══════════════════════════════════════════════════════════════════

    def _compute_metrics(self, y_true, y_pred, n_classes, class_names):
        """Compute accuracy, per-class metrics, confusion matrix."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix as sklearn_cm
        )

        accuracy = float(accuracy_score(y_true, y_pred))
        labels = list(range(n_classes))
        prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        f1_vals = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

        per_class = {}
        for i, name in enumerate(class_names):
            n_true = int(np.sum(y_true == i))
            n_pred_count = int(np.sum(y_pred == i))
            n_correct = int(np.sum((y_true == i) & (y_pred == i)))
            per_class[name] = {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1_vals[i]), 4),
                "support": n_true,
                "predicted": n_pred_count,
                "correct": n_correct,
            }

        cm = sklearn_cm(y_true, y_pred, labels=labels)
        return accuracy, per_class, cm.tolist()

    def _parse_hidden_layers(self, s: str) -> tuple:
        try:
            parts = [int(x.strip()) for x in s.split(",") if x.strip()]
            return tuple(max(1, p) for p in parts) if parts else (128,)
        except ValueError:
            return (128,)

    def _augment_training_data(
        self, X: np.ndarray, y: np.ndarray,
        width: int, height: int, augment_factor: int = 10
    ) -> tuple:
        """MLP data augmentation: random shift, gaussian noise, brightness jitter."""
        rng = np.random.default_rng(42)
        augmented_X = [X.copy()]
        augmented_y = [y.copy()]

        for _ in range(augment_factor - 1):
            batch = X.copy()
            for i in range(len(batch)):
                img = batch[i].reshape(height, width)

                shift_x = rng.integers(-2, 3)
                shift_y = rng.integers(-2, 3)
                if shift_x != 0 or shift_y != 0:
                    shifted = np.zeros_like(img)
                    src_y1 = max(0, -shift_y)
                    src_y2 = min(height, height - shift_y)
                    src_x1 = max(0, -shift_x)
                    src_x2 = min(width, width - shift_x)
                    dst_y1 = max(0, shift_y)
                    dst_x1 = max(0, shift_x)
                    h_size = src_y2 - src_y1
                    w_size = src_x2 - src_x1
                    shifted[dst_y1:dst_y1+h_size, dst_x1:dst_x1+w_size] = \
                        img[src_y1:src_y1+h_size, src_x1:src_x1+w_size]
                    img = shifted

                noise = rng.normal(0, 0.03, img.shape)
                img = img + noise

                brightness = rng.uniform(0.85, 1.15)
                img = img * brightness

                batch[i] = np.clip(img.flatten(), 0, 1)

            augmented_X.append(batch)
            augmented_y.append(y.copy())

        X_aug = np.concatenate(augmented_X)
        y_aug = np.concatenate(augmented_y)

        perm = rng.permutation(len(X_aug))
        return X_aug[perm], y_aug[perm]

    # ══════════════════════════════════════════════════════════════════
    #  CNN-specific helpers
    # ══════════════════════════════════════════════════════════════════

    def _build_cnn_architecture_info(self, model, width, height, n_classes):
        """Extract architecture dict from Keras model."""
        total_params = model.count_params()
        return {
            "layers": [width * height] + [64] + [n_classes],
            "layer_names": ["Input (image)", "CNN Feature Extractor", f"Output ({n_classes} classes)"],
            "activation": "relu",
            "total_params": total_params,
            "model_type": "cnn",
        }

    def _generate_cnn_architecture_diagram(self, model, width, height, n_classes):
        """Generate layer-by-layer architecture diagram data for CNN."""
        import tensorflow as tf

        layers_info = []
        for i, layer in enumerate(model.layers):
            layer_config = layer.get_config()
            layer_info = {
                "index": i,
                "name": layer.name,
                "params": layer.count_params(),
            }

            if isinstance(layer, tf.keras.layers.Conv2D):
                filters = layer_config.get("filters", 0)
                kernel = layer_config.get("kernel_size", (3, 3))
                layer_info["type"] = "conv2d"
                layer_info["size"] = filters
                layer_info["activation"] = layer_config.get("activation", "relu")
                layer_info["description"] = f"Conv2D: {filters} filters of {kernel[0]}x{kernel[1]}, detects local patterns"
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                pool = layer_config.get("pool_size", (2, 2))
                layer_info["type"] = "pooling"
                layer_info["size"] = 0
                layer_info["description"] = f"MaxPool {pool[0]}x{pool[1]}: keeps strongest activations, halves spatial size"
            elif isinstance(layer, tf.keras.layers.Flatten):
                out_shape = layer.output_shape[-1] if layer.output_shape and layer.output_shape[-1] else 0
                layer_info["type"] = "flatten"
                layer_info["size"] = out_shape
                layer_info["description"] = f"Flatten: convert feature maps to {out_shape}-dim vector"
            elif isinstance(layer, tf.keras.layers.Dense):
                units = layer_config.get("units", 0)
                act = layer_config.get("activation", "")
                is_output = (i == len(model.layers) - 1)
                layer_info["type"] = "output" if is_output else "dense"
                layer_info["size"] = units
                layer_info["activation"] = act
                layer_info["description"] = (
                    f"Output: {units} classes with softmax" if is_output
                    else f"Dense: {units} neurons with {act}"
                )
            elif isinstance(layer, tf.keras.layers.Dropout):
                rate = layer_config.get("rate", 0)
                layer_info["type"] = "dropout"
                layer_info["size"] = 0
                layer_info["description"] = f"Dropout({rate}): randomly disables {rate*100:.0f}% of neurons to prevent overfitting"
            else:
                layer_info["type"] = "other"
                layer_info["size"] = 0
                layer_info["description"] = layer.__class__.__name__

            layers_info.append(layer_info)

        return {
            "layers": layers_info,
            "total_params": model.count_params(),
            "input_shape": f"{width}x{height}x1 (grayscale)",
            "output_shape": f"{n_classes} classes",
            "model_type": "cnn",
        }

    def _generate_cnn_feature_importance(self, model, X_sample, width, height):
        """Compute a simple saliency map by averaging gradient magnitudes."""
        import tensorflow as tf
        try:
            X_tensor = tf.constant(X_sample, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = model(X_tensor, training=False)
                top_class = tf.reduce_max(predictions, axis=1)
                loss = tf.reduce_mean(top_class)
            grads = tape.gradient(loss, X_tensor)
            saliency = tf.reduce_mean(tf.abs(grads), axis=0).numpy()  # (H, W, 1)
            saliency = saliency.squeeze()  # (H, W)
            saliency_flat = saliency.flatten()
            sal_min, sal_max = saliency_flat.min(), saliency_flat.max()
            if sal_max - sal_min > 1e-8:
                saliency_flat = (saliency_flat - sal_min) / (sal_max - sal_min)
            return {
                "heatmap": saliency_flat.tolist(),
                "width": width,
                "height": height,
                "description": (
                    "CNN saliency map: brighter pixels have more influence on predictions. "
                    "Shows which spatial regions the CNN focuses on."
                ),
            }
        except Exception as e:
            logger.warning(f"CNN feature importance failed: {e}")
            return {"heatmap": [], "width": width, "height": height, "description": "Not available"}

    def _generate_cnn_quiz(
        self, accuracy, n_classes, class_names, per_class, cm_list, n_test,
        width, height, n_train, training_time, n_iter
    ):
        """CNN-specific quiz questions."""
        import random as _random
        questions = []
        q_id = 0

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why does a CNN work better than MLP for images?",
            "options": [
                "CNN preserves spatial structure — convolutional filters detect edges, corners, and patterns regardless of position",
                "CNN is just faster than MLP",
                "CNN uses more neurons so it's always better",
                "There is no difference between CNN and MLP for images",
            ],
            "correct_answer": 0,
            "explanation": "MLP flattens the image, losing all spatial information. CNN uses convolutional filters that slide across the image, detecting local patterns like edges and textures. This spatial awareness is key for image understanding.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This CNN processes {width}x{height} images. What does a Conv2D layer do?",
            "options": [
                "It slides small filters (e.g. 3x3) across the image to detect local patterns like edges and corners",
                "It resizes the image to be smaller",
                "It converts the image to color",
                "It flattens the image into a 1D vector",
            ],
            "correct_answer": 0,
            "explanation": f"A Conv2D layer uses small learned filters (kernels) that slide across the {width}x{height} image. Each filter detects a specific pattern. Early layers detect edges; deeper layers detect complex shapes.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is the purpose of MaxPooling in a CNN?",
            "options": [
                "It reduces spatial size by keeping only the strongest activations, making the network more efficient and translation-invariant",
                "It increases the image resolution",
                "It adds more color channels",
                "It's the same as a dense layer",
            ],
            "correct_answer": 0,
            "explanation": "MaxPooling (e.g. 2x2) takes the maximum value in each 2x2 region, halving the spatial dimensions. This reduces computation and makes the model more robust to small shifts in the input.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why is Dropout used in the CNN?",
            "options": [
                "It randomly disables neurons during training, forcing the network to not rely on any single neuron — this prevents overfitting",
                "It makes training faster",
                "It removes bad neurons permanently",
                "It's required for all neural networks",
            ],
            "correct_answer": 0,
            "explanation": "Dropout is a regularization technique. During training, it randomly 'drops' a percentage of neurons, preventing co-adaptation and reducing overfitting. At test time, all neurons are active.",
            "difficulty": "hard",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The model got {accuracy*100:.1f}% accuracy on {n_test} test images. Is accuracy alone enough to evaluate?",
            "options": [
                "No — accuracy can be misleading with imbalanced classes. Precision, recall, and F1 give a fuller picture.",
                "Yes — accuracy is the only metric that matters",
                "Only if accuracy is above 90%",
                "Accuracy is unreliable for all tasks",
            ],
            "correct_answer": 0,
            "explanation": f"With {n_classes} classes, if one class dominates, high accuracy can be achieved by always predicting that class. Per-class metrics reveal the true performance.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does data augmentation do for a CNN?",
            "options": [
                "It creates modified copies of training images (rotated, shifted, zoomed) to make the model more robust with limited data",
                "It increases the image resolution",
                "It removes bad images from the dataset",
                "It downloads more images from the internet",
            ],
            "correct_answer": 0,
            "explanation": f"With only {n_train} training images, the model can easily memorize. Data augmentation generates variations (rotation, shift, zoom) so the model learns to recognize objects regardless of exact position or angle.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]

    # ══════════════════════════════════════════════════════════════════
    #  Shared Explorer Data Generators (used by both MLP and CNN)
    # ══════════════════════════════════════════════════════════════════

    def _generate_prediction_gallery(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
        y_proba, class_names: list, width: int, height: int,
        n_correct: int = 12, n_wrong: int = 12
    ) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(42)
        gallery = []
        correct_mask = y_true == y_pred
        wrong_mask = ~correct_mask

        correct_indices = np.where(correct_mask)[0]
        if len(correct_indices) > 0:
            chosen = rng.choice(correct_indices, size=min(n_correct, len(correct_indices)), replace=False)
            for idx in chosen:
                item = {
                    "pixels": X[idx].tolist(),
                    "width": width, "height": height,
                    "true_label": str(class_names[int(y_true[idx])] if int(y_true[idx]) < len(class_names) else y_true[idx]),
                    "pred_label": str(class_names[int(y_pred[idx])] if int(y_pred[idx]) < len(class_names) else y_pred[idx]),
                    "correct": True,
                }
                if y_proba is not None:
                    item["confidence"] = round(float(np.max(y_proba[idx])), 4)
                    item["probabilities"] = {
                        str(class_names[i]): round(float(y_proba[idx][i]), 4)
                        for i in range(min(len(class_names), y_proba.shape[1]))
                    }
                gallery.append(item)

        wrong_indices = np.where(wrong_mask)[0]
        if len(wrong_indices) > 0:
            chosen = rng.choice(wrong_indices, size=min(n_wrong, len(wrong_indices)), replace=False)
            for idx in chosen:
                item = {
                    "pixels": X[idx].tolist(),
                    "width": width, "height": height,
                    "true_label": str(class_names[int(y_true[idx])] if int(y_true[idx]) < len(class_names) else y_true[idx]),
                    "pred_label": str(class_names[int(y_pred[idx])] if int(y_pred[idx]) < len(class_names) else y_pred[idx]),
                    "correct": False,
                }
                if y_proba is not None:
                    item["confidence"] = round(float(np.max(y_proba[idx])), 4)
                    item["probabilities"] = {
                        str(class_names[i]): round(float(y_proba[idx][i]), 4)
                        for i in range(min(len(class_names), y_proba.shape[1]))
                    }
                gallery.append(item)

        return gallery

    def _generate_confusion_analysis(
        self, cm: np.ndarray, class_names: list,
        X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
        width: int, height: int
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(42)
        n = len(class_names)

        confused_pairs = []
        for i in range(n):
            for j in range(n):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append({
                        "true_class": str(class_names[i]),
                        "predicted_class": str(class_names[j]),
                        "count": int(cm[i][j]),
                    })
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)

        for pair in confused_pairs[:5]:
            true_idx = class_names.index(pair["true_class"]) if pair["true_class"] in class_names else -1
            pred_idx = class_names.index(pair["predicted_class"]) if pair["predicted_class"] in class_names else -1
            if true_idx >= 0 and pred_idx >= 0:
                mask = (y_true == true_idx) & (y_pred == pred_idx)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    chosen = rng.choice(indices, size=min(3, len(indices)), replace=False)
                    pair["example_images"] = [X[i].tolist() for i in chosen]
                    pair["width"] = width
                    pair["height"] = height

        per_class_accuracy = []
        for i, name in enumerate(class_names):
            total = int(cm[i].sum())
            correct = int(cm[i][i])
            acc = round(correct / total, 4) if total > 0 else 0
            per_class_accuracy.append({
                "class_name": str(name), "accuracy": acc,
                "correct": correct, "total": total,
            })

        return {
            "confusion_matrix": cm.tolist(),
            "class_names": [str(c) for c in class_names],
            "most_confused_pairs": confused_pairs[:10],
            "per_class_accuracy": per_class_accuracy,
        }

    def _generate_confidence_distribution(
        self, y_proba, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        if y_proba is None:
            return {"has_data": False}

        max_proba = np.max(y_proba, axis=1)
        correct_mask = y_true == y_pred
        correct_conf = max_proba[correct_mask]
        wrong_conf = max_proba[~correct_mask]

        bins = np.linspace(0, 1, 21)
        h_correct, _ = np.histogram(correct_conf, bins=bins)
        h_wrong, _ = np.histogram(wrong_conf, bins=bins)

        return {
            "has_data": True,
            "bins": [round(float(b), 2) for b in bins.tolist()],
            "correct_counts": h_correct.tolist(),
            "wrong_counts": h_wrong.tolist(),
            "correct_mean_confidence": round(float(np.mean(correct_conf)), 4) if len(correct_conf) > 0 else 0,
            "wrong_mean_confidence": round(float(np.mean(wrong_conf)), 4) if len(wrong_conf) > 0 else 0,
            "insight": (
                f"Correct predictions have higher average confidence "
                f"({np.mean(correct_conf):.1%}) than incorrect ones "
                f"({np.mean(wrong_conf):.1%} if any). The model is "
                f"{'well-calibrated' if np.mean(correct_conf) > 0.7 else 'somewhat uncertain'}."
            ) if len(correct_conf) > 0 else "No correct predictions available.",
        }

    def _generate_metrics_summary(
        self, accuracy: float, per_class: Dict, class_names: list, n_test: int
    ) -> Dict[str, Any]:
        sorted_classes = sorted(
            [(name, data) for name, data in per_class.items()],
            key=lambda x: x[1]["f1"], reverse=True
        )
        best_class = sorted_classes[0] if sorted_classes else None
        worst_class = sorted_classes[-1] if sorted_classes else None

        macro_precision = np.mean([d["precision"] for d in per_class.values()])
        macro_recall = np.mean([d["recall"] for d in per_class.values()])
        macro_f1 = np.mean([d["f1"] for d in per_class.values()])

        return {
            "overall_accuracy": round(accuracy, 4),
            "macro_precision": round(float(macro_precision), 4),
            "macro_recall": round(float(macro_recall), 4),
            "macro_f1": round(float(macro_f1), 4),
            "n_test_samples": n_test,
            "n_classes": len(class_names),
            "best_class": {"name": best_class[0], "f1": best_class[1]["f1"]} if best_class else None,
            "worst_class": {"name": worst_class[0], "f1": worst_class[1]["f1"]} if worst_class else None,
            "grade": (
                "A+" if accuracy >= 0.95 else
                "A" if accuracy >= 0.90 else
                "B" if accuracy >= 0.80 else
                "C" if accuracy >= 0.70 else
                "D" if accuracy >= 0.60 else "F"
            ),
        }

    # MLP-specific architecture diagram
    def _generate_architecture_diagram(
        self, architecture: Dict[str, Any], hidden_layers: tuple,
        activation: str, width: int, height: int
    ) -> Dict[str, Any]:
        layers_info = []
        sizes = architecture["layers"]
        names = architecture["layer_names"]

        for i, (size, name) in enumerate(zip(sizes, names)):
            layer = {
                "index": i, "name": name, "size": size,
                "type": "input" if i == 0 else "output" if i == len(sizes) - 1 else "hidden",
            }
            if i > 0:
                params = sizes[i - 1] * size + size
                layer["params"] = params
                layer["activation"] = activation if i < len(sizes) - 1 else "softmax"

            if i == 0:
                layer["description"] = f"Each {width}x{height} image flattened to {size} pixel values"
            elif i == len(sizes) - 1:
                layer["description"] = f"One neuron per class, softmax gives probability for each class"
            else:
                layer["description"] = f"{size} neurons with {activation} activation"

            layers_info.append(layer)

        return {
            "layers": layers_info,
            "total_params": architecture["total_params"],
            "input_shape": f"{width}x{height} = {width * height} pixels",
            "output_shape": f"{sizes[-1]} classes",
            "model_type": "mlp",
        }

    def _generate_training_analysis(
        self, loss_curve: list, val_scores: list,
        n_iter: int, max_iter: int, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not loss_curve:
            return {"has_data": False}

        analysis = {
            "has_data": True,
            "loss_values": loss_curve,
            "validation_scores": val_scores,
            "n_iterations": n_iter,
            "max_iterations": max_iter,
            "initial_loss": round(loss_curve[0], 6),
            "final_loss": round(loss_curve[-1], 6),
            "loss_reduction_pct": round(
                (1 - loss_curve[-1] / loss_curve[0]) * 100, 1
            ) if loss_curve[0] > 0 else 0,
        }

        if n_iter < max_iter:
            analysis["convergence"] = "converged"
            analysis["convergence_message"] = (
                f"Training converged in {n_iter} iterations (limit was {max_iter})."
            )
        else:
            analysis["convergence"] = "hit_limit"
            analysis["convergence_message"] = (
                f"Training reached the maximum of {max_iter} iterations without fully converging. "
                f"Try increasing max_iter or adjusting the learning rate."
            )

        return analysis

    # MLP-specific feature importance
    def _generate_feature_importance(
        self, model, width: int, height: int, n_features: int
    ) -> Dict[str, Any]:
        try:
            first_layer_weights = model.coefs_[0]
            importance = np.sum(np.abs(first_layer_weights), axis=1)
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            return {
                "heatmap": importance.tolist(),
                "width": width, "height": height,
                "description": (
                    "Brighter pixels = more important for classification. "
                    "This shows which parts of the image the model pays most attention to."
                ),
            }
        except Exception as e:
            logger.warning(f"Feature importance generation failed: {e}")
            return {"heatmap": [], "width": width, "height": height, "description": "Not available"}

    # MLP-specific quiz
    def _generate_quiz(
        self, accuracy: float, n_classes: int, class_names: list,
        per_class: Dict, cm: list, n_test: int,
        hidden_layers: tuple, activation: str, n_train: int,
        n_features: int, training_time: float, n_iter: int
    ) -> List[Dict[str, Any]]:
        import random as _random
        questions = []
        q_id = 0

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The model got {accuracy*100:.1f}% accuracy on {n_test} test images. Is accuracy alone enough to evaluate a model?",
            "options": [
                "No — accuracy can be misleading with imbalanced classes. Precision, recall, and F1 give a fuller picture.",
                "Yes — accuracy is the only metric that matters",
                "Only if accuracy is above 90%",
                "Accuracy is unreliable for all tasks",
            ],
            "correct_answer": 0,
            "explanation": f"With {n_classes} classes, if one class has 90% of samples, a model predicting only that class gets 90% accuracy while failing on all others!",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does a confusion matrix show?",
            "options": [
                "For each true class (row), how many images were predicted as each class (column)",
                "How confused the model is overall",
                "The training loss over time",
                "Which images are most similar",
            ],
            "correct_answer": 0,
            "explanation": f"The {n_classes}x{n_classes} confusion matrix: rows = true labels, columns = predicted. Diagonal = correct, off-diagonal = errors.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This network has {n_features} input features. Why so many?",
            "options": [
                f"Each pixel is a feature — a {int(np.sqrt(n_features))}x{int(np.sqrt(n_features))} image has {n_features} pixels",
                "The model created extra features",
                "It's always 784 for image models",
                "The number is random",
            ],
            "correct_answer": 0,
            "explanation": f"When we flatten a {int(np.sqrt(n_features))}x{int(np.sqrt(n_features))} image, each pixel becomes a separate feature.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The architecture uses layers {hidden_layers}. Why do later layers have fewer neurons?",
            "options": [
                "Each layer extracts higher-level, more abstract features — fewer neurons needed",
                "Fewer neurons means faster training, no other reason",
                "It's just a convention with no technical reason",
                "Layers must always decrease in size",
            ],
            "correct_answer": 0,
            "explanation": f"The 'funnel' shape ({' -> '.join(str(h) for h in hidden_layers)}) is a common pattern: early layers detect low-level features, later layers combine them into abstract concepts.",
            "difficulty": "hard",
        })

        if per_class:
            classes_sorted = sorted(per_class.items(), key=lambda x: x[1]["f1"])
            worst_name, worst_data = classes_sorted[0]
            best_name, best_data = classes_sorted[-1]
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"'{best_name}' has F1={best_data['f1']:.2f} but '{worst_name}' has F1={worst_data['f1']:.2f}. What could cause this?",
                "options": [
                    f"'{worst_name}' might look similar to other classes, have fewer examples, or more visual variation",
                    "The model always performs equally on all classes",
                    "F1 scores are random",
                    f"'{worst_name}' images are larger",
                ],
                "correct_answer": 0,
                "explanation": f"Some classes are harder. '{worst_name}' might be confused with visually similar classes. Check the confusion matrix!",
                "difficulty": "hard",
            })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What is the difference between precision and recall?",
            "options": [
                "Precision = 'of predictions for class X, how many are correct'. Recall = 'of actual class X items, how many were found'",
                "They are the same metric",
                "Precision is for training, recall is for testing",
                "Precision is always higher than recall",
            ],
            "correct_answer": 0,
            "explanation": "Precision: 'when I predict X, how often am I right?' Recall: 'of all actual X, how many did I catch?'",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
