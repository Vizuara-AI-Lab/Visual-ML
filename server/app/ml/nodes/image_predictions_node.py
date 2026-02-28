"""
Image Predictions Node — Train a model on training images, then evaluate on test set.
Uses Transfer Learning (MobileNetV2 pretrained on ImageNet) for image classification.
Pose landmark data is handled separately with a dedicated MLP.
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

    # Transfer Learning config
    epochs: int = Field(20, description="Training epochs")
    learning_rate: float = Field(0.0001, description="Learning rate (lower is safer for pretrained weights)")
    fine_tune_layers: int = Field(0, description="MobileNetV2 layers to unfreeze for fine-tuning (0 = freeze all)")
    use_augmentation: bool = Field(True, description="Apply data augmentation during training")

    # Pose MLP config (used only when pose data is detected)
    hidden_layers: str = Field("128,64", description="Hidden layer sizes for pose MLP")
    activation: str = Field("relu", description="Activation: relu, tanh, logistic")
    max_iter: int = Field(500, description="Maximum training iterations for pose MLP")
    learning_rate_init: float = Field(0.001, description="Initial learning rate for pose MLP")
    early_stopping: bool = Field(True, description="Stop when validation score plateaus")

    # Pass-through
    image_width: Optional[int] = Field(None, description="Image width")
    image_height: Optional[int] = Field(None, description="Image height")
    n_classes: Optional[int] = Field(None, description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Class names")


class ImagePredictionsOutput(NodeOutput):
    """Output schema for Image Predictions node."""
    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")
    algorithm: str = Field("transfer_learning", description="Which algorithm was used")

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
        None, description="Which pixels matter most (saliency heatmap)"
    )

    # Pass-through for live testing tabs
    image_width: Optional[int] = Field(None, description="Image width for model input")
    image_height: Optional[int] = Field(None, description="Image height for model input")


class ImagePredictionsNode(BaseNode):
    """
    Image Predictions Node — Train + Evaluate in one step.
    Uses Transfer Learning (MobileNetV2) for images, MLP for pose landmarks.
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

    # ── Main execution ──

    async def _execute(self, input_data: ImagePredictionsInput) -> ImagePredictionsOutput:
        try:
            # Load datasets
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
                return await self._train_pose_mlp(
                    input_data, X_train, y_train, X_test, y_true,
                    n_classes, class_names, feature_cols
                )
            else:
                width = input_data.image_width or int(np.sqrt(len(feature_cols)))
                height = input_data.image_height or int(np.sqrt(len(feature_cols)))
                return await self._train_transfer_learning(
                    input_data, X_train, y_train, X_test, y_true,
                    n_classes, class_names, width, height, feature_cols
                )

        except Exception as e:
            logger.error(f"Image predictions failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # ══════════════════════════════════════════════════════════════════
    #  Transfer Learning (MobileNetV2) — Main Image Training Path
    # ══════════════════════════════════════════════════════════════════

    async def _train_transfer_learning(
        self, input_data, X_train_flat, y_train, X_test_flat, y_true,
        n_classes, class_names, width, height, pixel_cols
    ) -> ImagePredictionsOutput:
        """Transfer learning using MobileNetV2 pretrained on ImageNet with data augmentation."""
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        TARGET_SIZE = 96  # Smallest size MobileNetV2 accepts

        # Reshape flat pixels to (batch, H, W, 1) grayscale
        X_train_img = X_train_flat.reshape(-1, height, width, 1).astype(np.float32)
        X_test_img = X_test_flat.reshape(-1, height, width, 1).astype(np.float32)

        # Resize to 96x96 and convert grayscale to 3-channel RGB
        X_train_resized = tf.image.resize(X_train_img, [TARGET_SIZE, TARGET_SIZE]).numpy()
        X_test_resized = tf.image.resize(X_test_img, [TARGET_SIZE, TARGET_SIZE]).numpy()
        X_train_rgb = np.repeat(X_train_resized, 3, axis=-1)
        X_test_rgb = np.repeat(X_test_resized, 3, axis=-1)

        # MobileNetV2 preprocess_input expects [0,255] and maps to [-1,1]
        X_train_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(X_train_rgb * 255.0)
        X_test_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(X_test_rgb * 255.0)

        # One-hot encode labels
        y_train_int = y_train.astype(int)
        y_train_cat = tf.keras.utils.to_categorical(y_train_int, n_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_true, n_classes)

        # Build MobileNetV2 transfer learning model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            include_top=False,
            weights='imagenet',
        )
        base_model.trainable = False

        # Optionally unfreeze last N layers for fine-tuning
        fine_tune_layers = input_data.fine_tune_layers
        if fine_tune_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False

        n_frozen = sum(1 for l in base_model.layers if not l.trainable)
        n_trainable_base = sum(1 for l in base_model.layers if l.trainable)

        # Improved classification head with BatchNormalization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation='softmax'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=input_data.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0
            ),
        ]

        logger.info(
            f"Transfer Learning Training: {width}x{height} -> {TARGET_SIZE}x{TARGET_SIZE} RGB, "
            f"{len(X_train_rgb)} train / {len(X_test_rgb)} test, "
            f"{n_classes} classes, {input_data.epochs} epochs, "
            f"MobileNetV2 frozen={n_frozen} trainable={n_trainable_base}, "
            f"augmentation={'ON' if input_data.use_augmentation else 'OFF'}"
        )

        # Train with optional data augmentation
        training_start = datetime.utcnow()

        if input_data.use_augmentation:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zoom_range=0.15,
                horizontal_flip=False,  # Don't flip digits/letters
                brightness_range=[0.8, 1.2],
            )
            datagen.fit(X_train_rgb)
            history = model.fit(
                datagen.flow(X_train_rgb, y_train_cat, batch_size=32),
                epochs=input_data.epochs,
                validation_data=(X_test_rgb, y_test_cat),
                callbacks=callbacks,
                verbose=0,
            )
        else:
            history = model.fit(
                X_train_rgb, y_train_cat,
                batch_size=32,
                epochs=input_data.epochs,
                validation_data=(X_test_rgb, y_test_cat),
                callbacks=callbacks,
                verbose=0,
            )

        training_time = (datetime.utcnow() - training_start).total_seconds()

        # Training curves
        loss_curve = [float(v) for v in history.history.get('loss', [])]
        val_acc_curve = [float(v) for v in history.history.get('val_accuracy', [])]
        n_iter = len(loss_curve)

        # Training metrics
        y_train_pred_proba = model.predict(X_train_rgb, verbose=0)
        y_train_pred = np.argmax(y_train_pred_proba, axis=1)
        train_accuracy = float(accuracy_score(y_train_int, y_train_pred))
        training_metrics = {
            "accuracy": round(train_accuracy, 4),
            "precision_macro": round(float(precision_score(y_train_int, y_train_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_train_int, y_train_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_train_int, y_train_pred, average="macro", zero_division=0)), 4),
            "final_loss": round(loss_curve[-1], 6) if loss_curve else 0,
        }

        # Save model
        model_id = generate_id("model_tl")
        model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "image_predictions"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_v{model_version}.keras"
        model_path = model_dir / model_filename
        model.save(str(model_path))

        # Save meta
        meta_path = model_path.with_suffix(".meta.json")
        meta = {
            "algorithm": "transfer_learning",
            "base_model": "MobileNetV2",
            "target_size": TARGET_SIZE,
            "original_width": width,
            "original_height": height,
            "train_pixel_min": float(X_train_flat.min()),
            "train_pixel_max": float(X_train_flat.max()),
            "train_pixel_mean": float(X_train_flat.mean()),
            "train_pixel_std": float(X_train_flat.std()),
            "n_features": len(pixel_cols),
            "n_classes": n_classes,
            "class_names": class_names,
            "image_width": width,
            "image_height": height,
            "fine_tune_layers": fine_tune_layers,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        logger.info(f"Transfer Learning done: train_acc={train_accuracy:.4f}, {n_iter} epochs, {training_time:.1f}s")

        # Evaluate on test set
        y_proba = model.predict(X_test_rgb, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        accuracy, per_class, cm_list = self._compute_metrics(
            y_true, y_pred, n_classes, class_names
        )
        cm = np.array(cm_list)

        logger.info(f"Transfer Learning eval: test_acc={accuracy:.4f}, {len(X_test_rgb)} test, {n_classes} classes")

        # Architecture info
        base_params = base_model.count_params()
        total_params = model.count_params()
        trainable_params = sum(
            int(tf.keras.backend.count_params(w)) for w in model.trainable_weights
        )
        architecture = {
            "layers": [width * height] + [1280] + [256] + [128] + [n_classes],
            "layer_names": [
                f"Input ({width}x{height} -> {TARGET_SIZE}x{TARGET_SIZE} RGB)",
                f"MobileNetV2 ({n_frozen} frozen layers)",
                "Dense Head (256)",
                "Dense Head (128)",
                f"Output ({n_classes} classes)",
            ],
            "activation": "relu",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_type": "transfer_learning",
            "base_model": "MobileNetV2",
        }
        architecture_diagram = self._generate_transfer_learning_architecture_diagram(
            model, base_model, width, height, n_classes, TARGET_SIZE,
            n_frozen, fine_tune_layers, base_params, total_params, trainable_params
        )

        # Explorer data (use flat arrays for gallery rendering)
        prediction_gallery = self._generate_prediction_gallery(
            X_test_flat, y_true, y_pred, y_proba, class_names, width, height
        )
        confusion_analysis = self._generate_confusion_analysis(
            cm, class_names, X_test_flat, y_true, y_pred, width, height
        )
        confidence_distribution = self._generate_confidence_distribution(y_proba, y_true, y_pred)
        metrics_summary = self._generate_metrics_summary(accuracy, per_class, class_names, len(X_test_flat))
        training_analysis = self._generate_training_analysis(
            loss_curve, val_acc_curve, n_iter, input_data.epochs, training_metrics
        )
        feature_importance = self._generate_saliency_map(model, X_test_rgb[:20], TARGET_SIZE, TARGET_SIZE)
        quiz_questions = self._generate_transfer_learning_quiz(
            accuracy, n_classes, class_names, per_class, cm_list, len(X_test_flat),
            width, height, len(X_train_flat), training_time, n_iter,
            n_frozen, fine_tune_layers
        )

        return ImagePredictionsOutput(
            node_type=self.node_type,
            execution_time_ms=int(training_time * 1000),
            model_id=model_id,
            model_path=str(model_path),
            algorithm="transfer_learning",
            training_samples=len(X_train_flat),
            training_time_seconds=training_time,
            training_metrics=training_metrics,
            test_samples=len(X_test_flat),
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
        n_landmarks = n_features // 4  # 33 landmarks x 4 values each

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
                [f"Input ({n_landmarks} landmarks x 4)"]
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

    # ══════════════════════════════════════════════════════════════════
    #  Pose-specific helpers
    # ══════════════════════════════════════════════════════════════════

    def _parse_hidden_layers(self, s: str) -> tuple:
        try:
            parts = [int(x.strip()) for x in s.split(",") if x.strip()]
            return tuple(max(1, p) for p in parts) if parts else (128, 64)
        except ValueError:
            return (128, 64)

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
                layer["description"] = f"{n_landmarks} body landmarks x 4 values (x, y, z, visibility) = {n_features} features"
            elif i == len(sizes) - 1:
                layer["description"] = f"One neuron per pose class, softmax gives probability"
            else:
                layer["description"] = f"{size} neurons with {activation} activation"

            layers_info.append(layer)

        return {
            "layers": layers_info,
            "total_params": architecture["total_params"],
            "input_shape": f"{n_landmarks} landmarks x 4 = {n_features} features",
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
            first_layer_weights = model.coefs_[0]
            feature_importance = np.sum(np.abs(first_layer_weights), axis=1)

            landmark_importance = []
            for i in range(n_landmarks):
                base = i * 4
                lm_imp = float(np.sum(feature_importance[base:base + 4]))
                landmark_importance.append(lm_imp)

            landmark_importance = np.array(landmark_importance)
            lm_min, lm_max = landmark_importance.min(), landmark_importance.max()
            if lm_max - lm_min > 1e-8:
                landmark_importance = (landmark_importance - lm_min) / (lm_max - lm_min)

            ranked = sorted(enumerate(landmark_importance), key=lambda x: -x[1])
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
                "type": "landmark", "landmarks": [], "landmark_importance": [],
                "n_landmarks": n_landmarks, "description": "Not available",
            }

    def _generate_pose_quiz(
        self, accuracy: float, n_classes: int, class_names: list,
        per_class: Dict, cm: list, n_test: int,
        hidden_layers: tuple, activation: str, n_train: int,
        n_features: int, training_time: float, n_iter: int, n_landmarks: int
    ) -> List[Dict[str, Any]]:
        """Pose-specific quiz questions."""
        import random as _random
        questions = [
            {
                "id": "q1",
                "question": f"The model uses {n_features} input features from {n_landmarks} body landmarks. Why is this better than using raw camera pixels?",
                "options": [
                    f"Landmarks are a compact representation ({n_features} vs thousands of pixels) that captures body shape regardless of background, clothing, or lighting",
                    "Pixels are always better than landmarks",
                    "There is no difference",
                    "Landmarks only work for faces, not full body",
                ],
                "correct_answer": 0,
                "explanation": f"MediaPipe extracts {n_landmarks} body joint positions, giving us a clean {n_features}-feature representation invariant to background, lighting, and appearance.",
                "difficulty": "medium",
            },
            {
                "id": "q2",
                "question": f"The model got {accuracy*100:.1f}% accuracy on {n_test} test poses. What does the confusion matrix reveal?",
                "options": [
                    "Which poses are most often confused with each other",
                    "The overall accuracy number",
                    "Which body parts are most important",
                    "The training speed",
                ],
                "correct_answer": 0,
                "explanation": f"The {n_classes}x{n_classes} confusion matrix shows exactly which poses get mixed up.",
                "difficulty": "medium",
            },
            {
                "id": "q3",
                "question": "Why does an MLP work well for pose classification?",
                "options": [
                    "Pose landmarks are tabular data (132 numbers) — perfect for MLP. Raw images need CNNs to preserve spatial structure.",
                    "MLPs always work better than CNNs",
                    "The choice of model doesn't matter",
                    "MLP can't handle pose data",
                ],
                "correct_answer": 0,
                "explanation": f"Pose landmarks are already high-quality features. The {n_features} joint positions are tabular data where MLPs excel.",
                "difficulty": "hard",
            },
            {
                "id": "q4",
                "question": "What does the 'landmark importance' visualization show?",
                "options": [
                    "Which body joints the model relies on most for classification",
                    "Which landmarks are detected most reliably",
                    "The speed of each landmark detection",
                    "The visibility of each body part",
                ],
                "correct_answer": 0,
                "explanation": "By analyzing the MLP's first-layer weights, we see which landmarks carry the most discriminative power.",
                "difficulty": "hard",
            },
        ]

        if per_class:
            classes_sorted = sorted(per_class.items(), key=lambda x: x[1]["f1"])
            worst_name, worst_data = classes_sorted[0]
            best_name, best_data = classes_sorted[-1]
            questions.append({
                "id": "q5",
                "question": f"'{best_name}' has F1={best_data['f1']:.2f} but '{worst_name}' has F1={worst_data['f1']:.2f}. Why?",
                "options": [
                    f"'{worst_name}' might look similar to other poses in body joint positions",
                    "The model always performs equally on all poses",
                    "F1 scores are random",
                    "Some poses use fewer landmarks",
                ],
                "correct_answer": 0,
                "explanation": f"Some poses share similar body positions. '{worst_name}' might overlap with other poses in landmark space.",
                "difficulty": "hard",
            })

        _random.shuffle(questions)
        return questions[:5]

    # ══════════════════════════════════════════════════════════════════
    #  Transfer Learning helpers
    # ══════════════════════════════════════════════════════════════════

    def _generate_transfer_learning_architecture_diagram(
        self, model, base_model, width, height, n_classes, target_size,
        n_frozen, fine_tune_layers, base_params, total_params, trainable_params
    ) -> Dict[str, Any]:
        """Generate architecture diagram showing frozen MobileNetV2 base + trainable head."""
        layers_info = [
            {
                "index": 0,
                "name": f"Input ({width}x{height} grayscale)",
                "size": width * height,
                "type": "input",
                "params": 0,
                "description": f"Original {width}x{height} image resized to {target_size}x{target_size} and converted to 3-channel RGB",
            },
            {
                "index": 1,
                "name": "MobileNetV2 (pretrained on ImageNet)",
                "size": 1280,
                "type": "frozen_block",
                "params": base_params,
                "description": (
                    f"Pretrained on 1.4M ImageNet images. "
                    f"{n_frozen} layers frozen. "
                    + (f"{fine_tune_layers} layers unfrozen for fine-tuning." if fine_tune_layers > 0 else "All layers frozen — only the head trains.")
                ),
                "frozen": n_frozen,
                "trainable": fine_tune_layers,
            },
            {
                "index": 2,
                "name": "GlobalAveragePooling2D",
                "size": 1280,
                "type": "pooling",
                "params": 0,
                "description": "Average all spatial positions into a single 1280-dim vector",
            },
            {
                "index": 3,
                "name": "BatchNormalization",
                "size": 1280,
                "type": "dense",
                "params": 1280 * 4,
                "description": "Normalizes activations for stable training and faster convergence",
            },
            {
                "index": 4,
                "name": "Dense (256, relu)",
                "size": 256,
                "type": "dense",
                "params": 1280 * 256 + 256,
                "activation": "relu",
                "description": "First classification layer — learns to map ImageNet features to your domain",
            },
            {
                "index": 5,
                "name": "Dropout (0.4)",
                "size": 0,
                "type": "dropout",
                "params": 0,
                "description": "Drops 40% of neurons during training to prevent overfitting",
            },
            {
                "index": 6,
                "name": "Dense (128, relu)",
                "size": 128,
                "type": "dense",
                "params": 256 * 128 + 128,
                "activation": "relu",
                "description": "Second classification layer — refines class-specific features",
            },
            {
                "index": 7,
                "name": "Dropout (0.3)",
                "size": 0,
                "type": "dropout",
                "params": 0,
                "description": "Drops 30% of neurons during training to prevent overfitting",
            },
            {
                "index": 8,
                "name": f"Dense ({n_classes}, softmax)",
                "size": n_classes,
                "type": "output",
                "params": 128 * n_classes + n_classes,
                "activation": "softmax",
                "description": f"Output: {n_classes} classes with softmax probabilities",
            },
        ]

        return {
            "layers": layers_info,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_shape": f"{width}x{height}x1 -> {target_size}x{target_size}x3",
            "output_shape": f"{n_classes} classes",
            "model_type": "transfer_learning",
            "base_model": "MobileNetV2",
        }

    def _generate_transfer_learning_quiz(
        self, accuracy, n_classes, class_names, per_class, cm_list, n_test,
        width, height, n_train, training_time, n_iter, n_frozen, fine_tune_layers
    ) -> List[Dict[str, Any]]:
        """Transfer learning-specific quiz questions."""
        import random as _random
        questions = [
            {
                "id": "q1",
                "question": "What is transfer learning?",
                "options": [
                    "Using a model pretrained on a large dataset (ImageNet) and adapting it to a new task by training only a small classification head on top",
                    "Training a model from scratch on multiple datasets at once",
                    "Transferring data between different computers",
                    "A technique that only works with text models",
                ],
                "correct_answer": 0,
                "explanation": f"MobileNetV2 was pretrained on 1.4 million ImageNet images. We freeze its learned features and only train a small head on your {n_train} images.",
                "difficulty": "medium",
            },
            {
                "id": "q2",
                "question": f"Why did we resize {width}x{height} images to 96x96 and convert from grayscale to RGB?",
                "options": [
                    "MobileNetV2 was trained on RGB images of at least 96x96 — it expects that input format",
                    "Larger images always give better results",
                    "RGB is required for all neural networks",
                    "96x96 is the only size any CNN can accept",
                ],
                "correct_answer": 0,
                "explanation": f"MobileNetV2's convolutional filters were learned on RGB images. We upscale from {width}x{height} and replicate the channel 3x.",
                "difficulty": "hard",
            },
            {
                "id": "q3",
                "question": f"The model has {n_frozen} frozen layers. Why freeze them?",
                "options": [
                    f"With only {n_train} training images, updating millions of pretrained parameters would cause severe overfitting",
                    "Frozen layers are faster to compute",
                    "It's impossible to train pretrained layers",
                    "Frozen layers always give better accuracy",
                ],
                "correct_answer": 0,
                "explanation": f"MobileNetV2 has ~2.2M parameters learned from 1.4M images. Your dataset has {n_train} images — freezing prevents overfitting.",
                "difficulty": "hard",
            },
            {
                "id": "q4",
                "question": "What does GlobalAveragePooling2D do?",
                "options": [
                    "Averages each feature map into a single number, converting spatial maps to a compact 1280-dim vector",
                    "It increases the image resolution",
                    "It's the same as Flatten",
                    "It applies a global filter to detect objects",
                ],
                "correct_answer": 0,
                "explanation": "MobileNetV2 outputs feature maps. GlobalAveragePooling2D takes the mean of each, producing a 1280-dim vector.",
                "difficulty": "hard",
            },
            {
                "id": "q5",
                "question": f"This model achieved {accuracy*100:.1f}% accuracy. How does transfer learning compare to training from scratch?",
                "options": [
                    "Transfer learning typically achieves higher accuracy with less data because the pretrained features capture universal visual patterns",
                    "Transfer learning always gets the same accuracy",
                    "Training from scratch is always better",
                    "The comparison depends only on the number of epochs",
                ],
                "correct_answer": 0,
                "explanation": f"With {n_train} images, training from scratch must learn edges, textures, etc. MobileNetV2 already knows these from ImageNet.",
                "difficulty": "medium",
            },
            {
                "id": "q6",
                "question": "Why does the model use data augmentation during training?",
                "options": [
                    "It creates modified copies of images (rotated, shifted, zoomed) to make the model more robust with limited data",
                    "It increases the image resolution",
                    "It removes bad images from the dataset",
                    "It downloads more images from the internet",
                ],
                "correct_answer": 0,
                "explanation": f"With {n_train} images, the model can easily memorize. Augmentation generates variations so the model generalizes better.",
                "difficulty": "medium",
            },
        ]

        if fine_tune_layers > 0:
            questions.append({
                "id": "q7",
                "question": f"You unfroze the last {fine_tune_layers} layers for fine-tuning. What does this do?",
                "options": [
                    "Allows top MobileNetV2 layers to adapt features for your dataset, at the risk of overfitting",
                    "Makes training faster",
                    "Reduces the number of parameters",
                    "Has no effect on the model",
                ],
                "correct_answer": 0,
                "explanation": f"Fine-tuning lets the last {fine_tune_layers} layers update. Use a very small learning rate to avoid destroying pretrained knowledge.",
                "difficulty": "hard",
            })

        _random.shuffle(questions)
        return questions[:5]

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

    def _generate_saliency_map(self, model, X_sample, width, height):
        """Compute a saliency map by averaging gradient magnitudes."""
        import tensorflow as tf
        try:
            X_tensor = tf.constant(X_sample, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = model(X_tensor, training=False)
                top_class = tf.reduce_max(predictions, axis=1)
                loss = tf.reduce_mean(top_class)
            grads = tape.gradient(loss, X_tensor)
            saliency = tf.reduce_mean(tf.abs(grads), axis=0).numpy()  # (H, W, 3)
            saliency = np.mean(saliency, axis=-1)  # (H, W) — average across RGB channels
            saliency_flat = saliency.flatten()
            sal_min, sal_max = saliency_flat.min(), saliency_flat.max()
            if sal_max - sal_min > 1e-8:
                saliency_flat = (saliency_flat - sal_min) / (sal_max - sal_min)
            return {
                "heatmap": saliency_flat.tolist(),
                "width": width,
                "height": height,
                "description": (
                    "Saliency map: brighter pixels have more influence on predictions. "
                    "Shows which spatial regions the model focuses on."
                ),
            }
        except Exception as e:
            logger.warning(f"Saliency map generation failed: {e}")
            return {"heatmap": [], "width": width, "height": height, "description": "Not available"}

    # ══════════════════════════════════════════════════════════════════
    #  Shared Explorer Data Generators
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
                f"Training converged in {n_iter} epochs (limit was {max_iter})."
            )
        else:
            analysis["convergence"] = "hit_limit"
            analysis["convergence_message"] = (
                f"Training used all {max_iter} epochs. "
                f"Try increasing epochs or adjusting the learning rate."
            )

        return analysis
