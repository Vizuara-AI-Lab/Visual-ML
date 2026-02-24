"""
CNN Classifier Node — Train a neural network classifier on image data.
Uses sklearn's MLPClassifier on flattened pixel features (no PyTorch needed).
Returns rich explorer data: architecture visualization, training curves, and quizzes.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
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


class CNNClassifierInput(NodeInput):
    """Input schema for CNN Classifier node."""
    train_dataset_id: str = Field(..., description="Training dataset ID")
    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: Optional[str] = Field(
        None, description="Target column name (default: 'label')"
    )

    # Architecture
    hidden_layer_sizes: str = Field(
        "256, 128, 64", description="Hidden layer sizes (comma-separated)"
    )
    activation: str = Field("relu", description="Activation: relu, tanh, logistic")

    # Training
    solver: str = Field("adam", description="Optimizer: adam, sgd, lbfgs")
    max_iter: int = Field(100, description="Maximum training iterations")
    learning_rate_init: float = Field(0.001, description="Initial learning rate")
    alpha: float = Field(0.0001, description="L2 regularization strength")
    batch_size: str = Field("auto", description="Mini-batch size ('auto' or integer)")
    early_stopping: bool = Field(True, description="Stop when validation score plateaus")
    random_state: int = Field(42, description="Random seed")

    # Pass-through
    image_width: Optional[int] = Field(None, description="Image width")
    image_height: Optional[int] = Field(None, description="Image height")
    n_channels: Optional[int] = Field(None, description="Channels")
    n_classes: Optional[int] = Field(None, description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Class names")


class CNNClassifierOutput(NodeOutput):
    """Output schema for CNN Classifier node."""
    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")
    training_samples: int = Field(..., description="Training sample count")
    n_features: int = Field(..., description="Number of features (pixels)")
    n_classes: int = Field(..., description="Number of classes")
    training_metrics: Dict[str, Any] = Field(..., description="Training metrics")
    training_time_seconds: float = Field(..., description="Training duration")
    class_names: list = Field(..., description="Class names")

    # Pass-through
    test_dataset_id: Optional[str] = Field(None, description="Test dataset ID")
    target_column: Optional[str] = Field(None, description="Target column")
    image_width: Optional[int] = Field(None, description="Image width")
    image_height: Optional[int] = Field(None, description="Image height")

    # Training details
    loss_curve: Optional[List[float]] = Field(None, description="Training loss curve")
    validation_scores: Optional[List[float]] = Field(None, description="Validation scores")
    architecture: Optional[Dict[str, Any]] = Field(None, description="Architecture details")
    n_iterations: Optional[int] = Field(None, description="Actual training iterations")

    # Explorer data
    architecture_diagram: Optional[Dict[str, Any]] = Field(
        None, description="Layer-by-layer architecture visualization data"
    )
    training_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Training curve analysis"
    )
    feature_importance: Optional[Dict[str, Any]] = Field(
        None, description="Which pixels matter most (weight magnitude heatmap)"
    )
    class_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Training data class distribution"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions"
    )


class CNNClassifierNode(BaseNode):
    """
    CNN Classifier Node — Train neural network on image pixel features.
    Uses sklearn's MLPClassifier for simplicity (no PyTorch dependency).
    """

    node_type = "cnn_classifier"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.ML_ALGORITHM,
            primary_output_field="model_id",
            output_fields={
                "model_id": "Model identifier",
                "model_path": "Saved model path",
                "training_metrics": "Training metrics",
                "test_dataset_id": "Test dataset ID",
                "target_column": "Target column name",
                "image_width": "Image width",
                "image_height": "Image height",
            },
            requires_input=True,
            can_branch=True,
            produces_dataset=False,
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.DATA_TRANSFORM,
                NodeCategory.PREPROCESSING,
            ],
        )

    def get_input_schema(self):
        return CNNClassifierInput

    def get_output_schema(self):
        return CNNClassifierOutput

    def _parse_hidden_layers(self, s: str) -> tuple:
        try:
            parts = [int(x.strip()) for x in s.split(",") if x.strip()]
            return tuple(max(1, p) for p in parts) if parts else (128,)
        except ValueError:
            return (128,)

    def _parse_batch_size(self, s: str):
        if s.strip().lower() == "auto":
            return "auto"
        try:
            return max(1, int(s.strip()))
        except ValueError:
            return "auto"

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

    async def _execute(self, input_data: CNNClassifierInput) -> CNNClassifierOutput:
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            import joblib

            logger.info("Training CNN Classifier on image data")

            # Load training data
            df_train = await self._load_dataset(input_data.train_dataset_id)
            if df_train is None or df_train.empty:
                raise ValueError(f"Training dataset {input_data.train_dataset_id} not found or empty")

            target_col = input_data.target_column or "label"
            if target_col not in df_train.columns:
                # Try last column
                target_col = df_train.columns[-1]

            pixel_cols = [c for c in df_train.columns if c != target_col]
            X_train = df_train[pixel_cols].values
            y_train = df_train[target_col].values

            n_classes = len(np.unique(y_train))
            class_names = input_data.class_names or [str(i) for i in range(n_classes)]
            width = input_data.image_width or int(np.sqrt(len(pixel_cols)))
            height = input_data.image_height or int(np.sqrt(len(pixel_cols)))

            hidden_layers = self._parse_hidden_layers(input_data.hidden_layer_sizes)
            batch_size = self._parse_batch_size(input_data.batch_size)

            logger.info(
                f"Architecture: {len(pixel_cols)} input → {hidden_layers} → {n_classes} classes, "
                f"{len(X_train)} training samples"
            )

            # Train
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=input_data.activation,
                solver=input_data.solver,
                max_iter=input_data.max_iter,
                learning_rate_init=input_data.learning_rate_init,
                alpha=input_data.alpha,
                batch_size=batch_size,
                early_stopping=input_data.early_stopping,
                random_state=input_data.random_state,
                verbose=False,
            )

            training_start = datetime.utcnow()
            model.fit(X_train, y_train)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Metrics on training data
            y_pred = model.predict(X_train)
            metrics = {
                "accuracy": round(float(accuracy_score(y_train, y_pred)), 4),
                "precision_macro": round(float(precision_score(y_train, y_pred, average="macro", zero_division=0)), 4),
                "recall_macro": round(float(recall_score(y_train, y_pred, average="macro", zero_division=0)), 4),
                "f1_macro": round(float(f1_score(y_train, y_pred, average="macro", zero_division=0)), 4),
            }

            # Save model
            model_id = generate_id("model_cnn")
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "cnn_classifier"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename
            joblib.dump(model, model_path)

            loss_curve = list(model.loss_curve_) if hasattr(model, "loss_curve_") else []
            val_scores = list(model.validation_scores_) if hasattr(model, "validation_scores_") and model.validation_scores_ else []
            n_iter = model.n_iter_ if hasattr(model, "n_iter_") else 0

            # Architecture info
            layers = [len(pixel_cols)] + list(hidden_layers) + [n_classes]
            architecture = {
                "layers": layers,
                "layer_names": ["Input (pixels)"] + [f"Hidden ({s} neurons)" for s in hidden_layers] + [f"Output ({n_classes} classes)"],
                "activation": input_data.activation,
                "total_params": sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1)),
            }

            logger.info(
                f"CNN Classifier trained: accuracy={metrics['accuracy']:.4f}, "
                f"{n_iter} iterations, {training_time:.1f}s"
            )

            # --- Explorer data ---
            architecture_diagram = self._generate_architecture_diagram(
                architecture, hidden_layers, input_data.activation, width, height
            )
            training_analysis = self._generate_training_analysis(
                loss_curve, val_scores, n_iter, input_data.max_iter, metrics
            )
            feature_importance = self._generate_feature_importance(
                model, width, height, len(pixel_cols)
            )
            class_distribution = self._generate_class_distribution(y_train, class_names)
            quiz_questions = self._generate_quiz(
                metrics, n_classes, class_names, hidden_layers, input_data.activation,
                len(X_train), len(pixel_cols), training_time, n_iter
            )

            return CNNClassifierOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=len(pixel_cols),
                n_classes=n_classes,
                training_metrics=metrics,
                training_time_seconds=training_time,
                class_names=class_names,
                test_dataset_id=input_data.test_dataset_id,
                target_column=target_col,
                image_width=width,
                image_height=height,
                loss_curve=loss_curve,
                validation_scores=val_scores,
                architecture=architecture,
                n_iterations=n_iter,
                architecture_diagram=architecture_diagram,
                training_analysis=training_analysis,
                feature_importance=feature_importance,
                class_distribution=class_distribution,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"CNN Classifier training failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Explorer Data Generators ---

    def _generate_architecture_diagram(
        self, architecture: Dict[str, Any], hidden_layers: tuple,
        activation: str, width: int, height: int
    ) -> Dict[str, Any]:
        """Architecture visualization data for layer-by-layer diagram."""
        layers_info = []
        sizes = architecture["layers"]
        names = architecture["layer_names"]

        for i, (size, name) in enumerate(zip(sizes, names)):
            layer = {
                "index": i,
                "name": name,
                "size": size,
                "type": "input" if i == 0 else "output" if i == len(sizes) - 1 else "hidden",
            }

            if i > 0:
                # Count params: weights + biases
                params = sizes[i-1] * size + size
                layer["params"] = params
                layer["activation"] = activation if i < len(sizes) - 1 else "softmax"

            if i == 0:
                layer["description"] = f"Each {width}×{height} image flattened to {size} pixel values"
            elif i == len(sizes) - 1:
                layer["description"] = f"One neuron per class, softmax gives probability for each class"
            else:
                layer["description"] = f"{size} neurons with {activation} activation"

            layers_info.append(layer)

        return {
            "layers": layers_info,
            "total_params": architecture["total_params"],
            "input_shape": f"{width}×{height} = {width*height} pixels",
            "output_shape": f"{sizes[-1]} classes",
        }

    def _generate_training_analysis(
        self, loss_curve: list, val_scores: list,
        n_iter: int, max_iter: int, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Training curve analysis."""
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
                f"✅ Training converged in {n_iter} iterations (limit was {max_iter}). "
                f"The model successfully found a good solution."
            )
        else:
            analysis["convergence"] = "hit_limit"
            analysis["convergence_message"] = (
                f"⚠️ Training reached the maximum of {max_iter} iterations without fully converging. "
                f"Try increasing max_iter or adjusting the learning rate."
            )

        # Detect overfitting signal
        if len(loss_curve) > 10:
            recent_loss = np.mean(loss_curve[-5:])
            early_loss = np.mean(loss_curve[:5])
            if recent_loss > early_loss * 0.99 and metrics.get("accuracy", 0) > 0.95:
                analysis["overfitting_warning"] = (
                    "⚠️ High training accuracy with flattening loss suggests possible overfitting. "
                    "Check test set performance!"
                )

        return analysis

    def _generate_feature_importance(
        self, model, width: int, height: int, n_features: int
    ) -> Dict[str, Any]:
        """Compute which pixels matter most (first layer weight magnitudes)."""
        try:
            first_layer_weights = model.coefs_[0]  # shape: (n_features, hidden_size)
            # Sum absolute weights per input feature
            importance = np.sum(np.abs(first_layer_weights), axis=1)
            # Normalize to 0-1
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

            return {
                "heatmap": importance.tolist(),
                "width": width,
                "height": height,
                "description": (
                    "Brighter pixels = more important for classification. "
                    "This shows which parts of the image the model pays most attention to."
                ),
            }
        except Exception as e:
            logger.warning(f"Feature importance generation failed: {e}")
            return {"heatmap": [], "width": width, "height": height, "description": "Not available"}

    def _generate_class_distribution(
        self, y: np.ndarray, class_names: list
    ) -> Dict[str, Any]:
        """Training data class balance."""
        total = len(y)
        classes = []
        for idx, name in enumerate(class_names):
            count = int(np.sum(y == idx))
            classes.append({
                "name": str(name),
                "count": count,
                "percentage": round(count / total * 100, 1),
            })
        return {"total": total, "classes": classes}

    def _generate_quiz(
        self, metrics: Dict, n_classes: int, class_names: list,
        hidden_layers: tuple, activation: str, n_samples: int,
        n_features: int, training_time: float, n_iter: int
    ) -> List[Dict[str, Any]]:
        """Quiz questions about CNN/MLP image classification."""
        import random as _random
        questions = []
        q_id = 0

        acc = metrics.get("accuracy", 0)
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The model achieved {acc*100:.1f}% accuracy. With {n_classes} classes, what would random guessing give?",
            "options": [
                f"~{100/n_classes:.1f}% (1 ÷ {n_classes} classes)",
                "50% always",
                "0% — random guessing never works",
                "100% — if you guess enough times",
            ],
            "correct_answer": 0,
            "explanation": f"Random guessing with {n_classes} classes gives 1/{n_classes} = {100/n_classes:.1f}% accuracy. Our model achieved {acc*100:.1f}%, which is {'much' if acc > 2/n_classes else 'slightly'} better than random, showing it has learned meaningful patterns.",
            "difficulty": "easy",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This network has {n_features} input features. Why so many?",
            "options": [
                f"Each pixel is a feature — a {int(np.sqrt(n_features))}×{int(np.sqrt(n_features))} image has {n_features} pixels",
                "The model created extra features",
                "It's always 784 for image models",
                "The number is random",
            ],
            "correct_answer": 0,
            "explanation": f"When we flatten a {int(np.sqrt(n_features))}×{int(np.sqrt(n_features))} image, each pixel becomes a separate feature. So {n_features} pixel values = {n_features} input features. This is why image models need many neurons!",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why do real CNN architectures use convolutional layers instead of flattening?",
            "options": [
                "Conv layers preserve spatial structure (neighbors matter) and share weights, using far fewer parameters",
                "Conv layers are slower but more accurate",
                "Flattening always gives better results",
                "There is no difference",
            ],
            "correct_answer": 0,
            "explanation": f"Flattening loses spatial info — pixel (0,0) and pixel ({int(np.sqrt(n_features))-1},{int(np.sqrt(n_features))-1}) look equally distant. Convolutional layers use small filters that slide over the image, detecting local patterns (edges, curves, shapes) while using shared weights — far fewer parameters than a fully-connected layer.",
            "difficulty": "hard",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"Training took {training_time:.1f} seconds for {n_samples} images. What would happen with 100x more data?",
            "options": [
                "Training would take much longer (potentially 100x+), but accuracy should improve if the model has enough capacity",
                "Training time wouldn't change",
                "More data always means worse accuracy",
                "The model would refuse to train",
            ],
            "correct_answer": 0,
            "explanation": f"More data = longer training but better generalization. With {n_samples} images, training took {training_time:.1f}s. With 100x data ({n_samples*100}), training could take minutes. This is why real CNN training uses GPUs — they process images in parallel.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The architecture uses layers {hidden_layers}. Why do later layers have fewer neurons?",
            "options": [
                "Each layer extracts higher-level, more abstract features — fewer neurons needed to represent them",
                "Fewer neurons means faster training, no other reason",
                "It's just a common convention with no technical reason",
                "Layers must always decrease in size",
            ],
            "correct_answer": 0,
            "explanation": f"The 'funnel' shape ({' → '.join(str(h) for h in hidden_layers)}) is a common pattern. Early layers detect low-level features (edges, corners), middle layers combine them into shapes, and final layers recognize high-level concepts. Fewer neurons suffice for abstract representations.",
            "difficulty": "hard",
        })

        _random.shuffle(questions)
        return questions[:5]
