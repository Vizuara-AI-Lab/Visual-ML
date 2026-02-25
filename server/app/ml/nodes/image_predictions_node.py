"""
Image Predictions Node — Run predictions on test set, generate evaluation metrics.
Produces confusion matrix, per-class metrics, sample prediction gallery.
Returns rich explorer data for interactive analysis.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
from pydantic import Field
from pathlib import Path
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
    model_id: str = Field(..., description="Model ID from CNN classifier")
    model_path: str = Field(..., description="Path to saved model file")
    test_dataset_id: str = Field(..., description="Test dataset ID")
    target_column: Optional[str] = Field("label", description="Target column name")

    # Pass-through
    image_width: Optional[int] = Field(None, description="Image width")
    image_height: Optional[int] = Field(None, description="Image height")
    n_classes: Optional[int] = Field(None, description="Number of classes")
    class_names: Optional[List[str]] = Field(None, description="Class names")


class ImagePredictionsOutput(NodeOutput):
    """Output schema for Image Predictions node."""
    model_id: str = Field(..., description="Model ID used")
    test_samples: int = Field(..., description="Number of test images evaluated")
    overall_accuracy: float = Field(..., description="Overall accuracy [0-1]")

    # Per-class metrics
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

    # Pass-through for live testing tabs (Draw / Upload / Live Camera)
    model_path: Optional[str] = Field(None, description="Path to saved model file (for live testing)")
    image_width: Optional[int] = Field(None, description="Image width for model input")
    image_height: Optional[int] = Field(None, description="Image height for model input")


class ImagePredictionsNode(BaseNode):
    """
    Image Predictions Node — Evaluate trained model on test images.
    """

    node_type = "image_predictions"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.VIEW,
            primary_output_field="overall_accuracy",
            output_fields={
                "overall_accuracy": "Overall test accuracy",
                "per_class_metrics": "Per-class precision/recall/f1",
                "confusion_matrix": "Confusion matrix",
            },
            requires_input=True,
            can_branch=False,
            produces_dataset=False,
            max_inputs=1,
            allowed_source_categories=[NodeCategory.ML_ALGORITHM],
        )

    def get_input_schema(self):
        return ImagePredictionsInput

    def get_output_schema(self):
        return ImagePredictionsOutput

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

    async def _execute(self, input_data: ImagePredictionsInput) -> ImagePredictionsOutput:
        try:
            import joblib
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                confusion_matrix as sklearn_cm
            )

            logger.info(f"Running predictions with model {input_data.model_id}")

            # Load model
            model_path = Path(input_data.model_path)
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
            model = joblib.load(model_path)

            # Load test data
            df_test = await self._load_dataset(input_data.test_dataset_id)
            if df_test is None or df_test.empty:
                raise ValueError(f"Test dataset {input_data.test_dataset_id} not found or empty")

            target_col = input_data.target_column or "label"
            if target_col not in df_test.columns:
                target_col = df_test.columns[-1]

            pixel_cols = [c for c in df_test.columns if c != target_col]
            X_test = df_test[pixel_cols].values
            y_true = df_test[target_col].values.astype(int)

            n_classes = input_data.n_classes or len(np.unique(y_true))
            class_names = input_data.class_names or [str(i) for i in range(n_classes)]
            width = input_data.image_width or int(np.sqrt(len(pixel_cols)))
            height = input_data.image_height or int(np.sqrt(len(pixel_cols)))

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            # Overall metrics
            accuracy = float(accuracy_score(y_true, y_pred))

            # Per-class metrics
            per_class = {}
            labels = list(range(n_classes))
            prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
            rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

            for i, name in enumerate(class_names):
                n_true = int(np.sum(y_true == i))
                n_pred = int(np.sum(y_pred == i))
                n_correct = int(np.sum((y_true == i) & (y_pred == i)))
                per_class[name] = {
                    "precision": round(float(prec[i]), 4),
                    "recall": round(float(rec[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": n_true,
                    "predicted": n_pred,
                    "correct": n_correct,
                }

            # Confusion matrix
            cm = sklearn_cm(y_true, y_pred, labels=labels)
            cm_list = cm.tolist()

            logger.info(
                f"Predictions complete: accuracy={accuracy:.4f}, "
                f"{len(X_test)} test images, {n_classes} classes"
            )

            # --- Explorer data ---
            prediction_gallery = self._generate_prediction_gallery(
                X_test, y_true, y_pred, y_proba, class_names, width, height
            )
            confusion_analysis = self._generate_confusion_analysis(
                cm, class_names, X_test, y_true, y_pred, width, height
            )
            confidence_distribution = self._generate_confidence_distribution(
                y_proba, y_true, y_pred
            )
            metrics_summary = self._generate_metrics_summary(
                accuracy, per_class, class_names, len(X_test)
            )
            quiz_questions = self._generate_quiz(
                accuracy, n_classes, class_names, per_class, cm_list, len(X_test)
            )

            return ImagePredictionsOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                model_id=input_data.model_id,
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
                model_path=input_data.model_path,
                image_width=width,
                image_height=height,
            )

        except Exception as e:
            logger.error(f"Image predictions failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Explorer Data Generators ---

    def _generate_prediction_gallery(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
        y_proba: Optional[np.ndarray], class_names: list,
        width: int, height: int, n_correct: int = 12, n_wrong: int = 12
    ) -> List[Dict[str, Any]]:
        """Sample predictions with images, showing correct and incorrect."""
        rng = np.random.default_rng(42)
        gallery = []

        correct_mask = y_true == y_pred
        wrong_mask = ~correct_mask

        # Correct predictions
        correct_indices = np.where(correct_mask)[0]
        if len(correct_indices) > 0:
            chosen = rng.choice(correct_indices, size=min(n_correct, len(correct_indices)), replace=False)
            for idx in chosen:
                item = {
                    "pixels": X[idx].tolist(),
                    "width": width,
                    "height": height,
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

        # Wrong predictions
        wrong_indices = np.where(wrong_mask)[0]
        if len(wrong_indices) > 0:
            chosen = rng.choice(wrong_indices, size=min(n_wrong, len(wrong_indices)), replace=False)
            for idx in chosen:
                item = {
                    "pixels": X[idx].tolist(),
                    "width": width,
                    "height": height,
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
        """Analyze confusion matrix: most confused pairs with examples."""
        rng = np.random.default_rng(42)
        n = len(class_names)

        # Find most confused pairs
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

        # Add example images for top confused pairs
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

        # Per-class accuracy
        per_class_accuracy = []
        for i, name in enumerate(class_names):
            total = int(cm[i].sum())
            correct = int(cm[i][i])
            acc = round(correct / total, 4) if total > 0 else 0
            per_class_accuracy.append({
                "class_name": str(name),
                "accuracy": acc,
                "correct": correct,
                "total": total,
            })

        return {
            "confusion_matrix": cm.tolist(),
            "class_names": [str(c) for c in class_names],
            "most_confused_pairs": confused_pairs[:10],
            "per_class_accuracy": per_class_accuracy,
        }

    def _generate_confidence_distribution(
        self, y_proba: Optional[np.ndarray],
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Distribution of prediction confidence for correct vs incorrect."""
        if y_proba is None:
            return {"has_data": False}

        max_proba = np.max(y_proba, axis=1)
        correct_mask = y_true == y_pred

        correct_conf = max_proba[correct_mask]
        wrong_conf = max_proba[~correct_mask]

        # Histograms
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
        """Visual dashboard metrics summary."""
        # Best/worst classes
        sorted_classes = sorted(
            [(name, data) for name, data in per_class.items()],
            key=lambda x: x[1]["f1"],
            reverse=True
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
            "best_class": {
                "name": best_class[0],
                "f1": best_class[1]["f1"],
            } if best_class else None,
            "worst_class": {
                "name": worst_class[0],
                "f1": worst_class[1]["f1"],
            } if worst_class else None,
            "grade": (
                "A+" if accuracy >= 0.95 else
                "A" if accuracy >= 0.90 else
                "B" if accuracy >= 0.80 else
                "C" if accuracy >= 0.70 else
                "D" if accuracy >= 0.60 else "F"
            ),
        }

    def _generate_quiz(
        self, accuracy: float, n_classes: int, class_names: list,
        per_class: Dict, cm: list, n_test: int
    ) -> List[Dict[str, Any]]:
        """Quiz about model evaluation."""
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
            "explanation": f"With {n_classes} classes, if one class has 90% of samples, a model predicting only that class gets 90% accuracy while failing on all others! Precision (of those predicted as X, how many are actually X), recall (of actual X, how many were found), and F1 (harmonic mean) give a complete picture.",
            "difficulty": "medium",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does a confusion matrix show?",
            "options": [
                "For each true class (row), how many images were predicted as each class (column) — showing where the model gets confused",
                "How confused the model is overall",
                "The training loss over time",
                "Which images are most similar",
            ],
            "correct_answer": 0,
            "explanation": f"The {n_classes}×{n_classes} confusion matrix has rows = true labels, columns = predicted labels. Diagonal cells are correct predictions. Off-diagonal cells show errors — e.g., if cell [3,8] is high, the model often confuses class '{class_names[3] if 3 < len(class_names) else 3}' as '{class_names[8] if 8 < len(class_names) else 8}'.",
            "difficulty": "medium",
        })

        # Find interesting per-class comparison
        if per_class:
            classes_sorted = sorted(per_class.items(), key=lambda x: x[1]["f1"])
            worst_name, worst_data = classes_sorted[0]
            best_name, best_data = classes_sorted[-1]

            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"'{best_name}' has F1={best_data['f1']:.2f} but '{worst_name}' has F1={worst_data['f1']:.2f}. What could cause this difference?",
                "options": [
                    f"'{worst_name}' might look similar to other classes, have fewer training examples, or have more visual variation",
                    "The model always performs equally on all classes",
                    "F1 scores are random",
                    f"'{worst_name}' images are larger",
                ],
                "correct_answer": 0,
                "explanation": f"Some classes are harder to classify. '{worst_name}' (F1={worst_data['f1']:.2f}) might be confused with visually similar classes, or the training set might have fewer/less-variety examples of this class. Check the confusion matrix to see which classes it's confused with!",
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
            "explanation": "Think of it as: Precision answers 'when I predict X, how often am I right?' (quality of predictions). Recall answers 'of all actual X items, how many did I catch?' (coverage). High precision + low recall = picky but accurate. Low precision + high recall = catches everything but with many false positives.",
            "difficulty": "hard",
        })

        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "If the model has 95% training accuracy but 60% test accuracy, what happened?",
            "options": [
                "Overfitting — the model memorized training data instead of learning generalizable patterns",
                "The model is working perfectly",
                "The test set is broken",
                "Underfitting — the model is too simple",
            ],
            "correct_answer": 0,
            "explanation": "A large gap between training accuracy (95%) and test accuracy (60%) is a classic sign of overfitting. The model learned to recognize specific training images rather than general class features. Solutions: more training data, augmentation, regularization, or simpler architecture.",
            "difficulty": "medium",
        })

        _random.shuffle(questions)
        return questions[:5]
