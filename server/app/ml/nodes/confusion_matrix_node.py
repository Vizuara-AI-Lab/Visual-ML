"""
Confusion Matrix Node - Computes confusion matrix for classification models.
Shows TP, TN, FP, FN and overall accuracy.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import io
import joblib
from pydantic import Field
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.ml.algorithms.classification.logistic_regression import LogisticRegression
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class ConfusionMatrixInput(NodeInput):
    """Input schema for Confusion Matrix node."""

    model_output_id: str = Field(..., description="Model output ID from classification model")

    # Optional fields from ML model output (auto-filled by engine)
    model_id: Optional[str] = Field(None, description="Model ID")
    model_path: Optional[str] = Field(None, description="Path to saved model")

    test_dataset_id: Optional[str] = Field(
        None, description="Test dataset ID (auto-filled from split node)"
    )
    target_column: Optional[str] = Field(
        None, description="Target column (auto-filled from split node)"
    )

    # UI options
    show_percentages: bool = Field(True, description="Show percentages in addition to counts")
    color_scheme: str = Field("blue", description="Color scheme for visualization")


class ConfusionMatrixOutput(NodeOutput):
    """Output schema for Confusion Matrix node."""

    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix as 2D array")
    class_labels: List[str] = Field(..., description="Class labels")
    total_samples: int = Field(..., description="Total number of test samples")
    accuracy: float = Field(..., description="Overall accuracy score")

    # Per-class metrics
    true_positives: Dict[str, int] = Field(..., description="True positives per class")
    false_positives: Dict[str, int] = Field(..., description="False positives per class")
    true_negatives: Dict[str, int] = Field(..., description="True negatives per class")
    false_negatives: Dict[str, int] = Field(..., description="False negatives per class")


class ConfusionMatrixNode(BaseNode):
    """
    Confusion Matrix Node - Compute confusion matrix for classification models.

    Responsibilities:
    - Load trained classification model
    - Load test dataset
    - Make predictions
    - Calculate confusion matrix
    - Return structured output with TP/TN/FP/FN
    """

    node_type = "confusion_matrix"

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return ConfusionMatrixInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return ConfusionMatrixOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            # Recognize common missing value indicators: ?, NA, N/A, null, empty strings
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            # FIRST: Try to load from uploads folder (for train/test datasets from split node)
            upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
            if upload_path.exists():
                logger.info(f"Loading dataset from uploads folder: {upload_path}")
                df = pd.read_csv(upload_path, na_values=missing_values, keep_default_na=True)
                return df

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

    async def _execute(self, input_data: ConfusionMatrixInput) -> ConfusionMatrixOutput:
        """
        Execute confusion matrix computation.

        Args:
            input_data: Validated input data

        Returns:
            Confusion matrix result with metrics
        """
        try:
            logger.info(f"Computing confusion matrix for model: {input_data.model_output_id}")

            # Extract test_dataset_id and target_column from input_data
            # These should be auto-filled from split node via the pipeline engine
            test_dataset_id = input_data.test_dataset_id
            target_column = input_data.target_column

            # If not in input_data, try to get from model_output_id context
            # The engine passes split node outputs (test_dataset_id, target_column) through ML nodes
            if not test_dataset_id:
                # Try to find in the merged input (check for test_dataset_id in the input dict)
                test_dataset_id = getattr(input_data, "test_dataset_id", None)

            if not target_column:
                target_column = getattr(input_data, "target_column", None)

            if not test_dataset_id:
                raise ValueError(
                    "Test dataset ID must be provided. "
                    "Connect the confusion_matrix node to a classification model that's connected to a split node."
                )

            if not target_column:
                raise ValueError(
                    "Target column must be provided. "
                    "Ensure the split node and model are configured correctly."
                )

            # Load test dataset
            df_test = await self._load_dataset(test_dataset_id)
            if df_test is None or df_test.empty:
                raise InvalidDatasetError(f"Dataset {test_dataset_id} not found or empty")

            # Validate target column
            if target_column not in df_test.columns:
                raise ValueError(f"Target column '{target_column}' not found in test data")

            # Load model - check if we have model_path in input_data (passed from ML node)
            # The ML node outputs both model_id and model_path, we need model_path
            model_path_str = getattr(input_data, "model_path", None) or input_data.model_output_id

            model_path = Path(model_path_str)
            if not model_path.exists():
                # Try relative to MODEL_ARTIFACTS_DIR
                model_path = Path(settings.MODEL_ARTIFACTS_DIR) / model_path_str
                if not model_path.exists():
                    raise ValueError(f"Model not found: {model_path_str}")

            logger.info(f"Loading model from: {model_path}")
            model = LogisticRegression.load(model_path)

            # Separate features and target
            X_test = df_test.drop(columns=[target_column])
            y_test = df_test[target_column]

            # Make predictions
            y_pred = model.predict(X_test)

            # Get class labels (keep original types for sklearn compatibility)
            if hasattr(model, "classes_"):
                labels_for_cm = list(model.classes_)  # Keep original types (int, str, etc.)
            else:
                labels_for_cm = sorted(y_test.unique().tolist())  # Keep original types

            # Convert to strings only for output/display
            class_labels = [str(label) for label in labels_for_cm]

            # Compute confusion matrix using original type labels
            cm = confusion_matrix(y_test, y_pred, labels=labels_for_cm)

            # Compute accuracy
            acc = accuracy_score(y_test, y_pred)

            # Compute per-class metrics (TP, TN, FP, FN)
            tp_dict = {}
            tn_dict = {}
            fp_dict = {}
            fn_dict = {}

            for idx, label in enumerate(class_labels):
                tp = int(cm[idx, idx])
                fp = int(cm[:, idx].sum() - cm[idx, idx])
                fn = int(cm[idx, :].sum() - cm[idx, idx])
                tn = int(cm.sum() - tp - fp - fn)

                tp_dict[label] = tp
                tn_dict[label] = tn
                fp_dict[label] = fp
                fn_dict[label] = fn

            logger.info(
                f"Confusion matrix computed - Accuracy: {acc:.4f}, Classes: {len(class_labels)}"
            )

            return ConfusionMatrixOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                confusion_matrix=cm.tolist(),
                class_labels=class_labels,
                total_samples=len(y_test),
                accuracy=float(acc),
                true_positives=tp_dict,
                false_positives=fp_dict,
                true_negatives=tn_dict,
                false_negatives=fn_dict,
            )

        except Exception as e:
            logger.error(f"Confusion matrix computation failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )
