"""
Linear Regression Node - Individual node wrapper for Linear Regression algorithm.
Supports training linear regression models with configurable hyperparameters.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.ml.algorithms.regression.linear_regression import LinearRegression
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class LinearRegressionInput(NodeInput):
    """Input schema for Linear Regression node."""

    train_dataset_id: str = Field(..., description="Training dataset ID")
    target_column: Optional[str] = Field(
        None, description="Name of target column (auto-filled from split node)"
    )

    # Hyperparameters
    fit_intercept: bool = Field(True, description="Calculate intercept for the model")


class LinearRegressionOutput(NodeOutput):
    """Output schema for Linear Regression node."""

    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")

    training_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")

    training_metrics: Dict[str, float] = Field(
        ..., description="Training metrics (MAE, MSE, RMSE, R²)"
    )
    training_time_seconds: float = Field(..., description="Training duration")

    coefficients: list = Field(..., description="Model coefficients")
    intercept: float = Field(..., description="Model intercept")

    metadata: Dict[str, Any] = Field(..., description="Training metadata")

    # Learning activity data (all Optional for backward compatibility)
    coefficient_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Feature coefficient analysis for visualization"
    )
    prediction_playground: Optional[Dict[str, Any]] = Field(
        None, description="Data for interactive prediction playground"
    )
    equation_data: Optional[Dict[str, Any]] = Field(
        None, description="Regression equation breakdown"
    )
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions about linear regression"
    )


class LinearRegressionNode(BaseNode):
    """
    Linear Regression Node - Train linear regression models.

    Responsibilities:
    - Load training dataset from database/S3
    - Train linear regression model
    - Calculate regression metrics
    - Save model artifacts
    - Return model metadata

    Production features:
    - Dataset loading from multiple sources
    - Hyperparameter configuration
    - Model versioning
    - Comprehensive metrics
    """

    node_type = "linear_regression"

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
                "coefficients": "Model coefficients",
                "intercept": "Model intercept",
            },
            requires_input=True,
            can_branch=True,  # Can feed into multiple metric nodes
            produces_dataset=False,  # Produces model, not dataset
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.DATA_TRANSFORM,  # From split node
                NodeCategory.PREPROCESSING,
                NodeCategory.FEATURE_ENGINEERING,
            ],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return LinearRegressionInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return LinearRegressionOutput

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

    async def _execute(self, input_data: LinearRegressionInput) -> LinearRegressionOutput:
        """
        Execute linear regression training.

        Args:
            input_data: Validated input data

        Returns:
            Training result with model metadata
        """
        try:
            logger.info(f"Training Linear Regression model")

            # Load training data
            df_train = await self._load_dataset(input_data.train_dataset_id)
            if df_train is None or df_train.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.train_dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            # Validate target column
            if not input_data.target_column:
                raise ValueError("Target column must be provided (auto-filled from split node)")

            if input_data.target_column not in df_train.columns:
                raise ValueError(f"Target column '{input_data.target_column}' not found")

            # Separate features and target
            X_train = df_train.drop(columns=[input_data.target_column])
            y_train = df_train[input_data.target_column]

            # Validate dataset has data
            if X_train.empty:
                raise ValueError("Training dataset is empty")

            # Ensure all features are numeric
            non_numeric = X_train.select_dtypes(exclude=["number"]).columns.tolist()
            if non_numeric:
                raise ValueError(
                    f"Non-numeric columns found: {non_numeric}. "
                    f"Please encode categorical variables before training."
                )

            # Initialize and train model
            model = LinearRegression(
                fit_intercept=input_data.fit_intercept,
            )

            training_start = datetime.utcnow()
            training_metadata = model.train(X_train, y_train, validate=True)
            training_time = (datetime.utcnow() - training_start).total_seconds()

            # Generate model ID
            model_id = generate_id("model_linear_regression")
            model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_dir = Path(settings.MODEL_ARTIFACTS_DIR) / "linear_regression"
            model_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"{model_id}_v{model_version}.joblib"
            model_path = model_dir / model_filename

            model.save(model_path)

            logger.info(f"Model saved to {model_path}")

            coefficients = training_metadata.get("coefficients", [])
            intercept = training_metadata.get("intercept", 0.0)
            feature_names = X_train.columns.tolist()
            metrics = training_metadata.get("training_metrics", {})

            # Generate learning activity data (non-blocking)
            coefficient_analysis = None
            try:
                coefficient_analysis = self._generate_coefficient_analysis(
                    feature_names, coefficients, X_train
                )
            except Exception as e:
                logger.warning(f"Coefficient analysis generation failed: {e}")

            prediction_playground = None
            try:
                prediction_playground = self._generate_prediction_playground(
                    feature_names, coefficients, intercept, X_train
                )
            except Exception as e:
                logger.warning(f"Prediction playground generation failed: {e}")

            equation_data = None
            try:
                equation_data = self._generate_equation_data(
                    feature_names, coefficients, intercept, input_data.target_column
                )
            except Exception as e:
                logger.warning(f"Equation data generation failed: {e}")

            quiz_questions = None
            try:
                quiz_questions = self._generate_linear_regression_quiz(
                    feature_names, coefficients, intercept, metrics, input_data.target_column
                )
            except Exception as e:
                logger.warning(f"Linear regression quiz generation failed: {e}")

            return LinearRegressionOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                model_id=model_id,
                model_path=str(model_path),
                training_samples=len(X_train),
                n_features=len(X_train.columns),
                training_metrics=metrics,
                training_time_seconds=training_time,
                coefficients=coefficients,
                intercept=intercept,
                metadata={
                    "target_column": input_data.target_column,
                    "feature_names": feature_names,
                    "hyperparameters": {
                        "fit_intercept": input_data.fit_intercept,
                    },
                    "full_training_metadata": training_metadata,
                },
                coefficient_analysis=coefficient_analysis,
                prediction_playground=prediction_playground,
                equation_data=equation_data,
                quiz_questions=quiz_questions,
            )

        except Exception as e:
            logger.error(f"Linear Regression training failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # --- Learning Activity Helpers ---

    def _generate_coefficient_analysis(
        self, feature_names: list, coefficients: list, X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        """Feature coefficient analysis for bar chart visualization."""
        if not coefficients or not feature_names:
            return {"features": []}

        coef_list = []
        max_abs = max(abs(c) for c in coefficients) if coefficients else 1

        for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
            coef_val = float(coef)
            bar_width = abs(coef_val) / max_abs * 100 if max_abs > 0 else 0

            feat_stats = {}
            if name in X_train.columns:
                col = X_train[name].dropna()
                if len(col) > 0:
                    feat_stats = {
                        "mean": round(float(col.mean()), 4),
                        "std": round(float(col.std()), 4),
                        "min": round(float(col.min()), 4),
                        "max": round(float(col.max()), 4),
                    }

            coef_list.append({
                "name": name,
                "coefficient": round(coef_val, 6),
                "abs_coefficient": round(abs(coef_val), 6),
                "direction": "positive" if coef_val > 0 else "negative",
                "bar_width_pct": round(bar_width, 1),
                "interpretation": f"For every 1 unit increase in {name}, the prediction {'increases' if coef_val > 0 else 'decreases'} by {abs(round(coef_val, 4))}",
                "stats": feat_stats,
            })

        # Sort by absolute coefficient
        coef_list.sort(key=lambda x: x["abs_coefficient"], reverse=True)
        for rank, item in enumerate(coef_list, 1):
            item["rank"] = rank

        return {"features": coef_list, "total_features": len(coef_list)}

    def _generate_prediction_playground(
        self, feature_names: list, coefficients: list, intercept: float,
        X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        """Data for interactive prediction sliders."""
        if not feature_names or not coefficients:
            return {"features": [], "intercept": intercept}

        features = []
        for name, coef in zip(feature_names, coefficients):
            if name in X_train.columns:
                col = X_train[name].dropna()
                if len(col) > 0:
                    features.append({
                        "name": name,
                        "coefficient": round(float(coef), 6),
                        "min": round(float(col.min()), 4),
                        "max": round(float(col.max()), 4),
                        "mean": round(float(col.mean()), 4),
                        "step": round(float((col.max() - col.min()) / 50), 4) if col.max() != col.min() else 1,
                    })

        return {
            "features": features[:10],
            "intercept": round(float(intercept), 6),
            "formula": "y = intercept + sum(coefficient_i * feature_i)",
        }

    def _generate_equation_data(
        self, feature_names: list, coefficients: list, intercept: float,
        target_column: str
    ) -> Dict[str, Any]:
        """Regression equation breakdown."""
        terms = []
        for name, coef in zip(feature_names, coefficients):
            terms.append({
                "feature": name,
                "coefficient": round(float(coef), 4),
                "sign": "+" if coef >= 0 else "-",
            })

        # Build equation string
        eq_parts = [f"{round(float(intercept), 4)}"]
        for t in terms:
            sign = "+" if t["coefficient"] >= 0 else "-"
            eq_parts.append(f"{sign} {abs(t['coefficient'])} x {t['feature']}")

        equation_string = f"{target_column} = " + " ".join(eq_parts)

        return {
            "target": target_column,
            "intercept": round(float(intercept), 4),
            "terms": terms,
            "equation_string": equation_string,
        }

    def _generate_linear_regression_quiz(
        self, feature_names: list, coefficients: list, intercept: float,
        metrics: Dict[str, float], target_column: str
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about linear regression."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: Coefficient interpretation
        if coefficients and feature_names:
            # Find most negative coefficient
            neg_pairs = [(n, c) for n, c in zip(feature_names, coefficients) if c < 0]
            if neg_pairs:
                feat, coef = max(neg_pairs, key=lambda x: abs(x[1]))
                q_id += 1
                questions.append({
                    "id": f"q{q_id}",
                    "question": f"Feature '{feat}' has a coefficient of {round(float(coef), 4)}. What happens when '{feat}' increases by 1?",
                    "options": [
                        f"{target_column} decreases by {abs(round(float(coef), 4))}",
                        f"{target_column} increases by {abs(round(float(coef), 4))}",
                        f"{target_column} stays the same",
                        "Cannot determine from the coefficient",
                    ],
                    "correct_answer": 0,
                    "explanation": f"A negative coefficient ({round(float(coef), 4)}) means the feature and target move in opposite directions. When '{feat}' goes up by 1, '{target_column}' goes down by {abs(round(float(coef), 4))}.",
                    "difficulty": "medium",
                })

        # Q2: Strongest feature
        if len(coefficients) >= 2:
            abs_coefs = [(n, abs(c)) for n, c in zip(feature_names, coefficients)]
            strongest = max(abs_coefs, key=lambda x: x[1])
            others = [x for x in abs_coefs if x[0] != strongest[0]]
            _random.shuffle(others)
            q_id += 1
            options = [strongest[0]] + [o[0] for o in others[:3]]
            _random.shuffle(options)
            questions.append({
                "id": f"q{q_id}",
                "question": "Which feature has the strongest influence on the prediction?",
                "options": options,
                "correct_answer": options.index(strongest[0]),
                "explanation": f"'{strongest[0]}' has the largest absolute coefficient ({round(strongest[1], 4)}), meaning it has the biggest impact on the prediction for each unit change.",
                "difficulty": "easy",
            })

        # Q3: R² interpretation
        r2 = metrics.get("r2_score", metrics.get("r2", None))
        if r2 is not None:
            r2_pct = round(float(r2) * 100, 1)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"The R² score is {round(float(r2), 4)}. What does this mean?",
                "options": [
                    f"The model explains {r2_pct}% of the variation in {target_column}",
                    f"The model is {r2_pct}% accurate",
                    f"The model makes errors {r2_pct}% of the time",
                    f"The model uses {r2_pct}% of the features",
                ],
                "correct_answer": 0,
                "explanation": f"R² (coefficient of determination) tells you what fraction of the target's variation your model captures. R²={round(float(r2), 4)} means {r2_pct}% of the changes in '{target_column}' are explained by the features.",
                "difficulty": "medium",
            })

        # Q4: Intercept meaning
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"The intercept is {round(float(intercept), 4)}. What does it represent?",
            "options": [
                f"The predicted {target_column} when all features are 0",
                "The average of all coefficients",
                "The maximum possible prediction",
                "The error of the model",
            ],
            "correct_answer": 0,
            "explanation": f"The intercept ({round(float(intercept), 4)}) is the baseline prediction — what the model predicts when every feature equals zero. Each feature then adds or subtracts from this baseline.",
            "difficulty": "easy",
        })

        # Q5: What is linear regression
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does linear regression try to find?",
            "options": [
                "The best straight line (or plane) that fits the data",
                "The most common value in the data",
                "Groups of similar data points",
                "The probability of each class",
            ],
            "correct_answer": 0,
            "explanation": "Linear regression finds the line (or hyperplane in multiple dimensions) that minimizes the distance between actual values and predicted values. It assumes a linear relationship between features and target.",
            "difficulty": "easy",
        })

        _random.shuffle(questions)
        return questions[:5]
