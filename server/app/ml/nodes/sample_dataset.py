"""
Sample Dataset Node - Loads built-in sample datasets from scikit-learn.
Saves them as CSV to local storage so downstream nodes can use them like any uploaded file.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from pydantic import Field
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id


# ─── Dataset Loaders ──────────────────────────────────────────────

def _load_iris() -> pd.DataFrame:
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def _load_wine() -> pd.DataFrame:
    from sklearn.datasets import load_wine
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def _load_diabetes() -> pd.DataFrame:
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def _load_breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def _load_california_housing() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def _load_digits() -> pd.DataFrame:
    from sklearn.datasets import load_digits
    data = load_digits()
    df = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
    df["target"] = data.target
    return df


def _load_tips() -> pd.DataFrame:
    """Classic tips dataset for regression (total_bill prediction)."""
    data = {
        "total_bill": [], "tip": [], "sex": [], "smoker": [],
        "day": [], "time": [], "size": [],
    }
    # Seaborn-style tips dataset via a small embedded copy
    import csv, io
    # Use pandas built-in if available, otherwise generate
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
        df = pd.read_csv(url, timeout=5)
        return df
    except Exception:
        # Fallback: generate synthetic tips data
        import numpy as np
        rng = np.random.default_rng(42)
        n = 244
        total_bill = rng.uniform(3, 50, n).round(2)
        tip = (total_bill * rng.uniform(0.1, 0.3, n)).round(2)
        sex = rng.choice(["Male", "Female"], n)
        smoker = rng.choice(["Yes", "No"], n)
        day = rng.choice(["Sun", "Sat", "Thur", "Fri"], n)
        time = rng.choice(["Lunch", "Dinner"], n)
        size = rng.choice([1, 2, 3, 4, 5, 6], n)
        return pd.DataFrame({
            "total_bill": total_bill, "tip": tip, "sex": sex,
            "smoker": smoker, "day": day, "time": time, "size": size,
        })


def _load_titanic() -> pd.DataFrame:
    """Titanic survival dataset for classification."""
    try:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url, timeout=5)
        return df
    except Exception:
        # Fallback: generate synthetic titanic-like data
        import numpy as np
        rng = np.random.default_rng(42)
        n = 891
        pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
        sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
        age = rng.normal(30, 14, n).clip(0.5, 80).round(1)
        sibsp = rng.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.05, 0.03, 0.01])
        parch = rng.choice([0, 1, 2, 3], n, p=[0.76, 0.13, 0.09, 0.02])
        fare = (rng.exponential(30, n) + 5).round(2)
        survived = ((sex == "female").astype(int) * 0.4 + (pclass == 1).astype(int) * 0.2 + rng.random(n) * 0.4 > 0.5).astype(int)
        return pd.DataFrame({
            "Survived": survived, "Pclass": pclass, "Sex": sex,
            "Age": age, "SibSp": sibsp, "Parch": parch, "Fare": fare,
        })


def _load_penguins() -> pd.DataFrame:
    """Palmer Penguins dataset for classification."""
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
        df = pd.read_csv(url, timeout=5)
        return df
    except Exception:
        import numpy as np
        rng = np.random.default_rng(42)
        n = 344
        species = rng.choice(["Adelie", "Chinstrap", "Gentoo"], n, p=[0.44, 0.20, 0.36])
        island = rng.choice(["Torgersen", "Biscoe", "Dream"], n)
        bill_length = rng.normal(44, 5, n).round(1)
        bill_depth = rng.normal(17, 2, n).round(1)
        flipper_length = rng.normal(201, 14, n).round(0).astype(int)
        body_mass = rng.normal(4200, 800, n).round(0).astype(int)
        sex = rng.choice(["Male", "Female"], n)
        return pd.DataFrame({
            "species": species, "island": island, "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth, "flipper_length_mm": flipper_length,
            "body_mass_g": body_mass, "sex": sex,
        })


def _load_heart_disease() -> pd.DataFrame:
    """Heart disease dataset for classification."""
    try:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/heart.csv"
        df = pd.read_csv(url, timeout=5)
        return df
    except Exception:
        import numpy as np
        rng = np.random.default_rng(42)
        n = 303
        age = rng.integers(29, 77, n)
        sex = rng.choice([0, 1], n, p=[0.32, 0.68])
        cp = rng.choice([0, 1, 2, 3], n)
        trestbps = rng.normal(131, 17, n).clip(94, 200).round(0).astype(int)
        chol = rng.normal(246, 52, n).clip(126, 564).round(0).astype(int)
        fbs = rng.choice([0, 1], n, p=[0.85, 0.15])
        restecg = rng.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02])
        thalach = rng.normal(150, 23, n).clip(71, 202).round(0).astype(int)
        exang = rng.choice([0, 1], n, p=[0.67, 0.33])
        oldpeak = rng.exponential(1.0, n).clip(0, 6.2).round(1)
        target = rng.choice([0, 1], n, p=[0.46, 0.54])
        return pd.DataFrame({
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "target": target,
        })


def _load_auto_mpg() -> pd.DataFrame:
    """Auto MPG dataset for regression (fuel efficiency prediction)."""
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
        df = pd.read_csv(url, timeout=5)
        df = df.dropna()
        return df
    except Exception:
        import numpy as np
        rng = np.random.default_rng(42)
        n = 392
        cylinders = rng.choice([4, 6, 8], n, p=[0.51, 0.25, 0.24])
        displacement = (cylinders * rng.uniform(15, 55, n)).round(1)
        horsepower = (displacement * rng.uniform(0.4, 1.0, n)).round(0).astype(int)
        weight = (displacement * rng.uniform(8, 16, n)).round(0).astype(int)
        acceleration = rng.normal(15.5, 2.8, n).clip(8, 25).round(1)
        model_year = rng.integers(70, 83, n)
        origin = rng.choice([1, 2, 3], n, p=[0.63, 0.20, 0.17])
        mpg = (50 - cylinders * 2 - horsepower * 0.05 + acceleration * 0.3 + rng.normal(0, 3, n)).clip(9, 47).round(1)
        return pd.DataFrame({
            "mpg": mpg, "cylinders": cylinders, "displacement": displacement,
            "horsepower": horsepower, "weight": weight, "acceleration": acceleration,
            "model_year": model_year, "origin": origin,
        })


def _load_student_performance() -> pd.DataFrame:
    """Student performance dataset for regression (grade prediction)."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 395
    study_hours = rng.uniform(1, 14, n).round(1)
    failures = rng.choice([0, 1, 2, 3], n, p=[0.66, 0.22, 0.08, 0.04])
    absences = rng.integers(0, 75, n)
    parent_edu = rng.choice([0, 1, 2, 3, 4], n, p=[0.10, 0.20, 0.25, 0.25, 0.20])
    internet = rng.choice(["yes", "no"], n, p=[0.84, 0.16])
    activities = rng.choice(["yes", "no"], n, p=[0.51, 0.49])
    health = rng.integers(1, 6, n)
    age = rng.integers(15, 22, n)
    # Grade depends on study hours and failures
    grade = (study_hours * 1.5 - failures * 3.0 + parent_edu * 0.8 - absences * 0.05 + rng.normal(5, 2, n)).clip(0, 20).round(0).astype(int)
    return pd.DataFrame({
        "age": age, "study_hours": study_hours, "failures": failures,
        "absences": absences, "parent_education": parent_edu, "internet": internet,
        "activities": activities, "health": health, "final_grade": grade,
    })


def _load_linnerud() -> pd.DataFrame:
    """Linnerud dataset for multivariate regression (exercise → physiological)."""
    from sklearn.datasets import load_linnerud
    data = load_linnerud()
    df_x = pd.DataFrame(data.data, columns=data.feature_names)
    df_y = pd.DataFrame(data.target, columns=data.target_names)
    return pd.concat([df_x, df_y], axis=1)


SAMPLE_DATASETS = {
    "iris":               ("Iris", _load_iris),
    "wine":               ("Wine Quality", _load_wine),
    "diabetes":           ("Diabetes", _load_diabetes),
    "breast_cancer":      ("Breast Cancer", _load_breast_cancer),
    "boston":              ("California Housing", _load_california_housing),
    "california_housing": ("California Housing", _load_california_housing),
    "digits":             ("Digits", _load_digits),
    "tips":               ("Tips", _load_tips),
    "titanic":            ("Titanic", _load_titanic),
    "penguins":           ("Palmer Penguins", _load_penguins),
    "heart_disease":      ("Heart Disease", _load_heart_disease),
    "auto_mpg":           ("Auto MPG", _load_auto_mpg),
    "student":            ("Student Performance", _load_student_performance),
    "linnerud":           ("Linnerud", _load_linnerud),
}


# ─── Node Implementation ──────────────────────────────────────────

class SampleDatasetInput(NodeInput):
    """Input schema for SampleDataset node."""
    dataset_name: str = Field("iris", description="Name of the sample dataset to load")


class SampleDatasetOutput(NodeOutput):
    """Output schema for SampleDataset node."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Generated filename")
    file_path: str = Field(..., description="Path to stored CSV file")
    storage_backend: str = Field(default="local", description="Storage backend")
    n_rows: int = Field(..., description="Number of rows")
    n_columns: int = Field(..., description="Number of columns")
    columns: list[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")


class SampleDatasetNode(BaseNode):
    """
    Sample Dataset Node - Loads built-in datasets from scikit-learn.

    Loads the chosen dataset, saves it as a CSV in the uploads directory,
    and returns the same metadata format as upload_file so downstream
    nodes work seamlessly.
    """

    node_type = "sample_dataset"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.DATA_SOURCE,
            primary_output_field="dataset_id",
            output_fields={
                "dataset_id": "Unique identifier for the sample dataset",
                "n_rows": "Number of rows",
                "n_columns": "Number of columns",
                "columns": "Column names",
                "dtypes": "Data types of each column",
            },
            requires_input=False,
            can_branch=True,
            produces_dataset=True,
            max_inputs=0,
            allowed_source_categories=[],
        )

    def get_input_schema(self):
        return SampleDatasetInput

    def get_output_schema(self):
        return SampleDatasetOutput

    async def _execute(self, input_data: SampleDatasetInput) -> SampleDatasetOutput:
        dataset_name = input_data.dataset_name

        if dataset_name not in SAMPLE_DATASETS:
            available = ", ".join(sorted(SAMPLE_DATASETS.keys()))
            raise ValueError(
                f"Unknown sample dataset: '{dataset_name}'. "
                f"Available: {available}"
            )

        label, loader_fn = SAMPLE_DATASETS[dataset_name]
        logger.info(f"Loading sample dataset: {label} ({dataset_name})")

        df = loader_fn()

        # Save to uploads directory so downstream nodes can read it
        # IMPORTANT: filename must be {dataset_id}.csv — downstream nodes look for exactly that
        dataset_id = generate_id("sample")
        filename = f"{dataset_id}.csv"
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        df.to_csv(file_path, index=False)

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        logger.info(
            f"Sample dataset loaded: {label} — "
            f"{len(df)} rows × {len(df.columns)} cols, saved to {file_path}"
        )

        return SampleDatasetOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            dataset_id=dataset_id,
            filename=filename,
            file_path=str(file_path),
            storage_backend="local",
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=df.columns.tolist(),
            dtypes=dtypes,
            memory_usage_mb=round(memory_mb, 2),
        )
