"""
K-Means Clustering Node — Unsupervised clustering pipeline node.
Runs scikit-learn KMeans on real data with rich output for interactive exploration.
"""

from typing import Type, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
from pydantic import Field
from pathlib import Path
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput, NodeMetadata, NodeCategory
from app.core.exceptions import NodeExecutionError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.s3_service import s3_service


class KMeansInput(NodeInput):
    """Input schema for K-Means node."""

    train_dataset_id: str = Field(..., description="Dataset ID")
    exclude_columns: Optional[str] = Field(
        "", description="Comma-separated column names to exclude from clustering"
    )

    # Hyperparameters
    n_clusters: int = Field(3, ge=2, le=20, description="Number of clusters")
    max_iter: int = Field(300, ge=50, le=1000, description="Maximum iterations per run")
    n_init: int = Field(10, ge=1, le=50, description="Number of initializations")
    init_method: str = Field("k-means++", description="Initialization: k-means++ or random")
    random_state: int = Field(42, description="Random seed")


class KMeansOutput(NodeOutput):
    """Output schema for K-Means node."""

    # Core results
    n_clusters: int = Field(..., description="Number of clusters used")
    n_samples: int = Field(..., description="Number of samples clustered")
    n_features: int = Field(..., description="Number of features used")
    feature_names: List[str] = Field(..., description="Feature column names")

    # Metrics
    silhouette_score: float = Field(..., description="Overall silhouette score (-1 to 1)")
    inertia: float = Field(..., description="Within-cluster sum of squares (WCSS)")
    training_time_seconds: float = Field(..., description="Training duration")

    # Cluster data
    cluster_sizes: List[int] = Field(..., description="Number of points per cluster")
    centroids: List[List[float]] = Field(..., description="Centroid coordinates (k x n_features)")

    # Elbow method data
    elbow_data: List[Dict[str, Any]] = Field(
        ..., description="Inertia and silhouette for K=2..max_k"
    )

    # Cluster profiles
    cluster_profiles: Dict[str, Any] = Field(
        ..., description="Per-cluster feature means and descriptions"
    )

    # PCA 2D projection for scatter
    pca_projection: Dict[str, Any] = Field(
        ..., description="2D PCA projection: points, centroids_2d, explained_variance"
    )

    # Convergence history
    convergence_history: Optional[Dict[str, Any]] = Field(
        None, description="Centroid movement across iterations"
    )

    # Per-sample silhouette scores
    sample_silhouette_scores: Optional[List[float]] = Field(
        None, description="Silhouette score per data point"
    )

    # Feature importance
    feature_importance: Optional[Dict[str, Any]] = Field(
        None, description="Feature relevance to clustering"
    )

    # Learning activity data
    quiz_questions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Auto-generated quiz questions"
    )

    metadata: Dict[str, Any] = Field(..., description="Clustering metadata")


class KMeansNode(BaseNode):
    """
    K-Means Clustering Node — unsupervised clustering on real pipeline data.
    Produces rich output for interactive SVG-based exploration.
    """

    node_type = "kmeans"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            category=NodeCategory.ML_ALGORITHM,
            primary_output_field=None,
            output_fields={
                "n_clusters": "Number of clusters",
                "silhouette_score": "Silhouette score",
                "inertia": "Within-cluster sum of squares",
            },
            requires_input=True,
            can_branch=False,
            produces_dataset=False,
            max_inputs=1,
            allowed_source_categories=[
                NodeCategory.DATA_TRANSFORM,
                NodeCategory.PREPROCESSING,
                NodeCategory.FEATURE_ENGINEERING,
            ],
        )

    def get_input_schema(self) -> Type[NodeInput]:
        return KMeansInput

    def get_output_schema(self) -> Type[NodeOutput]:
        return KMeansOutput

    async def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage (uploads folder first, then database)."""
        try:
            missing_values = ["?", "NA", "N/A", "null", "NULL", "", " ", "NaN", "nan"]

            upload_path = Path(settings.UPLOAD_DIR) / f"{dataset_id}.csv"
            if upload_path.exists():
                logger.info(f"Loading dataset from uploads folder: {upload_path}")
                return pd.read_csv(upload_path, na_values=missing_values, keep_default_na=True)

            db = SessionLocal()
            dataset = (
                db.query(Dataset)
                .filter(Dataset.dataset_id == dataset_id, Dataset.is_deleted == False)
                .first()
            )

            if not dataset:
                logger.error(f"Dataset not found: {dataset_id}")
                db.close()
                return None

            if dataset.storage_backend == "s3" and dataset.s3_key:
                file_content = await s3_service.download_file(dataset.s3_key)
                df = pd.read_csv(
                    io.BytesIO(file_content), na_values=missing_values, keep_default_na=True
                )
            elif dataset.local_path:
                df = pd.read_csv(dataset.local_path, na_values=missing_values, keep_default_na=True)
            else:
                db.close()
                return None

            db.close()
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None

    async def _execute(self, input_data: KMeansInput) -> KMeansOutput:
        """Execute K-Means clustering."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, silhouette_samples
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        try:
            logger.info(f"Running K-Means clustering (K={input_data.n_clusters})")

            # 1. Load dataset
            df = await self._load_dataset(input_data.train_dataset_id)
            if df is None or df.empty:
                raise InvalidDatasetError(
                    reason=f"Dataset {input_data.train_dataset_id} not found or empty",
                    expected_format="Valid dataset ID",
                )

            # 2. Exclude columns
            exclude_cols = []
            if input_data.exclude_columns:
                exclude_cols = [c.strip() for c in input_data.exclude_columns.split(",") if c.strip()]
            for col in exclude_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # 3. Keep only numeric columns
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                logger.info(f"Dropping non-numeric columns: {non_numeric}")
                df = df.select_dtypes(include=[np.number])

            if df.empty or len(df.columns) == 0:
                raise InvalidDatasetError(
                    reason="No numeric columns remaining after filtering",
                    expected_format="Dataset with numeric features",
                )

            # 4. Handle missing values
            df = df.fillna(df.mean())

            feature_names = df.columns.tolist()
            X_original = df.values.astype(float)
            n_samples, n_features = X_original.shape

            # 5. Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_original)

            # 6. Run K-Means
            n_clusters = min(input_data.n_clusters, n_samples - 1)
            training_start = datetime.utcnow()

            km = KMeans(
                n_clusters=n_clusters,
                init=input_data.init_method,
                max_iter=input_data.max_iter,
                n_init=input_data.n_init,
                random_state=input_data.random_state,
            )
            km.fit(X_scaled)

            training_time = (datetime.utcnow() - training_start).total_seconds()

            # 7. Compute metrics
            labels = km.labels_
            sil_score = float(silhouette_score(X_scaled, labels)) if n_clusters < n_samples else 0.0
            inertia_val = float(km.inertia_)

            # Per-sample silhouette
            sample_sil = None
            try:
                sample_sil = silhouette_samples(X_scaled, labels).tolist()
                # Cap for payload size
                if len(sample_sil) > 3000:
                    sample_sil = sample_sil[:3000]
            except Exception:
                pass

            # Cluster sizes
            cluster_sizes = [int(np.sum(labels == c)) for c in range(n_clusters)]

            # Centroids (unscaled for interpretability)
            centroids_scaled = km.cluster_centers_
            centroids_original = scaler.inverse_transform(centroids_scaled)

            # 8. Elbow data
            elbow_data = self._compute_elbow_data(
                X_scaled, n_samples, input_data.init_method,
                input_data.n_init, input_data.random_state, input_data.max_iter,
            )

            # 9. Cluster profiles
            cluster_profiles = self._compute_cluster_profiles(
                X_original, X_scaled, labels, n_clusters, feature_names, centroids_original,
            )

            # 10. PCA 2D projection
            pca_projection = self._compute_pca_projection(
                X_scaled, labels, n_clusters, centroids_scaled, n_samples,
            )

            # 11. Convergence history
            convergence_history = None
            try:
                convergence_history = self._compute_convergence_history(
                    X_scaled, n_clusters, input_data.init_method,
                    input_data.random_state, pca_projection.get("_pca_model"),
                )
            except Exception as e:
                logger.warning(f"Convergence history failed: {e}")

            # Remove internal PCA model ref from projection
            pca_projection.pop("_pca_model", None)

            # 12. Feature importance
            feature_importance = None
            try:
                feature_importance = self._compute_feature_importance(
                    X_scaled, labels, n_clusters, feature_names, n_samples, centroids_scaled,
                )
            except Exception as e:
                logger.warning(f"Feature importance failed: {e}")

            # 13. Quiz
            quiz_questions = None
            try:
                quiz_questions = self._generate_quiz(
                    n_clusters, sil_score, inertia_val, n_samples,
                    n_features, feature_names, cluster_sizes,
                    feature_importance, input_data.init_method,
                )
            except Exception as e:
                logger.warning(f"Quiz generation failed: {e}")

            return KMeansOutput(
                node_type=self.node_type,
                execution_time_ms=int(training_time * 1000),
                n_clusters=n_clusters,
                n_samples=n_samples,
                n_features=n_features,
                feature_names=feature_names,
                silhouette_score=sil_score,
                inertia=inertia_val,
                training_time_seconds=training_time,
                cluster_sizes=cluster_sizes,
                centroids=centroids_original.tolist(),
                elbow_data=elbow_data,
                cluster_profiles=cluster_profiles,
                pca_projection=pca_projection,
                convergence_history=convergence_history,
                sample_silhouette_scores=sample_sil,
                feature_importance=feature_importance,
                quiz_questions=quiz_questions,
                metadata={
                    "excluded_columns": exclude_cols,
                    "dropped_non_numeric": non_numeric,
                    "hyperparameters": {
                        "n_clusters": n_clusters,
                        "init_method": input_data.init_method,
                        "max_iter": input_data.max_iter,
                        "n_init": input_data.n_init,
                        "random_state": input_data.random_state,
                    },
                    "n_iterations": int(km.n_iter_),
                },
            )

        except Exception as e:
            logger.error(f"K-Means clustering failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_type=self.node_type, reason=str(e), input_data=input_data.model_dump()
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _compute_elbow_data(
        self, X_scaled: np.ndarray, n_samples: int,
        init_method: str, n_init: int, random_state: int, max_iter: int,
    ) -> List[Dict[str, Any]]:
        """Compute inertia + silhouette for a range of K values."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        max_k = min(15, n_samples - 1)
        elbow = []

        for k in range(2, max_k + 1):
            try:
                km_temp = KMeans(
                    n_clusters=k, init=init_method,
                    n_init=min(n_init, 3), random_state=random_state,
                    max_iter=max_iter,
                )
                km_temp.fit(X_scaled)
                sil = float(silhouette_score(X_scaled, km_temp.labels_)) if k < n_samples else 0.0
                elbow.append({
                    "k": k,
                    "inertia": round(float(km_temp.inertia_), 2),
                    "silhouette": round(sil, 4),
                })
            except Exception:
                continue

        return elbow

    def _compute_cluster_profiles(
        self, X_original: np.ndarray, X_scaled: np.ndarray,
        labels: np.ndarray, n_clusters: int,
        feature_names: List[str], centroids_original: np.ndarray,
    ) -> Dict[str, Any]:
        """Per-cluster feature means and normalized values."""
        global_means = X_original.mean(axis=0)
        global_stds = X_original.std(axis=0)
        global_stds[global_stds == 0] = 1

        clusters = []
        for c in range(n_clusters):
            mask = labels == c
            cluster_means = X_original[mask].mean(axis=0)

            # Deviation from global mean (in std units)
            deviations = (cluster_means - global_means) / global_stds

            # Find distinguishing features (highest absolute deviation)
            sorted_idxs = np.argsort(np.abs(deviations))[::-1]
            distinguishing = []
            for idx in sorted_idxs[:3]:
                direction = "higher" if deviations[idx] > 0 else "lower"
                distinguishing.append({
                    "feature": feature_names[idx],
                    "direction": direction,
                    "deviation": round(float(deviations[idx]), 2),
                    "cluster_mean": round(float(cluster_means[idx]), 3),
                    "global_mean": round(float(global_means[idx]), 3),
                })

            clusters.append({
                "cluster_id": c,
                "size": int(mask.sum()),
                "feature_means": {
                    name: round(float(val), 3)
                    for name, val in zip(feature_names, cluster_means)
                },
                "normalized_means": {
                    name: round(float(val), 3)
                    for name, val in zip(feature_names, X_scaled[mask].mean(axis=0))
                },
                "distinguishing_features": distinguishing,
            })

        return {
            "clusters": clusters,
            "global_means": {
                name: round(float(val), 3) for name, val in zip(feature_names, global_means)
            },
        }

    def _compute_pca_projection(
        self, X_scaled: np.ndarray, labels: np.ndarray,
        n_clusters: int, centroids_scaled: np.ndarray, n_samples: int,
    ) -> Dict[str, Any]:
        """2D PCA projection for scatter plot."""
        from sklearn.decomposition import PCA

        n_components = min(2, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X_scaled)

        # Sample points if dataset is large
        max_points = 2000
        if n_samples > max_points:
            rng = np.random.RandomState(42)
            indices = rng.choice(n_samples, max_points, replace=False)
            indices.sort()
        else:
            indices = np.arange(n_samples)

        points = []
        for i in indices:
            pt = {"x": round(float(coords[i, 0]), 4), "cluster": int(labels[i])}
            if n_components >= 2:
                pt["y"] = round(float(coords[i, 1]), 4)
            else:
                pt["y"] = 0.0
            points.append(pt)

        centroids_2d = pca.transform(centroids_scaled).tolist()
        for i in range(len(centroids_2d)):
            centroids_2d[i] = [round(v, 4) for v in centroids_2d[i]]

        evr = pca.explained_variance_ratio_.tolist()

        return {
            "points": points,
            "centroids_2d": centroids_2d,
            "explained_variance_ratio": evr,
            "pc1_label": f"PC1 ({evr[0]*100:.1f}%)" if len(evr) > 0 else "PC1",
            "pc2_label": f"PC2 ({evr[1]*100:.1f}%)" if len(evr) > 1 else "PC2",
            "_pca_model": pca,  # Internal, removed before serialization
        }

    def _compute_convergence_history(
        self, X_scaled: np.ndarray, n_clusters: int,
        init_method: str, random_state: int, pca_model,
    ) -> Dict[str, Any]:
        """Manual K-Means loop to track centroid positions per iteration."""
        from sklearn.cluster import KMeans

        # Get initial centroids
        km_init = KMeans(
            n_clusters=n_clusters, init=init_method,
            max_iter=1, n_init=1, random_state=random_state,
        )
        km_init.fit(X_scaled)
        current_centroids = km_init.cluster_centers_.copy()

        iterations = []
        max_history = 50

        for iteration in range(max_history):
            # Assign points to nearest centroid
            distances = np.linalg.norm(
                X_scaled[:, np.newaxis, :] - current_centroids[np.newaxis, :, :], axis=2
            )
            assigned = np.argmin(distances, axis=1)

            # Compute WCSS
            wcss = float(np.sum([
                np.sum((X_scaled[assigned == c] - current_centroids[c]) ** 2)
                for c in range(n_clusters)
                if np.any(assigned == c)
            ]))

            iterations.append({
                "iteration": iteration,
                "centroids": current_centroids.tolist(),
                "wcss": round(wcss, 2),
            })

            # Update centroids
            new_centroids = np.array([
                X_scaled[assigned == c].mean(axis=0) if np.any(assigned == c)
                else current_centroids[c]
                for c in range(n_clusters)
            ])

            if np.allclose(new_centroids, current_centroids, atol=1e-6):
                break
            current_centroids = new_centroids

        # Project iteration centroids to 2D
        pca_iterations = []
        if pca_model is not None:
            for step in iterations:
                projected = pca_model.transform(np.array(step["centroids"])).tolist()
                pca_iterations.append([
                    [round(v, 4) for v in c] for c in projected
                ])

        return {
            "iterations": iterations,
            "converged_at": len(iterations),
            "pca_iterations": pca_iterations,
        }

    def _compute_feature_importance(
        self, X_scaled: np.ndarray, labels: np.ndarray,
        n_clusters: int, feature_names: List[str],
        n_samples: int, centroids: np.ndarray,
    ) -> Dict[str, Any]:
        """Feature importance based on inter-cluster variance ratio."""
        global_means = X_scaled.mean(axis=0)
        total_var = X_scaled.var(axis=0)
        total_var[total_var == 0] = 1

        cluster_sizes = [int(np.sum(labels == c)) for c in range(n_clusters)]
        features = []

        for j, fname in enumerate(feature_names):
            inter_var = sum(
                cluster_sizes[c] * (centroids[c, j] - global_means[j]) ** 2
                for c in range(n_clusters)
            ) / n_samples
            ratio = float(inter_var / total_var[j])
            features.append({"feature": fname, "importance": round(ratio, 4)})

        features.sort(key=lambda x: x["importance"], reverse=True)
        max_imp = features[0]["importance"] if features else 1
        for f in features:
            f["bar_width_pct"] = round(f["importance"] / max_imp * 100, 1) if max_imp > 0 else 0

        return {"features": features, "total_features": len(features)}

    def _generate_quiz(
        self, n_clusters: int, sil_score: float, inertia: float,
        n_samples: int, n_features: int, feature_names: List[str],
        cluster_sizes: List[int], feature_importance: Optional[Dict],
        init_method: str,
    ) -> List[Dict[str, Any]]:
        """Auto-generate quiz questions about K-Means using real data."""
        import random as _random
        questions = []
        q_id = 0

        # Q1: What is K-Means
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What type of machine learning is K-Means Clustering?",
            "options": [
                "Unsupervised learning — it finds patterns without labeled data",
                "Supervised learning — it needs labeled training examples",
                "Reinforcement learning — it learns from rewards",
                "Semi-supervised learning — it needs some labels",
            ],
            "correct_answer": 0,
            "explanation": f"K-Means is unsupervised — it grouped your {n_samples} data points into {n_clusters} clusters without needing any labels. It discovers hidden structure by finding groups of similar data points.",
            "difficulty": "easy",
        })

        # Q2: Silhouette score interpretation
        q_id += 1
        sil_pct = round(sil_score, 2)
        quality = "strong" if sil_score >= 0.7 else "reasonable" if sil_score >= 0.5 else "weak" if sil_score >= 0.25 else "poor"
        questions.append({
            "id": f"q{q_id}",
            "question": f"Your silhouette score is {sil_pct}. What does this tell you?",
            "options": [
                f"The cluster separation is {quality} — silhouette ranges from -1 to 1, higher is better",
                "The model is overfitting to the training data",
                "The number of features is too high",
                "The dataset needs more data points to cluster",
            ],
            "correct_answer": 0,
            "explanation": f"Silhouette score measures how similar points are to their own cluster vs neighboring clusters. Score of {sil_pct} indicates {quality} cluster structure. Scores near 1 mean well-separated clusters, near 0 means overlapping, and negative means points may be in the wrong cluster.",
            "difficulty": "medium",
        })

        # Q3: What inertia measures
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "What does 'inertia' (WCSS) measure in K-Means?",
            "options": [
                "The total distance of each point to its cluster center — lower means tighter clusters",
                "The number of iterations needed to converge",
                "The distance between cluster centers",
                "The number of points that changed clusters",
            ],
            "correct_answer": 0,
            "explanation": f"Inertia (Within-Cluster Sum of Squares) = {round(inertia, 1)}. It's the sum of squared distances from each point to its assigned centroid. Lower inertia means points are closer to their centroids = tighter, more compact clusters.",
            "difficulty": "medium",
        })

        # Q4: Why scale features
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "Why is feature scaling important before running K-Means?",
            "options": [
                "K-Means uses distance — features with larger ranges would dominate the clustering",
                "It makes the algorithm run faster",
                "It reduces the number of clusters needed",
                "It prevents the algorithm from overfitting",
            ],
            "correct_answer": 0,
            "explanation": f"K-Means measures distance between points. With {n_features} features, if one feature ranges 0-1000 and another 0-1, the large-range feature would completely dominate cluster assignments. StandardScaler ensures all features contribute equally.",
            "difficulty": "medium",
        })

        # Q5: Elbow method
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": "How does the Elbow Method help choose the best K?",
            "options": [
                "Plot inertia vs K — the 'elbow' point where inertia stops dropping sharply suggests the optimal K",
                "Always pick the K with the lowest inertia",
                "Choose the K where silhouette score first becomes positive",
                "Use K equal to the number of features",
            ],
            "correct_answer": 0,
            "explanation": "Inertia always decreases as K increases (more clusters = tighter fit). The Elbow Method looks for the K where adding more clusters stops giving significant improvement — the 'bend' in the curve. Beyond this point, you're overfitting.",
            "difficulty": "medium",
        })

        # Q6: K-Means++ vs random
        q_id += 1
        questions.append({
            "id": f"q{q_id}",
            "question": f"This model used '{init_method}' initialization. Why is K-Means++ preferred over random?",
            "options": [
                "It spreads initial centroids apart, leading to better and more consistent results",
                "It always finds the globally optimal solution",
                "It uses fewer iterations to converge",
                "It automatically determines the best K",
            ],
            "correct_answer": 0,
            "explanation": "K-Means++ picks the first centroid randomly, then places each subsequent centroid far from existing ones (probability proportional to distance²). This avoids poor initializations where centroids start too close together, which can lead to bad local optima.",
            "difficulty": "hard",
        })

        # Q7: Cluster sizes (data-specific)
        if cluster_sizes:
            largest = max(cluster_sizes)
            smallest = min(cluster_sizes)
            largest_idx = cluster_sizes.index(largest)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": f"Cluster {largest_idx} has {largest} points while the smallest has {smallest}. What might this indicate?",
                "options": [
                    "The data naturally has groups of different sizes — some patterns are more common",
                    "K-Means always produces equal-sized clusters",
                    "The algorithm failed to converge properly",
                    "The random seed was chosen incorrectly",
                ],
                "correct_answer": 0,
                "explanation": f"K-Means does NOT force equal-sized clusters. Cluster sizes reflect the data's natural structure. Having clusters of size {largest} and {smallest} suggests some groups in your data are naturally larger than others.",
                "difficulty": "easy",
            })

        # Q8: Most important feature
        if feature_importance and feature_importance.get("features"):
            top = feature_importance["features"][0]
            others = [f["feature"] for f in feature_importance["features"][1:4]]
            opts = [top["feature"]] + others
            _random.shuffle(opts)
            q_id += 1
            questions.append({
                "id": f"q{q_id}",
                "question": "Which feature contributes most to separating the clusters?",
                "options": opts,
                "correct_answer": opts.index(top["feature"]),
                "explanation": f"'{top['feature']}' has the highest inter-cluster variance ratio ({round(top['importance'] * 100, 1)}%), meaning it varies most between clusters. Features with high inter-cluster variance are the ones that most distinguish one cluster from another.",
                "difficulty": "easy",
            })

        _random.shuffle(questions)
        return questions[:6]
