"""
Pipeline Analyzer Service

Analyzes ML pipeline structure to provide intelligent guidance:
- Detect missing pipeline steps
- Validate node connections
- Suggest next appropriate nodes
- Check model compatibility with data
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from app.core.logging import logger


class PipelineAnalyzer:
    """Intelligent pipeline structure analysis for mentor system."""

    # Node categories for validation
    DATA_SOURCE_NODES = {"upload_file", "select_dataset"}
    PREPROCESSING_NODES = {"missing_value_handler", "encoding", "scaling", "feature_selection"}
    SPLIT_NODES = {"split"}
    ML_ALGORITHM_NODES = {
        "linear_regression",
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }
    RESULT_NODES = {"metrics", "confusion_matrix", "feature_importance", "predictions"}
    VIEW_NODES = {"view_data", "chart_view"}

    def analyze_pipeline(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze current pipeline structure.

        Args:
            nodes: List of pipeline nodes
            edges: List of pipeline edges
            dataset_metadata: Optional dataset information

        Returns:
            Analysis results with gaps, warnings, and recommendations
        """
        logger.info(f"Analyzing pipeline with {len(nodes)} nodes and {len(edges)} edges")

        if not nodes:
            return {
                "status": "empty",
                "message": "Your workspace is empty. Let's start by uploading a dataset!",
                "suggested_nodes": ["upload_file"],
                "gaps": ["No nodes added yet"],
                "warnings": [],
                "next_steps": ["Add an Upload Dataset node to begin"],
            }

        # Extract node types and build graph
        node_types = [node.get("node_type", node.get("type")) for node in nodes]
        node_map = {node.get("node_id", node.get("id")): node for node in nodes}

        # Build adjacency structure
        graph = self._build_graph(edges)

        # Detect pipeline components
        has_data_source = any(nt in self.DATA_SOURCE_NODES for nt in node_types)
        has_split = any(nt in self.SPLIT_NODES for nt in node_types)
        has_ml_model = any(nt in self.ML_ALGORITHM_NODES for nt in node_types)
        has_preprocessing = any(nt in self.PREPROCESSING_NODES for nt in node_types)

        # Detect gaps
        gaps = self._detect_gaps(
            has_data_source,
            has_split,
            has_ml_model,
            has_preprocessing,
            node_types,
            dataset_metadata,
        )

        # Validate connections
        warnings = self._validate_connections(nodes, graph, node_map)

        # Suggest next steps
        next_steps = self._suggest_next_steps(
            node_types, has_data_source, has_split, has_ml_model, dataset_metadata
        )

        # Determine pipeline status
        if has_data_source and has_split and has_ml_model:
            status = "complete"
            message = "Your pipeline looks complete! Ready to execute and see results."
        elif has_data_source and has_ml_model:
            status = "partial"
            message = "Pipeline setup in progress. A few more steps recommended."
        elif has_data_source:
            status = "started"
            message = "Great start! Now let's prepare your data for training."
        else:
            status = "incomplete"
            message = "Add nodes to build your ML pipeline."

        return {
            "status": status,
            "message": message,
            "gaps": gaps,
            "warnings": warnings,
            "next_steps": next_steps,
            "node_count": len(nodes),
            "has_data_source": has_data_source,
            "has_split": has_split,
            "has_ml_model": has_ml_model,
        }

    def _build_graph(self, edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        graph = {}
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                if source not in graph:
                    graph[source] = []
                graph[source].append(target)
        return graph

    def _detect_gaps(
        self,
        has_data_source: bool,
        has_split: bool,
        has_ml_model: bool,
        has_preprocessing: bool,
        node_types: List[str],
        dataset_metadata: Optional[Dict] = None,
    ) -> List[str]:
        """Detect missing pipeline components."""
        gaps = []

        if not has_data_source:
            gaps.append("No data source - start by uploading a dataset")

        if has_ml_model and not has_split:
            gaps.append("Training a model requires a Train/Test Split node")

        # Check for categorical encoding if dataset has categorical columns
        if dataset_metadata and has_data_source:
            categorical_cols = dataset_metadata.get("categorical_columns", [])
            has_encoding = "encoding" in node_types

            if categorical_cols and not has_encoding and has_ml_model:
                gaps.append(
                    f"Detected {len(categorical_cols)} categorical columns but no Encoding node - "
                    "ML models need numeric inputs"
                )

        # Check for missing value handling if needed
        if dataset_metadata:
            missing_values = dataset_metadata.get("missing_values", {})
            total_missing = sum(missing_values.values())
            has_missing_handler = "missing_value_handler" in node_types

            if total_missing > 0 and not has_missing_handler:
                gaps.append(
                    f"Dataset has {total_missing} missing values but no Missing Value Handler node"
                )

        return gaps

    def _validate_connections(
        self, nodes: List[Dict], graph: Dict[str, List[str]], node_map: Dict[str, Dict]
    ) -> List[str]:
        """Validate node connections for logical errors."""
        warnings = []

        # Check each node's position in pipeline
        for node in nodes:
            node_id = node.get("node_id", node.get("id"))
            node_type = node.get("node_type", node.get("type"))

            # Check if ML model comes before split
            if node_type in self.ML_ALGORITHM_NODES:
                # Find all predecessors
                predecessors = self._find_all_predecessors(node_id, graph, node_map)
                predecessor_types = [
                    node_map[pid].get("node_type", node_map[pid].get("type"))
                    for pid in predecessors
                    if pid in node_map
                ]

                if not any(pt in self.SPLIT_NODES for pt in predecessor_types):
                    warnings.append(
                        f"Model '{node_type}' should be connected after a Split node for proper train/test separation"
                    )

            # Check if split has a data source upstream
            if node_type in self.SPLIT_NODES:
                predecessors = self._find_all_predecessors(node_id, graph, node_map)
                predecessor_types = [
                    node_map[pid].get("node_type", node_map[pid].get("type"))
                    for pid in predecessors
                    if pid in node_map
                ]

                if not any(pt in self.DATA_SOURCE_NODES for pt in predecessor_types):
                    warnings.append(
                        "Split node should be connected to a data source (directly or through preprocessing)"
                    )

        return warnings

    def _find_all_predecessors(
        self,
        node_id: str,
        graph: Dict[str, List[str]],
        node_map: Dict[str, Dict],
        visited: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Find all predecessor nodes recursively."""
        if visited is None:
            visited = set()

        predecessors = set()

        # Find direct predecessors
        for source, targets in graph.items():
            if node_id in targets and source not in visited:
                visited.add(source)
                predecessors.add(source)
                # Recursively find predecessors of this predecessor
                predecessors.update(self._find_all_predecessors(source, graph, node_map, visited))

        return predecessors

    def _suggest_next_steps(
        self,
        node_types: List[str],
        has_data_source: bool,
        has_split: bool,
        has_ml_model: bool,
        dataset_metadata: Optional[Dict] = None,
    ) -> List[str]:
        """Suggest logical next steps based on current state."""
        next_steps = []

        if not has_data_source:
            next_steps.append("Upload a dataset to begin building your pipeline")
            return next_steps

        if has_data_source and not has_ml_model:
            # Suggest data preparation
            if dataset_metadata:
                missing_values = dataset_metadata.get("missing_values", {})
                categorical_cols = dataset_metadata.get("categorical_columns", [])

                if sum(missing_values.values()) > 0 and "missing_value_handler" not in node_types:
                    next_steps.append("Add Missing Value Handler to clean your data")

                if categorical_cols and "encoding" not in node_types:
                    next_steps.append(
                        f"Add Encoding node to convert {len(categorical_cols)} categorical columns to numeric"
                    )

            if not has_split:
                next_steps.append("Add Split node to separate training and test data")

            next_steps.append(
                "Choose a model to train (Linear Regression, Logistic Regression, Decision Tree, etc.)"
            )

        elif has_ml_model and not has_split:
            next_steps.insert(0, "Add Split node before your model for proper validation")

        elif has_data_source and has_split and has_ml_model:
            # Pipeline is ready
            if not any(nt in self.RESULT_NODES for nt in node_types):
                next_steps.append("Add Metrics or Visualization nodes to analyze results")
            next_steps.append("Execute your pipeline to train and evaluate the model")

        return next_steps

    def get_model_specific_requirements(self, model_type: str) -> Dict[str, Any]:
        """Get specific requirements for a model type."""
        requirements = {
            "linear_regression": {
                "requires_numeric": True,
                "requires_split": True,
                "target_type": "continuous",
                "preprocessing": [
                    "Handle missing values",
                    "Encode categorical features",
                    "Consider scaling",
                ],
                "evaluation": ["R² Score", "RMSE", "MAE"],
            },
            "logistic_regression": {
                "requires_numeric": True,
                "requires_split": True,
                "target_type": "binary or multiclass categorical",
                "preprocessing": [
                    "Handle missing values",
                    "Encode categorical features",
                    "Balance classes if needed",
                ],
                "evaluation": ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"],
            },
            "decision_tree": {
                "requires_numeric": False,  # Can handle categorical after encoding
                "requires_split": True,
                "target_type": "any",
                "preprocessing": ["Handle missing values", "Encode categorical features"],
                "evaluation": ["Accuracy/R² depending on task", "Feature Importance"],
            },
            "random_forest": {
                "requires_numeric": False,
                "requires_split": True,
                "target_type": "any",
                "preprocessing": ["Handle missing values", "Encode categorical features"],
                "evaluation": ["Accuracy/R²", "Feature Importance", "OOB Score"],
            },
        }

        return requirements.get(model_type, {})


# Singleton instance
pipeline_analyzer = PipelineAnalyzer()
