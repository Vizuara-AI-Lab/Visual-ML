"""
ML Pipeline Engine - Orchestrates node execution and manages pipelines.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from app.ml.nodes.upload import UploadFileNode
from app.ml.nodes.select import SelectDatasetNode
from app.ml.nodes.clean import PreprocessNode
from app.ml.nodes.missing_value_handler import MissingValueHandlerNode
from app.ml.nodes.encoding import EncodingNode
from app.ml.nodes.transformation import TransformationNode
from app.ml.nodes.scaling import ScalingNode
from app.ml.nodes.feature_selection import FeatureSelectionNode
from app.ml.nodes.split import SplitNode
from app.ml.nodes.confusion_matrix_node import ConfusionMatrixNode

# Individual ML Algorithm Nodes
from app.ml.nodes.linear_regression_node import LinearRegressionNode
from app.ml.nodes.logistic_regression_node import LogisticRegressionNode
from app.ml.nodes.decision_tree_node import DecisionTreeNode
from app.ml.nodes.random_forest_node import RandomForestNode

# Result/Metrics Nodes
from app.ml.nodes.results_and_metrics import (
    R2ScoreNode,
    MSEScoreNode,
    RMSEScoreNode,
    MAEScoreNode,
)

from app.ml.nodes.view import (
    TableViewNode,
    DataPreviewNode,
    StatisticsViewNode,
    ColumnInfoNode,
    ChartViewNode,
)
from app.core.logging import logger, log_ml_operation
from app.core.exceptions import NodeExecutionError


class MLPipelineEngine:
    """
    ML Pipeline orchestration engine.

    Manages:
    - Node registration and discovery
    - Pipeline execution
    - Result aggregation
    - Error handling
    """

    def __init__(self):
        """Initialize pipeline engine."""
        self.nodes = {
            "upload_file": UploadFileNode,
            "select_dataset": SelectDatasetNode,
            "preprocess": PreprocessNode,
            "missing_value_handler": MissingValueHandlerNode,
            "encoding": EncodingNode,
            "transformation": TransformationNode,
            "scaling": ScalingNode,
            "feature_selection": FeatureSelectionNode,
            "split": SplitNode,
            "confusion_matrix": ConfusionMatrixNode,
            # Individual ML Algorithm Nodes
            "linear_regression": LinearRegressionNode,
            "logistic_regression": LogisticRegressionNode,
            "decision_tree": DecisionTreeNode,
            "random_forest": RandomForestNode,
            # Result/Metrics Nodes
            "r2_score": R2ScoreNode,
            "mse_score": MSEScoreNode,
            "rmse_score": RMSEScoreNode,
            "mae_score": MAEScoreNode,
            # View Nodes
            "table_view": TableViewNode,
            "data_preview": DataPreviewNode,
            "statistics_view": StatisticsViewNode,
            "column_info": ColumnInfoNode,
            "chart_view": ChartViewNode,
        }
        self.execution_history: List[Dict[str, Any]] = []

    async def execute_node(
        self,
        node_type: str,
        input_data: Dict[str, Any],
        node_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a single node.

        Args:
            node_type: Type of node to execute
            input_data: Input data for the node
            node_id: Optional custom node ID
            dry_run: If True, validate but don't execute

        Returns:
            Node output as dictionary

        Raises:
            NodeExecutionError: If node execution fails
        """
        if node_type not in self.nodes:
            raise NodeExecutionError(
                node_type=node_type,
                reason=f"Unknown node type: {node_type}. Available: {list(self.nodes.keys())}",
                input_data=input_data,
            )

        try:
            logger.info(f"Executing node: {node_type} (dry_run={dry_run})")

            # Create node instance
            node_class = self.nodes[node_type]
            node = node_class(node_id=node_id)

            # Execute node
            start_time = datetime.utcnow()
            result = await node.execute(input_data, dry_run=dry_run)
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Convert to dictionary
            result_dict = result.model_dump()

            # Log execution
            log_ml_operation(
                operation=f"node_execution_{node_type}",
                details={
                    "node_type": node_type,
                    "node_id": node.node_id,
                    "execution_time_seconds": execution_time,
                    "dry_run": dry_run,
                    "success": True,
                },
                level="info",
            )

            # Store in history
            self.execution_history.append(
                {
                    "node_type": node_type,
                    "node_id": node.node_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "execution_time_seconds": execution_time,
                    "success": True,
                }
            )

            return result_dict

        except NodeExecutionError:
            raise
        except Exception as e:
            logger.error(f"Node execution failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(node_type=node_type, reason=str(e), input_data=input_data)

    def _get_connected_source_output(
        self, node_id: str, edges: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get output from the node that's directly connected to this node.

        Args:
            node_id: Current node ID
            edges: List of edges in the pipeline graph
            results: List of results from previously executed nodes

        Returns:
            Output dictionary from connected source node, or empty dict if no connection
        """
        if not edges or not node_id:
            logger.info(
                f"_get_connected_source_output: No edges or node_id (node_id={node_id}, edges count={len(edges) if edges else 0})"
            )
            return {}

        # Find edge where target is current node
        incoming_edge = next((e for e in edges if e.get("target") == node_id), None)

        if not incoming_edge:
            logger.info(f"_get_connected_source_output: No incoming edge found for {node_id}")
            return {}

        source_node_id = incoming_edge.get("source")
        if not source_node_id:
            logger.info(f"_get_connected_source_output: Incoming edge has no source for {node_id}")
            return {}

        logger.info(
            f"_get_connected_source_output: Looking for source {source_node_id} in {len(results)} results"
        )

        # Log all result node_ids for debugging
        result_node_ids = [r.get("node_id", "NO_ID") for r in results]
        logger.info(
            f"_get_connected_source_output: Available node_ids in results: {result_node_ids}"
        )

        # Find result from source node (search by node_id in results)
        source_result = next((r for r in results if r.get("node_id") == source_node_id), {})

        logger.info(
            f"Found connected source for {node_id}: {source_node_id} "
            f"(found result: {bool(source_result)}, has dataset_id: {bool(source_result.get('dataset_id'))})"
        )

        return source_result

    async def execute_pipeline(
        self,
        pipeline: List[Dict[str, Any]],
        edges: List[Dict[str, Any]] = None,
        dry_run: bool = False,
        current_user: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a complete pipeline of nodes.

        Args:
            pipeline: List of node configurations
            edges: List of edge connections (source -> target)
            dry_run: If True, validate but don't execute
            current_user: Current user context for authentication

        Returns:
            List of node outputs

        Example pipeline:
            [
                {"node_type": "upload_file", "input": {...}},
                {"node_type": "preprocess", "input": {...}},
                {"node_type": "split", "input": {...}},
                {"node_type": "train", "input": {...}},
                {"node_type": "evaluate", "input": {...}},
            ]
        """
        # Clear execution history to ensure fresh pipeline execution
        self.execution_history.clear()

        # Initialize edges (backward compatibility - default to empty if not provided)
        edges = edges or []

        logger.info(
            f"Executing pipeline with {len(pipeline)} nodes, {len(edges)} edges (dry_run={dry_run})"
        )

        results = []
        previous_output = {}  # Store output from previous node for data flow

        # Extract user context for nodes that need it
        user_context = {}
        if current_user:
            user_context["user_id"] = current_user.get("id")
            user_context["project_id"] = current_user.get("project_id")  # May be None

        for i, node_config in enumerate(pipeline):
            node_type = node_config.get("node_type")
            input_data = node_config.get("input", {})
            node_id = node_config.get("node_id")

            # View nodes should NOT inherit dataset_id from previous_output
            # They should use their own configured dataset_id
            # This allows multiple view nodes to connect to the same source independently
            view_node_types = [
                "table_view",
                "data_preview",
                "statistics_view",
                "column_info",
                "chart_view",
            ]

            # Preprocessing nodes that can operate in parallel or sequential
            preprocessing_node_types = [
                "missing_value_handler",
                "encoding",
                "transformation",
                "scaling",
                "feature_selection",
            ]

            # For view nodes, prefer connected source output over configured dataset_id
            if node_type in view_node_types:
                # FIRST: Check if connected to a preprocessing node
                connected_source = self._get_connected_source_output(node_id, edges, results)

                # Check if connected source has preprocessed/encoded/etc dataset_id
                has_preprocessing_output = any(
                    key in connected_source
                    for key in [
                        "preprocessed_dataset_id",
                        "encoded_dataset_id",
                        "transformed_dataset_id",
                        "scaled_dataset_id",
                        "selected_dataset_id",
                    ]
                )

                if has_preprocessing_output:
                    # Connected to preprocessing node - use its output
                    # Normalize the dataset ID field name
                    normalized_source = dict(connected_source)
                    for old_field in [
                        "preprocessed_dataset_id",
                        "encoded_dataset_id",
                        "transformed_dataset_id",
                        "scaled_dataset_id",
                        "selected_dataset_id",
                    ]:
                        if old_field in normalized_source:
                            normalized_source["dataset_id"] = normalized_source[old_field]
                            dataset_field_used = old_field
                            break

                    merged_input = {**user_context, **input_data, **normalized_source}
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(view node - using {dataset_field_used}: {normalized_source['dataset_id']})"
                    )
                elif connected_source.get("dataset_id"):
                    # Connected to dataset node or other source with dataset_id
                    merged_input = {**user_context, **input_data, **connected_source}
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(view node - using connected dataset_id: {connected_source.get('dataset_id')})"
                    )
                else:
                    # No connection or connected source has no dataset - use configured dataset_id
                    configured_dataset_id = input_data.get("dataset_id")
                    merged_input = {**user_context, **input_data}
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(view node - using configured dataset_id: {configured_dataset_id})"
                    )
            elif node_type in preprocessing_node_types:
                # Graph-aware preprocessing logic:
                # Use the connection graph to find the ACTUAL connected source
                # This works for both sequential (Handler→Encoding) and parallel (Dataset→[Handler1, Handler2])

                connected_source = self._get_connected_source_output(node_id, edges, results)

                # Check if connected source is a preprocessing node
                is_sequential_preprocessing = any(
                    key in connected_source
                    for key in [
                        "preprocessed_dataset_id",
                        "encoded_dataset_id",
                        "transformed_dataset_id",
                        "scaled_dataset_id",
                        "selected_dataset_id",
                    ]
                )

                if is_sequential_preprocessing:
                    # Sequential: Inherit from connected preprocessing node
                    # Normalize the dataset ID field name so downstream nodes can find it
                    normalized_source = dict(connected_source)
                    for old_field in [
                        "preprocessed_dataset_id",
                        "encoded_dataset_id",
                        "transformed_dataset_id",
                        "scaled_dataset_id",
                        "selected_dataset_id",
                    ]:
                        if old_field in normalized_source:
                            normalized_source["dataset_id"] = normalized_source[old_field]
                            break

                    # CRITICAL FIX: Merge order matters!
                    # normalized_source first (dataset metadata), then input_data (user config)
                    # This ensures user's column selection isn't overwritten by dataset metadata
                    merged_input = {**user_context, **normalized_source, **input_data}

                    # Find which preprocessing dataset field is present for logging
                    dataset_field = next(
                        (
                            key
                            for key in [
                                "preprocessed_dataset_id",
                                "encoded_dataset_id",
                                "transformed_dataset_id",
                                "scaled_dataset_id",
                                "selected_dataset_id",
                            ]
                            if key in connected_source
                        ),
                        "dataset_id",
                    )
                    dataset_value = connected_source.get(dataset_field, "N/A")

                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(sequential preprocessing - using {dataset_field}: {dataset_value})"
                    )
                else:
                    # Parallel or first in chain: Use configured dataset_id
                    merged_input = {**user_context, **input_data}
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(parallel/first preprocessing - using configured dataset_id: {input_data.get('dataset_id')})"
                    )

            # Metric/Result nodes should inherit from their connected ML algorithm node
            elif node_type in [
                "r2_score",
                "mae_score",
                "mse_score",
                "rmse_score",
                "confusion_matrix",
            ]:
                # Get output from connected ML algorithm node
                connected_source = self._get_connected_source_output(node_id, edges, results)

                if connected_source:
                    # Merge connected source (ML algorithm output) with input_data
                    merged_input = {**user_context, **input_data, **connected_source}
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(metric node - inheriting from connected ML algorithm)"
                    )
                else:
                    # Fallback to previous_output for backward compatibility
                    merged_input = {**user_context, **input_data, **previous_output}
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(metric node - using previous output)"
                    )

            # Split and ML algorithm nodes should also use graph-based connections
            elif node_type in [
                "split",  # Data splitting
                "linear_regression",
                "logistic_regression",  # ML algorithms
                "decision_tree",
                "random_forest",  # Tree-based
                "svm",
                "knn",
                "naive_bayes",  # Other algorithms
            ]:
                # Get output from connected node (could be preprocessing, encoding, etc.)
                connected_source = self._get_connected_source_output(node_id, edges, results)

                if connected_source:
                    # Normalize dataset field names (preprocessing nodes use different field names)
                    normalized_source = dict(connected_source)
                    for old_field in [
                        "preprocessed_dataset_id",
                        "encoded_dataset_id",
                        "transformed_dataset_id",
                        "scaled_dataset_id",
                        "selected_dataset_id",
                    ]:
                        if old_field in normalized_source:
                            normalized_source["dataset_id"] = normalized_source[old_field]
                            break

                    # Use graph-based connection with normalized field names
                    merged_input = {**user_context, **input_data, **normalized_source}
                    source_dataset = (
                        normalized_source.get("dataset_id")
                        or normalized_source.get("train_dataset_id")
                        or "N/A"
                    )
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(graph-based - using dataset from connected node: {source_dataset})"
                    )
                else:
                    # Fallback to previous_output for backward compatibility
                    merged_input = {**user_context, **input_data, **previous_output}
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(fallback - using previous output)"
                    )
            else:
                # For all other nodes, merge previous output (normal sequential data flow)
                # This is the fallback for any nodes not explicitly categorized above
                merged_input = {**user_context, **input_data, **previous_output}

                # Log which dataset is being used
                if "dataset_id" in previous_output and previous_output["dataset_id"]:
                    logger.info(
                        f"Pipeline step {i+1}/{len(pipeline)}: {node_type} "
                        f"(using dataset from previous node: {previous_output['dataset_id']})"
                    )
                else:
                    logger.info(f"Pipeline step {i+1}/{len(pipeline)}: {node_type}")

            try:
                result = await self.execute_node(
                    node_type=node_type, input_data=merged_input, node_id=node_id, dry_run=dry_run
                )

                # Ensure result includes node_id for graph-aware lookup
                if node_id and "node_id" not in result:
                    result["node_id"] = node_id

                results.append(result)

                # Store output for next node (exclude metadata fields)
                previous_output = {
                    k: v
                    for k, v in result.items()
                    if k not in ["node_type", "execution_time_ms", "timestamp", "success", "error"]
                }

                # Normalize dataset ID field names for data flow between nodes
                # All preprocessing nodes output different field names but next nodes expect 'dataset_id'
                dataset_id_mappings = {
                    "preprocessed_dataset_id": "dataset_id",  # missing_value_handler, clean
                    "encoded_dataset_id": "dataset_id",  # encoding
                    "transformed_dataset_id": "dataset_id",  # transformation
                    "scaled_dataset_id": "dataset_id",  # scaling
                    "selected_dataset_id": "dataset_id",  # feature_selection
                }

                for old_key, new_key in dataset_id_mappings.items():
                    if old_key in previous_output and new_key not in previous_output:
                        previous_output[new_key] = previous_output[old_key]
                        logger.debug(
                            f"Mapped {old_key} -> {new_key} for next node: {previous_output[new_key]}"
                        )

            except NodeExecutionError as e:
                logger.error(f"Pipeline failed at step {i+1}: {node_type}")
                # Add error to results
                results.append({"node_type": node_type, "success": False, "error": e.to_dict()})
                # Stop pipeline execution
                break

        logger.info(f"Pipeline execution complete - {len(results)} nodes executed")

        return results

    def get_available_nodes(self) -> List[str]:
        """Get list of available node types."""
        return list(self.nodes.keys())

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history."""
        return self.execution_history


# Global pipeline engine instance
pipeline_engine = MLPipelineEngine()
