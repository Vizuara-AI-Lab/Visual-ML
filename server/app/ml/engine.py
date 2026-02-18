"""
ML Pipeline Engine - Orchestrates node execution and manages pipelines.
DAG-based execution with dynamic node registration.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple, Type, Callable
from datetime import datetime
from collections import defaultdict, deque

from app.core.logging import logger, log_ml_operation
from app.core.exceptions import NodeExecutionError
from app.ml.nodes.base import BaseNode


class DAGValidationError(Exception):
    """DAG validation error."""

    pass


class MLPipelineEngine:

    NODE_REGISTRY: Dict[str, Type[BaseNode]] = {}

    def __init__(self):
        """Initialize pipeline engine."""
        self.execution_context: Dict[str, Any] = {}  # Node outputs by node_id
        self.execution_history: List[Dict[str, Any]] = []

        # Auto-discover and register nodes on first initialization
        if not self.NODE_REGISTRY:
            self._auto_discover_nodes()

    @classmethod
    def register_node(cls, node_type: str, node_class: Type[BaseNode]):
        """
        Register a custom node type.

        Args:
            node_type: Unique node type identifier
            node_class: Node class (must inherit from BaseNode)
        """
        if not issubclass(node_class, BaseNode):
            raise ValueError(f"Node class must inherit from BaseNode: {node_class}")

        cls.NODE_REGISTRY[node_type] = node_class
        logger.info(f"Registered ML node: {node_type}")

    def _auto_discover_nodes(self):
        """
        Auto-discover and register all nodes from app.ml.nodes package.

        This eliminates the need for manual imports and registration.
        """
        from app.ml.nodes.upload import UploadFileNode
        from app.ml.nodes.select import SelectDatasetNode
        from app.ml.nodes.sample_dataset import SampleDatasetNode
        from app.ml.nodes.clean import PreprocessNode
        from app.ml.nodes.missing_value_handler import MissingValueHandlerNode
        from app.ml.nodes.encoding import EncodingNode
        from app.ml.nodes.transformation import TransformationNode
        from app.ml.nodes.scaling import ScalingNode
        from app.ml.nodes.feature_selection import FeatureSelectionNode
        from app.ml.nodes.split import SplitNode
        from app.ml.nodes.confusion_matrix_node import ConfusionMatrixNode
        from app.ml.nodes.linear_regression_node import LinearRegressionNode
        from app.ml.nodes.logistic_regression_node import LogisticRegressionNode
        from app.ml.nodes.decision_tree_node import DecisionTreeNode
        from app.ml.nodes.random_forest_node import RandomForestNode
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

        # Register all nodes
        nodes_to_register = {
            "upload_file": UploadFileNode,
            "select_dataset": SelectDatasetNode,
            "sample_dataset": SampleDatasetNode,
            "preprocess": PreprocessNode,
            "missing_value_handler": MissingValueHandlerNode,
            "encoding": EncodingNode,
            "transformation": TransformationNode,
            "scaling": ScalingNode,
            "feature_selection": FeatureSelectionNode,
            "split": SplitNode,
            "confusion_matrix": ConfusionMatrixNode,
            "linear_regression": LinearRegressionNode,
            "logistic_regression": LogisticRegressionNode,
            "decision_tree": DecisionTreeNode,
            "random_forest": RandomForestNode,
            "r2_score": R2ScoreNode,
            "mse_score": MSEScoreNode,
            "rmse_score": RMSEScoreNode,
            "mae_score": MAEScoreNode,
            "table_view": TableViewNode,
            "data_preview": DataPreviewNode,
            "statistics_view": StatisticsViewNode,
            "column_info": ColumnInfoNode,
            "chart_view": ChartViewNode,
        }

        for node_type, node_class in nodes_to_register.items():
            self.register_node(node_type, node_class)

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
        if node_type not in self.NODE_REGISTRY:
            raise NodeExecutionError(
                node_type=node_type,
                reason=f"Unknown node type: {node_type}. Available: {list(self.NODE_REGISTRY.keys())}",
                input_data=input_data,
            )

        try:
            logger.info(f"Executing node: {node_type} (dry_run={dry_run})")

            # Create node instance
            node_class = self.NODE_REGISTRY[node_type]
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

    def _build_dag(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Build adjacency list representation of DAG.

        Args:
            nodes: List of node configs
            edges: List of edge connections

        Returns:
            {node_id: [dependent_node_id, ...]}
        """
        dag: Dict[str, List[str]] = defaultdict(list)

        for edge in edges:
            source = edge.get("source") or edge.get("sourceNodeId")
            target = edge.get("target") or edge.get("targetNodeId")

            if source and target:
                dag[source].append(target)

        return dict(dag)

    def _validate_dag(self, dag: Dict[str, List[str]], nodes: List[Dict[str, Any]]):
        """
        Validate DAG structure.

        Checks:
        1. No cycles (directed acyclic)
        2. All referenced nodes exist
        3. At least one starting node (no incoming edges)

        Args:
            dag: Adjacency list representation
            nodes: List of node configs

        Raises:
            DAGValidationError: If validation fails
        """
        node_ids = {node.get("node_id") or node.get("id") for node in nodes}

        # Check all edges reference valid nodes
        for source, targets in dag.items():
            if source not in node_ids:
                raise DAGValidationError(f"Edge references non-existent node: {source}")
            for target in targets:
                if target not in node_ids:
                    raise DAGValidationError(f"Edge references non-existent node: {target}")

        # Check for cycles using DFS
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in dag.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node in nodes:
            node_id = node.get("node_id") or node.get("id")
            if node_id and node_id not in visited:
                if has_cycle(node_id):
                    raise DAGValidationError("Pipeline contains cycles")

    def _topological_sort(
        self, dag: Dict[str, List[str]], nodes: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Topological sort using Kahn's algorithm for execution order.

        Args:
            dag: Adjacency list representation
            nodes: List of node configs

        Returns:
            List of node IDs in execution order

        Raises:
            DAGValidationError: If DAG contains unreachable nodes
        """
        # Calculate in-degrees
        in_degree: Dict[str, int] = {}
        for node in nodes:
            node_id = node.get("node_id") or node.get("id")
            if node_id:
                in_degree[node_id] = 0

        for source, targets in dag.items():
            for target in targets:
                in_degree[target] = in_degree.get(target, 0) + 1

        # Start with nodes having no dependencies
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        execution_order: List[str] = []

        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)

            # Reduce in-degree of neighbors
            for neighbor in dag.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) != len(nodes):
            raise DAGValidationError("Pipeline contains unreachable nodes")

        return execution_order

    def _prepare_node_input(
        self, node_id: str, node_config: Dict[str, Any], dag: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Prepare input data for a node by merging outputs from predecessor nodes.

        Uses node metadata to extract primary outputs and normalize field names.

        Args:
            node_id: Current node ID
            node_config: Node configuration
            dag: DAG adjacency list

        Returns:
            Merged input data dictionary
        """
        # Find predecessor nodes
        predecessors = [src for src, targets in dag.items() if node_id in targets]

        # Start with node's configured input
        input_data = node_config.get("input", {})

        if not predecessors:
            # No predecessors - use configured input only
            return input_data

        # Merge outputs from all predecessors
        # Strategy: Predecessor's dataset_id always wins, user explicit input for other fields
        merged_data = {}
        dataset_id_from_dag = None  # Track dataset_id from DAG flow
        target_column_from_dag = None  # Track target_column from split node
        feature_columns_from_dag = None  # Track feature_columns from split node
        columns_from_dag = None  # Track columns from upstream

        for pred_id in predecessors:
            pred_result = self.execution_context.get(pred_id)

            if pred_result and isinstance(pred_result, dict):
                # Get the node instance to access metadata
                pred_node_type = pred_result.get("node_type")

                if pred_node_type and pred_node_type in self.NODE_REGISTRY:
                    node_class = self.NODE_REGISTRY[pred_node_type]
                    temp_node = node_class()

                    # Use metadata to extract primary output
                    normalized_output = temp_node.extract_primary_output(pred_result)

                    # DEBUG: Log what we're extracting
                    logger.info(
                        f"[DAG DEBUG] Extracting from {pred_node_type} -> {node_id}: "
                        f"dataset_id={normalized_output.get('dataset_id', 'NOT FOUND')}, "
                        f"target_column={normalized_output.get('target_column', 'NOT FOUND')}, "
                        f"columns={normalized_output.get('columns', 'NOT FOUND')}"
                    )

                    # Capture critical fields from DAG (take absolute priority)
                    if "dataset_id" in normalized_output and dataset_id_from_dag is None:
                        dataset_id_from_dag = normalized_output["dataset_id"]

                    if "target_column" in normalized_output and target_column_from_dag is None:
                        target_column_from_dag = normalized_output["target_column"]

                    if "feature_columns" in normalized_output and feature_columns_from_dag is None:
                        feature_columns_from_dag = normalized_output["feature_columns"]

                    if "columns" in normalized_output and columns_from_dag is None:
                        columns_from_dag = normalized_output["columns"]

                    # Only merge fields that don't exist in user's explicit input
                    for key, value in normalized_output.items():
                        if key not in input_data:
                            merged_data[key] = value
                else:
                    # Fallback: merge all fields that don't conflict with user input
                    for key, value in pred_result.items():
                        if key not in input_data:
                            merged_data[key] = value

        # Overlay user's explicit input on top (takes priority for non-critical fields)
        merged_data.update(input_data)

        # CRITICAL: Override empty/falsy values with DAG values for auto-fillable fields
        if dataset_id_from_dag is not None:
            merged_data["dataset_id"] = dataset_id_from_dag
            # Also map to train_dataset_id for ML algorithm nodes that expect this field
            if not merged_data.get("train_dataset_id"):
                merged_data["train_dataset_id"] = dataset_id_from_dag
            logger.info(f"[DAG DEBUG] Forcing dataset_id from DAG: {dataset_id_from_dag}")

        # Override empty target_column with value from split node
        if target_column_from_dag is not None and not merged_data.get("target_column"):
            merged_data["target_column"] = target_column_from_dag
            logger.info(
                f"[DAG DEBUG] Auto-filling target_column from DAG: {target_column_from_dag}"
            )

        # Override empty feature_columns
        if feature_columns_from_dag is not None and not merged_data.get("feature_columns"):
            merged_data["feature_columns"] = feature_columns_from_dag

        # Override empty columns
        if columns_from_dag is not None and not merged_data.get("columns"):
            merged_data["columns"] = columns_from_dag

        # DEBUG: Log final merged data
        logger.info(
            f"[DAG DEBUG] Final input for {node_id}: "
            f"dataset_id={merged_data.get('dataset_id', 'NOT FOUND')}, "
            f"train_dataset_id={merged_data.get('train_dataset_id', 'NOT FOUND')}, "
            f"target_column={merged_data.get('target_column', 'NOT FOUND')}, "
            f"columns={merged_data.get('columns', 'NOT FOUND')}, "
            f"total_keys={len(merged_data)}"
        )

        return merged_data

    async def _execute_dag(
        self,
        execution_order: List[str],
        nodes: List[Dict[str, Any]],
        dag: Dict[str, List[str]],
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """
        Execute nodes in DAG order.

        Future enhancement: This can be optimized for parallel execution
        by executing nodes with satisfied dependencies concurrently.

        Args:
            execution_order: Topologically sorted node IDs
            nodes: List of node configs
            dag: DAG adjacency list
            progress_callback: Optional async callback for progress updates
        """
        # Create node lookup by ID
        node_lookup = {}
        for node in nodes:
            node_id = node.get("node_id") or node.get("id")
            node_lookup[node_id] = node

        # Execute nodes in order
        for node_id in execution_order:
            node_config = node_lookup[node_id]
            node_type = node_config.get("node_type")
            node_label = node_config.get("label", node_type)

            # Prepare input from predecessors
            input_data = self._prepare_node_input(node_id, node_config, dag)

            # Notify node started
            if progress_callback:
                await progress_callback(
                    {
                        "event": "node_started",
                        "node_id": node_id,
                        "node_type": node_type,
                        "label": node_label,
                    }
                )
                # Yield to event loop so SSE generator can send the event before
                # CPU-bound node execution blocks the loop
                await asyncio.sleep(0)

            # Execute node
            try:
                result = await self.execute_node(
                    node_type=node_type, input_data=input_data, node_id=node_id, dry_run=False
                )

                # Store result in execution context
                self.execution_context[node_id] = result

                # DEBUG: Log what was stored
                if "columns" in result:
                    logger.info(
                        f"[DAG DEBUG] Stored result for {node_id} ({node_type}): "
                        f"columns={result.get('columns')}"
                    )

                logger.info(f"Node {node_id} ({node_type}) executed successfully")

                # Notify node completed (include result so client can display incrementally)
                if progress_callback:
                    await progress_callback(
                        {
                            "event": "node_completed",
                            "node_id": node_id,
                            "node_type": node_type,
                            "label": node_label,
                            "success": True,
                            "result": result,
                        }
                    )
                    # Yield to event loop so SSE generator can send the event
                    await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Node {node_id} ({node_type}) execution failed: {str(e)}")

                # Store error in context
                self.execution_context[node_id] = {
                    "success": False,
                    "error": str(e),
                    "node_type": node_type,
                    "node_id": node_id,
                }

                # Notify node failed
                if progress_callback:
                    await progress_callback(
                        {
                            "event": "node_failed",
                            "node_id": node_id,
                            "node_type": node_type,
                            "label": node_label,
                            "error": str(e),
                        }
                    )
                    # Yield to event loop so SSE generator can send the event
                    await asyncio.sleep(0)

                # Stop execution on error
                raise

    async def execute_pipeline(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        dry_run: bool = False,
        current_user: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete pipeline using DAG-based execution.

        This new implementation:
        - Uses DAG for automatic execution order
        - No hardcoded node type handling
        - Supports parallel execution (future)
        - Metadata-driven input preparation

        Args:
            nodes: List of node configs [{"node_id": "1", "node_type": "upload", "input": {...}}, ...]
            edges: List of edges [{"source": "1", "target": "2"}, ...]
            dry_run: If True, validate but don't execute
            current_user: Current user context
            progress_callback: Optional async callback for progress updates

        Returns:
            {
                "success": bool,
                "results": [node outputs],
                "execution_time_ms": int,
                "nodes_executed": int
            }
        """
        import time

        start_time = time.time()

        # Clear execution context
        self.execution_context.clear()
        self.execution_history.clear()

        logger.info(
            f"Executing pipeline with {len(nodes)} nodes, {len(edges)} edges (dry_run={dry_run})"
        )

        # Add user context to all node inputs
        if current_user:
            user_context = {
                "user_id": current_user.get("id"),
                "project_id": current_user.get("project_id"),
            }
            for node in nodes:
                node.setdefault("input", {}).update(user_context)

        try:
            # Build DAG
            dag = self._build_dag(nodes, edges)

            # Validate DAG (cycle detection, node existence)
            self._validate_dag(dag, nodes)

            # Get execution order via topological sort
            execution_order = self._topological_sort(dag, nodes)

            logger.info(f"Execution order: {execution_order}")

            # Execute DAG with progress callback
            await self._execute_dag(execution_order, nodes, dag, progress_callback)

            # Collect results
            results = [self.execution_context.get(node_id, {}) for node_id in execution_order]

            execution_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Pipeline execution complete - {len(results)} nodes executed in {execution_time}ms"
            )

            return {
                "success": True,
                "results": results,
                "execution_time_ms": execution_time,
                "nodes_executed": len(results),
            }

        except (DAGValidationError, NodeExecutionError) as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Pipeline execution failed: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "nodes_executed": len(self.execution_context),
            }
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(
                f"Pipeline execution failed with unexpected error: {str(e)}", exc_info=True
            )

            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "execution_time_ms": execution_time,
                "nodes_executed": len(self.execution_context),
            }

    def get_available_nodes(self) -> List[str]:
        """Get list of available node types."""
        return list(self.NODE_REGISTRY.keys())

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history."""
        return self.execution_history


# Global pipeline engine instance
pipeline_engine = MLPipelineEngine()
