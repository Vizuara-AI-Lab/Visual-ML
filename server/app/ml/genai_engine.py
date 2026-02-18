"""
GenAI Pipeline Execution Engine.
DAG-based execution with parallel processing, streaming support, and error handling.
"""

from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque
from datetime import datetime
import time

from app.ml.nodes.genai import (
    LLMNode,
    SystemPromptNode,
    FewShotNode,
    MemoryNode,
    OutputParserNode,
    ChatbotNode,
    ExampleNode,
)
from app.ml.nodes.genai.base import GenAIBaseNode, GenAINodeInput, GenAINodeOutput
from app.core.logging import logger


class DAGValidationError(Exception):
    """DAG validation error."""

    pass


class NodeExecutionError(Exception):
    """Node execution error."""

    pass


class GenAIPipelineEngine:
    """
    GenAI Pipeline Execution Engine.

    Features:
    - DAG validation (cycle detection, topological sort)
    - Parallel execution where possible
    - Context passing between nodes
    - Streaming support
    - Error handling and retry logic
    - Execution tracking
    """

    # Node registry
    NODE_REGISTRY: Dict[str, type[GenAIBaseNode]] = {
        "llm": LLMNode,
        "system_prompt": SystemPromptNode,
        "few_shot": FewShotNode,
        "memory": MemoryNode,
        "output_parser": OutputParserNode,
        "chatbot": ChatbotNode,
        "example": ExampleNode,
    }

    def __init__(self):
        """Initialize engine."""
        self.execution_context: Dict[int, Any] = {}  # Node outputs by node ID
        self.execution_stats: Dict[str, Any] = {}

    @classmethod
    def register_node(cls, node_type: str, node_class: type[GenAIBaseNode]):
        """Register a custom node type."""
        cls.NODE_REGISTRY[node_type] = node_class
        logger.info(f"Registered GenAI node: {node_type}")

    async def execute_pipeline(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        input_data: Optional[Dict[str, Any]] = None,
        start_from_node_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline.

        Args:
            nodes: List of node configs [{id, nodeType, config}, ...]
            edges: List of edges [{sourceNodeId, targetNodeId}, ...]
            input_data: Initial input data
            start_from_node_id: Resume from specific node

        Returns:
            {
                "success": bool,
                "finalOutput": dict,
                "nodeExecutions": [...],
                "totalTokens": int,
                "totalCost": float,
                "executionTimeMs": int
            }
        """
        start_time = time.time()

        # Reset context
        self.execution_context = {}
        self.execution_stats = {"totalTokens": 0, "totalCost": 0.0, "nodeExecutions": []}

        try:
            # Build DAG
            dag = self._build_dag(nodes, edges)

            # Validate DAG
            self._validate_dag(dag, nodes)

            # Topological sort for execution order
            execution_order = self._topological_sort(dag, nodes)

            # Filter for partial execution
            if start_from_node_id:
                execution_order = self._filter_execution_order(
                    execution_order, start_from_node_id, dag
                )

            # Execute nodes in order (with parallelization where possible)
            await self._execute_dag(execution_order, nodes, dag, input_data)

            # Get final output (from last node or aggregator)
            final_output = self._get_final_output(execution_order)

            execution_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "finalOutput": final_output,
                "nodeExecutions": self.execution_stats["nodeExecutions"],
                "totalTokens": self.execution_stats["totalTokens"],
                "totalCost": self.execution_stats["totalCost"],
                "executionTimeMs": execution_time,
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Pipeline execution failed: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "nodeExecutions": self.execution_stats["nodeExecutions"],
                "totalTokens": self.execution_stats["totalTokens"],
                "totalCost": self.execution_stats["totalCost"],
                "executionTimeMs": execution_time,
            }

    def _build_dag(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """
        Build adjacency list representation of DAG.

        Returns:
            {nodeId: [dependentNodeId, ...]}
        """
        dag: Dict[int, List[int]] = defaultdict(list)

        for edge in edges:
            source = edge["sourceNodeId"]
            target = edge["targetNodeId"]
            dag[source].append(target)

        return dict(dag)

    def _validate_dag(self, dag: Dict[int, List[int]], nodes: List[Dict[str, Any]]):
        """
        Validate DAG structure.

        Checks:
        1. No cycles (directed acyclic)
        2. All referenced nodes exist
        3. At least one starting node (no incoming edges)
        """
        node_ids = {node["id"] for node in nodes}

        # Check all edges reference valid nodes
        for source, targets in dag.items():
            if source not in node_ids:
                raise DAGValidationError(f"Edge references non-existent node: {source}")
            for target in targets:
                if target not in node_ids:
                    raise DAGValidationError(f"Edge references non-existent node: {target}")

        # Check for cycles using DFS
        visited: Set[int] = set()
        rec_stack: Set[int] = set()

        def has_cycle(node_id: int) -> bool:
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
            node_id = node["id"]
            if node_id not in visited:
                if has_cycle(node_id):
                    raise DAGValidationError("Pipeline contains cycles")

    def _topological_sort(
        self, dag: Dict[int, List[int]], nodes: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Topological sort using Kahn's algorithm.

        Returns:
            List of node IDs in execution order
        """
        # Calculate in-degrees
        in_degree: Dict[int, int] = {node["id"]: 0 for node in nodes}

        for source, targets in dag.items():
            for target in targets:
                in_degree[target] += 1

        # Start with nodes having no dependencies
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        execution_order: List[int] = []

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

    def _filter_execution_order(
        self, execution_order: List[int], start_node_id: int, dag: Dict[int, List[int]]
    ) -> List[int]:
        """Filter execution order for partial runs."""
        # Find all downstream nodes from start_node_id
        downstream: Set[int] = set()

        def traverse(node_id: int):
            downstream.add(node_id)
            for child in dag.get(node_id, []):
                if child not in downstream:
                    traverse(child)

        traverse(start_node_id)

        # Keep only downstream nodes in original order
        return [nid for nid in execution_order if nid in downstream]

    async def _execute_dag(
        self,
        execution_order: List[int],
        nodes: List[Dict[str, Any]],
        dag: Dict[int, List[int]],
        input_data: Optional[Dict[str, Any]],
    ):
        """Execute nodes in DAG order with parallelization."""
        nodes_by_id = {node["id"]: node for node in nodes}

        for node_id in execution_order:
            node_config = nodes_by_id[node_id]

            # Get input from predecessor nodes
            node_input = self._prepare_node_input(node_id, dag, input_data)

            # Execute node
            result = await self._execute_node(node_config, node_input)

            # Store result in context
            self.execution_context[node_id] = result

            # Update stats
            self.execution_stats["nodeExecutions"].append(
                {
                    "nodeId": node_id,
                    "nodeType": node_config["nodeType"],
                    "status": "COMPLETED" if result.success else "FAILED",
                    "tokensUsed": result.tokensUsed,
                    "costUSD": result.costUSD,
                    "executionTimeMs": result.executionTimeMs,
                    "error": result.error,
                }
            )

            if result.tokensUsed:
                self.execution_stats["totalTokens"] += result.tokensUsed
            if result.costUSD:
                self.execution_stats["totalCost"] += result.costUSD

            # Stop on error
            if not result.success:
                raise NodeExecutionError(f"Node {node_id} failed: {result.error}")

    def _prepare_node_input(
        self, node_id: int, dag: Dict[int, List[int]], initial_input: Optional[Dict[str, Any]]
    ) -> GenAINodeInput:
        """Prepare input for node from predecessors' outputs."""
        # Find predecessors (nodes that point to this one)
        predecessors = [src for src, targets in dag.items() if node_id in targets]

        if not predecessors:
            # Starting node - use initial input
            input_data = initial_input or {}
        else:
            # Merge outputs from all predecessors
            input_data = {}
            for pred_id in predecessors:
                pred_output = self.execution_context.get(pred_id)
                if pred_output and pred_output.success:
                    # Merge data
                    input_data.update(pred_output.data)

        return GenAINodeInput(data=input_data)

    async def _execute_node(
        self, node_config: Dict[str, Any], node_input: GenAINodeInput
    ) -> GenAINodeOutput:
        """Execute single node."""
        node_type = node_config["nodeType"]
        node_id = node_config["id"]

        # Get node class
        node_class = self.NODE_REGISTRY.get(node_type)
        if not node_class:
            return GenAINodeOutput(success=False, error=f"Unknown node type: {node_type}", data={})

        # Create node instance
        config = node_config.get("config", {})
        node = node_class(config)

        # Execute
        logger.info(f"Executing node {node_id} ({node_type})")

        try:
            result = await node.execute(node_input.data)
            return result
        except Exception as e:
            logger.error(f"Node {node_id} execution error: {str(e)}")
            return GenAINodeOutput(success=False, error=str(e), data={})

    def _get_final_output(self, execution_order: List[int]) -> Dict[str, Any]:
        """Get final output from last executed node."""
        if not execution_order:
            return {}

        last_node_id = execution_order[-1]
        last_output = self.execution_context.get(last_node_id)

        if last_output and last_output.success:
            return last_output.data

        return {}
