"""
Base classes for ML pipeline nodes.
All nodes inherit from BaseNode for consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum
from app.core.logging import logger, log_ml_operation
from app.core.exceptions import NodeExecutionError


class NodeCategory(str, Enum):
    """Node category enumeration."""

    DATA_SOURCE = "data_source"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_TRANSFORM = "data_transform"
    ML_ALGORITHM = "ml_algorithm"
    METRIC = "metric"
    VIEW = "view"
    UTILITY = "utility"


class NodeMetadata(BaseModel):
    """
    Node metadata for DAG-based execution.

    Nodes declare their behavior instead of engine hardcoding knowledge about node types.
    """

    model_config = ConfigDict(extra="allow")

    category: NodeCategory
    """Node category for logical grouping"""

    primary_output_field: Optional[str] = None
    """Primary dataset field in output (e.g., 'encoded_dataset_id', 'train_dataset_id')"""

    output_fields: Dict[str, str] = Field(default_factory=dict)
    """All output fields with descriptions: {'encoded_dataset_id': 'Encoded dataset', ...}"""

    requires_input: bool = True
    """Whether this node requires at least one input connection"""

    can_branch: bool = False
    """Whether this node can output to multiple downstream nodes"""

    max_inputs: Optional[int] = None
    """Maximum number of input connections (None = unlimited)"""

    allowed_source_categories: Optional[List[NodeCategory]] = None
    """Allowed source node categories (None = any)"""

    produces_dataset: bool = False
    """Whether this node produces a dataset output"""


class NodeInput(BaseModel):
    """Base class for node input validation."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


class NodeOutput(BaseModel):
    """Base class for node output typing."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    node_type: str
    execution_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error: Optional[Dict[str, Any]] = None


class BaseNode(ABC):
    """
    Abstract base class for all pipeline nodes.

    Provides:
    - Input validation
    - Error handling
    - Execution timing
    - Logging
    - Dry run support
    - Node metadata for DAG execution
    """

    node_type: str = "base"

    def __init__(self, node_id: Optional[str] = None):
        """
        Initialize node.

        Args:
            node_id: Unique identifier for this node instance
        """
        self.node_id = node_id or f"{self.node_type}_{datetime.utcnow().timestamp()}"
        self.execution_count = 0

    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        """
        Return node metadata for DAG-based execution.

        Override this in each node to declare:
        - Category (preprocessing, view, ml_algorithm, etc.)
        - Primary output field (encoded_dataset_id, train_dataset_id, etc.)
        - Connection requirements
        - Other behavioral metadata

        Returns:
            NodeMetadata instance
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> Type[NodeInput]:
        """Return the input schema for this node."""
        pass

    @abstractmethod
    def get_output_schema(self) -> Type[NodeOutput]:
        """Return the output schema for this node."""
        pass

    @abstractmethod
    async def _execute(self, input_data: NodeInput) -> NodeOutput:
        """
        Execute the node logic.

        Args:
            input_data: Validated input data

        Returns:
            Node output
        """
        pass

    def validate_input(self, data: Dict[str, Any]) -> NodeInput:
        """
        Validate input data against schema.

        Args:
            data: Raw input data

        Returns:
            Validated input instance

        Raises:
            ValidationError: If validation fails
        """
        schema = self.get_input_schema()
        try:
            return schema(**data)
        except Exception as e:
            logger.error(f"Input validation failed for {self.node_type}: {str(e)}")
            raise NodeExecutionError(
                node_type=self.node_type,
                reason=f"Input validation failed: {str(e)}",
                input_data=data,
            )

    async def execute(self, input_data: Dict[str, Any], dry_run: bool = False) -> NodeOutput:
        """
        Execute node with validation and error handling.

        Args:
            input_data: Input data dictionary
            dry_run: If True, validate but don't execute

        Returns:
            Node output
        """
        start_time = datetime.utcnow()

        try:
            # Validate input
            validated_input = self.validate_input(input_data)

            logger.info(f"Executing node: {self.node_type} (ID: {self.node_id})")

            if dry_run:
                logger.info(f"Dry run mode - skipping execution for {self.node_type}")
                return self.get_output_schema()(
                    node_type=self.node_type, execution_time_ms=0, success=True
                )

            # Execute node logic
            result = await self._execute(validated_input)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time

            self.execution_count += 1

            log_ml_operation(
                operation=f"node_execution_{self.node_type}",
                details={
                    "node_id": self.node_id,
                    "execution_time_ms": execution_time,
                    "execution_count": self.execution_count,
                },
                level="info",
            )

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            logger.error(f"Node execution failed [{self.node_type}]: {str(e)}", exc_info=True)
            raise NodeExecutionError(node_type=self.node_type, reason=str(e), input_data=input_data)

    def get_metadata(self) -> Dict[str, Any]:
        """Get node metadata."""
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "execution_count": self.execution_count,
        }

    def extract_primary_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract primary output for downstream nodes using metadata.

        Maps node-specific output fields (e.g., 'encoded_dataset_id') to
        canonical 'dataset_id' field for uniform DAG execution.

        Args:
            result: Node execution result dictionary

        Returns:
            Normalized output with canonical field names
        """
        normalized = dict(result)

        # If node declares a primary output field, map it to canonical 'dataset_id'
        if self.metadata.primary_output_field and self.metadata.primary_output_field in result:
            primary_value = result[self.metadata.primary_output_field]
            normalized["dataset_id"] = primary_value

            # DEBUG
            logger.info(
                f"[DAG DEBUG] extract_primary_output: {self.node_type} - "
                f"Mapped {self.metadata.primary_output_field}={primary_value} -> dataset_id"
            )

        return normalized
