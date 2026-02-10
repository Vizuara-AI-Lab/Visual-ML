"""
GenAI Base Node.
Abstract base class for all GenAI nodes with LLM provider abstraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List, Type
from pydantic import BaseModel, Field
import time
import json

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput


class GenAINodeInput(NodeInput):
    """GenAI node input."""

    data: Dict[str, Any] = Field(default_factory=dict)
    sessionId: Optional[str] = None


class GenAINodeOutput(NodeOutput):
    """GenAI node output."""

    node_type: str = "genai"
    execution_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    provider: Optional[str] = None
    model: Optional[str] = None
    responseMetadata: Optional[Dict[str, Any]] = None


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    provider: str
    model: str
    apiKey: Optional[str] = None
    temperature: float = 0.7
    maxTokens: int = 1000
    topP: float = 1.0
    frequencyPenalty: float = 0.0
    presencePenalty: float = 0.0
    stopSequences: Optional[List[str]] = None


class GenAIBaseNode(BaseNode, ABC):
    """
    Base class for GenAI nodes.
    Provides provider abstraction.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize GenAI node."""
        super().__init__()
        self.config = config
        self.provider_config = self._parse_provider_config(config)

    @abstractmethod
    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Parse node config into provider config."""
        pass

    async def _execute(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Execute node."""
        start_time = time.time()

        try:
            result = await self._execute_genai(input_data)
            execution_time = int((time.time() - start_time) * 1000)

            # Add metadata
            result.execution_time_ms = execution_time

            return result

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return GenAINodeOutput(
                node_type=self.node_type,
                execution_time_ms=execution_time,
                success=False,
                error=str(e),
                data={},
            )

    @abstractmethod
    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Execute GenAI-specific logic."""
        pass

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema for GenAI nodes."""
        return GenAINodeInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema for GenAI nodes."""
        return GenAINodeOutput


class PromptNode(GenAIBaseNode, ABC):
    """
    Base class for prompt-building nodes.
    These nodes don't call LLM APIs but build prompts.
    """

    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Prompt nodes don't need provider config."""
        return ProviderConfig(provider="none", model="none")
