"""
GenAI Base Node.
Abstract base class for all GenAI nodes with LLM provider abstraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
from pydantic import BaseModel
import time
import json

from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput


class GenAINodeInput(NodeInput):
    """GenAI node input with streaming support."""

    streamResponse: bool = False
    sessionId: Optional[str] = None  # For memory/chat nodes


class GenAINodeOutput(NodeOutput):
    """GenAI node output with tokens and cost."""

    tokensUsed: Optional[int] = None
    costUSD: Optional[float] = None
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
    Provides provider abstraction, token counting, cost calculation.
    """

    # Token pricing (USD per 1K tokens) - update periodically
    TOKEN_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
        "dall-e-3": {"fixed": 0.04},  # per image
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize GenAI node."""
        super().__init__(config)
        self.provider_config = self._parse_provider_config(config)
        self.supports_streaming = config.get("stream", False)

    @abstractmethod
    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Parse node config into provider config."""
        pass

    async def _execute(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Execute node with token counting and cost tracking."""
        start_time = time.time()

        try:
            # Check if streaming is requested
            if input_data.streamResponse and self.supports_streaming:
                # For streaming, we'll return a placeholder and handle streaming separately
                result = await self._execute_streaming(input_data)
            else:
                result = await self._execute_genai(input_data)

            execution_time = int((time.time() - start_time) * 1000)

            # Add metadata
            result.executionTimeMs = execution_time

            return result

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return GenAINodeOutput(
                success=False, error=str(e), data={}, executionTimeMs=execution_time
            )

    @abstractmethod
    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Execute GenAI-specific logic."""
        pass

    async def _execute_streaming(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """
        Execute with streaming.
        By default, falls back to non-streaming.
        Override in subclasses for true streaming support.
        """
        return await self._execute_genai(input_data)

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int = 0, is_fixed_cost: bool = False
    ) -> float:
        """
        Calculate cost in USD.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            is_fixed_cost: If True, use fixed pricing (e.g., images)

        Returns:
            Cost in USD
        """
        if model not in self.TOKEN_PRICING:
            # Default fallback pricing
            pricing = {"input": 0.001, "output": 0.002}
        else:
            pricing = self.TOKEN_PRICING[model]

        if is_fixed_cost:
            return pricing.get("fixed", 0.0)

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).
        For production, use tiktoken library.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    async def get_api_key(self, db, student_id: int, api_key_ref: Optional[int]) -> Optional[str]:
        """
        Retrieve API key from encrypted storage.

        Args:
            db: Database session
            student_id: Student ID
            api_key_ref: Reference to APISecret.id

        Returns:
            Decrypted API key or None
        """
        if not api_key_ref:
            return None

        # Import here to avoid circular dependency
        from app.models.genai import APISecret
        from app.core.security import decrypt_api_key

        secret = (
            db.query(APISecret)
            .filter(
                APISecret.id == api_key_ref,
                APISecret.studentId == student_id,
                APISecret.isActive == True,
            )
            .first()
        )

        if not secret:
            raise ValueError(f"API key {api_key_ref} not found or inactive")

        # Decrypt the API key
        return decrypt_api_key(secret.encryptedKey)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate node configuration.
        Override in subclasses for specific validation.
        """
        return True

    def get_input_schema(self) -> Dict[str, Any]:
        """Return input schema for GenAI nodes."""
        return {
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Input data for the node"},
                "streamResponse": {
                    "type": "boolean",
                    "description": "Enable streaming response",
                    "default": False,
                },
                "sessionId": {"type": "string", "description": "Session ID for memory/chat nodes"},
            },
            "required": ["data"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Return output schema for GenAI nodes."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {"type": "object"},
                "error": {"type": "string"},
                "tokensUsed": {"type": "integer"},
                "costUSD": {"type": "number"},
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "executionTimeMs": {"type": "integer"},
                "responseMetadata": {"type": "object"},
            },
            "required": ["success", "data"],
        }


class PromptNode(GenAIBaseNode, ABC):
    """
    Base class for prompt-building nodes.
    These nodes don't call LLM APIs but build prompts.
    """

    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Prompt nodes don't need provider config."""
        return ProviderConfig(provider="none", model="none")

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int = 0, is_fixed_cost: bool = False
    ) -> float:
        """Prompt nodes have no cost."""
        return 0.0
