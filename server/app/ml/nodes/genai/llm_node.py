"""
LLM Node implementation.
Unified node for calling OpenAI, Anthropic, HuggingFace, and local models.
"""

from typing import Dict, Any, List
from pydantic import Field

from app.ml.nodes.genai.base import GenAIBaseNode, GenAINodeInput, GenAINodeOutput, ProviderConfig
from app.ml.providers import ProviderFactory, Message


class LLMNode(GenAIBaseNode):
    """
    LLM Node - calls language models with unified interface.

    Config:
        provider: "openai", "anthropic", "huggingface"
        model: Model name (e.g., "gpt-4", "claude-3-opus")
        temperature: 0.0-2.0
        maxTokens: Max output tokens
        topP: Nucleus sampling
        frequencyPenalty: -2.0 to 2.0
        presencePenalty: -2.0 to 2.0
        stopSequences: List of stop sequences
        useOwnApiKey: If True, use student's API key
        apiKeyRef: Reference to APISecret.id
        stream: Enable streaming (if supported)

    Input:
        messages: List of {role, content} messages
        OR
        prompt: Single user prompt (will be converted to messages)
        systemPrompt: Optional system message

    Output:
        response: Generated text
        tokensUsed: Total tokens
        inputTokens: Input tokens
        outputTokens: Output tokens
        model: Model used
        provider: Provider used
    """

    node_type = "llm"

    # Default models for each provider
    DEFAULT_MODELS = {
        "gemini": "gemini-1.5-flash",
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
        "grok": "grok-beta",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
    }

    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Parse LLM config."""
        provider = config.get("provider", "gemini")
        
        # Auto-select default model if not specified or use default for provider
        model = config.get("model")
        if not model or model == "gemini-1.5-flash":  # If default or not set
            model = self.DEFAULT_MODELS.get(provider, "gpt-3.5-turbo")
        
        return ProviderConfig(
            provider=provider,
            model=model,
            temperature=config.get("temperature", 0.7),
            maxTokens=config.get("maxTokens", 1000),
            topP=config.get("topP", 1.0),
            frequencyPenalty=config.get("frequencyPenalty", 0.0),
            presencePenalty=config.get("presencePenalty", 0.0),
            stopSequences=config.get("stopSequences"),
            apiKey=None,  # Will be fetched later
        )

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Execute LLM call."""
        data = input_data.data

        # Parse input messages
        messages = self._parse_messages(data)

        if not messages:
            return GenAINodeOutput(success=False, error="No messages provided", data={})

        # Get API key if needed
        api_key = None
        if self.config.get("useOwnApiKey") and self.config.get("apiKeyRef"):
            # Note: In production, pass db session through context
            # For now, use default provider keys
            pass

        # Create provider
        try:
            provider = ProviderFactory.create(self.provider_config.provider, api_key=api_key)
        except ValueError as e:
            return GenAINodeOutput(success=False, error=str(e), data={})

        # Generate completion
        try:
            response = await provider.generate(
                messages=messages,
                model=self.provider_config.model,
                temperature=self.provider_config.temperature,
                max_tokens=self.provider_config.maxTokens,
                top_p=self.provider_config.topP,
                frequency_penalty=self.provider_config.frequencyPenalty,
                presence_penalty=self.provider_config.presencePenalty,
                stop_sequences=self.provider_config.stopSequences,
            )
        except Exception as e:
            return GenAINodeOutput(success=False, error=f"LLM API error: {str(e)}", data={})

        # Calculate cost
        cost = self.calculate_cost(
            self.provider_config.model, response.inputTokens, response.outputTokens
        )

        return GenAINodeOutput(
            success=True,
            data={
                "response": response.content,
                "messages": [msg.dict() for msg in messages]
                + [{"role": "assistant", "content": response.content}],
                "finishReason": response.finishReason,
            },
            tokensUsed=response.tokensUsed,
            costUSD=cost,
            provider=self.provider_config.provider,
            model=self.provider_config.model,
            responseMetadata=response.metadata,
        )

    def _parse_messages(self, data: Dict[str, Any]) -> List[Message]:
        """
        Parse input data into messages.

        Supports:
        1. messages: [{role, content}, ...]
        2. prompt + systemPrompt
        """
        messages = []

        # Option 1: Explicit messages array
        if "messages" in data:
            for msg in data["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(Message(role=msg["role"], content=msg["content"]))
            return messages

        # Option 2: Simple prompt + system
        if "systemPrompt" in data:
            messages.append(Message(role="system", content=data["systemPrompt"]))

        if "prompt" in data:
            messages.append(Message(role="user", content=data["prompt"]))

        # Option 3: Inherit from previous node's messages
        if "previousMessages" in data:
            for msg in data["previousMessages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(Message(role=msg["role"], content=msg["content"]))

        return messages

    def get_input_schema(self) -> Dict[str, Any]:
        """LLM input schema."""
        return {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                    "description": "Chat messages",
                },
                "prompt": {
                    "type": "string",
                    "description": "Single user prompt (alternative to messages)",
                },
                "systemPrompt": {
                    "type": "string",
                    "description": "System message (used with prompt)",
                },
            },
            "oneOf": [{"required": ["messages"]}, {"required": ["prompt"]}],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """LLM output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "messages": {"type": "array"},
                        "finishReason": {"type": "string"},
                    },
                },
                "tokensUsed": {"type": "integer"},
                "costUSD": {"type": "number"},
                "provider": {"type": "string"},
                "model": {"type": "string"},
            },
        }
