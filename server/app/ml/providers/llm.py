"""
LLM Provider abstraction layer.
Unified interface for OpenAI, Anthropic, HuggingFace, and local models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from pydantic import BaseModel
import os


class Message(BaseModel):
    """Chat message."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """Unified LLM response."""

    content: str
    tokensUsed: int
    inputTokens: int
    outputTokens: int
    finishReason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize provider."""
        self.api_key = api_key or self._get_default_api_key()

    @abstractmethod
    def _get_default_api_key(self) -> Optional[str]:
        """Get default API key from environment."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def _get_default_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")

    async def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = AsyncOpenAI(api_key=self.api_key)

        # Convert messages to OpenAI format
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Call API
        response = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            stop=kwargs.get("stop_sequences"),
        )

        choice = response.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            tokensUsed=response.usage.total_tokens,
            inputTokens=response.usage.prompt_tokens,
            outputTokens=response.usage.completion_tokens,
            finishReason=choice.finish_reason,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using OpenAI."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = AsyncOpenAI(api_key=self.api_key)

        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        stream = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            top_p=kwargs.get("top_p", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def _get_default_api_key(self) -> Optional[str]:
        """Get Anthropic API key from environment."""
        return os.getenv("ANTHROPIC_API_KEY")

    async def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using Anthropic."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        client = AsyncAnthropic(api_key=self.api_key)

        # Anthropic requires system message separate
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})

        # Call API
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=conversation_messages,
            top_p=kwargs.get("top_p", 1.0),
            stop_sequences=kwargs.get("stop_sequences"),
        )

        return LLMResponse(
            content=response.content[0].text,
            tokensUsed=response.usage.input_tokens + response.usage.output_tokens,
            inputTokens=response.usage.input_tokens,
            outputTokens=response.usage.output_tokens,
            finishReason=response.stop_reason,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using Anthropic."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        client = AsyncAnthropic(api_key=self.api_key)

        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})

        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=conversation_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider."""

    def _get_default_api_key(self) -> Optional[str]:
        """Get HuggingFace API key from environment."""
        return os.getenv("HUGGINGFACE_API_KEY")

    async def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using HuggingFace."""
        try:
            from huggingface_hub import AsyncInferenceClient
        except ImportError:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

        client = AsyncInferenceClient(token=self.api_key)

        # Format prompt (most HF models expect string, not chat format)
        prompt = self._format_prompt(messages)

        response = await client.text_generation(
            prompt=prompt,
            model=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=kwargs.get("top_p", 0.95),
        )

        # Estimate tokens (HF doesn't always return token count)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        return LLMResponse(
            content=response,
            tokensUsed=input_tokens + output_tokens,
            inputTokens=input_tokens,
            outputTokens=output_tokens,
            metadata={"model": model},
        )

    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using HuggingFace."""
        try:
            from huggingface_hub import AsyncInferenceClient
        except ImportError:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

        client = AsyncInferenceClient(token=self.api_key)
        prompt = self._format_prompt(messages)

        async for token in client.text_generation(
            prompt=prompt,
            model=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            yield token

    def _format_prompt(self, messages: List[Message]) -> str:
        """Format messages as a single prompt string."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        return "\n\n".join(parts) + "\n\nAssistant:"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def _get_default_api_key(self) -> Optional[str]:
        """Get Gemini API key from environment."""
        return os.getenv("GEMINI_API_KEY")

    async def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using Gemini."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        genai.configure(api_key=self.api_key)
        
        # Create model instance
        gemini_model = genai.GenerativeModel(model)
        
        # Convert messages to Gemini format
        # Gemini uses a different format - system message is separate
        system_instruction = None
        conversation_parts = []
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                conversation_parts.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                conversation_parts.append({"role": "model", "parts": [msg.content]})
        
        # Build prompt from messages
        if system_instruction:
            prompt = f"{system_instruction}\n\n"
        else:
            prompt = ""
        
        for part in conversation_parts:
            role = "User" if part["role"] == "user" else "Assistant"
            prompt += f"{role}: {part['parts'][0]}\n\n"
        
        # Generate response
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=kwargs.get("top_p", 1.0),
        )
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        # Estimate tokens (Gemini API may not always provide exact counts)
        input_tokens = len(prompt) // 4
        output_tokens = len(response.text) // 4
        
        return LLMResponse(
            content=response.text,
            tokensUsed=input_tokens + output_tokens,
            inputTokens=input_tokens,
            outputTokens=output_tokens,
            finishReason=str(response.candidates[0].finish_reason) if response.candidates else None,
            metadata={"model": model},
        )

    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using Gemini."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        genai.configure(api_key=self.api_key)
        gemini_model = genai.GenerativeModel(model)
        
        # Build prompt
        system_instruction = None
        conversation_parts = []
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                conversation_parts.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                conversation_parts.append({"role": "model", "parts": [msg.content]})
        
        if system_instruction:
            prompt = f"{system_instruction}\n\n"
        else:
            prompt = ""
        
        for part in conversation_parts:
            role = "User" if part["role"] == "user" else "Assistant"
            prompt += f"{role}: {part['parts'][0]}\n\n"
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class GrokProvider(BaseLLMProvider):
    """xAI Grok API provider."""

    def _get_default_api_key(self) -> Optional[str]:
        """Get Grok API key from environment."""
        return os.getenv("GROK_API_KEY")

    async def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using Grok."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Grok uses OpenAI-compatible API
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

        # Convert messages to OpenAI format
        grok_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Call API
        response = await client.chat.completions.create(
            model=model,
            messages=grok_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            stop=kwargs.get("stop_sequences"),
        )

        choice = response.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            tokensUsed=response.usage.total_tokens,
            inputTokens=response.usage.prompt_tokens,
            outputTokens=response.usage.completion_tokens,
            finishReason=choice.finish_reason,
            metadata={
                "model": response.model,
                "id": response.id,
            },
        )

    async def generate_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using Grok."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

        grok_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        stream = await client.chat.completions.create(
            model=model,
            messages=grok_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            top_p=kwargs.get("top_p", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ProviderFactory:
    """Factory for creating LLM providers."""

    _providers: Dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "gemini": GeminiProvider,
        "grok": GrokProvider,
    }

    @classmethod
    def create(cls, provider: str, api_key: Optional[str] = None) -> BaseLLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider: Provider name (openai, anthropic, huggingface)
            api_key: Optional API key override

        Returns:
            Provider instance

        Raises:
            ValueError: If provider not supported
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {', '.join(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_lower]
        return provider_class(api_key=api_key)

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLMProvider]):
        """Register a custom provider."""
        cls._providers[name.lower()] = provider_class
