"""
Image Generation Node - DALL-E, Stable Diffusion, etc.
"""

from typing import Dict, Any
import base64

from app.ml.nodes.genai.base import GenAIBaseNode, GenAINodeInput, GenAINodeOutput, ProviderConfig


class ImageGenerationNode(GenAIBaseNode):
    """
    Image Generation Node - generates images from text prompts.

    Config:
        provider: "openai" (DALL-E), "stability" (Stable Diffusion)
        model: "dall-e-3", "dall-e-2", "stable-diffusion-xl"
        size: "1024x1024", "1792x1024", "1024x1792"
        quality: "standard" or "hd" (DALL-E 3 only)
        n: Number of images (default: 1)
        useOwnApiKey: Use student's API key
        apiKeyRef: Reference to APISecret

    Input:
        prompt: Image generation prompt
        negativePrompt: What to avoid (Stable Diffusion)

    Output:
        images: [{url, base64}, ...]
        revisedPrompt: DALL-E's revised prompt
    """

    node_type = "image_generation"

    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Parse image generation config."""
        return ProviderConfig(
            provider=config.get("provider", "openai"),
            model=config.get("model", "dall-e-3"),
        )

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Generate images."""
        data = input_data.data

        prompt = data.get("prompt", "")
        if not prompt:
            return GenAINodeOutput(success=False, error="prompt is required", data={})

        size = self.config.get("size", "1024x1024")
        quality = self.config.get("quality", "standard")
        n = self.config.get("n", 1)

        # Generate image
        try:
            if self.provider_config.provider == "openai":
                result = await self._generate_openai(prompt, size, quality, n)
            elif self.provider_config.provider == "stability":
                result = await self._generate_stability(prompt, size, n, data.get("negativePrompt"))
            else:
                return GenAINodeOutput(
                    success=False,
                    error=f"Unsupported provider: {self.provider_config.provider}",
                    data={},
                )
        except Exception as e:
            return GenAINodeOutput(
                success=False, error=f"Image generation error: {str(e)}", data={}
            )

        # Calculate cost (fixed per image for DALL-E)
        cost = self.calculate_cost(self.provider_config.model, 0, 0, is_fixed_cost=True) * n

        return GenAINodeOutput(
            success=True,
            data=result,
            costUSD=cost,
            provider=self.provider_config.provider,
            model=self.provider_config.model,
        )

    async def _generate_openai(
        self, prompt: str, size: str, quality: str, n: int
    ) -> Dict[str, Any]:
        """Generate image using DALL-E."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed")

        client = AsyncOpenAI()  # Uses OPENAI_API_KEY env var

        response = await client.images.generate(
            model=self.provider_config.model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )

        images = []
        for img in response.data:
            images.append({"url": img.url, "revisedPrompt": img.revised_prompt})

        return {
            "images": images,
            "revisedPrompt": response.data[0].revised_prompt if response.data else None,
        }

    async def _generate_stability(
        self, prompt: str, size: str, n: int, negative_prompt: str = None
    ) -> Dict[str, Any]:
        """Generate image using Stable Diffusion."""
        # Placeholder for Stability AI integration
        # In production:
        # import stability_sdk
        # ...

        return {
            "images": [{"url": "https://placeholder.com/image.png", "base64": None}],
            "message": "Stability AI integration placeholder",
        }

    def get_input_schema(self) -> Dict[str, Any]:
        """Image generation input schema."""
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image generation prompt"},
                "negativePrompt": {
                    "type": "string",
                    "description": "What to avoid (Stable Diffusion)",
                },
            },
            "required": ["prompt"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Image generation output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "base64": {"type": "string"},
                                    "revisedPrompt": {"type": "string"},
                                },
                            },
                        }
                    },
                },
                "costUSD": {"type": "number"},
            },
        }
