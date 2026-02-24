"""
GenAI Chat API endpoint - Multi-provider streaming support
Supports: Gemini, OpenAI, Anthropic, DynaRoute
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
from functools import partial
from google import genai
from dynaroute import DynaRouteClient, APIError

from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/genai", tags=["genai"])


class ChatRequest(BaseModel):
    message: str
    provider: str = "gemini"  # "gemini", "openai", "anthropic", or "dynaroute"
    apiKey: Optional[str] = None  # Optional user-provided API key
    systemPrompt: Optional[str] = None  # Optional system prompt from System Prompt node
    examples: Optional[List[Dict[str, Any]]] = None  # Optional examples from Example node


@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat: question → AI → streamed response
    Supports: Gemini, OpenAI, Anthropic, and DynaRoute
    """

    async def generate():
        try:
            if request.provider == "dynaroute":
                # DynaRoute streaming
                api_key = request.apiKey or settings.DYNAROUTE_API_KEY
                client = DynaRouteClient(api_key=api_key)

                # Build messages with system prompt and examples if provided
                messages = []
                if request.systemPrompt:
                    messages.append({"role": "system", "content": request.systemPrompt})

                # Add examples as conversation history
                if request.examples:
                    for example in request.examples:
                        if example.get("userInput") and example.get("expectedOutput"):
                            messages.append({"role": "user", "content": example["userInput"]})
                            messages.append(
                                {"role": "assistant", "content": example["expectedOutput"]}
                            )

                messages.append({"role": "user", "content": request.message})

                loop = asyncio.get_event_loop()
                stream = await loop.run_in_executor(
                    None, partial(client.chat, messages=messages, stream=True)
                )

                for chunk in stream:
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"

            elif request.provider == "openai":
                # OpenAI streaming (require apiKey from frontend)
                try:
                    from openai import OpenAI
                except ImportError:
                    yield f"data: {json.dumps({'error': 'OpenAI package not installed on server'})}\n\n"
                    return
                api_key = request.apiKey
                if not api_key:
                    yield f"data: {json.dumps({'error': 'OpenAI API key must be provided from frontend'})}\n\n"
                    return
                client = OpenAI(api_key=api_key)
                messages = []
                if request.systemPrompt:
                    messages.append({"role": "system", "content": request.systemPrompt})
                if request.examples:
                    for example in request.examples:
                        if example.get("userInput") and example.get("expectedOutput"):
                            messages.append({"role": "user", "content": example["userInput"]})
                            messages.append(
                                {"role": "assistant", "content": example["expectedOutput"]}
                            )
                messages.append({"role": "user", "content": request.message})
                loop = asyncio.get_event_loop()
                stream = await loop.run_in_executor(
                    None,
                    partial(
                        client.chat.completions.create,
                        model="gpt-4o-mini",
                        messages=messages,
                        stream=True,
                    ),
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content})}\n\n"

            elif request.provider == "anthropic":
                # Anthropic streaming (require apiKey from frontend)
                try:
                    from anthropic import Anthropic
                except ImportError:
                    yield f"data: {json.dumps({'error': 'Anthropic package not installed on server'})}\n\n"
                    return
                api_key = request.apiKey
                if not api_key:
                    yield f"data: {json.dumps({'error': 'Anthropic API key must be provided from frontend'})}\n\n"
                    return
                client = Anthropic(api_key=api_key)
                messages = []
                if request.examples:
                    for example in request.examples:
                        if example.get("userInput") and example.get("expectedOutput"):
                            messages.append({"role": "user", "content": example["userInput"]})
                            messages.append(
                                {"role": "assistant", "content": example["expectedOutput"]}
                            )
                messages.append({"role": "user", "content": request.message})
                system_prompt = request.systemPrompt or "You are a helpful assistant."
                loop = asyncio.get_event_loop()
                stream = await loop.run_in_executor(
                    None,
                    partial(
                        client.messages.create,
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        system=system_prompt,
                        messages=messages,
                        stream=True,
                    ),
                )
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield f"data: {json.dumps({'content': event.delta.text})}\n\n"

            else:
                # Gemini streaming (default)
                api_key = request.apiKey or settings.GEMINI_API_KEY
                if not api_key:
                    yield f"data: {json.dumps({'error': 'Gemini API key not configured'})}\n\n"
                    return

                client = genai.Client(api_key=api_key)

                # Build content with system prompt and examples if provided
                content_parts = []

                if request.systemPrompt:
                    content_parts.append(request.systemPrompt)

                # Add examples
                if request.examples:
                    examples_text = "Here are some examples:\n\n"
                    for i, example in enumerate(request.examples, 1):
                        if example.get("userInput") and example.get("expectedOutput"):
                            examples_text += f"Example {i}:\n"
                            examples_text += f"User: {example['userInput']}\n"
                            examples_text += f"Assistant: {example['expectedOutput']}\n\n"
                    content_parts.append(examples_text)

                content_parts.append(f"User: {request.message}")
                content = "\n\n".join(content_parts)

                loop = asyncio.get_event_loop()
                stream = await loop.run_in_executor(
                    None,
                    partial(
                        client.models.generate_content_stream,
                        # Hay ai don't change  the model here
                        model="gemma-3-27b-it",
                        contents=content,
                    ),
                )

                for chunk in stream:
                    if hasattr(chunk, "text") and chunk.text:
                        yield f"data: {json.dumps({'content': chunk.text})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"

        except APIError as e:
            logger.exception(f"DynaRoute API error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.exception(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
