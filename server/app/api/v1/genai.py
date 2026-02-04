"""
GenAI Chat API endpoint for chatbot functionality.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.ml.nodes.genai.llm_node import LLMNode
from app.ml.nodes.genai.chatbot_node import ChatbotNode
from app.ml.nodes.genai.base import GenAINodeInput

router = APIRouter(prefix="/genai", tags=["genai"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    sessionId: str = "default"
    llmConfig: Dict[str, Any]  # LLM Provider node config
    chatbotConfig: Optional[Dict[str, Any]] = None
    conversationHistory: List[ChatMessage] = []


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    sessionId: str
    tokensUsed: Optional[int] = None
    cost: Optional[float] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and get LLM response.
    
    This endpoint:
    1. Takes user message and LLM config
    2. Manages conversation history via ChatbotNode
    3. Calls LLM via LLMNode
    4. Returns the response
    """
    try:
        # Step 1: Process message through ChatbotNode
        chatbot_node = ChatbotNode(request.chatbotConfig or {})
        
        chatbot_input = GenAINodeInput(data={
            "userMessage": request.message,
            "sessionId": request.sessionId,
            "clearHistory": False,
        })
        
        chatbot_result = await chatbot_node.execute(chatbot_input.data)
        
        if not chatbot_result.success:
            return ChatResponse(
                success=False,
                error=chatbot_result.error,
                sessionId=request.sessionId,
            )
        
        # Get messages from chatbot (includes history + new user message)
        messages = chatbot_result.data.get("messages", [])
        
        # Step 2: Send to LLM
        llm_node = LLMNode(request.llmConfig)
        
        llm_input = GenAINodeInput(data={
            "messages": messages,
        })
        
        llm_result = await llm_node.execute(llm_input.data)
        
        if not llm_result.success:
            return ChatResponse(
                success=False,
                error=llm_result.error,
                sessionId=request.sessionId,
            )
        
        # Step 3: Add assistant response to chatbot history
        assistant_message = llm_result.data.get("response", "")
        
        # Update chatbot with assistant response
        update_input = GenAINodeInput(data={
            "userMessage": "",  # Empty, we're just adding assistant response
            "sessionId": request.sessionId,
            "llmMessages": messages + [{"role": "assistant", "content": assistant_message}],
        })
        
        # Note: In a real implementation, you'd want to persist this
        # For now, the frontend will manage the full conversation
        
        return ChatResponse(
            success=True,
            response=assistant_message,
            sessionId=request.sessionId,
            tokensUsed=llm_result.tokensUsed,
            cost=llm_result.costUSD,
        )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            error=f"Chat error: {str(e)}",
            sessionId=request.sessionId,
        )
