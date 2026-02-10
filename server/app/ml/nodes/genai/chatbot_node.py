"""
Chatbot Node implementation.
Simple chat interface that connects to LLM node for completions.
"""

from typing import Dict, Any, List, Optional

from app.ml.nodes.genai.base import GenAIBaseNode, GenAINodeInput, GenAINodeOutput, ProviderConfig


class ChatbotNode(GenAIBaseNode):
    """
    Chatbot Node - simple chat interface.

    This node provides a chat interface and must be connected to an LLM node
    for actual completions. It manages conversation history only.

    Config:
        maxHistory: Maximum messages to keep in history (default: 20)
        clearOnRun: Clear history on each run (default: false)

    Input:
        userMessage: User's message text
        sessionId: Optional session ID for conversation tracking
        clearHistory: If True, clear conversation history
        llmMessages: Messages from connected LLM/System Prompt nodes

    Output:
        messages: Full conversation history with new user message
        sessionId: Session identifier
        messageCount: Number of messages in history
    """

    node_type = "chatbot"

    # In-memory session storage (in production, use Redis or database)
    _sessions: Dict[str, List[Dict[str, str]]] = {}

    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Chatbot doesn't use provider directly."""
        return ProviderConfig(provider="none", model="none")

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Execute chatbot - manage conversation history."""
        data = input_data.data

        user_message = data.get("userMessage", "")
        if not user_message:
            return GenAINodeOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                success=False,
                error="userMessage is required",
                data={},
            )

        session_id = data.get("sessionId", "default")
        max_history = self.config.get("maxHistory", 20)

        # Get or create session
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        # Get current history
        messages = self._sessions[session_id].copy()

        # Add new user message
        messages.append({"role": "user", "content": user_message})

        # Trim if too long
        if len(messages) > max_history:
            messages = messages[-max_history:]

        # Save to session
        self._sessions[session_id] = messages

        return GenAINodeOutput(
            node_type=self.node_type,
            execution_time_ms=0,
            success=True,
            data={
                "messages": messages,
                "sessionId": session_id,
                "messageCount": len(messages),
            },
        )
