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
        """Execute chatbot - just manage conversation history."""
        data = input_data.data

        user_message = data.get("userMessage", "")
        if not user_message:
            return GenAINodeOutput(success=False, error="userMessage is required", data={})

        session_id = data.get("sessionId", "default")
        clear_history = data.get("clearHistory", False)
        max_history = self.config.get("maxHistory", 20)
        clear_on_run = self.config.get("clearOnRun", False)

        # Get or create session
        if clear_history or clear_on_run or session_id not in self._sessions:
            self._sessions[session_id] = []

        # Get existing messages from connected nodes (system prompt, examples, etc.)
        incoming_messages = data.get("llmMessages", [])
        
        # Start with session history
        messages = self._sessions[session_id].copy()
        
        # If we have incoming messages from connected nodes, prepend them
        if incoming_messages:
            # Only add if not already in history
            for msg in incoming_messages:
                if msg not in messages:
                    messages.insert(0, msg)

        # Add user message
        messages.append({"role": "user", "content": user_message})

        # Trim history if needed (keep system messages)
        if len(messages) > max_history:
            system_msgs = [msg for msg in messages if msg.get("role") == "system"]
            other_msgs = [msg for msg in messages if msg.get("role") != "system"]
            messages = system_msgs + other_msgs[-(max_history - len(system_msgs)):]

        # Save to session
        self._sessions[session_id] = messages

        return GenAINodeOutput(
            success=True,
            data={
                "messages": messages,
                "sessionId": session_id,
                "messageCount": len(messages),
                "userMessage": user_message,
            },
        )

    def get_input_schema(self) -> Dict[str, Any]:
        """Chatbot input schema."""
        return {
            "type": "object",
            "properties": {
                "userMessage": {"type": "string", "description": "User's message"},
                "sessionId": {"type": "string", "description": "Session ID (optional)"},
                "clearHistory": {
                    "type": "boolean",
                    "description": "Clear conversation history",
                    "default": False,
                },
                "llmMessages": {
                    "type": "array",
                    "description": "Messages from connected nodes",
                },
            },
            "required": ["userMessage"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Chatbot output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "messages": {"type": "array"},
                        "sessionId": {"type": "string"},
                        "messageCount": {"type": "integer"},
                        "userMessage": {"type": "string"},
                    },
                },
            },
        }
