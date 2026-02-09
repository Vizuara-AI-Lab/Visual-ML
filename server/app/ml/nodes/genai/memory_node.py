"""
Memory Node - manages conversation history and context.
"""

from typing import Dict, Any, List
from datetime import datetime

from app.ml.nodes.genai.base import GenAIBaseNode, GenAINodeInput, GenAINodeOutput, ProviderConfig


class MemoryNode(GenAIBaseNode):
    """
    Memory Node - manages conversation history.

    Config:
        sessionId: Session identifier
        maxTurns: Max messages to keep (default: 10)
        summarizeMemory: Summarize old messages (default: False)
        retrieveRelevant: Retrieve relevant past context (default: False)

    Input:
        newMessage: New message to add
        messages: Current conversation messages

    Output:
        messages: Messages with memory context
        memoryStats: {totalMessages, summarized, etc.}
    """

    node_type = "memory"

    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """Memory node doesn't use LLM provider directly."""
        return ProviderConfig(provider="none", model="none")

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Manage conversation memory."""
        data = input_data.data

        session_id = self.config.get("sessionId") or input_data.sessionId
        if not session_id:
            return GenAINodeOutput(success=False, error="sessionId is required", data={})

        max_turns = self.config.get("maxTurns", 10)
        summarize = self.config.get("summarizeMemory", False)

        # Get or create memory session
        # In production, this queries ConversationMemory table
        memory = await self._get_memory(session_id)

        # Get new messages
        new_message = data.get("newMessage")
        current_messages = data.get("messages", [])

        # Add new message to history
        if new_message:
            if isinstance(new_message, str):
                current_messages.append({"role": "user", "content": new_message})
            elif isinstance(new_message, dict):
                current_messages.append(new_message)

        # Trim to max_turns (keep recent)
        if len(current_messages) > max_turns * 2:  # *2 for user+assistant pairs
            if summarize:
                # Summarize old messages
                old_messages = current_messages[: -max_turns * 2]
                summary = self._summarize_messages(old_messages)

                # Keep summary + recent messages
                current_messages = [
                    {"role": "system", "content": f"Previous conversation summary: {summary}"}
                ] + current_messages[-max_turns * 2 :]
            else:
                # Just keep recent
                current_messages = current_messages[-max_turns * 2 :]

        # Save memory (in production)
        await self._save_memory(session_id, current_messages)

        return GenAINodeOutput(
            success=True,
            data={
                "messages": current_messages,
                "memoryStats": {
                    "sessionId": session_id,
                    "totalMessages": len(current_messages),
                    "maxTurns": max_turns,
                    "summarized": summarize,
                },
            },
        )

    async def _get_memory(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history from database."""
        # TODO: Query ConversationMemory table
        # from app.models.genai import ConversationMemory
        # memory = db.query(ConversationMemory).filter(
        #     ConversationMemory.sessionId == session_id
        # ).first()
        # return memory.messages if memory else []

        return []  # Placeholder

    async def _save_memory(self, session_id: str, messages: List[Dict[str, Any]]):
        """Save conversation history to database."""
        # TODO: Update ConversationMemory table
        # from app.models.genai import ConversationMemory
        # memory = db.query(ConversationMemory).filter(...).first()
        # if memory:
        #     memory.messages = messages
        #     memory.lastMessageAt = datetime.utcnow()
        # else:
        #     memory = ConversationMemory(sessionId=session_id, messages=messages)
        #     db.add(memory)
        # db.commit()

        pass  # Placeholder

    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize old messages (simple version)."""
        # In production, use LLM to generate summary
        # For now, just create a simple summary
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        return f"User discussed: {', '.join(user_msgs[:3])}... ({len(messages)} messages)"
