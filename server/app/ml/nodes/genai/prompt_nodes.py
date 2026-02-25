"""
Prompt building nodes.
System prompt node.
"""

from typing import Dict, Any

from app.ml.nodes.genai.base import PromptNode, GenAINodeInput, GenAINodeOutput


class SystemPromptNode(PromptNode):
    """
    System Prompt Node - adds system instruction with role presets.

    Config:
        role: Preset role (e.g., "domain_expert", "tutor", "code_reviewer", "custom")
        customRole: Custom role description (used when role="custom")
        systemPrompt: System instruction text

    Input:
        messages: Existing messages (optional)

    Output:
        messages: Messages with system prompt prepended
    """

    node_type = "system_prompt"

    # Role presets
    ROLE_PRESETS = {
        "domain_expert": "an expert in your field with deep knowledge and experience",
        "tutor": "a patient and helpful tutor who explains concepts clearly",
        "code_reviewer": "an experienced code reviewer who provides constructive feedback",
        "creative_writer": "a creative writer who crafts engaging and imaginative content",
        "data_analyst": "a skilled data analyst who interprets data and provides insights",
        "technical_writer": "a technical writer who creates clear and concise documentation",
        "problem_solver": "a logical problem solver who breaks down complex issues",
        "research_assistant": "a thorough research assistant who finds and summarizes information",
        "helpful_assistant": "a helpful assistant who provides accurate and useful responses",
    }

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Add system prompt to messages."""
        data = input_data.data

        role = self.config.get("role", "helpful_assistant")
        custom_role = self.config.get("customRole", "")
        system_prompt = self.config.get("systemPrompt", "")

        if not system_prompt:
            return GenAINodeOutput(success=False, error="systemPrompt is required", data={})

        # Get role description
        if role == "custom":
            if not custom_role:
                return GenAINodeOutput(
                    success=False, error="customRole is required when role is 'custom'", data={}
                )
            role_description = custom_role
        else:
            role_description = self.ROLE_PRESETS.get(role, self.ROLE_PRESETS["helpful_assistant"])

        # Build system message
        system_content = f"You are {role_description}. {system_prompt}"

        # Get existing messages
        existing_messages = data.get("messages", [])

        # Prepend system message
        messages = [{"role": "system", "content": system_content}]

        # Add existing messages (skip any existing system messages)
        for msg in existing_messages:
            if msg.get("role") != "system":
                messages.append(msg)

        return GenAINodeOutput(
            success=True, data={"messages": messages, "systemPrompt": system_content}
        )
