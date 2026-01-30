"""
Prompt building nodes.
System prompts, few-shot examples, and template nodes.
"""

from typing import Dict, Any, List
import re

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

    def get_input_schema(self) -> Dict[str, Any]:
        """System prompt input schema."""
        return {
            "type": "object",
            "properties": {
                "messages": {"type": "array", "description": "Existing messages (optional)"}
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """System prompt output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "messages": {"type": "array"},
                        "systemPrompt": {"type": "string"},
                    },
                },
            },
        }


class FewShotNode(PromptNode):
    """
    Few-Shot Examples Node - adds training examples to prompt.

    Config:
        examples: [{input, output}, ...]
        prefix: Text before examples (default: "Here are some examples:")

    Input:
        messages: Existing messages
        OR
        prompt: User prompt to append examples to

    Output:
        messages: Messages with few-shot examples
    """

    node_type = "few_shot"

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Add few-shot examples."""
        data = input_data.data

        examples = self.config.get("examples", [])
        prefix = self.config.get("prefix", "Here are some examples:")

        if not examples:
            return GenAINodeOutput(success=False, error="examples array is required", data={})

        # Build few-shot text
        few_shot_text = f"{prefix}\n\n"
        for i, example in enumerate(examples, 1):
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            few_shot_text += f"Example {i}:\nInput: {input_text}\nOutput: {output_text}\n\n"

        # Get existing messages
        existing_messages = data.get("messages", [])

        if existing_messages:
            # Append to last user message
            messages = existing_messages.copy()

            # Find last user message
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                messages[last_user_idx]["content"] = (
                    few_shot_text + messages[last_user_idx]["content"]
                )
            else:
                # No user message, create one
                messages.append({"role": "user", "content": few_shot_text})
        else:
            # Create new message
            prompt = data.get("prompt", "")
            content = few_shot_text + prompt
            messages = [{"role": "user", "content": content}]

        return GenAINodeOutput(
            success=True, data={"messages": messages, "fewShotExamples": examples}
        )

    def get_input_schema(self) -> Dict[str, Any]:
        """Few-shot input schema."""
        return {
            "type": "object",
            "properties": {"messages": {"type": "array"}, "prompt": {"type": "string"}},
        }


class PromptTemplateNode(PromptNode):
    """
    Prompt Template Node - fills variables in template.

    Config:
        template: "Hello {{name}}, your age is {{age}}"
        variables: {name: "John", age: 25}
        allowMissingVars: If False, error on missing vars
        defaultValues: {var: default_value}

    Input:
        variables: Runtime variable overrides
        messages: Existing messages (will append rendered template)

    Output:
        messages: Messages with rendered template
        renderedPrompt: Final rendered text
    """

    node_type = "prompt_template"

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Render prompt template."""
        data = input_data.data

        template = self.config.get("template", "")
        config_vars = self.config.get("variables", {})
        allow_missing = self.config.get("allowMissingVars", False)
        defaults = self.config.get("defaultValues", {})

        if not template:
            return GenAINodeOutput(success=False, error="template is required", data={})

        # Merge variables: config < defaults < runtime
        variables = {**config_vars, **defaults, **data.get("variables", {})}

        # Find all variables in template
        var_pattern = re.compile(r"\{\{(\w+)\}\}")
        required_vars = set(var_pattern.findall(template))

        # Check for missing variables
        missing_vars = required_vars - set(variables.keys())
        if missing_vars and not allow_missing:
            return GenAINodeOutput(
                success=False,
                error=f"Missing variables: {', '.join(missing_vars)}",
                data={"missingVariables": list(missing_vars)},
            )

        # Render template
        rendered = template
        for var, value in variables.items():
            rendered = rendered.replace(f"{{{{{var}}}}}", str(value))

        # Handle still-missing vars
        if allow_missing:
            rendered = var_pattern.sub("", rendered)  # Remove unfilled vars

        # Build messages
        existing_messages = data.get("messages", [])
        messages = existing_messages.copy()
        messages.append({"role": "user", "content": rendered})

        return GenAINodeOutput(
            success=True,
            data={"messages": messages, "renderedPrompt": rendered, "variables": variables},
        )

    def get_input_schema(self) -> Dict[str, Any]:
        """Template input schema."""
        return {
            "type": "object",
            "properties": {
                "variables": {"type": "object", "description": "Runtime variable values"},
                "messages": {"type": "array", "description": "Existing messages"},
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Template output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "messages": {"type": "array"},
                        "renderedPrompt": {"type": "string"},
                        "variables": {"type": "object"},
                    },
                },
            },
        }
