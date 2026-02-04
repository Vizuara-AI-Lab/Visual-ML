"""
Example Node implementation.
Adds few-shot examples to prompts for better LLM performance.
"""

from typing import Dict, Any, List

from app.ml.nodes.genai.base import PromptNode, GenAINodeInput, GenAINodeOutput


class ExampleNode(PromptNode):
    """
    Example Node - adds few-shot examples to prompts.

    Config:
        examples: Array of {input, output} example objects
        exampleFormat: Template for formatting each example
            Default: "Input: {input}\nOutput: {output}"
        prefix: Text before examples (default: "Here are some examples:")
        includeInstructions: Add instruction text (default: True)

    Input:
        messages: Existing messages (optional)
        userPrompt: User's prompt to append after examples

    Output:
        messages: Messages with examples injected
        formattedExamples: Rendered example text
        exampleCount: Number of examples added
    """

    node_type = "example"

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Add few-shot examples to messages."""
        data = input_data.data

        examples = self.config.get("examples", [])
        if not examples:
            return GenAINodeOutput(
                success=False, error="At least one example is required in config", data={}
            )

        example_format = self.config.get(
            "exampleFormat", "Input: {input}\nOutput: {output}"
        )
        prefix = self.config.get("prefix", "Here are some examples:")
        include_instructions = self.config.get("includeInstructions", True)

        # Build formatted examples
        formatted_parts = []
        if include_instructions:
            formatted_parts.append(prefix)
            formatted_parts.append("")

        for i, example in enumerate(examples, 1):
            example_input = example.get("input", "")
            example_output = example.get("output", "")

            if not example_input or not example_output:
                continue

            # Format the example
            formatted_example = example_format.format(
                input=example_input, output=example_output, number=i
            )
            formatted_parts.append(f"Example {i}:")
            formatted_parts.append(formatted_example)
            formatted_parts.append("")

        formatted_examples = "\n".join(formatted_parts)

        # Get existing messages or create new ones
        existing_messages = data.get("messages", [])
        user_prompt = data.get("userPrompt", "")

        messages = existing_messages.copy()

        # Add examples as a user message or append to existing user message
        if user_prompt:
            # Combine examples with user prompt
            combined_content = f"{formatted_examples}\n\nNow, please help with this:\n{user_prompt}"
            messages.append({"role": "user", "content": combined_content})
        elif messages:
            # Append to last user message if exists
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                messages[last_user_idx]["content"] = (
                    formatted_examples + "\n\n" + messages[last_user_idx]["content"]
                )
            else:
                # No user message found, add as new message
                messages.append({"role": "user", "content": formatted_examples})
        else:
            # No messages at all, create one with examples
            messages.append({"role": "user", "content": formatted_examples})

        return GenAINodeOutput(
            success=True,
            data={
                "messages": messages,
                "formattedExamples": formatted_examples,
                "exampleCount": len(examples),
            },
        )

    def get_input_schema(self) -> Dict[str, Any]:
        """Example node input schema."""
        return {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Existing messages (optional)",
                },
                "userPrompt": {
                    "type": "string",
                    "description": "User's prompt to append after examples",
                },
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Example node output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "messages": {"type": "array"},
                        "formattedExamples": {"type": "string"},
                        "exampleCount": {"type": "integer"},
                    },
                },
            },
        }
