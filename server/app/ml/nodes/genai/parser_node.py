"""
Output Parser Node - parse and validate LLM outputs.
"""

from typing import Dict, Any, Optional
import json
import re
from pydantic import BaseModel, ValidationError

from app.ml.nodes.genai.base import PromptNode, GenAINodeInput, GenAINodeOutput


class OutputParserNode(PromptNode):
    """
    Output Parser Node - extract structured data from LLM responses.

    Config:
        parserType: "json", "regex", "schema"
        schema: JSON schema for validation (for "schema" type)
        regexPattern: Regex pattern (for "regex" type)
        retryOnError: Retry with fix prompt on parse error
        maxRetries: Max retry attempts (default: 2)

    Input:
        response: LLM response text to parse
        OR
        messages: Messages with last assistant response

    Output:
        parsedData: Extracted/validated data
        parseErrors: Any parsing errors
        retryCount: Number of retries attempted
    """

    node_type = "output_parser"

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Parse LLM output."""
        data = input_data.data

        parser_type = self.config.get("parserType", "json")

        # Get response text
        response_text = self._extract_response(data)
        if not response_text:
            return GenAINodeOutput(success=False, error="No response text found", data={})

        # Parse based on type
        try:
            if parser_type == "json":
                parsed = self._parse_json(response_text)
            elif parser_type == "regex":
                parsed = self._parse_regex(response_text)
            elif parser_type == "schema":
                parsed = self._parse_schema(response_text)
            else:
                return GenAINodeOutput(
                    success=False, error=f"Unknown parserType: {parser_type}", data={}
                )
        except Exception as e:
            # Handle retry logic
            retry_on_error = self.config.get("retryOnError", True)
            if retry_on_error:
                return GenAINodeOutput(
                    success=False,
                    error=f"Parse error: {str(e)}",
                    data={
                        "parseErrors": [str(e)],
                        "needsRetry": True,
                        "fixPrompt": self._generate_fix_prompt(response_text, str(e)),
                    },
                )
            else:
                return GenAINodeOutput(
                    success=False, error=f"Parse error: {str(e)}", data={"parseErrors": [str(e)]}
                )

        return GenAINodeOutput(
            success=True,
            data={"parsedData": parsed, "parseErrors": [], "originalResponse": response_text},
        )

    def _extract_response(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract response text from input."""
        # Option 1: Direct response field
        if "response" in data:
            return data["response"]

        # Option 2: Last assistant message
        if "messages" in data:
            messages = data["messages"]
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")

        return None

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from response."""
        # Try to extract JSON from markdown code blocks
        json_pattern = r"```json\s*\n(.*?)\n```"
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            json_text = match.group(1)
        else:
            # Try to find JSON object
            json_obj_pattern = r"\{.*\}"
            match = re.search(json_obj_pattern, text, re.DOTALL)
            if match:
                json_text = match.group(0)
            else:
                json_text = text

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")

    def _parse_regex(self, text: str) -> Dict[str, Any]:
        """Parse using regex pattern."""
        pattern = self.config.get("regexPattern")
        if not pattern:
            raise ValueError("regexPattern is required for regex parser")

        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError(f"No match found for pattern: {pattern}")

        # Return groups as dict
        return match.groupdict() if match.groupdict() else {"match": match.group(0)}

    def _parse_schema(self, text: str) -> Dict[str, Any]:
        """Parse and validate against JSON schema."""
        schema = self.config.get("schema")
        if not schema:
            raise ValueError("schema is required for schema parser")

        # First parse JSON
        parsed = self._parse_json(text)

        # Then validate against schema
        # In production, use jsonschema library
        # For now, basic validation
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")

        return parsed

    def _generate_fix_prompt(self, response: str, error: str) -> str:
        """Generate prompt to fix parsing error."""
        return (
            f"The previous response had a parsing error:\n"
            f"Error: {error}\n\n"
            f"Previous response:\n{response}\n\n"
            f"Please provide the response in the correct format."
        )

    def get_input_schema(self) -> Dict[str, Any]:
        """Parser input schema."""
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string", "description": "Response text to parse"},
                "messages": {"type": "array", "description": "Messages with assistant response"},
            },
            "oneOf": [{"required": ["response"]}, {"required": ["messages"]}],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """Parser output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "parsedData": {"type": "object"},
                        "parseErrors": {"type": "array"},
                        "originalResponse": {"type": "string"},
                        "needsRetry": {"type": "boolean"},
                        "fixPrompt": {"type": "string"},
                    },
                },
            },
        }
