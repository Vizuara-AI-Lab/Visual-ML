/**
 * GenAI Node Definitions
 */

import type { NodeCategory } from "./nodeDefinitions";
import { Sparkles, Bot, MessageSquare, FileText } from "lucide-react";

export const genaiCategory: NodeCategory = {
  id: "genai",
  label: "GenAI",
  icon: Sparkles,
  nodes: [
    {
      type: "llm_node",
      label: "LLM Provider",
      description: "Configure LLM provider (GPT, Claude, Gemini, Grok)",
      category: "genai",
      icon: Bot,
      color: "#8B5CF6",
      defaultConfig: {
        provider: "gemini",
        model: "gemini-1.5-flash", // Set by backend based on provider
        temperature: 0.7,
        maxTokens: 1000,
        useOwnApiKey: false,
        apiKey: "",
      },
      configFields: [
        {
          name: "provider",
          label: "Provider",
          type: "select",
          options: [
            { value: "gemini", label: "Google Gemini" },
            { value: "openai", label: "OpenAI" },
            { value: "anthropic", label: "Anthropic Claude" },
            { value: "grok", label: "xAI Grok" },
            { value: "huggingface", label: "HuggingFace" },
          ],
          defaultValue: "gemini",
          required: true,
        },
        {
          name: "temperature",
          label: "Temperature",
          type: "number",
          min: 0,
          max: 2,
          step: 0.1,
          defaultValue: 0.7,
          description: "Controls randomness (0=focused, 2=creative)",
        },
        {
          name: "maxTokens",
          label: "Max Tokens",
          type: "number",
          min: 1,
          max: 4000,
          defaultValue: 1000,
        },
        {
          name: "useOwnApiKey",
          label: "Use Your Own API Key",
          type: "checkbox",
          defaultValue: false,
          description: "Use your own API key instead of platform key",
        },
        {
          name: "apiKey",
          label: "API Key",
          type: "password",
          conditionalDisplay: { field: "useOwnApiKey", equals: true },
          description: "Your API key (encrypted and stored securely)",
          placeholder: "Enter your API key...",
        },
      ],
    },
    {
      type: "system_prompt",
      label: "System Prompt",
      description: "Set AI role and behavior with presets",
      category: "genai",
      icon: MessageSquare,
      color: "#6366F1",
      defaultConfig: {
        role: "helpful_assistant",
        customRole: "",
        systemPrompt: "",
      },
      configFields: [
        {
          name: "role",
          label: "Role",
          type: "select",
          options: [
            { value: "helpful_assistant", label: "Helpful Assistant" },
            { value: "domain_expert", label: "Domain Expert" },
            { value: "tutor", label: "Tutor" },
            { value: "code_reviewer", label: "Code Reviewer" },
            { value: "creative_writer", label: "Creative Writer" },
            { value: "data_analyst", label: "Data Analyst" },
            { value: "technical_writer", label: "Technical Writer" },
            { value: "problem_solver", label: "Problem Solver" },
            { value: "research_assistant", label: "Research Assistant" },
            { value: "custom", label: "Custom Role" },
          ],
          defaultValue: "helpful_assistant",
          required: true,
        },
        {
          name: "customRole",
          label: "Custom Role Description",
          type: "text",
          placeholder: "e.g., an expert Python developer",
          conditionalDisplay: { field: "role", equals: "custom" },
          description: "Describe the custom role",
        },
        {
          name: "systemPrompt",
          label: "Instructions",
          type: "textarea",
          placeholder: "Enter specific instructions for the AI...",
          required: true,
          description: "Detailed instructions for the AI's behavior",
        },
      ],
    },
    {
      type: "chatbot_node",
      label: "Chatbot",
      description: "Chat interface - send and receive messages",
      category: "genai",
      icon: MessageSquare,
      color: "#10B981",
      defaultConfig: {
        sessionId: "default",
      },
      configFields: [
        {
          name: "sessionId",
          label: "Session ID",
          type: "text",
          defaultValue: "default",
          description: "Unique identifier for this conversation",
          placeholder: "default",
        },
      ],
    },
    {
      type: "example_node",
      label: "Examples (Few-Shot)",
      description: "Add examples for few-shot learning",
      category: "genai",
      icon: FileText,
      color: "#F59E0B",
      defaultConfig: {
        examples: [{ input: "Example input 1", output: "Example output 1" }],
        prefix: "Here are some examples:",
        includeInstructions: true,
      },
      configFields: [
        {
          name: "examples",
          label: "Examples",
          type: "array",
          description: "Add multiple input/output examples",
          itemFields: [
            {
              name: "input",
              label: "Input",
              type: "textarea",
              placeholder: "Example input...",
            },
            {
              name: "output",
              label: "Output",
              type: "textarea",
              placeholder: "Expected output...",
            },
          ],
        },
        {
          name: "prefix",
          label: "Prefix Text",
          type: "text",
          defaultValue: "Here are some examples:",
          description: "Text shown before examples",
        },
        {
          name: "includeInstructions",
          label: "Include Instructions",
          type: "boolean",
          defaultValue: true,
          description: "Add instructional text with examples",
        },
      ],
    },
  ],
};
