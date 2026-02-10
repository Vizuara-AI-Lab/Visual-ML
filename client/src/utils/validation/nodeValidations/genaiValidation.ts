/**
 * Validation rules for GenAI nodes
 * - llm_node
 * - system_prompt
 * - chatbot_node
 * - example_node
 * 
 * Note: GenAI flows are separate from ML flows
 */

import type { ValidationRegistry } from "./types";

export const genaiValidationRules: ValidationRegistry = {
  llm_node: {
    allowedSources: [], // No inputs - LLM provider is the start
    requiresInput: false,
    allowedTargets: [
      "system_prompt",
      "chatbot_node",
    ],
  },

  system_prompt: {
    allowedSources: [
      "llm_node",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [
      "chatbot_node",
      "example_node",
    ],
  },

  example_node: {
    allowedSources: [
      "system_prompt",
      "llm_node",
    ],
    requiresInput: false, // Optional
    allowedTargets: [
      "chatbot_node",
    ],
  },

  chatbot_node: {
    allowedSources: [
      "llm_node",
      "system_prompt",
      "example_node",
    ],
    requiresInput: true, // Needs at least LLM provider
    allowedTargets: [], // Terminal node
  },
};
