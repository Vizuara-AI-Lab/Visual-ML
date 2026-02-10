/**
 * Validation rules for Deployment nodes
 * - model_export
 * - api_endpoint
 */

import type { ValidationRegistry } from "./types";

export const deploymentValidationRules: ValidationRegistry = {
  model_export: {
    allowedSources: [
      "linear_regression",
      "logistic_regression",
      "decision_tree",
      "random_forest",
      "evaluate",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [], // Terminal node
  },

  api_endpoint: {
    allowedSources: [
      "linear_regression",
      "logistic_regression",
      "decision_tree",
      "random_forest",
      "evaluate",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [], // Terminal node
  },
};
