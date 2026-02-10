/**
 * Validation rules for Result & Metric nodes
 * - r2_score, mse_score, rmse_score, mae_score (regression)
 */

import type { ValidationRegistry } from "./types";

const regressionModelSources = [
  "linear_regression",
  "decision_tree", // when task_type = regression
  "random_forest", // when task_type = regression
  "evaluate",
];

export const resultValidationRules: ValidationRegistry = {
  // Regression metrics
  r2_score: {
    allowedSources: regressionModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [], // Terminal node
  },

  mse_score: {
    allowedSources: regressionModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },

  rmse_score: {
    allowedSources: regressionModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },

  mae_score: {
    allowedSources: regressionModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },
};
