/**
 * Validation rules for Result & Metric nodes
 * - r2_score, mse_score, rmse_score, mae_score (regression)
 * - confusion_matrix (classification)
 */

import type { ValidationRegistry } from "./types";

const regressionModelSources = [
  "linear_regression",
  "decision_tree", // when task_type = regression
  "random_forest", // when task_type = regression
  "evaluate",
];

const classificationModelSources = [
  "logistic_regression",
  "decision_tree", // when task_type = classification
  "random_forest", // when task_type = classification
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

  // Classification metrics
  confusion_matrix: {
    allowedSources: classificationModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },
};
