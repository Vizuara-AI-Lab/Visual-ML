/**
 * Validation rules for Result & Metric nodes
 * - r2_score, mse_score, rmse_score, mae_score (regression)
 * - confusion_matrix, classification_report, accuracy_score, roc_curve (classification)
 * - feature_importance, residual_plot, prediction_table (both)
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

const allModelSources = [
  "linear_regression",
  "logistic_regression",
  "decision_tree",
  "random_forest",
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

  residual_plot: {
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

  classification_report: {
    allowedSources: classificationModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },

  accuracy_score: {
    allowedSources: classificationModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },

  roc_curve: {
    allowedSources: classificationModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },

  // Universal metrics
  feature_importance: {
    allowedSources: [
      "decision_tree",
      "random_forest",
      "evaluate",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },

  prediction_table: {
    allowedSources: allModelSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [],
  },
};
