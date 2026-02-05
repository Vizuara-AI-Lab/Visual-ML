/**
 * Validation rules for Feature Engineering nodes
 * - encoding
 * - transformation
 * - scaling
 * - feature_selection
 */

import type { ValidationRegistry } from "./types";

const commonFeatureEngSources = [
  // Data sources
  "upload_file",
  "select_dataset",
  "sample_dataset",
  // View nodes
  "table_view",
  "data_preview",
  "statistics_view",
  "column_info",
  "chart_view",
  // Preprocessing
  "missing_value_handler",
  // Other feature engineering (chaining)
  "encoding",
  "transformation",
  "scaling",
  "feature_selection",
];

const commonFeatureEngTargets = [
  // View nodes
  "table_view",
  "data_preview",
  "statistics_view",
  "column_info",
  "chart_view",
  // Other feature engineering (chaining)
  "encoding",
  "transformation",
  "scaling",
  "feature_selection",
  // Target & Split (REQUIRED before ML)
  "split",
];

export const featureEngValidationRules: ValidationRegistry = {
  encoding: {
    allowedSources: commonFeatureEngSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: commonFeatureEngTargets,
  },

  transformation: {
    allowedSources: commonFeatureEngSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: commonFeatureEngTargets,
  },

  scaling: {
    allowedSources: commonFeatureEngSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: commonFeatureEngTargets,
  },

  feature_selection: {
    allowedSources: commonFeatureEngSources,
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: commonFeatureEngTargets,
  },
};
