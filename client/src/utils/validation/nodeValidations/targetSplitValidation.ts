/**
 * Validation rules for Target & Split node
 * - split
 * 
 * This is a CRITICAL node in ML pipelines - it must come before ML algorithms
 */

import type { ValidationRegistry } from "./types";

export const targetSplitValidationRules: ValidationRegistry = {
  split: {
    allowedSources: [
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
      // Feature engineering
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [
      // View nodes (to inspect split data)
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
      // ML Algorithms - THIS IS THE KEY STEP
      "linear_regression",
      "logistic_regression",
      "decision_tree",
      "random_forest",
      "evaluate",
    ],
  },
};
