/**
 * Validation rules for Preprocessing nodes
 * - missing_value_handler
 */

import type { ValidationRegistry } from "./types";

export const preprocessingValidationRules: ValidationRegistry = {
  missing_value_handler: {
    allowedSources: [
      // Data sources
      "upload_file",
      "select_dataset",
      "sample_dataset",
      // View nodes (user can inspect then preprocess)
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
      // Other preprocessing (chaining)
      "missing_value_handler",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [
      // View nodes
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
      // Feature engineering
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      // Target & Split
      "split",
      // Chain preprocessing
      "missing_value_handler",
    ],
  },
};
