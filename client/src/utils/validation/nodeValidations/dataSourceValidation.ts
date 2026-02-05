/**
 * Validation rules for Data Source nodes
 * - upload_file
 * - select_dataset
 * - sample_dataset
 */

import type { ValidationRegistry } from "./types";

export const dataSourceValidationRules: ValidationRegistry = {
  upload_file: {
    allowedSources: [], // No inputs - data sources are pipeline starts
    requiresInput: false,
    allowedTargets: [
      // View nodes
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
      // Preprocessing
      "missing_value_handler",
      // Feature Engineering
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      // Target & Split
      "split",
    ],
  },

  select_dataset: {
    allowedSources: [],
    requiresInput: false,
    allowedTargets: [
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
  },

  sample_dataset: {
    allowedSources: [],
    requiresInput: false,
    allowedTargets: [
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
  },
};
