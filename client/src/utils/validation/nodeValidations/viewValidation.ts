/**
 * Validation rules for View nodes
 * - table_view
 * - data_preview
 * - statistics_view
 * - column_info
 * - chart_view
 */

import type { ValidationRegistry } from "./types";

export const viewValidationRules: ValidationRegistry = {
  table_view: {
    allowedSources: [
      // Data sources
      "upload_file",
      "select_dataset",
      "sample_dataset",
      // Preprocessing
      "missing_value_handler",
      // Feature engineering
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      // Target & Split (to view split data)
      "split",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    // View nodes can connect to other processing nodes (user requested this)
    allowedTargets: [
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
  },

  data_preview: {
    allowedSources: [
      "upload_file",
      "select_dataset",
      "sample_dataset",
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
  },

  statistics_view: {
    allowedSources: [
      "upload_file",
      "select_dataset",
      "sample_dataset",
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
  },

  column_info: {
    allowedSources: [
      "upload_file",
      "select_dataset",
      "sample_dataset",
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
  },

  chart_view: {
    allowedSources: [
      "upload_file",
      "select_dataset",
      "sample_dataset",
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
    requiresInput: true,
    maxInputConnections: 1,
    allowedTargets: [
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
    ],
  },
};
