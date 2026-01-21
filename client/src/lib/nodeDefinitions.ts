/**
 * Node definitions and metadata for the ML Pipeline builder
 */

import type { NodeMetadata } from "../types/pipeline";

export const nodeDefinitions: NodeMetadata[] = [
  {
    type: "upload_file",
    label: "Upload Data",
    description: "Upload and validate a CSV dataset",
    color: "#3B82F6", // blue
    icon: "📤",
    category: "input",
    defaultConfig: {
      filename: "",
      content_type: "text/csv",
    },
  },
  {
    type: "preprocess",
    label: "Preprocess Data",
    description: "Clean data and handle missing values",
    color: "#8B5CF6", // purple
    icon: "🧹",
    category: "preprocessing",
    defaultConfig: {
      dataset_path: "",
      target_column: "",
      handle_missing: true,
      missing_strategy: "drop",
      text_columns: [],
      numeric_columns: [],
      lowercase: true,
      remove_stopwords: false,
      max_features: 1000,
      scale_features: false,
    },
  },
  {
    type: "split",
    label: "Split",
    description: "Split dataset into train/val/test sets",
    color: "#10B981", // green
    icon: "✂️",
    category: "preprocessing",
    defaultConfig: {
      dataset_path: "",
      target_column: "",
      train_ratio: 0.7,
      val_ratio: 0.15,
      test_ratio: 0.15,
      random_seed: 42,
      shuffle: true,
      stratify: false,
    },
  },
  {
    type: "train",
    label: "Train",
    description: "Train a machine learning model",
    color: "#F59E0B", // amber
    icon: "🎓",
    category: "model",
    defaultConfig: {
      train_dataset_path: "",
      target_column: "",
      algorithm: "linear_regression",
      task_type: "regression",
      hyperparameters: {
        fit_intercept: true,
        copy_X: true,
        n_jobs: null,
      },
      model_name: "",
    },
  },
  {
    type: "evaluate",
    label: "View Result",
    description: "Evaluate model performance",
    color: "#EF4444", // red
    icon: "📊",
    category: "output",
    defaultConfig: {
      model_path: "",
      test_dataset_path: "",
      target_column: "",
      task_type: "regression",
    },
  },
];

export const getNodeDefinition = (type: string): NodeMetadata | undefined => {
  return nodeDefinitions.find((node) => node.type === type);
};

export const getNodesByCategory = (category: string): NodeMetadata[] => {
  return nodeDefinitions.filter((node) => node.category === category);
};
