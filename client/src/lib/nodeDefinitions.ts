/**
 * Node definitions and metadata for the ML Pipeline builder
 */

import type { NodeMetadata } from "../types/pipeline";

export const nodeDefinitions: NodeMetadata[] = [
  {
    type: "missing_value_handler",
    label: "Missing Value Handler",
    description: "Handle missing values with column-wise control",
    color: "#EC4899", // pink
    icon: "🔧",
    category: "preprocessing",
    defaultConfig: {
      dataset_id: "",
      column_configs: {},
      default_strategy: "none",
      preview_mode: false,
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
];

export const getNodeDefinition = (type: string): NodeMetadata | undefined => {
  return nodeDefinitions.find((node) => node.type === type);
};

export const getNodesByCategory = (category: string): NodeMetadata[] => {
  return nodeDefinitions.filter((node) => node.category === category);
};
