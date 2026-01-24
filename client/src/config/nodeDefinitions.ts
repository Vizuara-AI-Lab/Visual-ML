/**
 * Complete node definitions based on backend ML and GenAI capabilities
 */

import type { NodeType } from "../types/pipeline";
import type { LucideIcon } from "lucide-react";
import {
  Link,
  Database,
  Split,
  Filter,
  Scale,
  Droplet,
  Brain,
  Network,
  Sparkles,
  Bot,
  MessageSquare,
  Image,
  FileText,
  Layers,
  Zap,
  Cloud,
  Upload,
} from "lucide-react";

export interface NodeCategory {
  id: string;
  label: string;
  icon: LucideIcon;
  nodes: NodeDefinition[];
}

export interface NodeDefinition {
  type: NodeType;
  label: string;
  description: string;
  category: string;
  icon: LucideIcon;
  color: string;
  defaultConfig: Record<string, unknown>;
  configFields?: ConfigField[];
}

export interface ConfigField {
  name: string;
  label: string;
  type:
    | "text"
    | "number"
    | "select"
    | "multiselect"
    | "checkbox"
    | "file"
    | "textarea";
  required?: boolean;
  options?: Array<{ value: string; label: string }>;
  defaultValue?: unknown;
  description?: string;
  min?: number;
  max?: number;
  step?: number;
  autoFill?: boolean; // Auto-fill from dataset metadata
}

export const nodeCategories: NodeCategory[] = [
  {
    id: "data-sources",
    label: "Data Sources",
    icon: Database,
    nodes: [
      {
        type: "upload_file",
        label: "Upload Dataset",
        description: "Upload your own CSV file",
        category: "data-sources",
        icon: Upload,
        color: "#3B82F6",
        defaultConfig: {
          dataset_id: "",
          filename: "",
          n_rows: 0,
          n_columns: 0,
          columns: [],
        },
        configFields: [
          {
            name: "filename",
            label: "Dataset File",
            type: "file",
            required: true,
            description: "Upload a CSV file",
          },
          {
            name: "dataset_id",
            label: "Dataset ID",
            type: "text",
            required: false,
            description: "Auto-filled after upload",
            autoFill: true,
          },
        ],
      },
      {
        type: "load_url",
        label: "Load from URL",
        description: "Load dataset from a URL",
        category: "data-sources",
        icon: Link,
        color: "#10B981",
        defaultConfig: {
          url: "",
          format: "csv",
        },
        configFields: [
          {
            name: "url",
            label: "Dataset URL",
            type: "text",
            required: true,
            description: "HTTP(S) URL to your dataset",
          },
          {
            name: "format",
            label: "Format",
            type: "select",
            options: [
              { value: "csv", label: "CSV" },
              { value: "json", label: "JSON" },
              { value: "excel", label: "Excel" },
            ],
            defaultValue: "csv",
          },
        ],
      },
      {
        type: "sample_dataset",
        label: "Sample Dataset",
        description: "Use a built-in sample dataset",
        category: "data-sources",
        icon: Database,
        color: "#8B5CF6",
        defaultConfig: {
          dataset_name: "iris",
        },
        configFields: [
          {
            name: "dataset_name",
            label: "Dataset",
            type: "select",
            options: [
              { value: "iris", label: "Iris (Classification)" },
              { value: "boston", label: "Boston Housing (Regression)" },
              { value: "wine", label: "Wine Quality (Classification)" },
              { value: "diabetes", label: "Diabetes (Regression)" },
            ],
            defaultValue: "iris",
          },
        ],
      },
    ],
  },
  {
    id: "feature-engineering",
    label: "Feature Engineering",
    icon: Layers,
    nodes: [
      {
        type: "split",
        label: "Train/Test Split",
        description: "Split dataset into training and testing sets",
        category: "feature-engineering",
        icon: Split,
        color: "#F59E0B",
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
        configFields: [
          {
            name: "dataset_path",
            label: "Dataset Path",
            type: "text",
            required: true,
            autoFill: true,
          },
          {
            name: "target_column",
            label: "Target Column",
            type: "select",
            required: true,
            autoFill: true,
            description: "Column to predict",
          },
          {
            name: "train_ratio",
            label: "Training Ratio",
            type: "number",
            min: 0.1,
            max: 0.9,
            step: 0.05,
            defaultValue: 0.7,
          },
          {
            name: "val_ratio",
            label: "Validation Ratio",
            type: "number",
            min: 0,
            max: 0.5,
            step: 0.05,
            defaultValue: 0.15,
          },
          {
            name: "test_ratio",
            label: "Test Ratio",
            type: "number",
            min: 0.1,
            max: 0.5,
            step: 0.05,
            defaultValue: 0.15,
          },
          {
            name: "shuffle",
            label: "Shuffle Data",
            type: "checkbox",
            defaultValue: true,
          },
          {
            name: "stratify",
            label: "Stratified Split",
            type: "checkbox",
            defaultValue: false,
            description: "Maintain class distribution (for classification)",
          },
        ],
      },
      {
        type: "preprocess",
        label: "Missing Value Handling",
        description: "Handle missing values and clean data",
        category: "feature-engineering",
        icon: Droplet,
        color: "#EC4899",
        defaultConfig: {
          dataset_path: "",
          target_column: "",
          handle_missing: true,
          missing_strategy: "drop",
          scale_features: false,
        },
        configFields: [
          {
            name: "dataset_path",
            label: "Dataset Path",
            type: "text",
            required: true,
            autoFill: true,
          },
          {
            name: "target_column",
            label: "Target Column",
            type: "select",
            required: true,
            autoFill: true,
          },
          {
            name: "missing_strategy",
            label: "Missing Value Strategy",
            type: "select",
            options: [
              { value: "drop", label: "Drop Rows" },
              { value: "mean", label: "Fill with Mean" },
              { value: "median", label: "Fill with Median" },
              { value: "mode", label: "Fill with Mode" },
              { value: "fill", label: "Fill with Value" },
            ],
            defaultValue: "drop",
          },
          {
            name: "scale_features",
            label: "Scale Features",
            type: "checkbox",
            defaultValue: false,
            description: "Apply StandardScaler to numeric features",
          },
        ],
      },
      {
        type: "feature_selection",
        label: "Feature Selection",
        description: "Select most important features",
        category: "feature-engineering",
        icon: Filter,
        color: "#06B6D4",
        defaultConfig: {
          method: "variance",
          n_features: 10,
        },
        configFields: [
          {
            name: "method",
            label: "Selection Method",
            type: "select",
            options: [
              { value: "variance", label: "Variance Threshold" },
              { value: "correlation", label: "Correlation" },
              { value: "mutual_info", label: "Mutual Information" },
            ],
            defaultValue: "variance",
          },
          {
            name: "n_features",
            label: "Number of Features",
            type: "number",
            min: 1,
            defaultValue: 10,
          },
        ],
      },
      {
        type: "scaling",
        label: "Scaling / Normalization",
        description: "Scale and normalize features",
        category: "feature-engineering",
        icon: Scale,
        color: "#14B8A6",
        defaultConfig: {
          method: "standard",
          columns: [],
        },
        configFields: [
          {
            name: "method",
            label: "Scaling Method",
            type: "select",
            options: [
              { value: "standard", label: "Standard Scaler (Z-score)" },
              { value: "minmax", label: "Min-Max Scaler" },
              { value: "robust", label: "Robust Scaler" },
              { value: "normalize", label: "Normalizer" },
            ],
            defaultValue: "standard",
          },
        ],
      },
    ],
  },
  {
    id: "ml-algorithms",
    label: "ML Algorithms",
    icon: Brain,
    nodes: [
      {
        type: "train",
        label: "Linear Regression",
        description: "Train linear regression model",
        category: "ml-algorithms",
        icon: Brain,
        color: "#EF4444",
        defaultConfig: {
          train_dataset_path: "",
          target_column: "",
          algorithm: "linear_regression",
          task_type: "regression",
          hyperparameters: {
            fit_intercept: true,
            copy_X: true,
          },
        },
      },
      {
        type: "evaluate",
        label: "Model Evaluation",
        description: "Evaluate model performance",
        category: "ml-algorithms",
        icon: Network,
        color: "#A855F7",
        defaultConfig: {
          model_path: "",
          test_dataset_path: "",
          target_column: "",
          task_type: "regression",
        },
      },
    ],
  },
  {
    id: "genai",
    label: "GenAI",
    icon: Sparkles,
    nodes: [
      {
        type: "llm_node",
        label: "LLM Chat",
        description: "Chat with Language Models (GPT, Claude, etc.)",
        category: "genai",
        icon: Bot,
        color: "#8B5CF6",
        defaultConfig: {
          provider: "openai",
          model: "gpt-4",
          temperature: 0.7,
          max_tokens: 1000,
        },
        configFields: [
          {
            name: "provider",
            label: "Provider",
            type: "select",
            options: [
              { value: "openai", label: "OpenAI" },
              { value: "anthropic", label: "Anthropic" },
              { value: "huggingface", label: "HuggingFace" },
            ],
            defaultValue: "openai",
          },
          {
            name: "model",
            label: "Model",
            type: "text",
            defaultValue: "gpt-4",
          },
          {
            name: "temperature",
            label: "Temperature",
            type: "number",
            min: 0,
            max: 2,
            step: 0.1,
            defaultValue: 0.7,
          },
          {
            name: "prompt",
            label: "Prompt",
            type: "textarea",
            required: true,
          },
        ],
      },
      {
        type: "prompt_template",
        label: "Prompt Template",
        description: "Create reusable prompt templates",
        category: "genai",
        icon: MessageSquare,
        color: "#6366F1",
        defaultConfig: {
          template: "",
          variables: [],
        },
      },
      {
        type: "rag_node",
        label: "RAG (Retrieval)",
        description: "Retrieval-Augmented Generation",
        category: "genai",
        icon: FileText,
        color: "#EC4899",
        defaultConfig: {
          vector_store: "faiss",
          top_k: 5,
        },
      },
      {
        type: "image_generation",
        label: "Image Generation",
        description: "Generate images from text",
        category: "genai",
        icon: Image,
        color: "#F59E0B",
        defaultConfig: {
          provider: "dall-e",
          size: "1024x1024",
        },
      },
    ],
  },
  {
    id: "deployment",
    label: "Deployment",
    icon: Cloud,
    nodes: [
      {
        type: "model_export",
        label: "Export Model",
        description: "Export trained model for deployment",
        category: "deployment",
        icon: Zap,
        color: "#10B981",
        defaultConfig: {
          format: "pickle",
          model_path: "",
        },
      },
      {
        type: "api_endpoint",
        label: "Create API",
        description: "Deploy model as REST API",
        category: "deployment",
        icon: Cloud,
        color: "#06B6D4",
        defaultConfig: {
          endpoint_name: "",
          model_path: "",
        },
      },
    ],
  },
];

export const getAllNodes = (): NodeDefinition[] => {
  return nodeCategories.flatMap((category) => category.nodes);
};

export const getNodeByType = (type: NodeType): NodeDefinition | undefined => {
  return getAllNodes().find((node) => node.type === type);
};

export const getNodesByCategory = (categoryId: string): NodeDefinition[] => {
  const category = nodeCategories.find((cat) => cat.id === categoryId);
  return category?.nodes || [];
};
