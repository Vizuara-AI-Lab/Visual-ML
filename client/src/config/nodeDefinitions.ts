/**
 * Complete node definitions based on backend ML and GenAI capabilities
 */

import type { NodeType } from "../types/pipeline";
import type { LucideIcon } from "lucide-react";
import {
  Database,
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
  Eye,
  Table,
  BarChart3,
  LineChart,
  Info,
  Hash,
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
  conditionalDisplay?: {
    field: string;
    equals?: string;
    notEquals?: string;
  }; // Show field conditionally based on another field's value
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
        type: "select_dataset",
        label: "Select Dataset",
        description: "Choose from previously uploaded datasets",
        category: "data-sources",
        icon: Database,
        color: "#10B981",
        defaultConfig: {
          dataset_id: "",
          filename: "",
          n_rows: 0,
          n_columns: 0,
          columns: [],
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset",
            type: "select",
            required: true,
            description: "Select from your uploaded datasets",
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
    id: "view",
    label: "View",
    icon: Eye,
    nodes: [
      {
        type: "table_view",
        label: "Table View",
        description: "Display dataset in table format",
        category: "view",
        icon: Table,
        color: "#06B6D4",
        defaultConfig: {
          dataset_id: "",
          max_rows: 100,
          columns_to_show: [],
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            description: "Connect to a data source node",
            autoFill: true,
          },
          {
            name: "max_rows",
            label: "Max Rows to Display",
            type: "number",
            min: 10,
            max: 1000,
            defaultValue: 100,
            description: "Number of rows to show",
          },
        ],
      },
      {
        type: "data_preview",
        label: "Data Preview",
        description: "Quick preview of first and last rows",
        category: "view",
        icon: Eye,
        color: "#8B5CF6",
        defaultConfig: {
          dataset_id: "",
          head_rows: 5,
          tail_rows: 5,
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            description: "Connect to a data source node",
            autoFill: true,
          },
          {
            name: "head_rows",
            label: "First N Rows",
            type: "number",
            min: 1,
            max: 50,
            defaultValue: 5,
          },
          {
            name: "tail_rows",
            label: "Last N Rows",
            type: "number",
            min: 1,
            max: 50,
            defaultValue: 5,
          },
        ],
      },
      {
        type: "statistics_view",
        label: "Statistics View",
        description: "Show statistical summary of data",
        category: "view",
        icon: BarChart3,
        color: "#10B981",
        defaultConfig: {
          dataset_id: "",
          include_all: true,
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            description: "Connect to a data source node",
            autoFill: true,
          },
          {
            name: "include_all",
            label: "Include All Columns",
            type: "checkbox",
            defaultValue: true,
            description: "Show stats for all columns",
          },
        ],
      },
      {
        type: "column_info",
        label: "Column Info",
        description: "Display column types and missing values",
        category: "view",
        icon: Info,
        color: "#F59E0B",
        defaultConfig: {
          dataset_id: "",
          show_dtypes: true,
          show_missing: true,
          show_unique: true,
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            description: "Connect to a data source node",
            autoFill: true,
          },
          {
            name: "show_dtypes",
            label: "Show Data Types",
            type: "checkbox",
            defaultValue: true,
          },
          {
            name: "show_missing",
            label: "Show Missing Values",
            type: "checkbox",
            defaultValue: true,
          },
          {
            name: "show_unique",
            label: "Show Unique Counts",
            type: "checkbox",
            defaultValue: true,
          },
        ],
      },
      {
        type: "chart_view",
        label: "Chart View",
        description: "Visualize data with charts",
        category: "view",
        icon: LineChart,
        color: "#EC4899",
        defaultConfig: {
          dataset_id: "",
          chart_type: "bar",
          x_column: "",
          y_column: "",
          y_columns: "",
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            description: "Connect to a data source node",
            autoFill: true,
          },
          {
            name: "chart_type",
            label: "Chart Type",
            type: "select",
            options: [
              { value: "bar", label: "Bar Chart" },
              { value: "line", label: "Line Chart" },
              { value: "scatter", label: "Scatter Plot" },
              { value: "histogram", label: "Histogram" },
              { value: "pie", label: "Pie Chart" },
            ],
            defaultValue: "bar",
          },
          {
            name: "x_column",
            label: "X-Axis Column",
            type: "select",
            required: false,
            autoFill: true,
            description: "Select column for X-axis",
            conditionalDisplay: { field: "chart_type", notEquals: "pie" },
          },
          {
            name: "y_columns",
            label: "Y-Axis Column",
            type: "select",
            required: false,
            autoFill: true,
            description: "Select column for Y-axis",
            conditionalDisplay: { field: "chart_type", notEquals: "pie" },
          },
          {
            name: "label_column",
            label: "Label Column",
            type: "select",
            required: false,
            autoFill: true,
            description: "Column for pie chart labels",
            conditionalDisplay: { field: "chart_type", equals: "pie" },
          },
          {
            name: "value_column",
            label: "Value Column",
            type: "select",
            required: false,
            autoFill: true,
            description: "Column for pie chart values",
            conditionalDisplay: { field: "chart_type", equals: "pie" },
          },
        ],
      },
    ],
  },
  {
    id: "preprocessing",
    label: "Preprocess Data",
    icon: Droplet,
    nodes: [
      {
        type: "missing_value_handler",
        label: "Missing Value Handler",
        description: "Handle missing values with column-wise control",
        category: "preprocessing",
        icon: Droplet,
        color: "#A855F7",
        defaultConfig: {
          dataset_id: "",
          column_configs: {},
          default_strategy: "none",
          preview_rows: 10,
          preview_mode: false,
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            autoFill: true,
            description: "Connect to a data source node",
          },
          {
            name: "default_strategy",
            label: "Default Strategy",
            type: "select",
            options: [
              { value: "none", label: "No Action" },
              { value: "drop", label: "Drop Rows" },
              { value: "mean", label: "Fill with Mean" },
              { value: "median", label: "Fill with Median" },
              { value: "mode", label: "Fill with Mode" },
              { value: "fill", label: "Fill with Value" },
              { value: "forward_fill", label: "Forward Fill" },
              { value: "backward_fill", label: "Backward Fill" },
            ],
            defaultValue: "none",
            description: "Default strategy for all columns",
          },
          {
            name: "preview_rows",
            label: "Preview Rows",
            type: "number",
            defaultValue: 10,
            min: 5,
            max: 100,
            description: "Number of rows to show in preview",
          },
          {
            name: "preview_mode",
            label: "Preview Mode",
            type: "checkbox",
            defaultValue: false,
            description: "Preview changes before applying",
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
        type: "encoding",
        label: "Encoding",
        description: "Encode categorical variables",
        category: "feature-engineering",
        icon: Hash,
        color: "#F59E0B",
        defaultConfig: {
          dataset_id: "",
          encoding_method: "onehot",
          columns: [],
          drop_first: false,
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            autoFill: true,
            description: "Connect to a data source node",
          },
          {
            name: "encoding_method",
            label: "Encoding Method",
            type: "select",
            options: [
              { value: "onehot", label: "One-Hot Encoding" },
              { value: "label", label: "Label Encoding" },
              { value: "ordinal", label: "Ordinal Encoding" },
              { value: "target", label: "Target Encoding" },
            ],
            defaultValue: "onehot",
          },
          {
            name: "drop_first",
            label: "Drop First Category",
            type: "checkbox",
            defaultValue: false,
            description: "Drop first category in one-hot encoding",
          },
        ],
      },
      {
        type: "transformation",
        label: "Transformation",
        description: "Apply mathematical transformations",
        category: "feature-engineering",
        icon: Zap,
        color: "#8B5CF6",
        defaultConfig: {
          dataset_id: "",
          transformation_type: "log",
          columns: [],
          degree: 2,
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            autoFill: true,
            description: "Connect to a data source node",
          },
          {
            name: "transformation_type",
            label: "Transformation Type",
            type: "select",
            options: [
              { value: "log", label: "Log Transform" },
              { value: "sqrt", label: "Square Root" },
              { value: "power", label: "Power Transform (Box-Cox)" },
              { value: "polynomial", label: "Polynomial Features" },
            ],
            defaultValue: "log",
          },
          {
            name: "degree",
            label: "Polynomial Degree",
            type: "number",
            defaultValue: 2,
            min: 2,
            max: 5,
            description: "Degree for polynomial features",
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
          dataset_id: "",
          method: "standard",
          columns: [],
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            autoFill: true,
            description: "Connect to a data source node",
          },
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
      {
        type: "feature_selection",
        label: "Feature Selection",
        description: "Select most important features",
        category: "feature-engineering",
        icon: Filter,
        color: "#06B6D4",
        defaultConfig: {
          dataset_id: "",
          method: "variance",
          n_features: 10,
        },
        configFields: [
          {
            name: "dataset_id",
            label: "Dataset Source",
            type: "text",
            required: true,
            autoFill: true,
            description: "Connect to a data source node",
          },
          {
            name: "method",
            label: "Selection Method",
            type: "select",
            options: [
              { value: "variance", label: "Variance Threshold" },
              { value: "correlation", label: "Correlation" },
              { value: "mutual_info", label: "Mutual Information" },
              { value: "kbest", label: "SelectKBest" },
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
