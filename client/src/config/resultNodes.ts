import type { NodeDefinition } from "./nodeDefinitions";
import { TrendingUp, BarChart3, Activity, Target } from "lucide-react";

/**
 * Result and Metrics Node Definitions
 * These nodes visualize model performance metrics and evaluation results
 */

export const resultNodes: NodeDefinition[] = [
  {
    type: "r2_score",
    label: "R² Score",
    description:
      "Coefficient of determination - measures how well predictions fit actual values (0-1, higher is better)",
    category: "result",
    icon: BarChart3,
    color: "#10B981",
    defaultConfig: {
      model_output_id: "",
      display_format: "percentage", // "percentage" | "decimal"
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected ML model node",
      },
      {
        name: "display_format",
        label: "Display Format",
        type: "select",
        required: true,
        options: [
          { value: "percentage", label: "Percentage (75.5%)" },
          { value: "decimal", label: "Decimal (0.755)" },
        ],
        description: "How to display the R² score",
      },
    ],
  },
  {
    type: "mse_score",
    label: "MSE",
    description:
      "Mean Squared Error - average squared difference between predictions and actual values (lower is better)",
    category: "result",
    icon: Activity,
    color: "#EF4444",
    defaultConfig: {
      model_output_id: "",
      precision: 4,
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected ML model node",
      },
      {
        name: "precision",
        label: "Decimal Precision",
        type: "number",
        required: true,
        description: "Number of decimal places to display",
      },
    ],
  },
  {
    type: "rmse_score",
    label: "RMSE",
    description:
      "Root Mean Squared Error - square root of MSE, in same units as target variable (lower is better)",
    category: "result",
    icon: TrendingUp,
    color: "#F59E0B",
    defaultConfig: {
      model_output_id: "",
      precision: 4,
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected ML model node",
      },
      {
        name: "precision",
        label: "Decimal Precision",
        type: "number",
        required: true,
        description: "Number of decimal places to display",
      },
    ],
  },
  {
    type: "mae_score",
    label: "MAE",
    description:
      "Mean Absolute Error - average absolute difference between predictions and actual values (lower is better)",
    category: "result",
    icon: BarChart3,
    color: "#F97316",
    defaultConfig: {
      model_output_id: "",
      precision: 4,
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected ML model node",
      },
      {
        name: "precision",
        label: "Decimal Precision",
        type: "number",
        required: true,
        description: "Number of decimal places to display",
      },
    ],
  },
  {
    type: "confusion_matrix",
    label: "Confusion Matrix",
    description:
      "Visualization of classification predictions vs actual values showing TP, TN, FP, FN",
    category: "result",
    icon: Target,
    color: "#8B5CF6",
    defaultConfig: {
      model_output_id: "",
      show_percentages: true,
      color_scheme: "blue", // "blue" | "green" | "purple"
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected classification model node",
      },
      {
        name: "show_percentages",
        label: "Show Percentages",
        type: "boolean",
        required: false,
        description: "Display percentages in addition to counts",
      },
      {
        name: "color_scheme",
        label: "Color Scheme",
        type: "select",
        required: true,
        options: [
          { value: "blue", label: "Blue" },
          { value: "green", label: "Green" },
          { value: "purple", label: "Purple" },
        ],
        description: "Color scheme for the matrix visualization",
      },
    ],
  },
];
