import type { NodeDefinition } from "./nodeDefinitions";
import {
  TrendingUp,
  BarChart3,
  Activity,
  Target,
  FileText,
  Crosshair,
  Award,
  LineChart,
  Layers,
  Search,
  Table,
} from "lucide-react";

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
  {
    type: "classification_report",
    label: "Classification Report",
    description:
      "Detailed metrics including precision, recall, F1-score for each class",
    category: "result",
    icon: FileText,
    color: "#6366F1",
    defaultConfig: {
      model_output_id: "",
      show_support: true,
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
        name: "show_support",
        label: "Show Support",
        type: "boolean",
        required: false,
        description: "Display number of samples per class",
      },
    ],
  },
  {
    type: "accuracy_score",
    label: "Accuracy",
    description:
      "Classification accuracy - percentage of correct predictions (0-100%, higher is better)",
    category: "result",
    icon: Crosshair,
    color: "#10B981",
    defaultConfig: {
      model_output_id: "",
      display_format: "percentage",
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
        name: "display_format",
        label: "Display Format",
        type: "select",
        required: true,
        options: [
          { value: "percentage", label: "Percentage (85.5%)" },
          { value: "decimal", label: "Decimal (0.855)" },
        ],
        description: "How to display the accuracy score",
      },
    ],
  },
  {
    type: "roc_curve",
    label: "ROC Curve",
    description:
      "Receiver Operating Characteristic curve - plots TPR vs FPR for binary classification",
    category: "result",
    icon: LineChart,
    color: "#EC4899",
    defaultConfig: {
      model_output_id: "",
      show_auc: true,
      show_threshold: false,
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected binary classification model",
      },
      {
        name: "show_auc",
        label: "Show AUC Score",
        type: "boolean",
        required: false,
        description: "Display Area Under Curve score on the plot",
      },
      {
        name: "show_threshold",
        label: "Show Decision Threshold",
        type: "boolean",
        required: false,
        description: "Display optimal decision threshold point",
      },
    ],
  },
  {
    type: "feature_importance",
    label: "Feature Importance",
    description:
      "Bar chart showing relative importance of each feature in the model",
    category: "result",
    icon: Layers,
    color: "#14B8A6",
    defaultConfig: {
      model_output_id: "",
      top_n: 10,
      sort_order: "descending", // "descending" | "ascending"
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected tree-based model node",
      },
      {
        name: "top_n",
        label: "Top N Features",
        type: "number",
        required: true,
        description: "Number of top features to display",
      },
      {
        name: "sort_order",
        label: "Sort Order",
        type: "select",
        required: true,
        options: [
          { value: "descending", label: "Descending (Highest First)" },
          { value: "ascending", label: "Ascending (Lowest First)" },
        ],
        description: "How to sort features by importance",
      },
    ],
  },
  {
    type: "residual_plot",
    label: "Residual Plot",
    description:
      "Scatter plot of residuals (errors) vs predicted values for regression diagnostics",
    category: "result",
    icon: Search,
    color: "#06B6D4",
    defaultConfig: {
      model_output_id: "",
      show_reference_line: true,
    },
    configFields: [
      {
        name: "model_output_id",
        label: "Model Output",
        type: "text",
        required: true,
        autoFill: true,
        description: "Auto-filled from connected regression model node",
      },
      {
        name: "show_reference_line",
        label: "Show Reference Line",
        type: "boolean",
        required: false,
        description: "Display y=0 reference line",
      },
    ],
  },
  {
    type: "prediction_table",
    label: "Predictions Table",
    description: "Tabular view of actual vs predicted values with sample IDs",
    category: "result",
    icon: Table,
    color: "#64748B",
    defaultConfig: {
      model_output_id: "",
      max_rows: 100,
      show_errors: true,
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
        name: "max_rows",
        label: "Max Rows to Display",
        type: "number",
        required: true,
        description: "Maximum number of predictions to show",
      },
      {
        name: "show_errors",
        label: "Show Error Column",
        type: "boolean",
        required: false,
        description: "Display difference between actual and predicted",
      },
    ],
  },
];
