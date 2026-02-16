/**
 * Template Configurations for ML Workflows
 * Pre-configured pipelines that can be loaded onto the canvas
 */

import { Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { NodeType } from "../types/pipeline";

export interface TemplateNode {
  type: NodeType;
  position: { x: number; y: number };
}

export interface TemplateEdge {
  sourceIndex: number;
  targetIndex: number;
}

export interface Template {
  id: string;
  label: string;
  description: string;
  icon: LucideIcon;
  color: string;
  nodes: TemplateNode[];
  edges: TemplateEdge[];
}

export const templates: Template[] = [
  {
    id: "linear_regression",
    label: "Linear Regression",
    description: "Complete Linear Regression workflow with data processing and metrics",
    icon: Zap,
    color: "#3B82F6",
    nodes: [
      { type: "upload_file", position: { x: 100, y: 100 } },
      { type: "table_view", position: { x: 100, y: 220 } },
      { type: "missing_value_handler", position: { x: 100, y: 340 } },
      { type: "encoding", position: { x: 100, y: 460 } },
      { type: "split", position: { x: 100, y: 580 } },
      { type: "linear_regression", position: { x: 100, y: 700 } },
      { type: "r2_score", position: { x: 400, y: 640 } },
      { type: "mse_score", position: { x: 400, y: 730 } },
      { type: "rmse_score", position: { x: 400, y: 820 } },
      { type: "mae_score", position: { x: 400, y: 910 } },
    ],
    edges: [
      { sourceIndex: 0, targetIndex: 1 }, // Upload → Table View
      { sourceIndex: 1, targetIndex: 2 }, // Table View → Missing Value Handler
      { sourceIndex: 2, targetIndex: 3 }, // Missing Value → Encoding
      { sourceIndex: 3, targetIndex: 4 }, // Encoding → Target & Split
      { sourceIndex: 4, targetIndex: 5 }, // Target & Split → Linear Regression
      { sourceIndex: 5, targetIndex: 6 }, // Linear Regression → R² Score
      { sourceIndex: 5, targetIndex: 7 }, // Linear Regression → MSE
      { sourceIndex: 5, targetIndex: 8 }, // Linear Regression → RMSE
      { sourceIndex: 5, targetIndex: 9 }, // Linear Regression → MAE
    ],
  },
  {
    id: "logistic_regression",
    label: "Logistic Regression",
    description: "Complete Logistic Regression workflow with confusion matrix",
    icon: Zap,
    color: "#8B5CF6",
    nodes: [
      { type: "upload_file", position: { x: 100, y: 100 } },
      { type: "table_view", position: { x: 100, y: 220 } },
      { type: "missing_value_handler", position: { x: 100, y: 340 } },
      { type: "encoding", position: { x: 100, y: 460 } },
      { type: "split", position: { x: 100, y: 580 } },
      { type: "logistic_regression", position: { x: 100, y: 700 } },
      { type: "confusion_matrix", position: { x: 400, y: 700 } },
    ],
    edges: [
      { sourceIndex: 0, targetIndex: 1 }, // Upload → Table View
      { sourceIndex: 1, targetIndex: 2 }, // Table View → Missing Value Handler
      { sourceIndex: 2, targetIndex: 3 }, // Missing Value → Encoding
      { sourceIndex: 3, targetIndex: 4 }, // Encoding → Target & Split
      { sourceIndex: 4, targetIndex: 5 }, // Target & Split → Logistic Regression
      { sourceIndex: 5, targetIndex: 6 }, // Logistic Regression → Confusion Matrix
    ],
  },
];

export const getTemplateById = (id: string): Template | undefined => {
  return templates.find((template) => template.id === id);
};
