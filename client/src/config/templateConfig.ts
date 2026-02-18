/**
 * Template Configurations for ML Workflows
 * Pre-configured pipelines that can be loaded onto the canvas
 */

import { Zap, TreePine, Trees, Bot } from "lucide-react";
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
  {
    id: "decision_tree",
    label: "Decision Tree",
    description: "Classification pipeline with Decision Tree and confusion matrix",
    icon: TreePine,
    color: "#10B981",
    nodes: [
      { type: "upload_file", position: { x: 100, y: 100 } },
      { type: "table_view", position: { x: 100, y: 220 } },
      { type: "missing_value_handler", position: { x: 100, y: 340 } },
      { type: "encoding", position: { x: 100, y: 460 } },
      { type: "split", position: { x: 100, y: 580 } },
      { type: "decision_tree", position: { x: 100, y: 700 } },
      { type: "confusion_matrix", position: { x: 400, y: 640 } },
      { type: "rmse_score", position: { x: 400, y: 730 } },
    ],
    edges: [
      { sourceIndex: 0, targetIndex: 1 }, // Upload → Table View
      { sourceIndex: 1, targetIndex: 2 }, // Table View → Missing Value Handler
      { sourceIndex: 2, targetIndex: 3 }, // Missing Value → Encoding
      { sourceIndex: 3, targetIndex: 4 }, // Encoding → Target & Split
      { sourceIndex: 4, targetIndex: 5 }, // Target & Split → Decision Tree
      { sourceIndex: 5, targetIndex: 6 }, // Decision Tree → Confusion Matrix
      { sourceIndex: 5, targetIndex: 7 }, // Decision Tree → RMSE
    ],
  },
  {
    id: "random_forest",
    label: "Random Forest",
    description: "Ensemble classification with Random Forest, scaling, and evaluation",
    icon: Trees,
    color: "#8B5CF6",
    nodes: [
      { type: "upload_file", position: { x: 100, y: 100 } },
      { type: "table_view", position: { x: 100, y: 220 } },
      { type: "missing_value_handler", position: { x: 100, y: 340 } },
      { type: "encoding", position: { x: 100, y: 460 } },
      { type: "scaling", position: { x: 100, y: 580 } },
      { type: "split", position: { x: 100, y: 700 } },
      { type: "random_forest", position: { x: 100, y: 820 } },
      { type: "confusion_matrix", position: { x: 400, y: 760 } },
      { type: "mae_score", position: { x: 400, y: 850 } },
    ],
    edges: [
      { sourceIndex: 0, targetIndex: 1 }, // Upload → Table View
      { sourceIndex: 1, targetIndex: 2 }, // Table View → Missing Value Handler
      { sourceIndex: 2, targetIndex: 3 }, // Missing Value → Encoding
      { sourceIndex: 3, targetIndex: 4 }, // Encoding → Scaling
      { sourceIndex: 4, targetIndex: 5 }, // Scaling → Target & Split
      { sourceIndex: 5, targetIndex: 6 }, // Target & Split → Random Forest
      { sourceIndex: 6, targetIndex: 7 }, // Random Forest → Confusion Matrix
      { sourceIndex: 6, targetIndex: 8 }, // Random Forest → MAE
    ],
  },
  {
    id: "genai_chatbot",
    label: "GenAI Chatbot",
    description: "AI chatbot with LLM provider, system prompt, and few-shot examples",
    icon: Bot,
    color: "#F59E0B",
    nodes: [
      { type: "llm_node", position: { x: 100, y: 100 } },
      { type: "system_prompt", position: { x: 100, y: 260 } },
      { type: "example_node", position: { x: 100, y: 420 } },
      { type: "chatbot_node", position: { x: 450, y: 260 } },
    ],
    edges: [
      { sourceIndex: 0, targetIndex: 3 }, // LLM Provider → Chatbot
      { sourceIndex: 1, targetIndex: 3 }, // System Prompt → Chatbot
      { sourceIndex: 2, targetIndex: 3 }, // Examples → Chatbot
    ],
  },
];

export const getTemplateById = (id: string): Template | undefined => {
  return templates.find((template) => template.id === id);
};
