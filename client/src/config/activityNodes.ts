/**
 * Activity Node Definitions — standalone interactive learning activities
 * Ordered by incremental difficulty: Beginner → Intermediate → Advanced
 */

import type { NodeCategory } from "./nodeDefinitions";
import {
  Calculator,
  TrendingUp,
  TrendingDown,
  Binary,
  Circle,
  GitBranch,
  Table,
  Zap,
  Network,
  Undo2,
  Layers,
  Gamepad2,
} from "lucide-react";

export const activityCategory: NodeCategory = {
  id: "activities",
  label: "Activities",
  icon: Gamepad2,
  nodes: [
    // ── Beginner: Foundations ──────────────────────────────────────────
    {
      type: "activity_loss_functions",
      label: "Loss Functions",
      description:
        "Compare MSE, MAE, and Huber loss by dragging prediction points away from targets",
      category: "activities",
      icon: Calculator,
      color: "#D946EF",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_linear_regression",
      label: "Linear Regression",
      description:
        "Drag a best-fit line through points and watch the error minimize in real-time",
      category: "activities",
      icon: TrendingUp,
      color: "#3B82F6",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_gradient_descent",
      label: "Gradient Descent",
      description:
        "Visualize how gradient descent optimizes a loss function step by step",
      category: "activities",
      icon: TrendingDown,
      color: "#EF4444",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_logistic_regression",
      label: "Logistic Regression",
      description:
        "Adjust the decision threshold on a sigmoid curve and watch accuracy change",
      category: "activities",
      icon: Binary,
      color: "#F97316",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_kmeans_clustering",
      label: "K-Means Clustering",
      description:
        "Drop centroids, click step, and watch them converge as clusters form iteratively",
      category: "activities",
      icon: Circle,
      color: "#14B8A6",
      defaultConfig: {},
      configFields: [],
    },

    // ── Intermediate: Model Understanding ───────────────────────────────
    {
      type: "activity_decision_tree",
      label: "Decision Tree Builder",
      description:
        "Split data step-by-step by choosing features and thresholds, building the tree visually",
      category: "activities",
      icon: GitBranch,
      color: "#22C55E",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_confusion_matrix",
      label: "Confusion Matrix",
      description:
        "Adjust a classifier threshold and watch TP/FP/TN/FN update with precision/recall",
      category: "activities",
      icon: Table,
      color: "#0EA5E9",
      defaultConfig: {},
      configFields: [],
    },

    // ── Advanced: Deep Learning ─────────────────────────────────────────
    {
      type: "activity_activation_functions",
      label: "Activation Functions",
      description:
        "Compare ReLU, Sigmoid, Tanh, and Leaky ReLU with their derivative curves side by side",
      category: "activities",
      icon: Zap,
      color: "#FACC15",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_neural_network",
      label: "Neural Network Builder",
      description:
        "Build a neural network visually and watch data propagate through layers",
      category: "activities",
      icon: Network,
      color: "#EC4899",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_backpropagation",
      label: "Backpropagation",
      description:
        "Watch gradients flow backward through a small network layer by layer",
      category: "activities",
      icon: Undo2,
      color: "#EF4444",
      defaultConfig: {},
      configFields: [],
    },
    {
      type: "activity_cnn_filters",
      label: "CNN Filter Visualizer",
      description:
        "Apply convolution filters (edge, blur, sharpen) to a small image grid and see output",
      category: "activities",
      icon: Layers,
      color: "#8B5CF6",
      defaultConfig: {},
      configFields: [],
    },
  ],
};
