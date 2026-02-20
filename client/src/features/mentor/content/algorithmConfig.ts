/**
 * Algorithm Configuration
 *
 * Per-algorithm settings for the guided learning flow.
 * Display names, metric nodes, and classification flags.
 */

export interface AlgorithmConfig {
  displayName: string;
  nodeType: string;
  description: string;
  metricNodes: string[];
  metricDisplayText: string;
  metricVoiceText: string;
  isClassification: boolean;
  targetWarningThreshold?: number;
}

export const ALGORITHM_CONFIG: Record<string, AlgorithmConfig> = {
  linear_regression: {
    displayName: "Linear Regression",
    nodeType: "linear_regression",
    description:
      "Predicts continuous numbers like prices, scores, or temperatures.",
    metricNodes: ["r2_score", "rmse_score", "mae_score"],
    metricDisplayText:
      "R squared score, Root Mean Squared Error, and Mean Absolute Error",
    metricVoiceText:
      "R squared score, Root Mean Squared Error, and Mean Absolute Error",
    isClassification: false,
  },
  logistic_regression: {
    displayName: "Logistic Regression",
    nodeType: "logistic_regression",
    description:
      "Predicts categories like spam or not spam, disease or healthy.",
    metricNodes: ["confusion_matrix"],
    metricDisplayText: "a Confusion Matrix",
    metricVoiceText: "a Confusion Matrix",
    isClassification: true,
    targetWarningThreshold: 10,
  },
  decision_tree: {
    displayName: "Decision Tree",
    nodeType: "decision_tree",
    description:
      "Makes decisions like a flowchart with yes or no questions.",
    metricNodes: ["r2_score", "rmse_score"],
    metricDisplayText: "R squared score and Root Mean Squared Error",
    metricVoiceText: "R squared score and Root Mean Squared Error",
    isClassification: false,
  },
  random_forest: {
    displayName: "Random Forest",
    nodeType: "random_forest",
    description:
      "Builds many decision trees and combines their answers by voting.",
    metricNodes: ["r2_score", "rmse_score"],
    metricDisplayText: "R squared score and Root Mean Squared Error",
    metricVoiceText: "R squared score and Root Mean Squared Error",
    isClassification: false,
  },
};

export function getAlgorithmConfig(algorithm: string): AlgorithmConfig {
  return (
    ALGORITHM_CONFIG[algorithm] ?? {
      displayName: algorithm.replace(/_/g, " "),
      nodeType: algorithm,
      description: "",
      metricNodes: ["r2_score"],
      metricDisplayText: "R squared score",
      metricVoiceText: "R squared score",
      isClassification: false,
    }
  );
}
