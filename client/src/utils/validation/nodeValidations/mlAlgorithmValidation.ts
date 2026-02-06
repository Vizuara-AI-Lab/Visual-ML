/**
 * Validation rules for ML Algorithm nodes
 * - linear_regression
 * - logistic_regression
 * - decision_tree
 * - random_forest
 * - evaluate
 */

import type { ValidationRegistry, MLNode } from "./types";

const mlAlgorithmCommonRules = {
  allowedSources: [
    "split", // MUST come from split node
  ],
  requiresInput: true,
  maxInputConnections: 1,
  allowedTargets: [
    // Result nodes
    "r2_score",
    "mse_score",
    "rmse_score",
    "mae_score",
    "confusion_matrix",
    "classification_report",
    "accuracy_score",
    "roc_curve",
    "feature_importance",
    "residual_plot",
    "prediction_table",
    // Evaluation
    "evaluate",
    // Deployment
    "model_export",
    "api_endpoint",
  ],
};

export const mlAlgorithmValidationRules: ValidationRegistry = {
  linear_regression: {
    ...mlAlgorithmCommonRules,
    customValidator: (sourceNode: MLNode, targetNode: MLNode) => {
      // Only validate when Linear Regression is the SOURCE (connecting to metrics)
      // Don't validate when Linear Regression is the TARGET (receiving from Split)
      if (sourceNode.data.type !== "linear_regression") {
        return null; // Not our concern, let other validators handle it
      }

      // Regression model should only connect to regression metrics
      const regressionMetrics = [
        "r2_score",
        "mse_score",
        "rmse_score",
        "mae_score",
        "residual_plot",
        "prediction_table",
        "evaluate",
        "model_export",
        "api_endpoint",
        "feature_importance",
      ];

      if (!regressionMetrics.includes(targetNode.data.type)) {
        return {
          type: "error",
          nodeId: targetNode.id,
          message: `Linear Regression cannot connect to ${targetNode.data.label} (classification metric)`,
          suggestion:
            "Use regression metrics like R² Score, MSE, RMSE, or MAE. For classification metrics, use a classification model like Logistic Regression.",
        };
      }
      return null;
    },
  },

  logistic_regression: {
    ...mlAlgorithmCommonRules,
    customValidator: (sourceNode: MLNode, targetNode: MLNode) => {
      // Only validate when Logistic Regression is the SOURCE (connecting to metrics)
      // Don't validate when Logistic Regression is the TARGET (receiving from Split)
      if (sourceNode.data.type !== "logistic_regression") {
        return null; // Not our concern, let other validators handle it
      }

      // Classification model should only connect to classification metrics
      const classificationMetrics = [
        "confusion_matrix",
        "classification_report",
        "accuracy_score",
        "roc_curve",
        "prediction_table",
        "evaluate",
        "model_export",
        "api_endpoint",
        "feature_importance",
      ];

      if (!classificationMetrics.includes(targetNode.data.type)) {
        return {
          type: "error",
          nodeId: targetNode.id,
          message: `Logistic Regression cannot connect to ${targetNode.data.label} (regression metric)`,
          suggestion:
            "Use classification metrics like Confusion Matrix, Accuracy, or Classification Report. For regression metrics, use a regression model like Linear Regression.",
        };
      }
      return null;
    },
  },

  decision_tree: {
    ...mlAlgorithmCommonRules,
    customValidator: (sourceNode: MLNode, targetNode: MLNode) => {
      // Decision tree can be used for both - check task_type in config
      const taskType = sourceNode.data.config?.task_type || "classification";

      const classificationMetrics = [
        "confusion_matrix",
        "classification_report",
        "accuracy_score",
        "roc_curve",
      ];
      const regressionMetrics = [
        "r2_score",
        "mse_score",
        "rmse_score",
        "mae_score",
        "residual_plot",
      ];

      if (
        taskType === "classification" &&
        regressionMetrics.includes(targetNode.data.type)
      ) {
        return {
          type: "error",
          nodeId: targetNode.id,
          message: `${targetNode.data.label} cannot be used with Decision Tree configured for classification`,
          suggestion:
            "Use classification metrics like Confusion Matrix or Accuracy, or change the Decision Tree task_type to 'regression'.",
        };
      }

      if (
        taskType === "regression" &&
        classificationMetrics.includes(targetNode.data.type)
      ) {
        return {
          type: "error",
          nodeId: targetNode.id,
          message: `${targetNode.data.label} cannot be used with Decision Tree configured for regression`,
          suggestion:
            "Use regression metrics like R² Score or MSE, or change the Decision Tree task_type to 'classification'.",
        };
      }

      return null;
    },
  },

  random_forest: {
    ...mlAlgorithmCommonRules,
    customValidator: (sourceNode: MLNode, targetNode: MLNode) => {
      // Random forest can be used for both - check task_type in config
      const taskType = sourceNode.data.config?.task_type || "classification";

      const classificationMetrics = [
        "confusion_matrix",
        "classification_report",
        "accuracy_score",
        "roc_curve",
      ];
      const regressionMetrics = [
        "r2_score",
        "mse_score",
        "rmse_score",
        "mae_score",
        "residual_plot",
      ];

      if (
        taskType === "classification" &&
        regressionMetrics.includes(targetNode.data.type)
      ) {
        return {
          type: "error",
          nodeId: targetNode.id,
          message: `${targetNode.data.label} cannot be used with Random Forest configured for classification`,
          suggestion:
            "Use classification metrics like Confusion Matrix or Accuracy, or change the Random Forest task_type to 'regression'.",
        };
      }

      if (
        taskType === "regression" &&
        classificationMetrics.includes(targetNode.data.type)
      ) {
        return {
          type: "error",
          nodeId: targetNode.id,
          message: `${targetNode.data.label} cannot be used with Random Forest configured for regression`,
          suggestion:
            "Use regression metrics like R² Score or MSE, or change the Random Forest task_type to 'classification'.",
        };
      }

      return null;
    },
  },

  evaluate: {
    allowedSources: [
      "split", // For test dataset
      "linear_regression",
      "logistic_regression",
      "decision_tree",
      "random_forest",
    ],
    requiresInput: true,
    maxInputConnections: 2, // Model + test data
    allowedTargets: [
      "r2_score",
      "mse_score",
      "rmse_score",
      "mae_score",
      "confusion_matrix",
      "classification_report",
      "accuracy_score",
      "roc_curve",
      "feature_importance",
      "residual_plot",
      "prediction_table",
    ],
  },
};
