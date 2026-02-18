/**
 * Main validation coordinator and registry
 * Exports unified validation system for all node types
 */

import type { Edge } from "@xyflow/react";
import type {
  ValidationError,
  NodeConnectionRule,
  ValidationRegistry,
  MLNode,
} from "./types";

import { dataSourceValidationRules } from "./dataSourceValidation";
import { viewValidationRules } from "./viewValidation";
import { preprocessingValidationRules } from "./preprocessingValidation";
import { featureEngValidationRules } from "./featureEngValidation";
import { targetSplitValidationRules } from "./targetSplitValidation";
import { mlAlgorithmValidationRules } from "./mlAlgorithmValidation";
import { resultValidationRules } from "./resultValidation";
import { genaiValidationRules } from "./genaiValidation";

/**
 * Master validation registry - all node types and their rules
 */
export const validationRegistry: ValidationRegistry = {
  ...dataSourceValidationRules,
  ...viewValidationRules,
  ...preprocessingValidationRules,
  ...featureEngValidationRules,
  ...targetSplitValidationRules,
  ...mlAlgorithmValidationRules,
  ...resultValidationRules,
  ...genaiValidationRules,
};

/**
 * Get validation rules for a specific node type
 */
export function getValidationRules(
  nodeType: string,
): NodeConnectionRule | undefined {
  return validationRegistry[nodeType];
}

/**
 * Validate a single connection between two nodes
 */
export function validateConnection(
  sourceNode: MLNode,
  targetNode: MLNode,
  edges: Edge[],
): ValidationError | null {
  const targetRules = getValidationRules(targetNode.data.type);

  if (!targetRules) {
    // No rules defined - allow connection
    return null;
  }

  // Check if source is in allowed sources
  if (
    targetRules.allowedSources &&
    targetRules.allowedSources.length > 0 &&
    !targetRules.allowedSources.includes(sourceNode.data.type)
  ) {
    return {
      type: "error",
      nodeId: targetNode.id,
      message: `${targetNode.data.label} cannot connect to ${sourceNode.data.label}`,
      suggestion: getConnectionSuggestion(
        sourceNode.data.type,
        targetNode.data.type,
        targetRules,
      ),
    };
  }

  // Check if target is in allowed targets for source
  const sourceRules = getValidationRules(sourceNode.data.type);
  if (
    sourceRules?.allowedTargets &&
    sourceRules.allowedTargets.length > 0 &&
    !sourceRules.allowedTargets.includes(targetNode.data.type)
  ) {
    return {
      type: "error",
      nodeId: sourceNode.id,
      message: `${sourceNode.data.label} cannot connect to ${targetNode.data.label}`,
      suggestion: getConnectionSuggestion(
        sourceNode.data.type,
        targetNode.data.type,
        sourceRules,
      ),
    };
  }

  // Check max input connections
  if (targetRules.maxInputConnections !== undefined) {
    const incomingEdges = edges.filter((e) => e.target === targetNode.id);
    if (incomingEdges.length > targetRules.maxInputConnections) {
      return {
        type: "error",
        nodeId: targetNode.id,
        message: `${targetNode.data.label} can only have ${targetRules.maxInputConnections} input connection(s)`,
        suggestion: "Remove existing connections before adding a new one.",
      };
    }
  }

  // Run custom validator if defined
  if (targetRules.customValidator) {
    const customError = targetRules.customValidator(
      sourceNode,
      targetNode,
      edges,
    );
    if (customError) {
      return customError;
    }
  }

  // Run source custom validator as well
  if (sourceRules?.customValidator) {
    const customError = sourceRules.customValidator(
      sourceNode,
      targetNode,
      edges,
    );
    if (customError) {
      return customError;
    }
  }

  return null;
}

/**
 * Validate all connections in the pipeline
 */
export function validateAllConnections(
  nodes: MLNode[],
  edges: Edge[],
): ValidationError[] {
  const errors: ValidationError[] = [];

  // Validate each edge/connection
  for (const edge of edges) {
    const sourceNode = nodes.find((n) => n.id === edge.source);
    const targetNode = nodes.find((n) => n.id === edge.target);

    if (!sourceNode || !targetNode) {
      continue;
    }

    const error = validateConnection(sourceNode, targetNode, edges);
    if (error) {
      errors.push(error);
    }
  }

  // Validate nodes that require input have at least one
  for (const node of nodes) {
    const rules = getValidationRules(node.data.type);
    if (rules?.requiresInput) {
      const incomingEdges = edges.filter((e) => e.target === node.id);
      if (incomingEdges.length === 0) {
        errors.push({
          type: "error",
          nodeId: node.id,
          message: `${node.data.label} requires an input connection`,
          suggestion: getRequiredInputSuggestion(node.data.type, rules),
        });
      }
    }
  }

  return errors;
}

/**
 * Generate helpful suggestion based on node types and rules
 */
function getConnectionSuggestion(
  sourceType: string,
  targetType: string,
  rules: NodeConnectionRule,
): string {
  // Special case suggestions for common mistakes
  const mlAlgorithms = [
    "linear_regression",
    "logistic_regression",
    "decision_tree",
    "random_forest",
  ];
  const dataSources = ["upload_file", "select_dataset", "sample_dataset"];
  const featureEng = [
    "encoding",
    "transformation",
    "scaling",
    "feature_selection",
  ];

  // Dataset directly to ML Algorithm
  if (dataSources.includes(sourceType) && mlAlgorithms.includes(targetType)) {
    return `You cannot connect a dataset directly to an ML algorithm. Correct flow: ${getNodeLabel(sourceType)} → Target & Split → ${getNodeLabel(targetType)}`;
  }

  // Feature engineering directly to ML Algorithm
  if (featureEng.includes(sourceType) && mlAlgorithms.includes(targetType)) {
    return `Feature engineering must be followed by Target & Split before ML training. Correct flow: ${getNodeLabel(sourceType)} → Target & Split → ${getNodeLabel(targetType)}`;
  }

  // Preprocessing directly to ML Algorithm
  if (
    sourceType === "missing_value_handler" &&
    mlAlgorithms.includes(targetType)
  ) {
    return `Preprocessing must be followed by Target & Split before ML training. Correct flow: Missing Value Handler → Target & Split → ${getNodeLabel(targetType)}`;
  }

  // View node to ML Algorithm
  if (targetType.includes("view") && mlAlgorithms.includes(targetType)) {
    return "View nodes are for data inspection only and cannot be used as input to ML algorithms.";
  }

  // Generic suggestion based on allowed targets
  if (rules.allowedTargets && rules.allowedTargets.length > 0) {
    const exampleTargets = rules.allowedTargets
      .slice(0, 3)
      .map((t) => getNodeLabel(t))
      .join(", ");
    return `${getNodeLabel(sourceType)} can connect to: ${exampleTargets}${rules.allowedTargets.length > 3 ? ", ..." : ""}`;
  }

  // Generic suggestion based on allowed sources
  if (rules.allowedSources && rules.allowedSources.length > 0) {
    const exampleSources = rules.allowedSources
      .slice(0, 3)
      .map((t) => getNodeLabel(t))
      .join(", ");
    return `${getNodeLabel(targetType)} can only accept input from: ${exampleSources}${rules.allowedSources.length > 3 ? ", ..." : ""}`;
  }

  return "Check the node palette to see which nodes can connect together.";
}

/**
 * Generate suggestion for nodes requiring input
 */
function getRequiredInputSuggestion(
  nodeType: string,
  rules: NodeConnectionRule,
): string {
  if (rules.allowedSources && rules.allowedSources.length > 0) {
    const exampleSources = rules.allowedSources
      .slice(0, 3)
      .map((t) => getNodeLabel(t))
      .join(", ");
    return `Connect ${getNodeLabel(nodeType)} to one of: ${exampleSources}${rules.allowedSources.length > 3 ? ", ..." : ""}`;
  }
  return `${getNodeLabel(nodeType)} requires an input connection.`;
}

/**
 * Convert node type to readable label
 */
function getNodeLabel(nodeType: string): string {
  const labelMap: Record<string, string> = {
    upload_file: "Upload Dataset",
    select_dataset: "Select Dataset",
    sample_dataset: "Sample Dataset",
    table_view: "Table View",
    data_preview: "Data Preview",
    statistics_view: "Statistics View",
    column_info: "Column Info",
    chart_view: "Chart View",
    missing_value_handler: "Missing Value Handler",
    encoding: "Encoding",
    transformation: "Transformation",
    scaling: "Scaling/Normalization",
    feature_selection: "Feature Selection",
    split: "Target & Split",
    linear_regression: "Linear Regression",
    logistic_regression: "Logistic Regression",
    decision_tree: "Decision Tree",
    random_forest: "Random Forest",
    evaluate: "Model Evaluation",
    r2_score: "R² Score",
    mse_score: "MSE",
    rmse_score: "RMSE",
    mae_score: "MAE",
    confusion_matrix: "Confusion Matrix",
    classification_report: "Classification Report",
    accuracy_score: "Accuracy",
    roc_curve: "ROC Curve",
    feature_importance: "Feature Importance",
    residual_plot: "Residual Plot",
    prediction_table: "Predictions Table",
    llm_node: "LLM Provider",
    system_prompt: "System Prompt",
    chatbot_node: "Chatbot",
    example_node: "Examples",
  };

  return labelMap[nodeType] || nodeType;
}

// Export types
export type { ValidationError, NodeConnectionRule, ValidationRegistry };
