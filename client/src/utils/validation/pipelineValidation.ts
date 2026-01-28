/**
 * Pipeline Validation Utilities
 *
 * Validates node connections and pipeline structure before execution
 */

import type { MLNode } from "../../types/pipeline";
import type { Edge } from "@xyflow/react";

export interface ValidationError {
  type: "error" | "warning";
  nodeId: string;
  message: string;
  suggestion?: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

/**
 * Define node categories
 */
const NODE_CATEGORIES = {
  DATA_SOURCE: ["upload_file", "select_dataset", "load_url"],
  VIEW: [
    "table_view",
    "data_preview",
    "statistics_view",
    "column_info",
    "chart_view",
  ],
  PROCESSING: ["preprocess", "split", "train", "evaluate"],
  FEATURE_ENGINEERING: ["feature_engineering"],
};

/**
 * Check if a node is of a specific category
 */
function isNodeCategory(
  nodeType: string,
  category: keyof typeof NODE_CATEGORIES,
): boolean {
  return NODE_CATEGORIES[category].includes(nodeType);
}

/**
 * Get all incoming edges for a node
 */
function getIncomingEdges(nodeId: string, edges: Edge[]): Edge[] {
  return edges.filter((edge) => edge.target === nodeId);
}

/**
 * Get source node for an edge
 */
function getSourceNode(edge: Edge, nodes: MLNode[]): MLNode | undefined {
  return nodes.find((node) => node.id === edge.source);
}

/**
 * Validate that view nodes are only connected to data source nodes
 */
export function validateViewNodeConnections(
  nodes: MLNode[],
  edges: Edge[],
): ValidationError[] {
  const errors: ValidationError[] = [];

  // Find all view nodes
  const viewNodes = nodes.filter((node) =>
    isNodeCategory(node.data.type, "VIEW"),
  );

  for (const viewNode of viewNodes) {
    const incomingEdges = getIncomingEdges(viewNode.id, edges);

    // Check if view node has no connections
    if (incomingEdges.length === 0) {
      errors.push({
        type: "error",
        nodeId: viewNode.id,
        message: `${viewNode.data.label} must be connected to a data source`,
        suggestion:
          "Connect this view node to an Upload File or Select Dataset node",
      });
      continue;
    }

    // Check each incoming connection
    for (const edge of incomingEdges) {
      const sourceNode = getSourceNode(edge, nodes);

      if (!sourceNode) {
        errors.push({
          type: "error",
          nodeId: viewNode.id,
          message: `${viewNode.data.label} has invalid connection`,
          suggestion: "Remove and reconnect this edge",
        });
        continue;
      }

      // View nodes should only connect to data sources
      if (!isNodeCategory(sourceNode.data.type, "DATA_SOURCE")) {
        errors.push({
          type: "error",
          nodeId: viewNode.id,
          message: `${viewNode.data.label} cannot be connected to ${sourceNode.data.label}`,
          suggestion:
            "View nodes can only connect to data source nodes (Upload File, Select Dataset, Load from URL)",
        });
      }
    }

    // Check for multiple connections (view nodes should have only one input)
    if (incomingEdges.length > 1) {
      errors.push({
        type: "warning",
        nodeId: viewNode.id,
        message: `${viewNode.data.label} has multiple incoming connections`,
        suggestion:
          "View nodes should typically connect to only one data source",
      });
    }
  }

  return errors;
}

/**
 * Validate that all nodes are properly configured
 */
export function validateNodeConfiguration(nodes: MLNode[]): ValidationError[] {
  const errors: ValidationError[] = [];

  for (const node of nodes) {
    // Skip validation for data source nodes that don't require configuration
    if (
      node.data.type === "upload_file" ||
      node.data.type === "select_dataset"
    ) {
      if (!node.data.config?.dataset_id) {
        errors.push({
          type: "error",
          nodeId: node.id,
          message: `${node.data.label} is not configured`,
          suggestion: "Click on the node to configure it",
        });
      }
    }

    // View nodes must be configured
    if (isNodeCategory(node.data.type, "VIEW")) {
      if (!node.data.isConfigured && !node.data.config?.dataset_id) {
        errors.push({
          type: "error",
          nodeId: node.id,
          message: `${node.data.label} is not configured`,
          suggestion: "Click on the node to configure it",
        });
      }
    }
  }

  return errors;
}

/**
 * Validate that there are no circular dependencies
 */
export function validateNoCircularDependencies(
  nodes: MLNode[],
  edges: Edge[],
): ValidationError[] {
  const errors: ValidationError[] = [];
  const visited = new Set<string>();
  const recursionStack = new Set<string>();

  function hasCycle(nodeId: string, path: string[] = []): boolean {
    visited.add(nodeId);
    recursionStack.add(nodeId);

    const outgoingEdges = edges.filter((edge) => edge.source === nodeId);

    for (const edge of outgoingEdges) {
      if (!visited.has(edge.target)) {
        if (hasCycle(edge.target, [...path, nodeId])) {
          return true;
        }
      } else if (recursionStack.has(edge.target)) {
        // Found a cycle
        const cycleNodes = [...path, nodeId, edge.target];
        const cycleNodeLabels = cycleNodes
          .map((id) => nodes.find((n) => n.id === id)?.data.label)
          .join(" → ");

        errors.push({
          type: "error",
          nodeId,
          message: `Circular dependency detected: ${cycleNodeLabels}`,
          suggestion: "Remove one of the connections to break the cycle",
        });
        return true;
      }
    }

    recursionStack.delete(nodeId);
    return false;
  }

  // Check all nodes
  for (const node of nodes) {
    if (!visited.has(node.id)) {
      hasCycle(node.id);
    }
  }

  return errors;
}

/**
 * Main validation function - validates entire pipeline
 */
export function validatePipeline(
  nodes: MLNode[],
  edges: Edge[],
): ValidationResult {
  const allErrors: ValidationError[] = [];

  // Run all validations
  allErrors.push(...validateViewNodeConnections(nodes, edges));
  allErrors.push(...validateNodeConfiguration(nodes, edges));
  allErrors.push(...validateNoCircularDependencies(nodes, edges));

  return {
    isValid: allErrors.filter((e) => e.type === "error").length === 0,
    errors: allErrors,
  };
}

/**
 * Format validation errors for display
 */
export function formatValidationErrors(errors: ValidationError[]): string {
  const errorMessages = errors.filter((e) => e.type === "error");
  const warningMessages = errors.filter((e) => e.type === "warning");

  let message = "";

  if (errorMessages.length > 0) {
    message += "❌ Errors:\n";
    errorMessages.forEach((error, index) => {
      message += `${index + 1}. ${error.message}\n`;
      if (error.suggestion) {
        message += `   💡 ${error.suggestion}\n`;
      }
    });
  }

  if (warningMessages.length > 0) {
    message += "\n⚠️ Warnings:\n";
    warningMessages.forEach((warning, index) => {
      message += `${index + 1}. ${warning.message}\n`;
      if (warning.suggestion) {
        message += `   💡 ${warning.suggestion}\n`;
      }
    });
  }

  return message;
}
