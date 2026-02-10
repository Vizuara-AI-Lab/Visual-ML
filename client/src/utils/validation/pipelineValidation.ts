/**
 * Pipeline Validation Utilities
 *
 * Validates node connections and pipeline structure before execution
 */

import type { Edge } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import type { BaseNodeData } from "../../types/pipeline";
import { validateAllConnections } from "./nodeValidations";

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

// Type alias for consistency with old code
export type MLNode = Node<BaseNodeData>;

/**
 * Define node categories (legacy - now handled by modular validation)
 */
const NODE_CATEGORIES = {
  DATA_SOURCE: ["upload_file", "select_dataset", "load_url", "sample_dataset"],
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
 * Validate node connections using modular validation system
 */
export function validateNodeConnections(
  nodes: MLNode[],
  edges: Edge[],
): ValidationError[] {
  // Use the modular validation system
  return validateAllConnections(nodes, edges);
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
  allErrors.push(...validateNodeConnections(nodes, edges));
  allErrors.push(...validateNodeConfiguration(nodes));
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
