/**
 * Shared types for node connection validation system
 */

import type { Node } from "@xyflow/react";
import type { BaseNodeData } from "../../../types/pipeline";

export type MLNode = Node<BaseNodeData>;

export interface ValidationError {
  type: "error" | "warning";
  nodeId: string;
  message: string;
  suggestion?: string;
}

export interface NodeConnectionRule {
  /**
   * Node types that can connect as inputs to this node
   * Empty array means no inputs allowed
   * undefined means any input is allowed
   */
  allowedSources?: string[];

  /**
   * Whether this node requires at least one input connection
   */
  requiresInput?: boolean;

  /**
   * Maximum number of incoming connections allowed
   * undefined means unlimited
   */
  maxInputConnections?: number;

  /**
   * Node types that this node can connect to as outputs
   * Empty array means no outputs allowed (terminal node)
   * undefined means any output is allowed
   */
  allowedTargets?: string[];

  /**
   * Custom validation function for complex rules
   */
  customValidator?: (
    sourceNode: MLNode,
    targetNode: MLNode,
    edges: any[]
  ) => ValidationError | null;
}

export interface ValidationRegistry {
  [nodeType: string]: NodeConnectionRule;
}
