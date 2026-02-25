/**
 * Zustand store for ML Pipeline Playground
 */

import { create } from "zustand";
import { type Node, type Edge } from "@xyflow/react";
import type {
  BaseNodeData,
  NodeExecutionStatus,
  ExecutionLogEntry,
} from "../types/pipeline";
import { getNodeByType } from "../config/nodeDefinitions";

export interface DatasetMetadata {
  dataset_id: string;
  filename: string;
  file_path: string;
  n_rows: number;
  n_columns: number;
  columns: string[];
  dtypes: Record<string, string>;
  numeric_columns: string[];
  categorical_columns: string[];
  missing_values: Record<string, number>;
  preview: Array<Record<string, unknown>>;
  statistics?: Record<string, unknown>;
  suggested_target?: string;
  memory_usage_mb: number;
}

export interface PipelineExecutionResult {
  success: boolean;
  pipeline_name?: string;
  nodeResults?: Record<
    string,
    { success: boolean; output?: unknown; error?: unknown }
  >;
  metrics?: Record<string, number | string>;
  error?: string;
  errorDetails?: Record<string, unknown>;
  errorSuggestion?: string;
  timestamp: string;
}

interface PlaygroundStore {
  // React Flow state
  nodes: Node<BaseNodeData>[];
  edges: Edge[];
  setNodes: (
    nodes:
      | Node<BaseNodeData>[]
      | ((prev: Node<BaseNodeData>[]) => Node<BaseNodeData>[]),
  ) => void;
  setEdges: (edges: Edge[] | ((prev: Edge[]) => Edge[])) => void;
  addNode: (node: Node<BaseNodeData>) => void;
  updateNode: (nodeId: string, data: Partial<BaseNodeData>) => void;
  deleteNode: (nodeId: string) => void;
  addEdge: (edge: Edge) => void;

  // Selected node for configuration
  selectedNodeId: string | null;
  setSelectedNodeId: (nodeId: string | null) => void;

  // Dataset metadata
  datasetMetadata: DatasetMetadata | null;
  setDatasetMetadata: (metadata: DatasetMetadata | null) => void;

  // Execution state
  isExecuting: boolean;
  setIsExecuting: (executing: boolean) => void;
  executionResult: PipelineExecutionResult | null;
  setExecutionResult: (result: PipelineExecutionResult | null) => void;
  executionResults: PipelineExecutionResult | null;
  setExecutionResults: (results: PipelineExecutionResult | null) => void;

  // Real-time execution status tracking
  nodeExecutionStatus: Record<string, NodeExecutionStatus>;
  setNodeExecutionStatus: (nodeId: string, status: NodeExecutionStatus) => void;
  setAllNodesPending: (nodeIds: string[]) => void;
  clearExecutionStatus: () => void;
  animateEdgesForNode: (nodeId: string) => void;
  addNodeResult: (nodeId: string, result: { success: boolean; output?: unknown; error?: unknown }) => void;

  // Project state
  currentProjectId: string | null;
  setCurrentProjectId: (id: string | null) => void;
  loadProjectState: (state: {
    nodes: any[];
    edges: any[];
    datasetMetadata?: any;
    executionResult?: any;
  }) => void;
  getProjectState: () => {
    nodes: any[];
    edges: any[];
    datasetMetadata: any;
    executionResult: any;
  };

  // Execution logs for results drawer timeline
  executionLogs: ExecutionLogEntry[];
  addExecutionLog: (entry: ExecutionLogEntry) => void;
  clearExecutionLogs: () => void;

  // Utility methods
  getNodeById: (nodeId: string) => Node<BaseNodeData> | undefined;
  updateNodeConfig: (nodeId: string, config: Record<string, unknown>) => void;
  clearAll: () => void;
  clearCanvas: () => void;
}

export const usePlaygroundStore = create<PlaygroundStore>((set, get) => ({
  // Initial state
  nodes: [],
  edges: [],
  selectedNodeId: null,
  datasetMetadata: null,
  isExecuting: false,
  executionResult: null,
  executionResults: null,
  nodeExecutionStatus: {},
  currentProjectId: null,
  executionLogs: [],

  // Node actions
  setNodes: (nodesOrUpdater) =>
    set((state) => {
      const newNodes =
        typeof nodesOrUpdater === "function"
          ? nodesOrUpdater(state.nodes)
          : nodesOrUpdater;
      return {
        nodes: Array.isArray(newNodes) ? newNodes : [],
      };
    }),

  setEdges: (edgesOrUpdater) =>
    set((state) => {
      const newEdges =
        typeof edgesOrUpdater === "function"
          ? edgesOrUpdater(state.edges)
          : edgesOrUpdater;
      return {
        edges: Array.isArray(newEdges) ? newEdges : [],
      };
    }),

  addNode: (node) =>
    set((state) => {
      console.log("🔧 Store: Adding node to state", node);
      console.log("🔧 Store: Current nodes:", state.nodes);
      const newNodes = [...state.nodes, node];
      console.log("🔧 Store: New nodes array:", newNodes);
      return { nodes: newNodes };
    }),

  updateNode: (nodeId, data) =>
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node,
      ),
    })),

  deleteNode: (nodeId) =>
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== nodeId),
      edges: state.edges.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId,
      ),
      selectedNodeId:
        state.selectedNodeId === nodeId ? null : state.selectedNodeId,
    })),

  addEdge: (edge) =>
    set((state) => ({
      edges: [...state.edges, edge],
    })),

  // Selection
  setSelectedNodeId: (nodeId) => set({ selectedNodeId: nodeId }),

  // Dataset
  setDatasetMetadata: (metadata) => set({ datasetMetadata: metadata }),

  // Execution
  setIsExecuting: (executing) => set({ isExecuting: executing }),
  setExecutionResult: (result) => set({ executionResult: result }),
  setExecutionResults: (results) => set({ executionResults: results }),

  // Execution logs
  addExecutionLog: (entry) =>
    set((state) => ({
      executionLogs: [...state.executionLogs, entry],
    })),
  clearExecutionLogs: () => set({ executionLogs: [] }),

  // Real-time execution status
  setNodeExecutionStatus: (nodeId, status) =>
    set((state) => ({
      nodeExecutionStatus: {
        ...state.nodeExecutionStatus,
        [nodeId]: status,
      },
    })),

  setAllNodesPending: (nodeIds) =>
    set({
      nodeExecutionStatus: Object.fromEntries(
        nodeIds.map((id) => [id, "pending" as NodeExecutionStatus]),
      ),
    }),

  clearExecutionStatus: () =>
    set({
      nodeExecutionStatus: {},
    }),

  addNodeResult: (nodeId, result) =>
    set((state) => ({
      executionResult: {
        ...(state.executionResult || { success: true, timestamp: new Date().toISOString() }),
        nodeResults: {
          ...(state.executionResult?.nodeResults || {}),
          [nodeId]: result,
        },
      },
    })),

  animateEdgesForNode: (nodeId) =>
    set((state) => {
      // Find outgoing edges from this node
      const outgoingEdges = state.edges.filter(
        (edge) => edge.source === nodeId,
      );

      if (outgoingEdges.length === 0) return {};

      // Temporarily animate outgoing edges
      const updatedEdges = state.edges.map((edge) => {
        if (edge.source === nodeId) {
          return {
            ...edge,
            animated: true,
            style: {
              ...edge.style,
              stroke: "#10B981", // Green color for data flow
              strokeWidth: 2,
            },
          };
        }
        return edge;
      });

      // Reset animation after 2 seconds
      setTimeout(() => {
        const currentState = usePlaygroundStore.getState();
        const resetEdges = currentState.edges.map((edge) => {
          if (edge.source === nodeId) {
            return {
              ...edge,
              animated: false,
              style: {
                ...edge.style,
                stroke: undefined,
                strokeWidth: undefined,
              },
            };
          }
          return edge;
        });
        usePlaygroundStore.setState({ edges: resetEdges });
      }, 2000);

      return { edges: updatedEdges };
    }),

  // Project state
  setCurrentProjectId: (id) => set({ currentProjectId: id }),

  loadProjectState: (state) => {
    // Enrich loaded nodes with icon/color/type from nodeDefinitions.
    // The backend doesn't persist React components (icon) or color,
    // so we reconstruct them from the node's top-level `type` field.
    const enrichedNodes = (Array.isArray(state.nodes) ? state.nodes : []).map(
      (node: Node<BaseNodeData>) => {
        const nodeType = node.data?.type || node.type;
        const nodeDef = nodeType ? getNodeByType(nodeType) : null;
        if (nodeDef) {
          return {
            ...node,
            data: {
              ...node.data,
              type: node.data?.type || nodeDef.type,
              color: node.data?.color || nodeDef.color,
              icon: node.data?.icon || nodeDef.icon,
              label: node.data?.label || nodeDef.label,
            },
          };
        }
        return node;
      }
    );

    set({
      nodes: enrichedNodes,
      edges: Array.isArray(state.edges) ? state.edges : [],
      datasetMetadata: state.datasetMetadata || null,
      executionResult: state.executionResult || null,
    });
  },

  getProjectState: () => {
    const state = get();
    return {
      nodes: state.nodes,
      edges: state.edges,
      datasetMetadata: state.datasetMetadata,
      executionResult: state.executionResult,
    };
  },

  // Utility methods
  getNodeById: (nodeId): Node<BaseNodeData> | undefined => {
    return usePlaygroundStore
      .getState()
      .nodes.find((node: Node<BaseNodeData>) => node.id === nodeId);
  },

  updateNodeConfig: (nodeId, config) => {
    console.log("🏪 Store - updateNodeConfig called for node:", nodeId);
    console.log("🏪 Store - Config received:", config);
    console.log("🏪 Store - target_column in config:", config.target_column);

    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, config, isConfigured: true } }
          : node,
      ),
      // Clear execution results for this node when config changes
      executionResult: state.executionResult
        ? {
            ...state.executionResult,
            nodeResults: state.executionResult.nodeResults
              ? Object.fromEntries(
                  Object.entries(state.executionResult.nodeResults).filter(
                    ([id]) => id !== nodeId,
                  ),
                )
              : {},
          }
        : null,
    }));
  },

  clearAll: () =>
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      datasetMetadata: null,
      executionResult: null,
      executionResults: null,
      nodeExecutionStatus: {},
      executionLogs: [],
    }),

  // Clear
  clearCanvas: () =>
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      executionResults: null,
      nodeExecutionStatus: {},
      executionLogs: [],
    }),
}));
