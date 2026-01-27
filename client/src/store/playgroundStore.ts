/**
 * Zustand store for ML Pipeline Playground
 */

import { create } from "zustand";
import { type Node, type Edge } from "@xyflow/react";
import type { BaseNodeData } from "../types/pipeline";

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
    { success: boolean; output?: unknown; error?: string }
  >;
  metrics?: Record<string, number | string>;
  error?: string;
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
  setEdges: (edges: Edge[]) => void;
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
  currentProjectId: null,

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

  setEdges: (edges) =>
    set(() => ({
      edges,
    })),

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

  // Project state
  setCurrentProjectId: (id) => set({ currentProjectId: id }),

  loadProjectState: (state) =>
    set({
      nodes: Array.isArray(state.nodes) ? state.nodes : [],
      edges: Array.isArray(state.edges) ? state.edges : [],
      datasetMetadata: state.datasetMetadata || null,
      executionResult: state.executionResult || null,
    }),

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

  updateNodeConfig: (nodeId, config) =>
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
    })),

  clearAll: () =>
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      datasetMetadata: null,
      executionResult: null,
      executionResults: null,
    }),

  // Clear
  clearCanvas: () =>
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      executionResults: null,
    }),
}));
