/**
 * API functions for ML Pipeline execution
 */

import apiClient from "../../lib/axios";
import type {
  NodeExecuteRequest,
  NodeExecuteResponse,
  PipelineExecuteRequest,
  PipelineExecuteResponse,
} from "../../types/pipeline";

export const pipelineApi = {
  /**
   * Execute a single node
   */
  executeNode: async (
    request: NodeExecuteRequest,
  ): Promise<NodeExecuteResponse> => {
    const response = await apiClient.post<NodeExecuteResponse>(
      "/ml/nodes/run",
      request,
    );
    return response.data;
  },

  /**
   * Execute a complete pipeline
   */
  executePipeline: async (
    request: PipelineExecuteRequest,
  ): Promise<PipelineExecuteResponse> => {
    const response = await apiClient.post<PipelineExecuteResponse>(
      "/ml/pipeline/run",
      request,
    );
    return response.data;
  },

  /**
   * Validate a node configuration (dry run)
   */
  validateNode: async (
    request: NodeExecuteRequest,
  ): Promise<NodeExecuteResponse> => {
    const response = await apiClient.post<NodeExecuteResponse>(
      "/ml/nodes/run",
      {
        ...request,
        dry_run: true,
      },
    );
    return response.data;
  },

  /**
   * Validate a pipeline (dry run)
   */
  validatePipeline: async (
    request: PipelineExecuteRequest,
  ): Promise<PipelineExecuteResponse> => {
    const response = await apiClient.post<PipelineExecuteResponse>(
      "/ml/pipeline/run",
      {
        ...request,
        dry_run: true,
      },
    );
    return response.data;
  },
};

// Named exports for convenience
export const executePipeline = pipelineApi.executePipeline;
export const executeNode = pipelineApi.executeNode;
export const validatePipeline = pipelineApi.validatePipeline;
export const validateNode = pipelineApi.validateNode;
