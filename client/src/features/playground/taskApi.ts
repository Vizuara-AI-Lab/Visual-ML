/**
 * API functions for async task execution and monitoring
 */

import apiClient from "../../lib/axios";

export interface TaskStatusResponse {
  task_id: string;
  status: "PENDING" | "PROGRESS" | "SUCCESS" | "FAILURE" | "REVOKED";
  result?: any;
}

export interface AsyncExecutionResponse {
  success: boolean;
  task_id: string;
  message: string;
  status_url: string;
  result_url: string;
  cancel_url: string;
}

export const taskApi = {
  /**
   * Get task status and progress
   */
  getTaskStatus: async (taskId: string): Promise<TaskStatusResponse> => {
    const response = await apiClient.get<TaskStatusResponse>(
      `/tasks/${taskId}/status`,
    );
    return response.data;
  },

  /**
   * Get task final result (only for completed tasks)
   */
  getTaskResult: async (taskId: string): Promise<any> => {
    const response = await apiClient.get(`/tasks/${taskId}/result`);
    return response.data;
  },

  /**
   * Cancel a running task
   */
  cancelTask: async (taskId: string): Promise<{ success: boolean }> => {
    const response = await apiClient.delete(`/tasks/${taskId}`);
    return response.data;
  },

  /**
   * Execute pipeline asynchronously
   */
  executePipelineAsync: async (request: {
    pipeline: Array<{ node_type: string; input: Record<string, any> }>;
    pipeline_name?: string;
    dry_run?: boolean;
  }): Promise<AsyncExecutionResponse> => {
    const response = await apiClient.post<AsyncExecutionResponse>(
      "/ml/pipeline/run-async",
      request,
    );
    return response.data;
  },

  /**
   * Execute single node asynchronously
   */
  executeNodeAsync: async (request: {
    node_type: string;
    input_data: Record<string, any>;
    dry_run?: boolean;
  }): Promise<AsyncExecutionResponse> => {
    const response = await apiClient.post<AsyncExecutionResponse>(
      "/ml/nodes/run-async",
      request,
    );
    return response.data;
  },

  /**
   * Poll task status until completion
   * Returns a promise that resolves when task is complete
   */
  pollTaskUntilComplete: async (
    taskId: string,
    onProgress?: (progress: TaskStatusResponse) => void,
    intervalMs: number = 1000,
  ): Promise<any> => {
    return new Promise((resolve, reject) => {
      const checkStatus = async () => {
        try {
          const status = await taskApi.getTaskStatus(taskId);

          // Call progress callback if provided
          if (onProgress) {
            onProgress(status);
          }

          if (status.status === "SUCCESS") {
            clearInterval(pollInterval);
            resolve(status.result);
          } else if (status.status === "FAILURE") {
            clearInterval(pollInterval);
            reject(new Error(status.result?.error || "Task failed"));
          } else if (status.status === "REVOKED") {
            clearInterval(pollInterval);
            reject(new Error("Task was cancelled"));
          }
          // Continue polling for PENDING and PROGRESS states
        } catch (error) {
          clearInterval(pollInterval);
          reject(error);
        }
      };

      // Initial check
      checkStatus();

      // Set up polling interval
      const pollInterval = setInterval(checkStatus, intervalMs);
    });
  },
};

export default taskApi;
