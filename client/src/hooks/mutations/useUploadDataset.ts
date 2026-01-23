/**
 * useUploadDataset - Hook for uploading datasets to S3
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../../lib/axios";

interface UploadDatasetParams {
  file: File;
  projectId: number;
}

interface UploadDatasetResponse {
  success: boolean;
  message: string;
  dataset: {
    dataset_id: string;
    filename: string;
    file_path: string;
    storage_backend: string;
    s3_bucket?: string;
    s3_key?: string;
    n_rows: number;
    n_columns: number;
    columns: string[];
    dtypes: Record<string, string>;
    memory_usage_mb: number;
    file_size: number;
    preview: Array<Record<string, any>>;
  };
}

export const useUploadDataset = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ file, projectId }: UploadDatasetParams) => {
      const formData = new FormData();
      formData.append("file", file);

      const response = await apiClient.post<UploadDatasetResponse>(
        `/datasets/upload?project_id=${projectId}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );

      return response.data;
    },
    onSuccess: (data, variables) => {
      // Invalidate project datasets query to refetch
      queryClient.invalidateQueries({
        queryKey: ["datasets", "project", variables.projectId],
      });

      // Invalidate all datasets query
      queryClient.invalidateQueries({
        queryKey: ["datasets"],
      });
    },
  });
};
