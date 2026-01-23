/**
 * useProjectDatasets - Hook for fetching datasets for a specific project
 */

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../../lib/axios";

interface Dataset {
  id: number;
  dataset_id: string;
  project_id: number;
  filename: string;
  content_type: string;
  file_size: number;
  storage_backend: string;
  n_rows: number;
  n_columns: number;
  columns: string[];
  dtypes: Record<string, string>;
  memory_usage_mb: number;
  preview_data: Array<Record<string, any>>;
  created_at: string;
  updated_at: string;
}

export const useProjectDatasets = (projectId: number | undefined) => {
  return useQuery({
    queryKey: ["datasets", "project", projectId],
    queryFn: async () => {
      if (!projectId) return [];

      const response = await apiClient.get<Dataset[]>(
        `/datasets/project/${projectId}`,
      );
      return response.data;
    },
    enabled: !!projectId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};
