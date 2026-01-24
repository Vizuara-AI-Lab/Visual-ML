/**
 * useAllDatasets - Hook for fetching all datasets for the current student
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

interface AllDatasetsResponse {
  datasets: Dataset[];
  total: number;
  limit: number;
  offset: number;
}

export const useAllDatasets = (limit = 50, offset = 0) => {
  return useQuery({
    queryKey: ["datasets", "all", limit, offset],
    queryFn: async () => {
      const response = await apiClient.get<AllDatasetsResponse>(
        `/datasets?limit=${limit}&offset=${offset}`,
      );
      return response.data;
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};
