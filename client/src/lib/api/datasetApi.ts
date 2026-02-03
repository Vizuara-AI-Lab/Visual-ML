/**
 * Dataset API - Upload and manage datasets
 */

import axios from "../axios";

export interface UploadDatasetResponse {
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
  };
}

export interface DatasetMetadata {
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
  created_at: string;
  updated_at: string;
}

/**
 * Upload dataset file to S3 and save metadata
 */
export const uploadDataset = async (
  file: File,
  projectId: number,
  onProgress?: (progress: number) => void,
): Promise<UploadDatasetResponse> => {
  console.log("📤 uploadDataset called with:", {
    fileName: file.name,
    fileSize: file.size,
    projectId,
  });

  const formData = new FormData();
  formData.append("file", file);

  console.log("📦 FormData created, sending POST request...");

  const response = await axios.post<UploadDatasetResponse>(
    `/datasets/upload?project_id=${projectId}`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total,
          );
          console.log("📊 Progress:", percentCompleted + "%");
          onProgress(percentCompleted);
        }
      },
    },
  );

  console.log("✅ Upload response received:", response.data);
  return response.data;
};

/**
 * List all datasets for a project
 */
export const listProjectDatasets = async (
  projectId: number,
): Promise<DatasetMetadata[]> => {
  const response = await axios.get<DatasetMetadata[]>(
    `/datasets/project/${projectId}`,
  );
  return response.data;
};

/**
 * Get dataset metadata by ID
 */
export const getDataset = async (
  datasetId: string,
): Promise<DatasetMetadata> => {
  const response = await axios.get<DatasetMetadata>(`/datasets/${datasetId}`);
  return response.data;
};

/**
 * Delete dataset
 */
export const deleteDataset = async (
  datasetId: string,
  permanent = false,
): Promise<{ message: string }> => {
  const response = await axios.delete<{ message: string }>(
    `/datasets/${datasetId}?permanent=${permanent}`,
  );
  return response.data;
};

/**
 * Download dataset as CSV
 */
export const downloadDataset = async (datasetId: string): Promise<void> => {
  try {
    const response = await axios.get(`/datasets/${datasetId}/download`, {
      responseType: "blob",
    });

    // Create blob link to download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;

    // Get filename from content-disposition header or use default
    const contentDisposition = response.headers["content-disposition"];
    let filename = `${datasetId}.csv`;
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
      if (filenameMatch) {
        filename = filenameMatch[1];
      }
    }

    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error("Download failed:", error);
    throw error;
  }
};

/**
 * Download file from uploads folder (for processed datasets like scaled, encoded, etc.)
 */
export const downloadFileFromUploads = async (
  fileId: string,
): Promise<void> => {
  try {
    const response = await axios.get(`/datasets/file/${fileId}/download`, {
      responseType: "blob",
    });

    // Create blob link to download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;

    // Get filename from content-disposition header or use file ID
    const contentDisposition = response.headers["content-disposition"];
    let filename = `${fileId}.csv`;
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
      if (filenameMatch) {
        filename = filenameMatch[1];
      }
    }

    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error("Download from uploads failed:", error);
    throw error;
  }
};

/**
 * List all datasets for the current user
 */
export const listAllUserDatasets = async (
  limit = 50,
  offset = 0,
): Promise<{
  datasets: DatasetMetadata[];
  total: number;
  limit: number;
  offset: number;
}> => {
  const response = await axios.get<{
    datasets: DatasetMetadata[];
    total: number;
    limit: number;
    offset: number;
  }>(`/datasets?limit=${limit}&offset=${offset}`);
  return response.data;
};
