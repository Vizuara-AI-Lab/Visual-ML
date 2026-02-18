/**
 * Upload Dataset Button - Handle file upload with progress
 */

import { useState, useRef } from "react";
import { Upload, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { uploadDataset } from "../../lib/api/datasetApi";

interface UploadDatasetButtonProps {
  nodeId: string;
  projectId: number;
  onUploadComplete?: (datasetData: {
    dataset_id: string;
    filename: string;
    n_rows: number;
    n_columns: number;
    columns: string[];
    dtypes: Record<string, string>;
  }) => void;
}

export const UploadDatasetButton = ({
  projectId,
  onUploadComplete,
}: UploadDatasetButtonProps) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (!file) {
      console.log("❌ No file selected");
      return;
    }

    // Validate file type
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Only CSV files are allowed");
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      setError("File size must be less than 100MB");
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(false);
    setProgress(0);

    try {
      const result = await uploadDataset(file, projectId, (progress) => {
        setProgress(progress);
      });

      setSuccess(true);
      setProgress(100);

      // Call callback with dataset info
      if (onUploadComplete) {
        onUploadComplete({
          dataset_id: result.dataset.dataset_id,
          filename: result.dataset.filename,
          n_rows: result.dataset.n_rows,
          n_columns: result.dataset.n_columns,
          columns: result.dataset.columns,
          dtypes: result.dataset.dtypes,
        });
      }

      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      // Reset success state after 3 seconds
      setTimeout(() => {
        setSuccess(false);
        setProgress(0);
      }, 3000);
    } catch (err: any) {
      console.error("Upload failed:", err);

      // Handle specific error codes
      if (err.response?.status === 401) {
        setError("Please log in to upload datasets");
      } else if (err.response?.status === 403) {
        setError("Your account is inactive. Please contact support.");
      } else if (err.response?.status === 404) {
        setError("Project not found or you don't have permission");
      } else if (err.response?.status === 400) {
        setError(err.response?.data?.detail || "Invalid file format");
      } else {
        setError(
          err.response?.data?.detail ||
            err.message ||
            "Upload failed. Please try again.",
        );
      }

      setProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-2">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={(e) => {
          console.log("🔄 File input onChange triggered!");
          handleFileSelect(e);
        }}
        className="hidden"
        onClick={() => console.log("🖱️ File input clicked!")}
      />

      <button
        onClick={(e) => {
          console.log("🔘 Button onClick event fired!");
          e.preventDefault();
          e.stopPropagation();
          handleClick();
        }}
        disabled={uploading}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        {uploading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Uploading... {progress}%
          </>
        ) : success ? (
          <>
            <CheckCircle className="w-4 h-4" />
            Upload Complete!
          </>
        ) : (
          <>
            <Upload className="w-4 h-4" />
            Upload CSV File
          </>
        )}
      </button>

      {uploading && (
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {success && !uploading && (
        <div className="flex items-start gap-2 p-3 bg-green-50 border border-green-200 rounded-lg text-green-700 text-sm">
          <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>Dataset uploaded successfully!</span>
        </div>
      )}
    </div>
  );
};
