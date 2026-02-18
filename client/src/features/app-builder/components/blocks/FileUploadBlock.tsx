/**
 * File Upload Block — CSV file upload widget.
 * In "edit"/"preview" mode: static display. In "live" mode: interactive upload.
 */

import { useCallback, useState } from "react";
import { Upload, File, X } from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { FileUploadConfig } from "../../types/appBuilder";

export default function FileUploadBlock({
  block,
  mode,
  theme,
  onFileUpload,
}: BlockRenderProps) {
  const config = block.config as FileUploadConfig;
  const [fileName, setFileName] = useState<string | null>(null);

  // Resolve nodeId from block-level or config-level mapping
  const uploadNodeId = block.nodeId || config.nodeId;

  const handleFile = useCallback(
    (file: globalThis.File) => {
      if (!onFileUpload) return;
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = (reader.result as string).split(",")[1];
        onFileUpload(base64, file.name, uploadNodeId);
        setFileName(file.name);
      };
      reader.readAsDataURL(file);
    },
    [onFileUpload, uploadNodeId],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const isInteractive = mode === "live";

  return (
    <div className="bg-white rounded-xl border p-6">
      <label className="block text-sm font-medium text-gray-700 mb-3">
        {config.label}
      </label>

      <div
        onDrop={isInteractive ? handleDrop : undefined}
        onDragOver={isInteractive ? (e) => e.preventDefault() : undefined}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isInteractive
            ? "border-gray-300 hover:border-indigo-400 cursor-pointer"
            : "border-gray-200"
        }`}
      >
        {fileName ? (
          <div className="flex items-center justify-center gap-2">
            <File className="h-5 w-5 text-indigo-600" />
            <span className="text-sm text-gray-700">{fileName}</span>
            {isInteractive && (
              <button
                onClick={() => {
                  setFileName(null);
                }}
                className="p-1 rounded hover:bg-gray-100"
              >
                <X className="h-4 w-4 text-gray-400" />
              </button>
            )}
          </div>
        ) : (
          <>
            <Upload className="h-8 w-8 mx-auto text-gray-400 mb-2" />
            <p className="text-sm text-gray-500 mb-1">
              {isInteractive ? "Drop a file here or click to upload" : "File upload area"}
            </p>
            {config.helpText && (
              <p className="text-xs text-gray-400">{config.helpText}</p>
            )}
          </>
        )}

        {isInteractive && !fileName && (
          <input
            type="file"
            accept={config.acceptTypes}
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            style={{ position: "absolute", inset: 0 }}
          />
        )}
      </div>
    </div>
  );
}
