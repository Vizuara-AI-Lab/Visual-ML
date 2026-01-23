/**
 * Node Configuration Panel - Configure node parameters with validation
 */

import { useState } from "react";
import type { BaseNodeData } from "../../types/pipeline";
import type { Node } from "@xyflow/react";
import { useUploadDataset } from "../../hooks/mutations/useUploadDataset";
import { useParams } from "react-router-dom";

interface NodeConfigPanelProps {
  node: Node<BaseNodeData>;
  onUpdate: (nodeId: string, config: Record<string, unknown>) => void;
  onClose: () => void;
}

const NodeConfigPanel = ({ node, onUpdate, onClose }: NodeConfigPanelProps) => {
  const [config, setConfig] = useState<Record<string, unknown>>(
    node.data.config || {},
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [uploading, setUploading] = useState(false);

  const { id: projectId } = useParams<{ id: string }>();
  const uploadDataset = useUploadDataset();

  const nodeData = node.data as BaseNodeData;

  const updateField = (field: string, value: unknown) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors((prev) => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
  };

  const validateConfig = (): boolean => {
    const newErrors: Record<string, string> = {};

    switch (nodeData.type) {
      case "upload_file":
        if (!config.filename) {
          newErrors.filename = "Filename is required";
        }
        break;

      case "preprocess":
        if (!config.dataset_path) {
          newErrors.dataset_path = "Dataset path is required";
        }
        if (!config.target_column) {
          newErrors.target_column = "Target column is required";
        }
        if (
          config.missing_strategy === "fill" &&
          config.fill_value === undefined
        ) {
          newErrors.fill_value =
            'Fill value is required when strategy is "fill"';
        }
        break;

      case "split": {
        if (!config.dataset_path) {
          newErrors.dataset_path = "Dataset path is required";
        }
        if (!config.target_column) {
          newErrors.target_column = "Target column is required";
        }
        const totalRatio =
          (config.train_ratio as number) +
          ((config.val_ratio as number) || 0) +
          (config.test_ratio as number);
        if (Math.abs(totalRatio - 1.0) > 0.01) {
          newErrors.ratios = "Ratios must sum to 1.0";
        }
        break;
      }

      case "train":
        if (!config.train_dataset_path) {
          newErrors.train_dataset_path = "Training dataset path is required";
        }
        if (!config.target_column) {
          newErrors.target_column = "Target column is required";
        }
        if (!config.algorithm) {
          newErrors.algorithm = "Algorithm is required";
        }
        if (!config.task_type) {
          newErrors.task_type = "Task type is required";
        }
        break;

      case "evaluate":
        if (!config.model_path) {
          newErrors.model_path = "Model path is required";
        }
        if (!config.test_dataset_path) {
          newErrors.test_dataset_path = "Test dataset path is required";
        }
        if (!config.target_column) {
          newErrors.target_column = "Target column is required";
        }
        break;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = () => {
    if (validateConfig()) {
      onUpdate(node.id, config);
      onClose();
    }
  };

  const renderField = (
    field: string,
    label: string,
    type: string = "text",
    options?:
      | { value: string; label: string }[]
      | { step?: number; min?: number; max?: number },
  ) => {
    const error = errors[field];

    if (type === "select") {
      return (
        <div key={field} className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {label}
          </label>
          <select
            value={(config[field] as string) || ""}
            onChange={(e) => updateField(field, e.target.value)}
            className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              error ? "border-red-500" : "border-gray-300"
            }`}
          >
            {Array.isArray(options) &&
              options.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
          </select>
          {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
        </div>
      );
    }

    if (type === "checkbox") {
      return (
        <div key={field} className="mb-4">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={(config[field] as boolean) || false}
              onChange={(e) => updateField(field, e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm font-medium text-gray-700">{label}</span>
          </label>
          {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
        </div>
      );
    }

    if (type === "number") {
      const numOptions = options && !Array.isArray(options) ? options : {};
      return (
        <div key={field} className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {label}
          </label>
          <input
            type="number"
            value={(config[field] as number) ?? ""}
            onChange={(e) =>
              updateField(field, parseFloat(e.target.value) || 0)
            }
            step={numOptions.step || 0.01}
            min={numOptions.min}
            max={numOptions.max}
            className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              error ? "border-red-500" : "border-gray-300"
            }`}
          />
          {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
        </div>
      );
    }

    if (type === "textarea") {
      return (
        <div key={field} className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {label}
          </label>
          <textarea
            value={(config[field] as string) || ""}
            onChange={(e) => updateField(field, e.target.value)}
            rows={3}
            className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              error ? "border-red-500" : "border-gray-300"
            }`}
          />
          {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
        </div>
      );
    }

    return (
      <div key={field} className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
        <input
          type={type}
          value={(config[field] as string) || ""}
          onChange={(e) => updateField(field, e.target.value)}
          className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            error ? "border-red-500" : "border-gray-300"
          }`}
        />
        {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
      </div>
    );
  };

  const renderConfigFields = () => {
    switch (nodeData.type) {
      case "upload_file":
        return (
          <>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Upload CSV File
              </label>
              <input
                type="file"
                accept=".csv"
                disabled={uploading}
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (file && projectId) {
                    try {
                      setUploading(true);
                      setErrors({});

                      // Upload to S3 via API
                      const result = await uploadDataset.mutateAsync({
                        file,
                        projectId: parseInt(projectId),
                      });

                      // Update node config with dataset info
                      updateField("filename", result.dataset.filename);
                      updateField("dataset_id", result.dataset.dataset_id);
                      updateField("file_path", result.dataset.file_path);
                      updateField(
                        "storage_backend",
                        result.dataset.storage_backend,
                      );
                      updateField("s3_key", result.dataset.s3_key);
                      updateField("n_rows", result.dataset.n_rows);
                      updateField("n_columns", result.dataset.n_columns);
                      updateField("columns", result.dataset.columns);
                      updateField("uploaded", true);

                      setUploading(false);
                    } catch (error: any) {
                      setUploading(false);
                      setErrors({
                        file: error.response?.data?.detail || "Upload failed",
                      });
                    }
                  }
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
              />
              {uploading && (
                <p className="mt-1 text-xs text-blue-600">
                  Uploading to S3... Please wait.
                </p>
              )}
              {errors.file && (
                <p className="mt-1 text-xs text-red-600">{errors.file}</p>
              )}
              {config.uploaded && (
                <div className="mt-2 p-2 bg-green-50 border border-green-200 rounded text-xs">
                  ✓ Uploaded: {config.filename as string}
                  <br />
                  Dataset ID: {config.dataset_id as string}
                  <br />
                  Rows: {config.n_rows as number}, Columns:{" "}
                  {config.n_columns as number}
                  <br />
                  Storage: {config.storage_backend as string}
                </div>
              )}
            </div>
            {config.uploaded && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dataset Info
                </label>
                <textarea
                  value={(config.dataset_info as string) || ""}
                  onChange={(e) => updateField("dataset_info", e.target.value)}
                  rows={2}
                  placeholder="Optional: Add notes about this dataset"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            )}
          </>
        );

      case "preprocess":
        return (
          <>
            {renderField("dataset_path", "Dataset Path")}
            {renderField("target_column", "Target Column")}
            {renderField("handle_missing", "Handle Missing Values", "checkbox")}
            {renderField("missing_strategy", "Missing Strategy", "select", [
              { value: "drop", label: "Drop" },
              { value: "mean", label: "Mean" },
              { value: "median", label: "Median" },
              { value: "mode", label: "Mode" },
              { value: "fill", label: "Fill Value" },
            ])}
            {config.missing_strategy === "fill" &&
              renderField("fill_value", "Fill Value", "number")}
            {renderField(
              "scale_features",
              "Scale Numeric Features",
              "checkbox",
            )}
            {renderField("max_features", "Max TF-IDF Features", "number")}
          </>
        );

      case "split":
        return (
          <>
            {renderField("dataset_path", "Dataset Path")}
            {renderField("target_column", "Target Column")}
            {renderField("train_ratio", "Train Ratio", "number", {
              step: 0.05,
              min: 0,
              max: 1,
            })}
            {renderField("val_ratio", "Validation Ratio", "number", {
              step: 0.05,
              min: 0,
              max: 1,
            })}
            {renderField("test_ratio", "Test Ratio", "number", {
              step: 0.05,
              min: 0,
              max: 1,
            })}
            {errors.ratios && (
              <p className="mb-4 text-xs text-red-600">{errors.ratios}</p>
            )}
            {renderField("random_seed", "Random Seed", "number")}
            {renderField("shuffle", "Shuffle Data", "checkbox")}
            {renderField("stratify", "Stratify Split", "checkbox")}
          </>
        );

      case "train":
        return (
          <>
            {renderField("train_dataset_path", "Training Dataset Path")}
            {renderField("target_column", "Target Column")}
            {renderField("task_type", "Task Type", "select", [
              { value: "regression", label: "Regression" },
              { value: "classification", label: "Classification" },
            ])}
            {renderField(
              "algorithm",
              "Algorithm",
              "select",
              config.task_type === "classification"
                ? [
                    {
                      value: "logistic_regression",
                      label: "Logistic Regression",
                    },
                  ]
                : [{ value: "linear_regression", label: "Linear Regression" }],
            )}
            {renderField("model_name", "Model Name (Optional)")}

            {/* Hyperparameters */}
            <div className="mb-4 p-3 bg-gray-50 rounded-md">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">
                Hyperparameters
              </h4>
              {config.algorithm === "linear_regression" && (
                <>
                  <div className="mb-2">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        checked={
                          (config.hyperparameters as any)?.fit_intercept ?? true
                        }
                        onChange={(e) =>
                          updateField("hyperparameters", {
                            ...(config.hyperparameters as object),
                            fit_intercept: e.target.checked,
                          })
                        }
                        className="rounded border-gray-300"
                      />
                      <span className="text-sm text-gray-700">
                        Fit Intercept
                      </span>
                    </label>
                  </div>
                </>
              )}
              {config.algorithm === "logistic_regression" && (
                <>
                  <div className="mb-2">
                    <label className="block text-xs text-gray-600 mb-1">
                      C (Inverse Regularization)
                    </label>
                    <input
                      type="number"
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      value={(config.hyperparameters as any)?.C ?? 1.0}
                      onChange={(e) =>
                        updateField("hyperparameters", {
                          ...(config.hyperparameters as object),
                          C: parseFloat(e.target.value) || 1.0,
                        })
                      }
                      step="0.1"
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded"
                    />
                  </div>
                  <div className="mb-2">
                    <label className="block text-xs text-gray-600 mb-1">
                      Max Iterations
                    </label>
                    <input
                      type="number"
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      value={(config.hyperparameters as any)?.max_iter ?? 1000}
                      onChange={(e) =>
                        updateField("hyperparameters", {
                          ...(config.hyperparameters as object),
                          max_iter: parseInt(e.target.value) || 1000,
                        })
                      }
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded"
                    />
                  </div>
                </>
              )}
            </div>
          </>
        );

      case "evaluate":
        return (
          <>
            {renderField("model_path", "Model Path")}
            {renderField("test_dataset_path", "Test Dataset Path")}
            {renderField("target_column", "Target Column")}
            {renderField("task_type", "Task Type", "select", [
              { value: "regression", label: "Regression" },
              { value: "classification", label: "Classification" },
            ])}
          </>
        );

      default:
        return <div>Unknown node type</div>;
    }
  };

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-white border-l border-gray-200 shadow-lg overflow-y-auto z-50">
      <div className="sticky top-0 bg-white border-b border-gray-200 px-4 py-3">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-gray-800">
            Configure {nodeData.label}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
          >
            ×
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-1">Type: {nodeData.type}</p>
      </div>

      <div className="p-4">{renderConfigFields()}</div>

      <div className="sticky bottom-0 bg-white border-t border-gray-200 px-4 py-3 flex gap-2">
        <button
          onClick={handleSave}
          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium"
        >
          Save Configuration
        </button>
        <button
          onClick={onClose}
          className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default NodeConfigPanel;
