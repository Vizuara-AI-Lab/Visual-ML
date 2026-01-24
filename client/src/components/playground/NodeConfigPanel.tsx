/**
 * Node Configuration Panel - Configure node parameters with validation
 */

import { useState } from "react";
import type { BaseNodeData } from "../../types/pipeline";
import type { Node } from "@xyflow/react";
import { UploadDatasetButton } from "./UploadDatasetButton";
import { usePlaygroundStore } from "../../store/playgroundStore";

interface NodeConfigPanelProps {
  node: Node<BaseNodeData>;
  onUpdate: (nodeId: string, config: Record<string, unknown>) => void;
  onClose: () => void;
}

const NodeConfigPanel = ({ node, onUpdate, onClose }: NodeConfigPanelProps) => {
  console.log("🎛️ NodeConfigPanel rendered for node:", node);
  console.log("📝 Node type:", node.data.type);
  console.log("⚙️ Node data:", node.data);

  const [config, setConfig] = useState<Record<string, unknown>>(
    node.data.config || {},
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const currentProjectId = usePlaygroundStore(
    (state) => state.currentProjectId,
  );

  console.log("🆔 Current project ID:", currentProjectId);

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
    console.log("🎨 Rendering config fields for node type:", nodeData.type);
    console.log("📍 Current project ID:", currentProjectId);

    switch (nodeData.type) {
      case "upload_file":
        console.log("📤 Rendering upload_file config");
        return (
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800 mb-3">
                Upload a CSV file to use in your ML pipeline
              </p>
              {currentProjectId ? (
                <UploadDatasetButton
                  projectId={parseInt(currentProjectId)}
                  onUploadComplete={(datasetData) => {
                    console.log(
                      "✅ Upload complete, auto-filling config:",
                      datasetData,
                    );
                    // Auto-fill dataset metadata
                    setConfig({
                      dataset_id: datasetData.dataset_id,
                      filename: datasetData.filename,
                      n_rows: datasetData.n_rows,
                      n_columns: datasetData.n_columns,
                      columns: datasetData.columns,
                      dtypes: datasetData.dtypes,
                    });
                  }}
                />
              ) : (
                <div className="text-red-600 text-sm">
                  No project selected. Please save your pipeline first.
                </div>
              )}
            </div>

            {config.dataset_id && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg space-y-2">
                <h4 className="font-semibold text-green-900 text-sm">
                  Dataset Loaded
                </h4>
                <div className="text-xs text-green-800 space-y-1">
                  <p>
                    <strong>File:</strong> {config.filename as string}
                  </p>
                  <p>
                    <strong>Rows:</strong> {config.n_rows as number}
                  </p>
                  <p>
                    <strong>Columns:</strong> {config.n_columns as number}
                  </p>
                  <p>
                    <strong>Dataset ID:</strong> {config.dataset_id as string}
                  </p>
                </div>
              </div>
            )}
          </div>
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
            {renderField("model_name", "Model Name (Optional)")}
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
