import { motion, AnimatePresence } from "framer-motion";
import { X, Save } from "lucide-react";
import { useState, useEffect } from "react";
import { getNodeByType } from "../../config/nodeDefinitions";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { UploadDatasetButton } from "./UploadDatasetButton";
import {
  listProjectDatasets,
  type DatasetMetadata,
} from "../../lib/api/datasetApi";

interface ConfigModalProps {
  nodeId: string | null;
  onClose: () => void;
}

export const ConfigModal = ({ nodeId, onClose }: ConfigModalProps) => {
  console.log("🎨 ConfigModal opened for nodeId:", nodeId);

  const {
    getNodeById,
    updateNodeConfig,
    datasetMetadata,
    currentProjectId,
    edges,
  } = usePlaygroundStore();
  const node = nodeId ? getNodeById(nodeId) : null;
  const nodeDef = node ? getNodeByType(node.type) : null;

  console.log("📦 Node:", node);
  console.log("📋 Node definition:", nodeDef);
  console.log("🆔 Current project ID:", currentProjectId);

  const [config, setConfig] = useState<Record<string, unknown>>(
    () => node?.data.config || {},
  );
  const [userDatasets, setUserDatasets] = useState<DatasetMetadata[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);

  // Fetch user's datasets for select_dataset node
  useEffect(() => {
    if (node?.type === "select_dataset" && currentProjectId) {
      setLoadingDatasets(true);
      listProjectDatasets(parseInt(currentProjectId))
        .then((datasets) => {
          console.log("📊 Loaded datasets:", datasets);
          setUserDatasets(datasets);
        })
        .catch((error) => {
          console.error("❌ Failed to load datasets:", error);
        })
        .finally(() => {
          setLoadingDatasets(false);
        });
    }
  }, [node?.type, currentProjectId]);

  if (!node || !nodeDef) return null;

  // Get connected source node for view nodes
  const getConnectedSourceNode = () => {
    const incomingEdge = edges.find((edge) => edge.target === nodeId);
    if (incomingEdge) {
      const sourceNode = getNodeById(incomingEdge.source);
      return sourceNode;
    }
    return null;
  };

  const connectedSourceNode = getConnectedSourceNode();

  // Get available columns from connected source node
  const getAvailableColumns = (): string[] => {
    if (!connectedSourceNode) return [];
    const sourceConfig = connectedSourceNode.data.config;
    return (sourceConfig?.columns as string[]) || [];
  };

  const availableColumns = getAvailableColumns();

  const handleSave = () => {
    updateNodeConfig(node.id, config);
    onClose();
  };

  const handleFieldChange = (fieldName: string, value: unknown) => {
    setConfig((prev) => ({ ...prev, [fieldName]: value }));
  };

  // Auto-fill logic based on dataset metadata
  const getAutoFilledValue = (fieldName: string): unknown => {
    // For view nodes, auto-fill dataset_id from connected source node
    if (fieldName === "dataset_id" && connectedSourceNode) {
      const sourceConfig = connectedSourceNode.data.config;
      if (sourceConfig?.dataset_id) {
        return sourceConfig.dataset_id;
      }
      // If source is an upload node, use its dataset_id
      if (
        connectedSourceNode.type === "upload_file" &&
        sourceConfig?.dataset_id
      ) {
        return sourceConfig.dataset_id;
      }
    }

    if (!datasetMetadata) return undefined;

    if (fieldName === "dataset_path" && datasetMetadata.dataset_id) {
      return datasetMetadata.dataset_id;
    }
    if (fieldName === "target_column" && datasetMetadata.suggested_target) {
      return datasetMetadata.suggested_target;
    }
    return undefined;
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          onClick={onClose}
        />

        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative bg-gray-900 border border-gray-700 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden z-10"
        >
          {/* Header */}
          <div
            className="px-6 py-4 border-b border-gray-700 flex items-center justify-between"
            style={{ borderTopColor: nodeDef.color, borderTopWidth: "3px" }}
          >
            <div className="flex items-center gap-3">
              <div
                className="p-2 rounded-lg"
                style={{ backgroundColor: `${nodeDef.color}20` }}
              >
                <nodeDef.icon
                  className="w-5 h-5"
                  style={{ color: nodeDef.color }}
                />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">
                  {nodeDef.label}
                </h3>
                <p className="text-sm text-gray-400">{nodeDef.description}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 overflow-y-auto max-h-[calc(80vh-180px)]">
            {/* Special handling for upload_file node */}
            {node.type === "upload_file" ? (
              <div className="space-y-4">
                <div className="p-4 bg-blue-900/20 border border-blue-700 rounded-lg">
                  <p className="text-sm text-blue-200 mb-3">
                    Upload a CSV file to use in your ML pipeline
                  </p>
                  {currentProjectId ? (
                    <UploadDatasetButton
                      projectId={parseInt(currentProjectId)}
                      onUploadComplete={(datasetData) => {
                        console.log(
                          "✅ Upload complete, updating config:",
                          datasetData,
                        );
                        // Auto-fill dataset metadata
                        const newConfig = {
                          dataset_id: datasetData.dataset_id,
                          filename: datasetData.filename,
                          n_rows: datasetData.n_rows,
                          n_columns: datasetData.n_columns,
                          columns: datasetData.columns,
                          dtypes: datasetData.dtypes,
                        };
                        setConfig(newConfig);
                        // Auto-save after upload
                        updateNodeConfig(node.id, newConfig);
                      }}
                    />
                  ) : (
                    <div className="text-red-400 text-sm p-3 bg-red-900/20 border border-red-700 rounded">
                      ⚠️ No project selected. Please save your pipeline first.
                    </div>
                  )}
                </div>

                {config.dataset_id && (
                  <div className="p-4 bg-green-900/20 border border-green-700 rounded-lg space-y-2">
                    <h4 className="font-semibold text-green-200 text-sm">
                      ✅ Dataset Loaded
                    </h4>
                    <div className="text-xs text-green-300 space-y-1">
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
                        <strong>Dataset ID:</strong>{" "}
                        {config.dataset_id as string}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : nodeDef.configFields && nodeDef.configFields.length > 0 ? (
              <div className="space-y-4">
                {nodeDef.configFields.map((field) => {
                  const autoFilledValue = field.autoFill
                    ? getAutoFilledValue(field.name)
                    : undefined;
                  const currentValue =
                    config[field.name] ?? autoFilledValue ?? field.defaultValue;

                  // Check if this is a dataset_id field for view nodes (should be read-only)
                  const isDatasetIdField =
                    field.name === "dataset_id" &&
                    [
                      "table_view",
                      "data_preview",
                      "statistics_view",
                      "column_info",
                      "chart_view",
                    ].includes(node.type);

                  return (
                    <div key={field.name}>
                      <label className="block text-sm font-medium text-gray-200 mb-2">
                        {field.label}
                        {field.required && (
                          <span className="text-red-400 ml-1">*</span>
                        )}
                      </label>

                      {field.type === "text" && (
                        <div>
                          <input
                            type="text"
                            value={(currentValue as string) || ""}
                            onChange={(e) =>
                              handleFieldChange(field.name, e.target.value)
                            }
                            readOnly={isDatasetIdField}
                            className={`w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                              isDatasetIdField
                                ? "cursor-not-allowed opacity-75"
                                : ""
                            }`}
                            placeholder={
                              isDatasetIdField && !currentValue
                                ? "Connect a data source node"
                                : field.description
                            }
                          />
                          {isDatasetIdField && connectedSourceNode && (
                            <p className="text-xs text-green-400 mt-1">
                              ✓ Connected to: {connectedSourceNode.data.label}
                            </p>
                          )}
                          {isDatasetIdField && !connectedSourceNode && (
                            <p className="text-xs text-amber-400 mt-1">
                              ⚠ Connect an upload dataset node to this view node
                            </p>
                          )}
                        </div>
                      )}

                      {field.type === "number" && (
                        <input
                          type="number"
                          value={(currentValue as number) || 0}
                          onChange={(e) =>
                            handleFieldChange(
                              field.name,
                              parseFloat(e.target.value),
                            )
                          }
                          min={field.min}
                          max={field.max}
                          step={field.step}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      )}

                      {field.type === "select" && (
                        <div>
                          <select
                            value={(currentValue as string) || ""}
                            onChange={(e) => {
                              const selectedValue = e.target.value;
                              handleFieldChange(field.name, selectedValue);

                              // For select_dataset, also store metadata
                              if (
                                node.type === "select_dataset" &&
                                field.name === "dataset_id"
                              ) {
                                const selectedDataset = userDatasets.find(
                                  (ds) => ds.dataset_id === selectedValue,
                                );
                                if (selectedDataset) {
                                  setConfig((prev) => ({
                                    ...prev,
                                    dataset_id: selectedDataset.dataset_id,
                                    filename: selectedDataset.filename,
                                    n_rows: selectedDataset.n_rows,
                                    n_columns: selectedDataset.n_columns,
                                    columns: selectedDataset.columns,
                                  }));
                                }
                              }
                            }}
                            disabled={
                              loadingDatasets &&
                              node.type === "select_dataset" &&
                              field.name === "dataset_id"
                            }
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <option value="">
                              {loadingDatasets &&
                              node.type === "select_dataset" &&
                              field.name === "dataset_id"
                                ? "Loading datasets..."
                                : `Select ${field.label}`}
                            </option>

                            {/* For select_dataset node, show user's uploaded datasets */}
                            {node.type === "select_dataset" &&
                            field.name === "dataset_id" ? (
                              userDatasets.length > 0 ? (
                                userDatasets.map((dataset) => (
                                  <option
                                    key={dataset.dataset_id}
                                    value={dataset.dataset_id}
                                  >
                                    {dataset.filename} ({dataset.n_rows} rows,{" "}
                                    {dataset.n_columns} cols)
                                  </option>
                                ))
                              ) : (
                                !loadingDatasets && (
                                  <option disabled>
                                    No datasets uploaded yet
                                  </option>
                                )
                              )
                            ) : (
                              <>
                                {/* Regular options from field definition */}
                                {field.options?.map((opt) => (
                                  <option key={opt.value} value={opt.value}>
                                    {opt.label}
                                  </option>
                                ))}
                                {/* Auto-fill columns for target_column field */}
                                {field.autoFill &&
                                  datasetMetadata?.columns &&
                                  field.name === "target_column" &&
                                  datasetMetadata.columns.map((col: string) => (
                                    <option key={col} value={col}>
                                      {col}
                                    </option>
                                  ))}
                                {/* Auto-fill columns for chart view column selection */}
                                {field.autoFill &&
                                  node.type === "chart_view" &&
                                  (field.name === "x_column" ||
                                    field.name === "y_column") &&
                                  availableColumns.map((col: string) => (
                                    <option key={col} value={col}>
                                      {col}
                                    </option>
                                  ))}
                              </>
                            )}
                          </select>

                          {/* Show selected dataset info for select_dataset */}
                          {node.type === "select_dataset" &&
                            field.name === "dataset_id" &&
                            config.dataset_id && (
                              <p className="text-xs text-green-400 mt-1">
                                ✓ Selected: {config.filename as string}
                              </p>
                            )}
                        </div>
                      )}

                      {field.type === "textarea" && (
                        <textarea
                          value={(currentValue as string) || ""}
                          onChange={(e) =>
                            handleFieldChange(field.name, e.target.value)
                          }
                          rows={4}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder={field.description}
                        />
                      )}

                      {field.type === "multiselect" && (
                        <div>
                          <select
                            multiple
                            value={
                              currentValue
                                ? (currentValue as string)
                                    .split(",")
                                    .map((s) => s.trim())
                                : []
                            }
                            onChange={(e) => {
                              const selected = Array.from(
                                e.target.selectedOptions,
                                (option) => option.value,
                              );
                              handleFieldChange(field.name, selected.join(","));
                            }}
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            size={Math.min(availableColumns.length || 5, 8)}
                          >
                            {availableColumns.length > 0 ? (
                              availableColumns.map((col: string) => (
                                <option key={col} value={col}>
                                  {col}
                                </option>
                              ))
                            ) : (
                              <option disabled>
                                Connect a data source to see columns
                              </option>
                            )}
                          </select>
                          {availableColumns.length > 0 && (
                            <p className="text-xs text-gray-400 mt-1">
                              💡 Hold Ctrl/Cmd to select multiple columns
                            </p>
                          )}
                          {!connectedSourceNode && (
                            <p className="text-xs text-amber-400 mt-1">
                              ⚠ Connect a data source node first
                            </p>
                          )}
                        </div>
                      )}

                      {field.type === "checkbox" && (
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={(currentValue as boolean) || false}
                            onChange={(e) =>
                              handleFieldChange(field.name, e.target.checked)
                            }
                            className="w-4 h-4 bg-gray-800 border-gray-600 rounded text-blue-500 focus:ring-2 focus:ring-blue-500"
                          />
                          <span className="text-sm text-gray-400">
                            {field.description}
                          </span>
                        </label>
                      )}

                      {field.type === "file" && (
                        <input
                          type="file"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                              handleFieldChange(field.name, file);
                            }
                          }}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white file:cursor-pointer hover:file:bg-blue-700"
                        />
                      )}

                      {field.description && field.type !== "checkbox" && (
                        <p className="text-xs text-gray-500 mt-1">
                          {field.description}
                        </p>
                      )}

                      {autoFilledValue && (
                        <p className="text-xs text-green-400 mt-1">
                          Auto-filled from dataset
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>No configuration required for this node</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-700 flex items-center justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <Save className="w-4 h-4" />
              Save Configuration
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
