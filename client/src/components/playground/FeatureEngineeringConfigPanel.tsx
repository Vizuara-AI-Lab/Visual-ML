/**
 * Feature Engineering Configuration Panel
 * Handles configuration for feature engineering nodes:
 * - missing_value_handler
 * - encoding
 * - transformation
 * - scaling
 * - feature_selection
 */

import { type Dispatch, type SetStateAction } from "react";
import type { Node } from "@xyflow/react";
import type { BaseNodeData } from "../../types/pipeline";

interface FeatureEngineeringConfigPanelProps {
  node: Node<BaseNodeData>;
  config: Record<string, unknown>;
  availableColumns: string[];
  updateField: (field: string, value: unknown) => void;
  setConfig: Dispatch<SetStateAction<Record<string, unknown>>>;
  renderField: (
    field: string,
    label: string,
    type?: string,
    options?:
      | { value: string; label: string }[]
      | { step?: number; min?: number; max?: number },
  ) => React.JSX.Element;
}

export const FeatureEngineeringConfigPanel = ({
  node,
  config,
  availableColumns,
  updateField,
  setConfig,
  renderField,
}: FeatureEngineeringConfigPanelProps) => {
  const nodeData = node.data as BaseNodeData;

  // Debug logging for column selection
  console.log("🔍 FeatureEngineeringConfigPanel:", {
    nodeType: nodeData.type,
    availableColumns,
    selectedColumns: config.columns,
    fullConfig: config,
  });

  switch (nodeData.type) {
    case "missing_value_handler":
      return (
        <div className="space-y-4">
          <div className="p-4 bg-pink-50 border border-pink-200 rounded-lg">
            <p className="text-sm text-pink-800 mb-3">
              🔧 Handle missing values with column-wise control
            </p>
            <p className="text-xs text-pink-700">
              Configure different strategies for each column
            </p>
          </div>
          {renderField("dataset_id", "Dataset Source", "text")}
          {renderField("default_strategy", "Default Strategy", "select", [
            { value: "none", label: "No Action" },
            { value: "drop", label: "Drop Rows" },
            { value: "mean", label: "Fill with Mean" },
            { value: "median", label: "Fill with Median" },
            { value: "mode", label: "Fill with Mode" },
            { value: "fill", label: "Fill with Value" },
            { value: "forward_fill", label: "Forward Fill" },
            { value: "backward_fill", label: "Backward Fill" },
          ])}
          {renderField("preview_mode", "Preview Mode", "checkbox")}
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-800">
              💡 <strong>Tip:</strong> Column-wise configuration will be
              available after connecting to a dataset
            </p>
          </div>
          {(config.dataset_id as string) && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800">
                ✓ Ready to handle missing values on pipeline execution
              </p>
            </div>
          )}
        </div>
      );

    case "encoding": {
      // Get column configs or initialize
      const columnConfigs =
        (config.column_configs as Record<string, Record<string, unknown>>) ||
        {};

      const handleEncodingMethodChange = (column: string, method: string) => {
        const updatedConfigs = { ...columnConfigs };
        if (!updatedConfigs[column]) {
          updatedConfigs[column] = {};
        }
        updatedConfigs[column] = {
          ...updatedConfigs[column],
          column_name: column,
          encoding_method: method,
          handle_unknown: updatedConfigs[column]?.handle_unknown || "error",
          handle_missing: updatedConfigs[column]?.handle_missing || "error",
          drop_first: updatedConfigs[column]?.drop_first || false,
        };
        setConfig((prev) => ({ ...prev, column_configs: updatedConfigs }));
      };

      const handleEncodingOptionChange = (
        column: string,
        option: string,
        value: unknown,
      ) => {
        const updatedConfigs = { ...columnConfigs };
        if (updatedConfigs[column]) {
          updatedConfigs[column][option] = value;
          setConfig((prev) => ({ ...prev, column_configs: updatedConfigs }));
        }
      };

      return (
        <div className="space-y-4">
          <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <p className="text-sm text-amber-800 mb-3">
              🔧 Encode categorical variables with per-column control
            </p>
            <p className="text-xs text-amber-700">
              Configure encoding method for each column individually
            </p>
          </div>

          {renderField("dataset_id", "Dataset Source", "text")}

          {/* Debug info */}
          {availableColumns.length > 0 && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-xs text-green-800">
                ✓ Found {availableColumns.length} columns:{" "}
                {availableColumns.join(", ")}
              </p>
            </div>
          )}
          {availableColumns.length === 0 && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-xs text-yellow-800">
                ⚠️ No columns detected. Please ensure the connected node has
                selected a dataset.
              </p>
            </div>
          )}

          {availableColumns.length > 0 ? (
            <>
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">
                  Column Encoding Configuration
                </h4>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {availableColumns.map((column) => (
                    <div
                      key={column}
                      className="border border-gray-200 rounded-lg p-3 bg-gray-50"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm text-gray-700">
                          {column}
                        </span>
                        <span className="text-xs text-gray-500">
                          {String(
                            columnConfigs[column]?.encoding_method || "none",
                          )}
                        </span>
                      </div>

                      <div className="space-y-2">
                        {/* Encoding Method */}
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">
                            Encoding Method
                          </label>
                          <select
                            value={String(
                              columnConfigs[column]?.encoding_method || "none",
                            )}
                            onChange={(e) =>
                              handleEncodingMethodChange(column, e.target.value)
                            }
                            className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-amber-500"
                          >
                            <option value="none">None (Skip)</option>
                            <option value="onehot">One-Hot Encoding</option>
                            <option value="label">Label Encoding</option>
                          </select>
                        </div>

                        {/* Show additional options only if encoding method is selected */}
                        {(() => {
                          const method = columnConfigs[column]?.encoding_method;
                          return method && String(method) !== "none" ? (
                            <>
                              {/* Handle Unknown */}
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <label className="block text-xs font-medium text-gray-600 mb-1">
                                    Handle Unknown
                                  </label>
                                  <select
                                    value={String(
                                      columnConfigs[column]?.handle_unknown ||
                                        "error",
                                    )}
                                    onChange={(e) =>
                                      handleEncodingOptionChange(
                                        column,
                                        "handle_unknown",
                                        e.target.value,
                                      )
                                    }
                                    className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-amber-400"
                                  >
                                    <option value="error">Error</option>
                                    <option value="ignore">Ignore</option>
                                    <option value="use_encoded_value">
                                      Use Encoded Value
                                    </option>
                                  </select>
                                </div>

                                {/* Handle Missing */}
                                <div>
                                  <label className="block text-xs font-medium text-gray-600 mb-1">
                                    Handle Missing
                                  </label>
                                  <select
                                    value={String(
                                      columnConfigs[column]?.handle_missing ||
                                        "error",
                                    )}
                                    onChange={(e) =>
                                      handleEncodingOptionChange(
                                        column,
                                        "handle_missing",
                                        e.target.value,
                                      )
                                    }
                                    className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-amber-400"
                                  >
                                    <option value="error">Error</option>
                                    <option value="most_frequent">
                                      Most Frequent
                                    </option>
                                    <option value="create_category">
                                      Create Category
                                    </option>
                                  </select>
                                </div>
                              </div>

                              {/* Drop First (only for one-hot) */}
                              {String(
                                columnConfigs[column]?.encoding_method,
                              ) === "onehot" && (
                                <div className="flex items-center">
                                  <input
                                    type="checkbox"
                                    checked={Boolean(
                                      columnConfigs[column]?.drop_first,
                                    )}
                                    onChange={(e) =>
                                      handleEncodingOptionChange(
                                        column,
                                        "drop_first",
                                        e.target.checked,
                                      )
                                    }
                                    className="mr-2"
                                  />
                                  <label className="text-xs text-gray-600">
                                    Drop First Category
                                  </label>
                                </div>
                              )}
                            </>
                          ) : null;
                        })()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-xs text-blue-800">
                💡 <strong>Connect to a dataset</strong> to configure column
                encoding
              </p>
            </div>
          )}

          {(config.dataset_id as string) && availableColumns.length > 0 && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800">
                ✓ Ready to encode{" "}
                {
                  Object.keys(columnConfigs).filter(
                    (col) => columnConfigs[col]?.encoding_method !== "none",
                  ).length
                }{" "}
                column(s)
              </p>
            </div>
          )}
        </div>
      );
    }

    case "transformation":
      return (
        <div className="space-y-4">
          <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
            <p className="text-sm text-purple-800 mb-3">
              ⚡ Apply mathematical transformations to features
            </p>
            <p className="text-xs text-purple-700">
              Transform data using log, sqrt, or power methods
            </p>
          </div>
          {renderField("dataset_id", "Dataset Source", "text")}
          {renderField("transformation_type", "Transformation Type", "select", [
            { value: "log", label: "Log Transform" },
            { value: "sqrt", label: "Square Root" },
            { value: "power", label: "Power Transform (Box-Cox)" },
          ])}
          {availableColumns.length > 0 && (
            <>
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-gray-800">
                    Select Columns to Transform
                  </h4>
                  <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded-full font-medium">
                    {((config.columns as string[]) || []).length} selected
                  </span>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {availableColumns.map((column) => {
                    const isSelected = (
                      (config.columns as string[]) || []
                    ).includes(column);
                    return (
                      <label
                        key={column}
                        className={`flex items-center gap-2 p-2 rounded transition-colors ${
                          isSelected
                            ? "bg-purple-50 border border-purple-200"
                            : "hover:bg-gray-50"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={(e) => {
                            const currentColumns =
                              (config.columns as string[]) || [];
                            const newColumns = e.target.checked
                              ? [...currentColumns, column]
                              : currentColumns.filter((c) => c !== column);
                            updateField("columns", newColumns);
                          }}
                          className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                        />
                        <span
                          className={`text-sm ${
                            isSelected
                              ? "text-purple-900 font-medium"
                              : "text-gray-700"
                          }`}
                        >
                          {column}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
              {((config.columns as string[]) || []).length > 0 && (
                <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
                  <p className="text-xs text-purple-900 font-medium mb-1">
                    ✓ Selected columns:
                  </p>
                  <p className="text-xs text-purple-700">
                    {((config.columns as string[]) || []).join(", ")}
                  </p>
                </div>
              )}
            </>
          )}
          {(config.dataset_id as string) && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800">
                ✓ Ready to transform{" "}
                {((config.columns as string[]) || []).length} column(s)
              </p>
            </div>
          )}
        </div>
      );

    case "scaling":
      return (
        <div className="space-y-4">
          <div className="p-4 bg-teal-50 border border-teal-200 rounded-lg">
            <p className="text-sm text-teal-800 mb-3">
              📏 Scale and normalize features
            </p>
            <p className="text-xs text-teal-700">
              Standardize features to improve model performance
            </p>
          </div>
          {renderField("dataset_id", "Dataset Source", "text")}
          {renderField("method", "Scaling Method", "select", [
            { value: "standard", label: "Standard Scaler (Z-score)" },
            { value: "minmax", label: "Min-Max Scaler" },
            { value: "robust", label: "Robust Scaler" },
            { value: "normalize", label: "Normalizer" },
          ])}
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-xs text-blue-800 space-y-1">
              <p>
                <strong>Standard:</strong> Mean=0, Std=1 (sensitive to outliers)
              </p>
              <p>
                <strong>Min-Max:</strong> Scale to [0,1] range
              </p>
              <p>
                <strong>Robust:</strong> Uses median and IQR (outlier resistant)
              </p>
              <p>
                <strong>Normalizer:</strong> Scale samples to unit norm
              </p>
            </div>
          </div>
          {availableColumns.length > 0 && (
            <>
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-gray-800">
                    Select Columns to Scale
                  </h4>
                  <span className="text-xs bg-teal-100 text-teal-800 px-2 py-1 rounded-full font-medium">
                    {((config.columns as string[]) || []).length} selected
                  </span>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {availableColumns.map((column) => {
                    const isSelected = (
                      (config.columns as string[]) || []
                    ).includes(column);
                    return (
                      <label
                        key={column}
                        className={`flex items-center gap-2 p-2 rounded transition-colors ${
                          isSelected
                            ? "bg-teal-50 border border-teal-200"
                            : "hover:bg-gray-50"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={(e) => {
                            const currentColumns =
                              (config.columns as string[]) || [];
                            const newColumns = e.target.checked
                              ? [...currentColumns, column]
                              : currentColumns.filter((c) => c !== column);
                            console.log("✏️ Scaling columns updated:", {
                              column,
                              checked: e.target.checked,
                              previousColumns: currentColumns,
                              newColumns,
                            });
                            updateField("columns", newColumns);
                          }}
                          className="rounded border-gray-300 text-teal-600 focus:ring-teal-500"
                        />
                        <span
                          className={`text-sm ${
                            isSelected
                              ? "text-teal-900 font-medium"
                              : "text-gray-700"
                          }`}
                        >
                          {column}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
              {((config.columns as string[]) || []).length > 0 && (
                <div className="bg-teal-50 border border-teal-200 rounded-lg p-3">
                  <p className="text-xs text-teal-900 font-medium mb-1">
                    ✓ Selected columns:
                  </p>
                  <p className="text-xs text-teal-700">
                    {((config.columns as string[]) || []).join(", ")}
                  </p>
                </div>
              )}
            </>
          )}
          {(config.dataset_id as string) && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800">
                ✓ Ready to scale {((config.columns as string[]) || []).length}{" "}
                column(s)
              </p>
            </div>
          )}
        </div>
      );

    case "feature_selection":
      return (
        <div className="space-y-4">
          <div className="p-4 bg-cyan-50 border border-cyan-200 rounded-lg">
            <p className="text-sm text-cyan-800 mb-3">
              🎯 Select most important features
            </p>
            <p className="text-xs text-cyan-700">
              Reduce dimensionality by selecting relevant features
            </p>
          </div>
          {renderField("dataset_id", "Dataset Source", "text")}
          {renderField("method", "Selection Method", "select", [
            { value: "variance", label: "Variance Threshold" },
            { value: "correlation", label: "Correlation Threshold" },
          ])}

          {config.method === "variance" && (
            <>
              {renderField(
                "variance_threshold",
                "Variance Threshold",
                "number",
                {
                  min: 0,
                  max: 1,
                  step: 0.01,
                },
              )}
              <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-800">
                  Features with variance below threshold will be removed
                </p>
              </div>
            </>
          )}

          {config.method === "correlation" && (
            <>
              {renderField("correlation_mode", "Correlation Mode", "select", [
                {
                  value: "threshold",
                  label: "Remove Highly Correlated (Threshold)",
                },
                { value: "topk", label: "Keep Top K Features" },
              ])}
              {renderField(
                "correlation_threshold",
                "Correlation Threshold",
                "number",
                { min: 0, max: 1, step: 0.05 },
              )}
              {config.correlation_mode === "topk" &&
                renderField("n_features", "Number of Features (K)", "number", {
                  min: 1,
                })}
              <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-800">
                  {config.correlation_mode === "topk"
                    ? "Keep top K features after removing highly correlated ones"
                    : "Remove features with correlation above threshold"}
                </p>
              </div>
            </>
          )}

          {(config.dataset_id as string) && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800">
                ✓ Ready to select features using {String(config.method)} method
              </p>
            </div>
          )}
        </div>
      );

    default:
      return <div>Unknown feature engineering node type</div>;
  }
};
