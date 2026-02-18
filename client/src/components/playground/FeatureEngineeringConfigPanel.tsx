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
import {
  Droplet,
  Hash,
  Scale,
  Filter,
  CheckCircle2,
  Columns3,
  Zap,
  BarChart3,
} from "lucide-react";

interface FeatureEngineeringConfigPanelProps {
  node: Node<BaseNodeData>;
  config: Record<string, unknown>;
  availableColumns: string[];
  updateField: (field: string, value: unknown) => void;
  setConfig: Dispatch<SetStateAction<Record<string, unknown>>>;
  connectedSourceNode?: { data: { label: string } } | null;
  renderField: (
    field: string,
    label: string,
    type?: string,
    options?:
      | { value: string; label: string }[]
      | { step?: number; min?: number; max?: number },
  ) => React.JSX.Element;
}

// Reusable connection status card
const ConnectionStatus = ({
  connectedSourceNode,
  datasetId,
  message,
}: {
  connectedSourceNode?: { data: { label: string } } | null;
  datasetId?: string;
  message: string;
}) => (
  <div
    className={`rounded-xl border p-3 ${connectedSourceNode ? "border-green-200 bg-green-50/60" : "border-amber-200 bg-amber-50/60"}`}
  >
    <div className="flex items-center gap-2">
      {connectedSourceNode ? (
        <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
      ) : (
        <span className="text-amber-500 text-sm">⚠</span>
      )}
      <span
        className={`text-xs font-bold uppercase tracking-wider ${connectedSourceNode ? "text-green-800" : "text-amber-800"}`}
      >
        {connectedSourceNode ? "Data Source Connected" : "No Data Source"}
      </span>
    </div>
    <p
      className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}
    >
      {connectedSourceNode
        ? `Connected to: ${connectedSourceNode.data.label}`
        : message}
    </p>
    {datasetId && (
      <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
        Dataset ID: {datasetId}
      </div>
    )}
  </div>
);

export const FeatureEngineeringConfigPanel = ({
  node,
  config,
  availableColumns,
  updateField,
  setConfig,
  connectedSourceNode,
  renderField,
}: FeatureEngineeringConfigPanelProps) => {
  const nodeData = node.data as BaseNodeData;

  switch (nodeData.type) {
    case "missing_value_handler": {
      const columnConfigs =
        (config.column_configs as Record<string, Record<string, unknown>>) ||
        {};

      const handleStrategyChange = (column: string, strategy: string) => {
        const updatedConfigs = { ...columnConfigs };
        if (!updatedConfigs[column]) {
          updatedConfigs[column] = {};
        }
        updatedConfigs[column] = {
          ...updatedConfigs[column],
          strategy: strategy,
          enabled: true,
          fill_value: updatedConfigs[column]?.fill_value || null,
        };
        setConfig((prev) => ({ ...prev, column_configs: updatedConfigs }));
      };

      const handleFillValueChange = (column: string, value: string) => {
        const updatedConfigs = { ...columnConfigs };
        if (updatedConfigs[column]) {
          updatedConfigs[column].fill_value = value;
          setConfig((prev) => ({ ...prev, column_configs: updatedConfigs }));
        }
      };

      const handleEnabledChange = (column: string, enabled: boolean) => {
        const updatedConfigs = { ...columnConfigs };
        if (updatedConfigs[column]) {
          updatedConfigs[column].enabled = enabled;
          setConfig((prev) => ({ ...prev, column_configs: updatedConfigs }));
        }
      };

      const strategies = [
        { value: "none", label: "No Action", color: "slate" },
        { value: "drop", label: "Drop Rows", color: "red" },
        { value: "drop_column", label: "Drop Column", color: "red" },
        { value: "mean", label: "Mean", color: "blue" },
        { value: "median", label: "Median", color: "blue" },
        { value: "mode", label: "Mode", color: "blue" },
        { value: "fill", label: "Custom Value", color: "purple" },
        { value: "forward_fill", label: "Forward Fill", color: "teal" },
        { value: "backward_fill", label: "Backward Fill", color: "teal" },
      ];

      const configuredCount = Object.keys(columnConfigs).filter(
        (col) =>
          columnConfigs[col]?.strategy &&
          columnConfigs[col]?.strategy !== "none" &&
          columnConfigs[col]?.enabled !== false,
      ).length;

      return (
        <div className="space-y-4">
          {/* Header */}
          <div className="rounded-xl border border-rose-200 bg-linear-to-br from-rose-50/80 to-pink-50/80 p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-rose-100 flex items-center justify-center">
                <Droplet className="w-4 h-4 text-rose-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Missing Value Handler
                </p>
                <p className="text-xs text-slate-500">
                  Handle missing values with column-wise control
                </p>
              </div>
            </div>
          </div>

          {/* Connection Status */}
          <ConnectionStatus
            connectedSourceNode={connectedSourceNode}
            datasetId={config.dataset_id as string}
            message="Connect a data source node to handle missing values"
          />

          {/* Default Strategy */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Default Strategy
            </label>
            <select
              value={(config.default_strategy as string) || "none"}
              onChange={(e) => updateField("default_strategy", e.target.value)}
              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-rose-500 focus:border-rose-500"
            >
              {strategies.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-500 mt-1.5">
              Applied to all columns unless overridden below
            </p>
          </div>

          {/* Preview Mode Toggle */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700">
                  Preview Mode
                </p>
                <p className="text-xs text-slate-500 mt-0.5">
                  Preview changes before applying them
                </p>
              </div>
              <button
                onClick={() =>
                  updateField(
                    "preview_mode",
                    !(config.preview_mode as boolean),
                  )
                }
                className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
                  (config.preview_mode as boolean)
                    ? "bg-rose-500"
                    : "bg-slate-300"
                }`}
              >
                <div
                  className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform duration-200 ${
                    (config.preview_mode as boolean)
                      ? "translate-x-5.5"
                      : "translate-x-0.5"
                  }`}
                />
              </button>
            </div>
          </div>

          {/* Column-wise Configuration */}
          {availableColumns.length > 0 ? (
            <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Columns3 className="w-3.5 h-3.5 text-slate-400" />
                  <h4 className="text-sm font-semibold text-slate-800">
                    Column Configuration
                  </h4>
                </div>
                <span className="text-[10px] font-bold text-rose-700 bg-rose-50 border border-rose-200 px-2 py-0.5 rounded-full uppercase tracking-wider">
                  {configuredCount} configured
                </span>
              </div>
              <div className="divide-y divide-slate-100 max-h-80 overflow-y-auto">
                {availableColumns.map((column) => {
                  const colConfig = columnConfigs[column];
                  const strategy = String(colConfig?.strategy || "none");
                  const isEnabled = colConfig?.enabled !== false;
                  const isConfigured = strategy !== "none" && isEnabled;

                  return (
                    <div key={column} className="px-4 py-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-2 h-2 rounded-full ${isConfigured ? "bg-rose-400" : "bg-slate-300"}`}
                          />
                          <span className="font-medium text-sm text-slate-700">
                            {column}
                          </span>
                        </div>
                        <button
                          onClick={() =>
                            handleEnabledChange(column, !isEnabled)
                          }
                          className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${
                            isEnabled ? "bg-rose-500" : "bg-slate-300"
                          }`}
                        >
                          <div
                            className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-sm transition-transform duration-200 ${
                              isEnabled
                                ? "translate-x-4.5"
                                : "translate-x-0.5"
                            }`}
                          />
                        </button>
                      </div>

                      {isEnabled && (
                        <div className="space-y-2 pl-4">
                          <select
                            value={strategy}
                            onChange={(e) =>
                              handleStrategyChange(column, e.target.value)
                            }
                            className="w-full px-2.5 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-rose-500 focus:border-rose-500"
                          >
                            {strategies.map((s) => (
                              <option key={s.value} value={s.value}>
                                {s.label}
                              </option>
                            ))}
                          </select>

                          {strategy === "drop_column" && (
                            <div className="px-2.5 py-1.5 bg-red-50 border border-red-200 rounded-lg">
                              <p className="text-[11px] text-red-700 font-medium">
                                This column will be removed from the dataset
                              </p>
                            </div>
                          )}

                          {strategy === "fill" && (
                            <input
                              type="text"
                              value={String(colConfig?.fill_value || "")}
                              onChange={(e) =>
                                handleFillValueChange(column, e.target.value)
                              }
                              placeholder="Enter fill value..."
                              className="w-full px-2.5 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-rose-500 focus:border-rose-500"
                            />
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-2 px-3 py-2.5 bg-amber-50/60 rounded-xl border border-amber-100">
              <span className="text-amber-500 text-sm mt-0.5">💡</span>
              <p className="text-xs text-amber-700 leading-relaxed">
                Column-wise configuration will be available after connecting to
                a data source and running the pipeline.
              </p>
            </div>
          )}

          {/* Ready Status */}
          {(config.dataset_id as string) && configuredCount > 0 && (
            <div className="flex items-center gap-2 px-4 py-3 bg-green-50/60 rounded-xl border border-green-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-medium text-green-800">
                Ready to handle missing values in {configuredCount} column(s)
              </p>
            </div>
          )}
        </div>
      );
    }

    case "encoding": {
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

      const encodedCount = Object.keys(columnConfigs).filter(
        (col) =>
          columnConfigs[col]?.encoding_method &&
          columnConfigs[col]?.encoding_method !== "none",
      ).length;

      return (
        <div className="space-y-4">
          {/* Header */}
          <div className="rounded-xl border border-amber-200 bg-linear-to-br from-amber-50/80 to-orange-50/80 p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-amber-100 flex items-center justify-center">
                <Hash className="w-4 h-4 text-amber-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Encoding
                </p>
                <p className="text-xs text-slate-500">
                  Encode categorical variables with per-column control
                </p>
              </div>
            </div>
          </div>

          {/* Connection Status */}
          <ConnectionStatus
            connectedSourceNode={connectedSourceNode}
            datasetId={config.dataset_id as string}
            message="Connect a data source node to configure encoding"
          />

          {/* Column Encoding Configuration */}
          {availableColumns.length > 0 ? (
            <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Columns3 className="w-3.5 h-3.5 text-slate-400" />
                  <h4 className="text-sm font-semibold text-slate-800">
                    Column Encoding
                  </h4>
                </div>
                <span className="text-[10px] font-bold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full uppercase tracking-wider">
                  {encodedCount} to encode
                </span>
              </div>
              <div className="divide-y divide-slate-100 max-h-80 overflow-y-auto">
                {availableColumns.map((column) => {
                  const colConfig = columnConfigs[column];
                  const method = String(
                    colConfig?.encoding_method || "none",
                  );
                  const isConfigured = method !== "none";

                  return (
                    <div key={column} className="px-4 py-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-2 h-2 rounded-full ${isConfigured ? "bg-amber-400" : "bg-slate-300"}`}
                          />
                          <span className="font-medium text-sm text-slate-700">
                            {column}
                          </span>
                        </div>
                        {isConfigured && (
                          <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 border border-amber-200">
                            {method === "onehot"
                              ? "One-Hot"
                              : method === "label"
                                ? "Label"
                                : method}
                          </span>
                        )}
                      </div>

                      <div className="space-y-2 pl-4">
                        {/* Encoding Method */}
                        <select
                          value={method}
                          onChange={(e) =>
                            handleEncodingMethodChange(column, e.target.value)
                          }
                          className="w-full px-2.5 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                        >
                          <option value="none">None (Skip)</option>
                          <option value="onehot">One-Hot Encoding</option>
                          <option value="label">Label Encoding</option>
                        </select>

                        {/* Additional Options */}
                        {isConfigured && (
                          <>
                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <label className="block text-[11px] font-medium text-slate-500 mb-1">
                                  Handle Unknown
                                </label>
                                <select
                                  value={String(
                                    colConfig?.handle_unknown || "error",
                                  )}
                                  onChange={(e) =>
                                    handleEncodingOptionChange(
                                      column,
                                      "handle_unknown",
                                      e.target.value,
                                    )
                                  }
                                  className="w-full px-2 py-1 text-[11px] bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-1 focus:ring-amber-400"
                                >
                                  <option value="error">Error</option>
                                  <option value="ignore">Ignore</option>
                                  <option value="use_encoded_value">
                                    Encoded Value
                                  </option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-[11px] font-medium text-slate-500 mb-1">
                                  Handle Missing
                                </label>
                                <select
                                  value={String(
                                    colConfig?.handle_missing || "error",
                                  )}
                                  onChange={(e) =>
                                    handleEncodingOptionChange(
                                      column,
                                      "handle_missing",
                                      e.target.value,
                                    )
                                  }
                                  className="w-full px-2 py-1 text-[11px] bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-1 focus:ring-amber-400"
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

                            {/* Drop First for one-hot */}
                            {method === "onehot" && (
                              <label className="flex items-center gap-2 py-1 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={Boolean(colConfig?.drop_first)}
                                  onChange={(e) =>
                                    handleEncodingOptionChange(
                                      column,
                                      "drop_first",
                                      e.target.checked,
                                    )
                                  }
                                  className="w-3.5 h-3.5 rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                                />
                                <span className="text-[11px] text-slate-600">
                                  Drop first category (avoid multicollinearity)
                                </span>
                              </label>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-2 px-3 py-2.5 bg-amber-50/60 rounded-xl border border-amber-100">
              <span className="text-amber-500 text-sm mt-0.5">💡</span>
              <p className="text-xs text-amber-700 leading-relaxed">
                Connect a data source and run the pipeline to configure column
                encoding.
              </p>
            </div>
          )}

          {/* Ready Status */}
          {(config.dataset_id as string) && encodedCount > 0 && (
            <div className="flex items-center gap-2 px-4 py-3 bg-green-50/60 rounded-xl border border-green-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-medium text-green-800">
                Ready to encode {encodedCount} column(s)
              </p>
            </div>
          )}
        </div>
      );
    }

    case "transformation": {
      const selectedColumns = (config.columns as string[]) || [];

      const handleSelectAll = () => {
        updateField("columns", [...availableColumns]);
      };
      const handleSelectNone = () => {
        updateField("columns", []);
      };

      const transformMethods = [
        {
          value: "log",
          label: "Log Transform",
          desc: "Natural logarithm - good for right-skewed data",
        },
        {
          value: "sqrt",
          label: "Square Root",
          desc: "Moderate skew correction",
        },
        {
          value: "power",
          label: "Power (Box-Cox)",
          desc: "Automatic optimal transformation",
        },
      ];

      return (
        <div className="space-y-4">
          {/* Header */}
          <div className="rounded-xl border border-purple-200 bg-linear-to-br from-purple-50/80 to-violet-50/80 p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-purple-100 flex items-center justify-center">
                <Zap className="w-4 h-4 text-purple-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Transformation
                </p>
                <p className="text-xs text-slate-500">
                  Apply mathematical transformations to features
                </p>
              </div>
            </div>
          </div>

          {/* Connection Status */}
          <ConnectionStatus
            connectedSourceNode={connectedSourceNode}
            datasetId={config.dataset_id as string}
            message="Connect a data source node to transform features"
          />

          {/* Transformation Method */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-medium text-slate-700 mb-3">
              Transformation Method
            </p>
            <div className="space-y-2">
              {transformMethods.map((m) => (
                <button
                  key={m.value}
                  onClick={() =>
                    updateField("transformation_type", m.value)
                  }
                  className={`w-full flex items-start gap-3 p-3 rounded-lg border-2 text-left transition-all duration-150 ${
                    (config.transformation_type as string) === m.value
                      ? "border-purple-500 bg-purple-50"
                      : "border-slate-200 bg-slate-50 hover:border-slate-300"
                  }`}
                >
                  <div
                    className={`w-4 h-4 rounded-full border-2 mt-0.5 shrink-0 flex items-center justify-center ${
                      (config.transformation_type as string) === m.value
                        ? "border-purple-500"
                        : "border-slate-300"
                    }`}
                  >
                    {(config.transformation_type as string) === m.value && (
                      <div className="w-2 h-2 rounded-full bg-purple-500" />
                    )}
                  </div>
                  <div>
                    <p
                      className={`text-sm font-medium ${(config.transformation_type as string) === m.value ? "text-purple-900" : "text-slate-700"}`}
                    >
                      {m.label}
                    </p>
                    <p className="text-[11px] text-slate-500">{m.desc}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Column Selection */}
          {availableColumns.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Columns3 className="w-3.5 h-3.5 text-slate-400" />
                  <h4 className="text-sm font-semibold text-slate-800">
                    Columns to Transform
                  </h4>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleSelectAll}
                    className="text-[10px] font-medium text-purple-600 hover:text-purple-800 px-1.5 py-0.5 rounded hover:bg-purple-50"
                  >
                    All
                  </button>
                  <span className="text-slate-300">|</span>
                  <button
                    onClick={handleSelectNone}
                    className="text-[10px] font-medium text-slate-500 hover:text-slate-700 px-1.5 py-0.5 rounded hover:bg-slate-50"
                  >
                    None
                  </button>
                  <span className="text-[10px] font-bold text-purple-700 bg-purple-50 border border-purple-200 px-2 py-0.5 rounded-full ml-1">
                    {selectedColumns.length}/{availableColumns.length}
                  </span>
                </div>
              </div>
              <div className="px-4 py-2 max-h-56 overflow-y-auto space-y-0.5">
                {availableColumns.map((column) => {
                  const isSelected = selectedColumns.includes(column);
                  return (
                    <label
                      key={column}
                      className={`flex items-center gap-2.5 px-2 py-1.5 rounded-md cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-purple-50"
                          : "hover:bg-slate-50"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={(e) => {
                          const newColumns = e.target.checked
                            ? [...selectedColumns, column]
                            : selectedColumns.filter((c) => c !== column);
                          updateField("columns", newColumns);
                        }}
                        className="w-3.5 h-3.5 rounded border-slate-300 text-purple-600 focus:ring-purple-500"
                      />
                      <span
                        className={`text-sm ${isSelected ? "text-purple-900 font-medium" : "text-slate-600"}`}
                      >
                        {column}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>
          )}

          {/* Ready Status */}
          {(config.dataset_id as string) && selectedColumns.length > 0 && (
            <div className="flex items-center gap-2 px-4 py-3 bg-green-50/60 rounded-xl border border-green-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-medium text-green-800">
                Ready to transform {selectedColumns.length} column(s) using{" "}
                {(config.transformation_type as string) || "log"}
              </p>
            </div>
          )}
        </div>
      );
    }

    case "scaling": {
      const selectedColumns = (config.columns as string[]) || [];
      const scalingMethod = (config.method as string) || "standard";

      const handleSelectAll = () => {
        updateField("columns", [...availableColumns]);
      };
      const handleSelectNone = () => {
        updateField("columns", []);
      };

      const scalingMethods = [
        {
          value: "standard",
          label: "Standard Scaler",
          desc: "Mean=0, Std=1 (Z-score normalization)",
          badge: "Z-score",
        },
        {
          value: "minmax",
          label: "Min-Max Scaler",
          desc: "Scale features to [0, 1] range",
          badge: "[0,1]",
        },
        {
          value: "robust",
          label: "Robust Scaler",
          desc: "Uses median & IQR (outlier resistant)",
          badge: "IQR",
        },
        {
          value: "normalize",
          label: "Normalizer",
          desc: "Scale each sample to unit norm",
          badge: "L2",
        },
      ];

      return (
        <div className="space-y-4">
          {/* Header */}
          <div className="rounded-xl border border-teal-200 bg-linear-to-br from-teal-50/80 to-emerald-50/80 p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-teal-100 flex items-center justify-center">
                <Scale className="w-4 h-4 text-teal-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Scaling / Normalization
                </p>
                <p className="text-xs text-slate-500">
                  Standardize features to improve model performance
                </p>
              </div>
            </div>
          </div>

          {/* Connection Status */}
          <ConnectionStatus
            connectedSourceNode={connectedSourceNode}
            datasetId={config.dataset_id as string}
            message="Connect a data source node to scale features"
          />

          {/* Scaling Method */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-medium text-slate-700 mb-3">
              Scaling Method
            </p>
            <div className="grid grid-cols-2 gap-2">
              {scalingMethods.map((m) => (
                <button
                  key={m.value}
                  onClick={() => updateField("method", m.value)}
                  className={`flex flex-col items-start p-3 rounded-lg border-2 text-left transition-all duration-150 ${
                    scalingMethod === m.value
                      ? "border-teal-500 bg-teal-50"
                      : "border-slate-200 bg-slate-50 hover:border-slate-300"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1 w-full">
                    <span
                      className={`text-xs font-semibold ${scalingMethod === m.value ? "text-teal-800" : "text-slate-700"}`}
                    >
                      {m.label}
                    </span>
                    <span
                      className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full ml-auto ${
                        scalingMethod === m.value
                          ? "bg-teal-200 text-teal-800"
                          : "bg-slate-200 text-slate-600"
                      }`}
                    >
                      {m.badge}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-500 leading-snug">
                    {m.desc}
                  </p>
                </button>
              ))}
            </div>
          </div>

          {/* Column Selection */}
          {availableColumns.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Columns3 className="w-3.5 h-3.5 text-slate-400" />
                  <h4 className="text-sm font-semibold text-slate-800">
                    Columns to Scale
                  </h4>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleSelectAll}
                    className="text-[10px] font-medium text-teal-600 hover:text-teal-800 px-1.5 py-0.5 rounded hover:bg-teal-50"
                  >
                    All
                  </button>
                  <span className="text-slate-300">|</span>
                  <button
                    onClick={handleSelectNone}
                    className="text-[10px] font-medium text-slate-500 hover:text-slate-700 px-1.5 py-0.5 rounded hover:bg-slate-50"
                  >
                    None
                  </button>
                  <span className="text-[10px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded-full ml-1">
                    {selectedColumns.length}/{availableColumns.length}
                  </span>
                </div>
              </div>
              <div className="px-4 py-2 max-h-56 overflow-y-auto space-y-0.5">
                {availableColumns.map((column) => {
                  const isSelected = selectedColumns.includes(column);
                  return (
                    <label
                      key={column}
                      className={`flex items-center gap-2.5 px-2 py-1.5 rounded-md cursor-pointer transition-colors ${
                        isSelected ? "bg-teal-50" : "hover:bg-slate-50"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={(e) => {
                          const newColumns = e.target.checked
                            ? [...selectedColumns, column]
                            : selectedColumns.filter((c) => c !== column);
                          updateField("columns", newColumns);
                        }}
                        className="w-3.5 h-3.5 rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                      />
                      <span
                        className={`text-sm ${isSelected ? "text-teal-900 font-medium" : "text-slate-600"}`}
                      >
                        {column}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>
          )}

          {/* Ready Status */}
          {(config.dataset_id as string) && selectedColumns.length > 0 && (
            <div className="flex items-center gap-2 px-4 py-3 bg-green-50/60 rounded-xl border border-green-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-medium text-green-800">
                Ready to scale {selectedColumns.length} column(s) using{" "}
                {scalingMethods.find((m) => m.value === scalingMethod)?.label ||
                  scalingMethod}
              </p>
            </div>
          )}
        </div>
      );
    }

    case "feature_selection": {
      const method = (config.method as string) || "variance";

      const selectionMethods = [
        {
          value: "variance",
          label: "Variance Threshold",
          desc: "Remove features with low variance (near-constant columns)",
          Icon: BarChart3,
        },
        {
          value: "correlation",
          label: "Correlation",
          desc: "Remove highly correlated features to reduce redundancy",
          Icon: Filter,
        },
      ];

      return (
        <div className="space-y-4">
          {/* Header */}
          <div className="rounded-xl border border-cyan-200 bg-linear-to-br from-cyan-50/80 to-sky-50/80 p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-cyan-100 flex items-center justify-center">
                <Filter className="w-4 h-4 text-cyan-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Feature Selection
                </p>
                <p className="text-xs text-slate-500">
                  Select the most important features for your model
                </p>
              </div>
            </div>
          </div>

          {/* Connection Status */}
          <ConnectionStatus
            connectedSourceNode={connectedSourceNode}
            datasetId={config.dataset_id as string}
            message="Connect a data source node to select features"
          />

          {/* Selection Method */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-sm font-medium text-slate-700 mb-3">
              Selection Strategy
            </p>
            <div className="space-y-2">
              {selectionMethods.map((m) => (
                <button
                  key={m.value}
                  onClick={() => updateField("method", m.value)}
                  className={`w-full flex items-start gap-3 p-3 rounded-lg border-2 text-left transition-all duration-150 ${
                    method === m.value
                      ? "border-cyan-500 bg-cyan-50"
                      : "border-slate-200 bg-slate-50 hover:border-slate-300"
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-md flex items-center justify-center shrink-0 ${
                      method === m.value ? "bg-cyan-100" : "bg-slate-200"
                    }`}
                  >
                    <m.Icon
                      className={`w-4 h-4 ${method === m.value ? "text-cyan-600" : "text-slate-400"}`}
                    />
                  </div>
                  <div>
                    <p
                      className={`text-sm font-medium ${method === m.value ? "text-cyan-900" : "text-slate-700"}`}
                    >
                      {m.label}
                    </p>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      {m.desc}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Method-specific Configuration */}
          {method === "variance" && (
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Variance Threshold
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  value={(config.variance_threshold as number) ?? 0}
                  onChange={(e) =>
                    updateField(
                      "variance_threshold",
                      parseFloat(e.target.value),
                    )
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #06B6D4 0%, #06B6D4 ${((config.variance_threshold as number) ?? 0) * 100}%, #E2E8F0 ${((config.variance_threshold as number) ?? 0) * 100}%, #E2E8F0 100%)`,
                  }}
                />
                <div className="w-14 px-2 py-1.5 bg-cyan-50 border border-cyan-200 rounded-lg text-center">
                  <span className="text-xs font-bold text-cyan-800">
                    {((config.variance_threshold as number) ?? 0).toFixed(2)}
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 mt-2">
                Features with variance below this threshold will be removed
              </p>
            </div>
          )}

          {method === "correlation" && (
            <>
              {/* Correlation Mode */}
              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Correlation Mode
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() =>
                      updateField("correlation_mode", "threshold")
                    }
                    className={`p-3 rounded-lg border-2 text-left transition-all ${
                      (config.correlation_mode as string) === "threshold" ||
                      !(config.correlation_mode as string)
                        ? "border-cyan-500 bg-cyan-50"
                        : "border-slate-200 bg-slate-50 hover:border-slate-300"
                    }`}
                  >
                    <p
                      className={`text-xs font-semibold ${
                        (config.correlation_mode as string) === "threshold" ||
                        !(config.correlation_mode as string)
                          ? "text-cyan-800"
                          : "text-slate-700"
                      }`}
                    >
                      Threshold
                    </p>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Remove by correlation
                    </p>
                  </button>
                  <button
                    onClick={() => updateField("correlation_mode", "topk")}
                    className={`p-3 rounded-lg border-2 text-left transition-all ${
                      (config.correlation_mode as string) === "topk"
                        ? "border-cyan-500 bg-cyan-50"
                        : "border-slate-200 bg-slate-50 hover:border-slate-300"
                    }`}
                  >
                    <p
                      className={`text-xs font-semibold ${
                        (config.correlation_mode as string) === "topk"
                          ? "text-cyan-800"
                          : "text-slate-700"
                      }`}
                    >
                      Top-K
                    </p>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Keep K best features
                    </p>
                  </button>
                </div>
              </div>

              {/* Correlation Threshold */}
              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Correlation Threshold
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    value={(config.correlation_threshold as number) ?? 0.95}
                    onChange={(e) =>
                      updateField(
                        "correlation_threshold",
                        parseFloat(e.target.value),
                      )
                    }
                    min={0}
                    max={1}
                    step={0.05}
                    className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                    style={{
                      background: `linear-gradient(to right, #06B6D4 0%, #06B6D4 ${((config.correlation_threshold as number) ?? 0.95) * 100}%, #E2E8F0 ${((config.correlation_threshold as number) ?? 0.95) * 100}%, #E2E8F0 100%)`,
                    }}
                  />
                  <div className="w-14 px-2 py-1.5 bg-cyan-50 border border-cyan-200 rounded-lg text-center">
                    <span className="text-xs font-bold text-cyan-800">
                      {(
                        (config.correlation_threshold as number) ?? 0.95
                      ).toFixed(2)}
                    </span>
                  </div>
                </div>
                <p className="text-[11px] text-slate-500 mt-2">
                  Features with correlation above this value will be removed
                </p>
              </div>

              {/* Top-K */}
              {(config.correlation_mode as string) === "topk" && (
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Number of Features (K)
                  </label>
                  <input
                    type="number"
                    value={(config.n_features as number) || 10}
                    onChange={(e) =>
                      updateField(
                        "n_features",
                        parseInt(e.target.value) || 1,
                      )
                    }
                    min={1}
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  />
                  <p className="text-[11px] text-slate-500 mt-1.5">
                    Number of features to keep after removing highly correlated
                    ones
                  </p>
                </div>
              )}
            </>
          )}

          {/* Ready Status */}
          {(config.dataset_id as string) && (
            <div className="flex items-center gap-2 px-4 py-3 bg-green-50/60 rounded-xl border border-green-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-medium text-green-800">
                Ready to select features using {method} method
              </p>
            </div>
          )}
        </div>
      );
    }

    default:
      return <div>Unknown feature engineering node type</div>;
  }
};
