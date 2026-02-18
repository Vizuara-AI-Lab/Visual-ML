/**
 * Node Configuration Panel - Configure node parameters with validation
 */

import { useState, useEffect } from "react";
import type { BaseNodeData } from "../../types/pipeline";
import type { Node } from "@xyflow/react";
import { UploadDatasetButton } from "./UploadDatasetButton";
import { usePlaygroundStore } from "../../store/playgroundStore";
import {
  listAllUserDatasets,
  type DatasetMetadata,
} from "../../lib/api/datasetApi";
import { FeatureEngineeringConfigPanel } from "./FeatureEngineeringConfigPanel";
import {
  Upload,
  Database,
  FileSpreadsheet,
  CheckCircle2,
  FlaskConical,
} from "lucide-react";

const SAMPLE_DATASETS: Record<
  string,
  { label: string; description: string; rows: string; cols: string; task: string }
> = {
  iris: { label: "Iris", description: "Classify iris flower species from petal & sepal measurements", rows: "150", cols: "5", task: "Classification" },
  wine: { label: "Wine Quality", description: "Predict wine quality from physicochemical properties", rows: "178", cols: "13", task: "Classification" },
  breast_cancer: { label: "Breast Cancer", description: "Classify tumors as malignant or benign using cell features", rows: "569", cols: "30", task: "Classification" },
  digits: { label: "Digits", description: "Handwritten digit recognition (0-9) from 8x8 pixel images", rows: "1,797", cols: "64", task: "Classification" },
  titanic: { label: "Titanic", description: "Predict passenger survival on the Titanic", rows: "891", cols: "12", task: "Classification" },
  penguins: { label: "Palmer Penguins", description: "Classify penguin species from body measurements", rows: "344", cols: "7", task: "Classification" },
  heart_disease: { label: "Heart Disease", description: "Predict the presence of heart disease from clinical features", rows: "303", cols: "14", task: "Classification" },
  diabetes: { label: "Diabetes", description: "Predict diabetes progression one year after baseline", rows: "442", cols: "10", task: "Regression" },
  boston: { label: "California Housing", description: "Predict median house values in California districts", rows: "20,640", cols: "8", task: "Regression" },
  tips: { label: "Tips", description: "Predict tip amount from restaurant billing data", rows: "244", cols: "7", task: "Regression" },
  auto_mpg: { label: "Auto MPG", description: "Predict fuel efficiency of automobiles", rows: "398", cols: "8", task: "Regression" },
  student: { label: "Student Performance", description: "Predict student academic performance", rows: "395", cols: "33", task: "Regression" },
  linnerud: { label: "Linnerud", description: "Relate exercise to physiological measurements", rows: "20", cols: "6", task: "Multivariate" },
};

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
  const [userDatasets, setUserDatasets] = useState<DatasetMetadata[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const currentProjectId = usePlaygroundStore(
    (state) => state.currentProjectId,
  );
  const edges = usePlaygroundStore((state) => state.edges);
  const nodes = usePlaygroundStore((state) => state.nodes);

  const nodeData = node.data as BaseNodeData;

  // Get connected source node for column information
  const getConnectedSourceNode = () => {
    const incomingEdge = edges.find((edge) => edge.target === node.id);
    if (incomingEdge) {
      return nodes.find((n) => n.id === incomingEdge.source);
    }
    return null;
  };

  const connectedSourceNode = getConnectedSourceNode();

  // Extract columns from various possible sources
  const getAvailableColumns = (): string[] => {
    if (!connectedSourceNode) return [];

    const config = connectedSourceNode.data.config;

    // Try to get columns from config.columns (for dataset nodes)
    if (config?.columns && Array.isArray(config.columns)) {
      return config.columns as string[];
    }

    // Try to get from execution results (for processing nodes)
    const result = connectedSourceNode.data.result as
      | Record<string, unknown>
      | undefined;
    if (result?.columns && Array.isArray(result.columns)) {
      return result.columns as string[];
    }

    return [];
  };

  const availableColumns = getAvailableColumns();

  // Re-render when nodes change (e.g., when parent node config updates)
  const [, forceUpdate] = useState({});
  useEffect(() => {
    forceUpdate({});
  }, [nodes, edges]);

  // Load user datasets for select_dataset node
  useEffect(() => {
    const loadDatasets = async () => {
      if (nodeData.type === "select_dataset") {
        setLoadingDatasets(true);
        try {
          const response = await listAllUserDatasets();
          setUserDatasets(response.datasets);
        } catch (error) {
          console.error("Failed to load datasets:", error);
          setErrors((prev) => ({
            ...prev,
            dataset_fetch: "Failed to load datasets",
          }));
        } finally {
          setLoadingDatasets(false);
        }
      }
    };
    loadDatasets();
  }, [nodeData.type]);

  const updateField = (field: string, value: unknown) => {
    console.log(`📝 Updating field "${field}" with value:`, value);
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
    console.log("💾 Saving config for node:", node.id, node.data.type);
    console.log("💾 Config being saved:", config);
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
            <option value="">-- Select {label} --</option>
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
        return (
          <div className="space-y-4">
            {/* Upload Area */}
            <div className="rounded-xl border-2 border-dashed border-blue-300 bg-linear-to-br from-blue-50/80 to-indigo-50/80 p-5">
              <div className="flex flex-col items-center gap-3 text-center">
                <div className="w-11 h-11 rounded-full bg-blue-100 flex items-center justify-center">
                  <Upload className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Upload CSV Dataset
                  </p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Select a CSV file from your computer
                  </p>
                </div>
                {currentProjectId ? (
                  <UploadDatasetButton
                    nodeId={node.id}
                    projectId={parseInt(currentProjectId)}
                    onUploadComplete={(datasetData) => {
                      const newConfig = {
                        dataset_id: datasetData.dataset_id,
                        filename: datasetData.filename,
                        n_rows: datasetData.n_rows,
                        n_columns: datasetData.n_columns,
                        columns: datasetData.columns,
                        dtypes: datasetData.dtypes,
                      };
                      setConfig(newConfig);
                      onUpdate(node.id, newConfig);
                    }}
                  />
                ) : (
                  <div className="px-3 py-2 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700 font-medium">
                    No project selected. Save your pipeline first.
                  </div>
                )}
              </div>
            </div>

            {/* Dataset Info Card */}
            {(config.dataset_id as string) && (
              <div className="rounded-xl border border-emerald-200 bg-linear-to-br from-emerald-50/80 to-green-50/80 overflow-hidden">
                <div className="px-4 py-2.5 border-b border-emerald-200/60 bg-emerald-100/40 flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="text-xs font-bold text-emerald-800 uppercase tracking-wider">
                    Dataset Loaded
                  </span>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4 text-slate-400 shrink-0" />
                    <span className="text-sm font-semibold text-slate-800 truncate">
                      {String(config.filename)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-white/80 rounded-lg border border-slate-200/60 px-3 py-2">
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        Rows
                      </div>
                      <div className="text-lg font-bold text-slate-900">
                        {Number(config.n_rows).toLocaleString()}
                      </div>
                    </div>
                    <div className="bg-white/80 rounded-lg border border-slate-200/60 px-3 py-2">
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        Columns
                      </div>
                      <div className="text-lg font-bold text-slate-900">
                        {String(config.n_columns)}
                      </div>
                    </div>
                  </div>
                  {Array.isArray(config.columns) &&
                    (config.columns as string[]).length > 0 && (
                      <div>
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1.5">
                          Column Preview
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {(config.columns as string[])
                            .slice(0, 6)
                            .map((col: string) => (
                              <span
                                key={col}
                                className="px-2 py-0.5 bg-white rounded-md text-[11px] text-slate-600 border border-slate-200/80 font-medium"
                              >
                                {col}
                              </span>
                            ))}
                          {(config.columns as string[]).length > 6 && (
                            <span className="px-2 py-0.5 bg-slate-100 rounded-md text-[11px] text-slate-500 font-medium">
                              +{(config.columns as string[]).length - 6} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  <div className="pt-2 border-t border-emerald-200/40">
                    <div className="text-[10px] text-slate-400 font-mono truncate">
                      ID: {String(config.dataset_id)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );

      case "select_dataset":
        return (
          <div className="space-y-4">
            {/* Header + Selector */}
            <div className="rounded-xl border border-emerald-200 bg-linear-to-br from-emerald-50/80 to-teal-50/80 p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-9 h-9 rounded-lg bg-emerald-100 flex items-center justify-center">
                  <Database className="w-4.5 h-4.5 text-emerald-600" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Select Existing Dataset
                  </p>
                  <p className="text-xs text-slate-500">
                    Choose from previously uploaded datasets
                  </p>
                </div>
              </div>

              {loadingDatasets ? (
                <div className="flex items-center gap-2 py-3 justify-center">
                  <div className="w-4 h-4 border-2 border-emerald-300 border-t-emerald-600 rounded-full animate-spin" />
                  <span className="text-sm text-slate-600">
                    Loading datasets...
                  </span>
                </div>
              ) : userDatasets.length > 0 ? (
                <select
                  value={(config.dataset_id as string) || ""}
                  onChange={(e) => {
                    const selectedDataset = userDatasets.find(
                      (ds) => ds.dataset_id === e.target.value,
                    );
                    if (selectedDataset) {
                      const newConfig = {
                        dataset_id: selectedDataset.dataset_id,
                        filename: selectedDataset.filename,
                        n_rows: selectedDataset.n_rows,
                        n_columns: selectedDataset.n_columns,
                        columns: selectedDataset.columns,
                        dtypes: selectedDataset.dtypes,
                      };
                      setConfig(newConfig);
                      onUpdate(node.id, newConfig);
                    }
                  }}
                  className="w-full px-3 py-2.5 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-colors"
                >
                  <option value="">-- Select a dataset --</option>
                  {userDatasets.map((dataset) => (
                    <option
                      key={dataset.dataset_id}
                      value={dataset.dataset_id}
                    >
                      {dataset.filename} ({dataset.n_rows} rows,{" "}
                      {dataset.n_columns} cols)
                    </option>
                  ))}
                </select>
              ) : (
                <div className="py-3 px-4 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-500 text-center">
                  No datasets available. Upload a dataset first.
                </div>
              )}
            </div>

            {/* Selected Dataset Info */}
            {(config.dataset_id as string) && (
              <div className="rounded-xl border border-blue-200 bg-linear-to-br from-blue-50/80 to-sky-50/80 overflow-hidden">
                <div className="px-4 py-2.5 border-b border-blue-200/60 bg-blue-100/40 flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" />
                  <span className="text-xs font-bold text-blue-800 uppercase tracking-wider">
                    Selected Dataset
                  </span>
                </div>
                <div className="p-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4 text-slate-400 shrink-0" />
                    <span className="text-sm font-semibold text-slate-800 truncate">
                      {String(config.filename)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-white/80 rounded-lg border border-slate-200/60 px-3 py-2">
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        Rows
                      </div>
                      <div className="text-lg font-bold text-slate-900">
                        {Number(config.n_rows).toLocaleString()}
                      </div>
                    </div>
                    <div className="bg-white/80 rounded-lg border border-slate-200/60 px-3 py-2">
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        Columns
                      </div>
                      <div className="text-lg font-bold text-slate-900">
                        {String(config.n_columns)}
                      </div>
                    </div>
                  </div>
                  {Array.isArray(config.columns) &&
                    (config.columns as string[]).length > 0 && (
                      <div>
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1.5">
                          Column Preview
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {(config.columns as string[])
                            .slice(0, 6)
                            .map((col: string) => (
                              <span
                                key={col}
                                className="px-2 py-0.5 bg-white rounded-md text-[11px] text-slate-600 border border-slate-200/80 font-medium"
                              >
                                {col}
                              </span>
                            ))}
                          {(config.columns as string[]).length > 6 && (
                            <span className="px-2 py-0.5 bg-slate-100 rounded-md text-[11px] text-slate-500 font-medium">
                              +{(config.columns as string[]).length - 6} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  <div className="pt-2 border-t border-blue-200/40">
                    <div className="text-[10px] text-slate-400 font-mono truncate">
                      ID: {String(config.dataset_id)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );

      case "sample_dataset": {
        const selectedName = (config.dataset_name as string) || "iris";
        const datasetInfo = SAMPLE_DATASETS[selectedName];
        return (
          <div className="space-y-4">
            {/* Header + Selector */}
            <div className="rounded-xl border border-violet-200 bg-linear-to-br from-violet-50/80 to-purple-50/80 p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-9 h-9 rounded-lg bg-violet-100 flex items-center justify-center">
                  <FlaskConical className="w-4.5 h-4.5 text-violet-600" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Sample Dataset
                  </p>
                  <p className="text-xs text-slate-500">
                    Choose a built-in dataset for learning
                  </p>
                </div>
              </div>

              <select
                value={selectedName}
                onChange={(e) => updateField("dataset_name", e.target.value)}
                className="w-full px-3 py-2.5 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-colors"
              >
                {Object.entries(SAMPLE_DATASETS).map(([value, info]) => (
                  <option key={value} value={value}>
                    {info.label} ({info.task})
                  </option>
                ))}
              </select>
            </div>

            {/* Dataset Info Card */}
            {datasetInfo && (
              <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
                <div className="px-4 py-2.5 border-b border-slate-100 flex items-center justify-between">
                  <span className="text-sm font-semibold text-slate-800">
                    {datasetInfo.label}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                      datasetInfo.task === "Classification"
                        ? "bg-blue-100 text-blue-700"
                        : datasetInfo.task === "Regression"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-purple-100 text-purple-700"
                    }`}
                  >
                    {datasetInfo.task}
                  </span>
                </div>
                <div className="p-4 space-y-3">
                  <p className="text-xs text-slate-600 leading-relaxed">
                    {datasetInfo.description}
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-slate-50 rounded-lg border border-slate-200/60 px-3 py-2">
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        Rows
                      </div>
                      <div className="text-lg font-bold text-slate-900">
                        {datasetInfo.rows}
                      </div>
                    </div>
                    <div className="bg-slate-50 rounded-lg border border-slate-200/60 px-3 py-2">
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        Columns
                      </div>
                      <div className="text-lg font-bold text-slate-900">
                        {datasetInfo.cols}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      }

      case "table_view":
        return (
          <div className="space-y-4">
            <div className="p-4 bg-cyan-50 border border-cyan-200 rounded-lg">
              <p className="text-sm text-cyan-800 mb-3">
                Display your dataset in a table format
              </p>
              <p className="text-xs text-cyan-700">
                Connect this node to a data source to view the data
              </p>
            </div>
            {renderField("dataset_id", "Dataset Source", "text")}
            {renderField("max_rows", "Max Rows to Display", "number", {
              min: 10,
              max: 1000,
            })}
            {(config.dataset_id as string) && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  ✓ Ready to display table view on pipeline execution
                </p>
              </div>
            )}
          </div>
        );

      case "data_preview":
        return (
          <div className="space-y-4">
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <p className="text-sm text-purple-800 mb-3">
                Quick preview of first and last rows
              </p>
              <p className="text-xs text-purple-700">
                Shows head and tail of the dataset
              </p>
            </div>
            {renderField("dataset_id", "Dataset Source", "text")}
            {renderField("head_rows", "First N Rows", "number", {
              min: 1,
              max: 50,
            })}
            {renderField("tail_rows", "Last N Rows", "number", {
              min: 1,
              max: 50,
            })}
            {(config.dataset_id as string) && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  ✓ Ready to preview data on pipeline execution
                </p>
              </div>
            )}
          </div>
        );

      case "statistics_view":
        return (
          <div className="space-y-4">
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800 mb-3">
                Statistical summary of your dataset
              </p>
              <p className="text-xs text-green-700">
                Shows mean, std, min, max, quartiles, etc.
              </p>
            </div>
            {renderField("dataset_id", "Dataset Source", "text")}
            {renderField("include_all", "Include All Columns", "checkbox")}
            {(config.dataset_id as string) && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  ✓ Ready to show statistics on pipeline execution
                </p>
              </div>
            )}
          </div>
        );

      case "column_info":
        return (
          <div className="space-y-4">
            <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
              <p className="text-sm text-amber-800 mb-3">
                Column information and data quality
              </p>
              <p className="text-xs text-amber-700">
                Shows data types, missing values, and unique counts
              </p>
            </div>
            {renderField("dataset_id", "Dataset Source", "text")}
            {renderField("show_dtypes", "Show Data Types", "checkbox")}
            {renderField("show_missing", "Show Missing Values", "checkbox")}
            {renderField("show_unique", "Show Unique Counts", "checkbox")}
            {(config.dataset_id as string) && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  ✓ Ready to display column info on pipeline execution
                </p>
              </div>
            )}
          </div>
        );

      case "chart_view": {
        // Filter y_column options to exclude x_column
        const xColumn = config.x_column as string;
        const yColumnOptions = availableColumns
          .filter((col) => col !== xColumn)
          .map((col) => ({ value: col, label: col }));

        return (
          <div className="space-y-4">
            <div className="p-4 bg-pink-50 border border-pink-200 rounded-lg">
              <p className="text-sm text-pink-800 mb-3">
                Visualize your data with charts
              </p>
              <p className="text-xs text-pink-700">
                Choose chart type and columns to visualize
              </p>
            </div>
            {renderField("dataset_id", "Dataset Source", "text")}
            {renderField("chart_type", "Chart Type", "select", [
              { value: "bar", label: "Bar Chart" },
              { value: "line", label: "Line Chart" },
              { value: "scatter", label: "Scatter Plot" },
              { value: "histogram", label: "Histogram" },
              { value: "pie", label: "Pie Chart" },
            ])}
            {availableColumns.length > 0 ? (
              <>
                {renderField(
                  "x_column",
                  "X-Axis Column",
                  "select",
                  availableColumns.map((col) => ({ value: col, label: col })),
                )}
                {xColumn && (
                  <div className="p-2 bg-blue-50 border border-blue-200 rounded text-xs text-blue-700">
                    ℹ️ Y-axis will exclude "{xColumn}" from available options
                  </div>
                )}
                {renderField(
                  "y_column",
                  "Y-Axis Column",
                  "select",
                  yColumnOptions,
                )}
              </>
            ) : (
              <>
                {renderField("x_column", "X-Axis Column", "text")}
                {renderField("y_column", "Y-Axis Column", "text")}
              </>
            )}
            {(config.dataset_id as string) && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  ✓ Ready to display chart on pipeline execution
                </p>
              </div>
            )}
          </div>
        );
      }

      case "missing_value_handler":
      case "encoding":
      case "transformation":
      case "scaling":
      case "feature_selection":
        return (
          <FeatureEngineeringConfigPanel
            node={node}
            config={config}
            availableColumns={availableColumns}
            updateField={updateField}
            setConfig={setConfig}
            renderField={renderField}
          />
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
