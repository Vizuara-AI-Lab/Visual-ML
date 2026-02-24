import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Save,
  Upload,
  Database,
  FileSpreadsheet,
  CheckCircle2,
  FlaskConical,
  Table,
  Eye,
  BarChart3,
  Info,
  LineChart,
  Rows3,
  Columns3,
  BarChart,
  ScatterChart,
  PieChart,
} from "lucide-react";
import { useState, useEffect, useCallback } from "react";
import { apiClient } from "../../lib/axios";
import { CameraCapturePanel } from "./CameraCapturePanel";
import { LiveCameraTestPanel } from "./LiveCameraTestPanel";
import { getNodeByType } from "../../config/nodeDefinitions";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { UploadDatasetButton } from "./UploadDatasetButton";
import { FeatureEngineeringConfigPanel } from "./FeatureEngineeringConfigPanel";
import { SplitConfigPanel } from "./SplitConfigPanel";
import { MLAlgorithmConfigPanel } from "./MLAlgorithmConfigPanel";
import { ResultMetricsConfigPanel } from "./ResultMetricsConfigPanel";
import { GenAIConfigPanel } from "./GenAIConfigPanel";
import {
  listProjectDatasets,
  type DatasetMetadata,
} from "../../lib/api/datasetApi";

const SAMPLE_DATASETS: Record<
  string,
  {
    label: string;
    description: string;
    rows: string;
    cols: string;
    task: string;
  }
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

interface ConfigModalProps {
  nodeId: string | null;
  onClose: () => void;
}

export const ConfigModal = ({ nodeId, onClose }: ConfigModalProps) => {
  const {
    getNodeById,
    updateNodeConfig,
    datasetMetadata,
    currentProjectId,
    edges,
  } = usePlaygroundStore();

  const node = nodeId ? getNodeById(nodeId) : null;
  const nodeDef = node ? getNodeByType(node.type) : null;

  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [userDatasets, setUserDatasets] = useState<DatasetMetadata[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [buildingDataset, setBuildingDataset] = useState(false);

  // Reset config when nodeId changes to prevent config sharing between nodes
  // ALWAYS load the latest config from the store (no caching)
  useEffect(() => {
    if (node) {
      setConfig({ ...node.data.config } || {});
    } else {
      setConfig({});
    }
  }, [nodeId, node?.data.config]);

  // Fetch user's datasets for select_dataset node
  useEffect(() => {
    if (node?.type === "select_dataset" && currentProjectId) {
      setLoadingDatasets(true);
      listProjectDatasets(parseInt(currentProjectId))
        .then((datasets) => {
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

  // Auto-fill dataset_id for view and preprocessing nodes when source is connected
  useEffect(() => {
    const isAutoFillNode = [
      "table_view",
      "data_preview",
      "statistics_view",
      "column_info",
      "chart_view",
      "missing_value_handler",
      "encoding",
      "transformation",
      "scaling",
      "feature_selection",
      "split",
      "linear_regression",
      "logistic_regression",
      "decision_tree",
      "random_forest",
      "r2_score",
      "mse_score",
      "rmse_score",
      "mae_score",
      "confusion_matrix",
      "classification_report",
      "accuracy_score",
      "roc_curve",
      "feature_importance",
      "residual_plot",
      "prediction_table",
      // Image pipeline nodes
      "image_preprocessing",
      "image_augmentation",
      "image_split",
      "cnn_classifier",
      "image_predictions",
    ].includes(node?.type || "");

    if (isAutoFillNode && nodeId) {
      const incomingEdge = edges.find((edge) => edge.target === nodeId);
      if (incomingEdge) {
        const sourceNode = getNodeById(incomingEdge.source);
        const sourceResult = sourceNode?.data.result as
          | Record<string, unknown>
          | undefined;
        const sourceConfig = sourceNode?.data.config;

        // Check if this is a result/metrics node (needs model_output_id from ML algorithm)
        const isResultNode = [
          "r2_score",
          "mse_score",
          "rmse_score",
          "mae_score",
          "confusion_matrix",
          "classification_report",
          "accuracy_score",
          "roc_curve",
          "feature_importance",
          "residual_plot",
          "prediction_table",
        ].includes(node?.type || "");

        // Check if source is an ML algorithm node
        const isMLAlgorithmNode = [
          "linear_regression",
          "logistic_regression",
          "decision_tree",
          "random_forest",
        ].includes(sourceNode?.type || "");

        if (isResultNode && isMLAlgorithmNode) {
          // Result nodes get model_output_id from ML algorithm node
          const modelId = sourceResult?.model_id;

          if (modelId) {
            setConfig((prev) => ({
              ...prev,
              model_output_id: modelId,
            }));
          }
        } else {
          // Check if this is an ML node (needs train_dataset_id from split)
          const isMLNode = [
            "linear_regression",
            "logistic_regression",
            "decision_tree",
            "random_forest",
          ].includes(node?.type || "");

          // CNN classifier gets train_dataset_id from image_split
          if (node?.type === "cnn_classifier" && sourceNode?.type === "image_split") {
            const trainDatasetId = sourceResult?.train_dataset_id;
            setConfig((prev) => ({
              ...prev,
              train_dataset_id: trainDatasetId || prev.train_dataset_id,
            }));
          } else if (node?.type === "image_predictions" && sourceNode?.type === "cnn_classifier") {
            // image_predictions gets test_dataset_id + model_path from cnn_classifier
            const testDatasetId = sourceResult?.test_dataset_id;
            const modelPath = sourceResult?.model_path;
            const modelId = sourceResult?.model_id;
            setConfig((prev) => ({
              ...prev,
              test_dataset_id: testDatasetId || prev.test_dataset_id,
              model_path: modelPath || prev.model_path,
              model_id: modelId || prev.model_id,
            }));
          } else if (isMLNode && sourceNode?.type === "split") {
            // ML nodes get train_dataset_id and target_column from split node
            const trainDatasetId = sourceResult?.train_dataset_id;
            const targetColumn = sourceResult?.target_column;

            setConfig((prev) => ({
              ...prev,
              train_dataset_id: trainDatasetId || prev.train_dataset_id,
              target_column: targetColumn || prev.target_column,
            }));
          } else {
            // For other nodes, use regular dataset_id auto-fill
            // Priority 1: Check result for specific dataset IDs from preprocessing nodes
            let datasetId =
              sourceResult?.preprocessed_dataset_id ||
              sourceResult?.encoded_dataset_id ||
              sourceResult?.transformed_dataset_id ||
              sourceResult?.scaled_dataset_id ||
              sourceResult?.selected_dataset_id ||
              // Image pipeline specific IDs
              sourceResult?.augmented_dataset_id ||
              sourceResult?.camera_dataset_id ||
              sourceResult?.dataset_id;

            // Priority 2: Check config for dataset_id (for upload/select nodes)
            if (!datasetId) {
              datasetId = sourceConfig?.dataset_id;
            }

            if (datasetId) {
              console.log(
                "✅ Auto-filling dataset_id from connected source:",
                datasetId,
              );
              setConfig((prev) => ({
                ...prev,
                dataset_id: datasetId,
              }));
            } else {
              console.warn(
                "⚠️ No dataset_id found in source node - connection may not be properly established",
              );
            }
          }
        }
      }
    }
  }, [node?.type, nodeId, edges, getNodeById]);

  // Build a camera dataset from captured pixel arrays and store dataset_id in config
  const buildCameraDataset = useCallback(
    async (payload: {
      class_names: string[];
      target_size: string;
      images_per_class: Record<string, number[][]>;
    }) => {
      if (!nodeId) return;
      setBuildingDataset(true);
      try {
        const response = await apiClient.post("/ml/camera/dataset", {
          class_names: payload.class_names,
          target_size: payload.target_size,
          images_per_class: payload.images_per_class,
        });
        const data = response.data as {
          dataset_id: string;
          n_rows: number;
          n_columns: number;
          image_width: number;
          image_height: number;
        };
        setConfig((prev) => {
          const updated = {
            ...prev,
            dataset_id: data.dataset_id,
            n_rows: data.n_rows,
            n_columns: data.n_columns,
            class_names: payload.class_names.join(","),
            image_width: data.image_width,
            image_height: data.image_height,
          };
          updateNodeConfig(nodeId, updated);
          return updated;
        });
      } catch (err) {
        console.error("❌ Camera dataset build failed:", err);
      } finally {
        setBuildingDataset(false);
      }
    },
    [nodeId, updateNodeConfig],
  );

  // Live camera predict — calls /ml/camera/predict with a flat pixel array
  const cameraLivePredict = useCallback(
    async (pixels: number[]) => {
      const modelPath = (config.model_path as string) || "";
      const response = await apiClient.post("/ml/camera/predict", {
        model_path: modelPath,
        pixels,
      });
      const data = response.data as {
        class_name: string;
        confidence: number;
        all_scores: { class_name: string; score: number }[];
      };
      // Resolve class index → class name from connected node's result
      const incomingEdge = edges.find((e) => e.target === nodeId);
      const srcResult = incomingEdge
        ? (getNodeById(incomingEdge.source)?.data.result as Record<string, unknown> | undefined)
        : undefined;
      const classNamesArr =
        (srcResult?.class_names as string[]) ||
        (config.class_names as string)?.split(",").map((s) => s.trim()).filter(Boolean) ||
        [];
      const mapIdx = (raw: string) => {
        const idx = parseInt(raw);
        return !isNaN(idx) && classNamesArr[idx] ? classNamesArr[idx] : raw;
      };
      return {
        class_name: mapIdx(data.class_name),
        confidence: data.confidence,
        all_scores: (data.all_scores as { class_name: string; score: number }[]).map((s) => ({
          class_name: mapIdx(s.class_name),
          score: s.score,
        })),
      };
    },
    [config, edges, nodeId, getNodeById],
  );

  // Early return AFTER ALL hooks have been called
  if (!nodeId || !node || !nodeDef) return null;

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

  // Recursively get available columns from connected source node or its parents
  const getAvailableColumns = (): string[] => {
    if (!connectedSourceNode) return [];

    console.log(
      "🔍 Getting columns from source node:",
      connectedSourceNode.type,
      connectedSourceNode.data,
    );

    // Helper function to recursively search for columns
    const searchForColumns = (
      node: typeof connectedSourceNode,
      visited = new Set<string>(),
    ): string[] => {
      if (!node || visited.has(node.id)) return [];
      visited.add(node.id);

      const sourceConfig = node.data.config;
      const sourceResult = node.data.result as
        | Record<string, unknown>
        | undefined;

      console.log(`  📦 Checking node ${node.type}:`, {
        config_columns: sourceConfig?.columns,
        result_columns: sourceResult?.columns,
        encoded_columns: sourceResult?.encoded_columns,
        final_columns: sourceResult?.final_columns,
        selected_feature_names: sourceResult?.selected_feature_names,
      });

      // Define preprocessing nodes where config.columns is a SELECTION, not all available columns
      const preprocessingNodesWithColumnSelection = [
        "scaling",
        "transformation",
        "feature_selection",
      ];

      // Priority 1: Check config.columns ONLY for dataset source nodes
      // (for preprocessing nodes, config.columns is what user selected to process, not all available)
      const isDatasetSourceNode = ["upload_file", "select_dataset"].includes(
        node.type,
      );

      if (
        isDatasetSourceNode &&
        sourceConfig?.columns &&
        Array.isArray(sourceConfig.columns)
      ) {
        console.log(
          `  ✅ Found columns from dataset source (${node.type}):`,
          sourceConfig.columns,
        );
        return sourceConfig.columns as string[];
      }

      // Priority 2: Check result.columns (for executed nodes like preprocessing, view nodes)
      if (sourceResult?.columns && Array.isArray(sourceResult.columns)) {
        console.log(
          `  ✅ Found columns from result (${node.type}):`,
          sourceResult.columns,
        );
        return sourceResult.columns as string[];
      }

      // Priority 3: Check for preprocessed_dataset columns (for preprocessing nodes)
      if (
        sourceConfig?.preprocessed_columns &&
        Array.isArray(sourceConfig.preprocessed_columns)
      ) {
        return sourceConfig.preprocessed_columns as string[];
      }

      // Priority 4: Check result.encoded_columns (for encoding nodes)
      if (
        sourceResult?.encoded_columns &&
        Array.isArray(sourceResult.encoded_columns)
      ) {
        return sourceResult.encoded_columns as string[];
      }

      // Priority 4.5: Check result.new_columns combined with original columns
      // For encoding nodes, we need to get ALL columns, not just encoded ones
      // If the node has been executed, try to load columns from the actual dataset
      if (
        sourceResult?.encoded_dataset_id ||
        sourceResult?.transformed_dataset_id ||
        sourceResult?.scaled_dataset_id ||
        sourceResult?.selected_dataset_id ||
        sourceResult?.preprocessed_dataset_id
      ) {
        // Node has been executed - columns should be in the result
        // If not found above, trace back to parent to get original columns
        const parentEdge = edges.find((edge) => edge.target === node.id);
        if (parentEdge) {
          const parentNode = getNodeById(parentEdge.source);
          if (parentNode) {
            const parentColumns = searchForColumns(parentNode, visited);
            if (parentColumns.length > 0) {
              return parentColumns;
            }
          }
        }
      }

      // Priority 5: Check result.final_columns (for transformation/scaling nodes)
      if (
        sourceResult?.final_columns &&
        Array.isArray(sourceResult.final_columns)
      ) {
        return sourceResult.final_columns as string[];
      }

      // Priority 6: Check result.selected_feature_names (for feature_selection nodes)
      if (
        sourceResult?.selected_feature_names &&
        Array.isArray(sourceResult.selected_feature_names)
      ) {
        return sourceResult.selected_feature_names as string[];
      }

      // Priority 6.5: Check split node columns (for ML nodes)
      if (
        node.type === "split" &&
        sourceResult?.columns &&
        Array.isArray(sourceResult.columns)
      ) {
        return sourceResult.columns as string[];
      }

      // Priority 6.6: Check split node feature_columns (for ML nodes - excludes target)
      if (
        node.type === "split" &&
        sourceResult?.feature_columns &&
        Array.isArray(sourceResult.feature_columns)
      ) {
        // Include target column too
        const targetCol = sourceResult?.target_column;
        const allColumns = targetCol
          ? [...(sourceResult.feature_columns as string[]), targetCol as string]
          : (sourceResult.feature_columns as string[]);
        return allColumns;
      }

      // Priority 7: If this is a view/processing node without columns, trace back to its source
      const nodeTypes = [
        "table_view",
        "data_preview",
        "statistics_view",
        "column_info",
        "chart_view",
        "missing_value_handler",
        "encoding",
        "transformation",
        "scaling",
        "feature_selection",
      ];
      if (nodeTypes.includes(node.type || "")) {
        // Find the parent node of this view/processing node
        const parentEdge = edges.find((edge) => edge.target === node.id);
        if (parentEdge) {
          const parentNode = getNodeById(parentEdge.source);
          if (parentNode) {
            return searchForColumns(parentNode, visited);
          }
        }
      }

      return [];
    };

    return searchForColumns(connectedSourceNode);
  };

  const availableColumns = getAvailableColumns();

  const handleSave = () => {
    updateNodeConfig(node.id, config);
    onClose();
  };

  const handleFieldChange = (fieldName: string, value: unknown) => {
    // Special handling for split node train/test ratios
    if (node.type === "split" && fieldName === "train_ratio") {
      const trainRatio = Number(value);
      const testRatio = Number((1 - trainRatio).toFixed(2));
      setConfig((prev) => ({
        ...prev,
        train_ratio: trainRatio,
        test_ratio: testRatio,
      }));
      return;
    }

    // Default behavior for all other fields
    setConfig((prev) => ({ ...prev, [fieldName]: value }));
  };

  // Auto-fill logic based on dataset metadata
  const getAutoFilledValue = (fieldName: string): unknown => {
    // For result nodes, auto-fill model_output_id from connected ML algorithm node
    if (fieldName === "model_output_id" && connectedSourceNode) {
      const sourceResult = connectedSourceNode.data.result as
        | Record<string, unknown>
        | undefined;

      // Check if source is an ML algorithm node and has model_id
      const isMLAlgorithmNode = [
        "linear_regression",
        "logistic_regression",
        "decision_tree",
        "random_forest",
      ].includes(connectedSourceNode.type);

      if (isMLAlgorithmNode && sourceResult?.model_id) {
        return sourceResult.model_id;
      }
    }

    // For view nodes, auto-fill dataset_id from connected source node
    if (fieldName === "dataset_id" && connectedSourceNode) {
      const sourceResult = connectedSourceNode.data.result as
        | Record<string, unknown>
        | undefined;
      const sourceConfig = connectedSourceNode.data.config;

      // Priority 1: Check result for preprocessing node dataset IDs
      const datasetId =
        sourceResult?.preprocessed_dataset_id ||
        sourceResult?.encoded_dataset_id ||
        sourceResult?.transformed_dataset_id ||
        sourceResult?.scaled_dataset_id ||
        sourceResult?.selected_dataset_id ||
        sourceResult?.augmented_dataset_id ||
        sourceResult?.camera_dataset_id ||
        sourceResult?.dataset_id ||
        sourceConfig?.dataset_id;

      if (datasetId) {
        return datasetId;
      }
    }

    // Image pipeline: auto-fill train_dataset_id for cnn_classifier from image_split
    if (fieldName === "train_dataset_id" && connectedSourceNode?.type === "image_split") {
      const sourceResult = connectedSourceNode.data.result as Record<string, unknown> | undefined;
      if (sourceResult?.train_dataset_id) return sourceResult.train_dataset_id;
    }

    // Image pipeline: auto-fill test_dataset_id + model_path for image_predictions from cnn_classifier
    if (fieldName === "test_dataset_id" && connectedSourceNode?.type === "cnn_classifier") {
      const sourceResult = connectedSourceNode.data.result as Record<string, unknown> | undefined;
      if (sourceResult?.test_dataset_id) return sourceResult.test_dataset_id;
    }
    if (fieldName === "model_path" && connectedSourceNode?.type === "cnn_classifier") {
      const sourceResult = connectedSourceNode.data.result as Record<string, unknown> | undefined;
      if (sourceResult?.model_path) return sourceResult.model_path;
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
          className="absolute inset-0 bg-black/40 backdrop-blur-sm"
          onClick={onClose}
        />

        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative bg-white border border-slate-200 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden z-10"
        >
          {/* Header */}
          <div
            className="px-6 py-4 border-b border-slate-200 flex items-center justify-between"
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
                <h3 className="text-lg font-semibold text-slate-900">
                  {nodeDef.label}
                </h3>
                <p className="text-sm text-slate-500">{nodeDef.description}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-500" />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 overflow-y-auto max-h-[calc(80vh-180px)]">
            {/* Special handling for feature engineering nodes */}
            {node?.type &&
            [
              "missing_value_handler",
              "encoding",
              "transformation",
              "scaling",
              "feature_selection",
            ].includes(node.type) ? (
              <FeatureEngineeringConfigPanel
                node={node}
                config={config}
                availableColumns={availableColumns}
                updateField={handleFieldChange}
                setConfig={setConfig}
                connectedSourceNode={connectedSourceNode}
                renderField={(field, label, type = "text", options) => {
                  const currentValue = config[field];

                  if (type === "select") {
                    return (
                      <div key={field} className="mb-4">
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                          {label}
                        </label>
                        <select
                          value={(currentValue as string) || ""}
                          onChange={(e) =>
                            handleFieldChange(field, e.target.value)
                          }
                          className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="">-- Select {label} --</option>
                          {Array.isArray(options) &&
                            options.map((opt) => (
                              <option key={opt.value} value={opt.value}>
                                {opt.label}
                              </option>
                            ))}
                        </select>
                      </div>
                    );
                  }

                  if (type === "checkbox") {
                    return (
                      <div key={field} className="mb-4">
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={(currentValue as boolean) || false}
                            onChange={(e) =>
                              handleFieldChange(field, e.target.checked)
                            }
                            className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 bg-white"
                          />
                          <span className="text-sm font-medium text-slate-700">
                            {label}
                          </span>
                        </label>
                      </div>
                    );
                  }

                  if (type === "number") {
                    const numOptions =
                      options && !Array.isArray(options) ? options : {};
                    return (
                      <div key={field} className="mb-4">
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                          {label}
                        </label>
                        <input
                          type="number"
                          value={(currentValue as number) ?? ""}
                          onChange={(e) =>
                            handleFieldChange(
                              field,
                              parseFloat(e.target.value) || 0,
                            )
                          }
                          step={numOptions.step || 0.01}
                          min={numOptions.min}
                          max={numOptions.max}
                          className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    );
                  }

                  // Default text input
                  return (
                    <div key={field} className="mb-4">
                      <label className="block text-sm font-medium text-slate-700 mb-2">
                        {label}
                      </label>
                      <input
                        type="text"
                        value={(currentValue as string) || ""}
                        onChange={(e) =>
                          handleFieldChange(field, e.target.value)
                        }
                        readOnly={field === "dataset_id"}
                        className={`w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 ${field === "dataset_id" ? "cursor-not-allowed opacity-75" : ""}`}
                      />
                      {field === "dataset_id" && connectedSourceNode && (
                        <p className="text-xs text-green-600 mt-1">
                          ✓ Connected to: {connectedSourceNode.data.label}
                        </p>
                      )}
                    </div>
                  );
                }}
              />
            ) : node.type === "upload_file" ? (
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
                          updateNodeConfig(node.id, newConfig);
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
                {config.dataset_id && (
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
                          {config.filename as string}
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
                                .slice(0, 8)
                                .map((col: string) => (
                                  <span
                                    key={col}
                                    className="px-2 py-0.5 bg-white rounded-md text-[11px] text-slate-600 border border-slate-200/80 font-medium"
                                  >
                                    {col}
                                  </span>
                                ))}
                              {(config.columns as string[]).length > 8 && (
                                <span className="px-2 py-0.5 bg-slate-100 rounded-md text-[11px] text-slate-500 font-medium">
                                  +{(config.columns as string[]).length - 8}{" "}
                                  more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      <div className="pt-2 border-t border-emerald-200/40">
                        <div className="text-[10px] text-slate-400 font-mono truncate">
                          ID: {config.dataset_id as string}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : node.type === "select_dataset" ? (
              <div className="space-y-4">
                {/* Header + Selector */}
                <div className="rounded-xl border border-emerald-200 bg-linear-to-br from-emerald-50/80 to-teal-50/80 p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-9 h-9 rounded-lg bg-emerald-100 flex items-center justify-center">
                      <Database className="w-4 h-4 text-emerald-600" />
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
                {config.dataset_id && (
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
                          {config.filename as string}
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
                                .slice(0, 8)
                                .map((col: string) => (
                                  <span
                                    key={col}
                                    className="px-2 py-0.5 bg-white rounded-md text-[11px] text-slate-600 border border-slate-200/80 font-medium"
                                  >
                                    {col}
                                  </span>
                                ))}
                              {(config.columns as string[]).length > 8 && (
                                <span className="px-2 py-0.5 bg-slate-100 rounded-md text-[11px] text-slate-500 font-medium">
                                  +{(config.columns as string[]).length - 8}{" "}
                                  more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      <div className="pt-2 border-t border-blue-200/40">
                        <div className="text-[10px] text-slate-400 font-mono truncate">
                          ID: {config.dataset_id as string}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : node.type === "sample_dataset" ? (
              (() => {
                const selectedName =
                  (config.dataset_name as string) || "iris";
                const datasetInfo = SAMPLE_DATASETS[selectedName];
                return (
                  <div className="space-y-4">
                    {/* Header + Selector */}
                    <div className="rounded-xl border border-violet-200 bg-linear-to-br from-violet-50/80 to-purple-50/80 p-4">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-9 h-9 rounded-lg bg-violet-100 flex items-center justify-center">
                          <FlaskConical className="w-4 h-4 text-violet-600" />
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
                        onChange={(e) =>
                          handleFieldChange("dataset_name", e.target.value)
                        }
                        className="w-full px-3 py-2.5 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-colors"
                      >
                        {Object.entries(SAMPLE_DATASETS).map(
                          ([value, info]) => (
                            <option key={value} value={value}>
                              {info.label} ({info.task})
                            </option>
                          ),
                        )}
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
              })()
            ) : node.type === "table_view" ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="rounded-xl border border-cyan-200 bg-linear-to-br from-cyan-50/80 to-sky-50/80 p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-cyan-100 flex items-center justify-center">
                      <Table className="w-4 h-4 text-cyan-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-800">Table View</p>
                      <p className="text-xs text-slate-500">Display your dataset as an interactive table</p>
                    </div>
                  </div>
                </div>

                {/* Connection Status */}
                <div className={`rounded-xl border p-3 ${connectedSourceNode ? "border-green-200 bg-green-50/60" : "border-amber-200 bg-amber-50/60"}`}>
                  <div className="flex items-center gap-2">
                    {connectedSourceNode ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
                    ) : (
                      <span className="text-amber-500 text-sm">⚠</span>
                    )}
                    <span className={`text-xs font-bold uppercase tracking-wider ${connectedSourceNode ? "text-green-800" : "text-amber-800"}`}>
                      {connectedSourceNode ? "Data Source Connected" : "No Data Source"}
                    </span>
                  </div>
                  <p className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}>
                    {connectedSourceNode
                      ? `Connected to: ${connectedSourceNode.data.label}`
                      : "Connect a data source node to enable table view"}
                  </p>
                  {config.dataset_id && (
                    <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
                      Dataset ID: {config.dataset_id as string}
                    </div>
                  )}
                </div>

                {/* Max Rows Setting */}
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <label className="block text-sm font-medium text-slate-700 mb-3">
                    Maximum Rows to Display
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      value={(config.max_rows as number) || 100}
                      onChange={(e) => handleFieldChange("max_rows", parseInt(e.target.value))}
                      min={10}
                      max={1000}
                      step={10}
                      className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, #06B6D4 0%, #06B6D4 ${(((config.max_rows as number) || 100) / 1000) * 100}%, #E2E8F0 ${(((config.max_rows as number) || 100) / 1000) * 100}%, #E2E8F0 100%)`,
                      }}
                    />
                    <div className="w-16 px-2 py-1.5 bg-cyan-50 border border-cyan-200 rounded-lg text-center">
                      <span className="text-sm font-bold text-cyan-800">{(config.max_rows as number) || 100}</span>
                    </div>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">Slide to adjust (10 - 1,000 rows)</p>
                </div>

                {/* Quick Info */}
                <div className="flex items-start gap-2 px-3 py-2.5 bg-slate-50 rounded-lg border border-slate-100">
                  <Rows3 className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
                  <p className="text-xs text-slate-500 leading-relaxed">
                    The table view displays your dataset with sortable columns. Connect a data source and run the pipeline to see your data.
                  </p>
                </div>
              </div>
            ) : node.type === "data_preview" ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="rounded-xl border border-violet-200 bg-linear-to-br from-violet-50/80 to-purple-50/80 p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-violet-100 flex items-center justify-center">
                      <Eye className="w-4 h-4 text-violet-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-800">Data Preview</p>
                      <p className="text-xs text-slate-500">Quick look at the first and last rows of your data</p>
                    </div>
                  </div>
                </div>

                {/* Connection Status */}
                <div className={`rounded-xl border p-3 ${connectedSourceNode ? "border-green-200 bg-green-50/60" : "border-amber-200 bg-amber-50/60"}`}>
                  <div className="flex items-center gap-2">
                    {connectedSourceNode ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
                    ) : (
                      <span className="text-amber-500 text-sm">⚠</span>
                    )}
                    <span className={`text-xs font-bold uppercase tracking-wider ${connectedSourceNode ? "text-green-800" : "text-amber-800"}`}>
                      {connectedSourceNode ? "Data Source Connected" : "No Data Source"}
                    </span>
                  </div>
                  <p className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}>
                    {connectedSourceNode
                      ? `Connected to: ${connectedSourceNode.data.label}`
                      : "Connect a data source node to enable preview"}
                  </p>
                  {config.dataset_id && (
                    <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
                      Dataset ID: {config.dataset_id as string}
                    </div>
                  )}
                </div>

                {/* Row Settings - Side by Side */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-6 h-6 rounded-md bg-violet-100 flex items-center justify-center">
                        <span className="text-[10px] font-bold text-violet-700">H</span>
                      </div>
                      <label className="text-sm font-medium text-slate-700">First N Rows</label>
                    </div>
                    <input
                      type="number"
                      value={(config.head_rows as number) || 5}
                      onChange={(e) => handleFieldChange("head_rows", parseInt(e.target.value) || 5)}
                      min={1}
                      max={50}
                      className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 text-center text-lg font-semibold focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                    />
                    <p className="text-[10px] text-slate-400 mt-1.5 text-center">Head rows (1-50)</p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-6 h-6 rounded-md bg-indigo-100 flex items-center justify-center">
                        <span className="text-[10px] font-bold text-indigo-700">T</span>
                      </div>
                      <label className="text-sm font-medium text-slate-700">Last N Rows</label>
                    </div>
                    <input
                      type="number"
                      value={(config.tail_rows as number) || 5}
                      onChange={(e) => handleFieldChange("tail_rows", parseInt(e.target.value) || 5)}
                      min={1}
                      max={50}
                      className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 text-center text-lg font-semibold focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                    />
                    <p className="text-[10px] text-slate-400 mt-1.5 text-center">Tail rows (1-50)</p>
                  </div>
                </div>

                {/* Preview Summary */}
                <div className="flex items-center justify-center gap-3 px-4 py-3 bg-violet-50/60 rounded-xl border border-violet-100">
                  <span className="text-xs text-violet-700 font-medium">
                    Showing first <span className="font-bold">{(config.head_rows as number) || 5}</span> and last <span className="font-bold">{(config.tail_rows as number) || 5}</span> rows = <span className="font-bold text-violet-900">{((config.head_rows as number) || 5) + ((config.tail_rows as number) || 5)}</span> total rows
                  </span>
                </div>
              </div>
            ) : node.type === "statistics_view" ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="rounded-xl border border-emerald-200 bg-linear-to-br from-emerald-50/80 to-green-50/80 p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-emerald-100 flex items-center justify-center">
                      <BarChart3 className="w-4 h-4 text-emerald-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-800">Statistics View</p>
                      <p className="text-xs text-slate-500">Statistical summary of your dataset</p>
                    </div>
                  </div>
                </div>

                {/* Connection Status */}
                <div className={`rounded-xl border p-3 ${connectedSourceNode ? "border-green-200 bg-green-50/60" : "border-amber-200 bg-amber-50/60"}`}>
                  <div className="flex items-center gap-2">
                    {connectedSourceNode ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
                    ) : (
                      <span className="text-amber-500 text-sm">⚠</span>
                    )}
                    <span className={`text-xs font-bold uppercase tracking-wider ${connectedSourceNode ? "text-green-800" : "text-amber-800"}`}>
                      {connectedSourceNode ? "Data Source Connected" : "No Data Source"}
                    </span>
                  </div>
                  <p className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}>
                    {connectedSourceNode
                      ? `Connected to: ${connectedSourceNode.data.label}`
                      : "Connect a data source node to compute statistics"}
                  </p>
                  {config.dataset_id && (
                    <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
                      Dataset ID: {config.dataset_id as string}
                    </div>
                  )}
                </div>

                {/* Include All Toggle */}
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-slate-700">Include All Columns</p>
                      <p className="text-xs text-slate-500 mt-0.5">Show statistics for every column in the dataset</p>
                    </div>
                    <button
                      onClick={() => handleFieldChange("include_all", !(config.include_all as boolean))}
                      className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
                        (config.include_all as boolean) !== false ? "bg-emerald-500" : "bg-slate-300"
                      }`}
                    >
                      <div
                        className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform duration-200 ${
                          (config.include_all as boolean) !== false ? "translate-x-5.5" : "translate-x-0.5"
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {/* Stats Info */}
                <div className="rounded-xl border border-emerald-100 bg-emerald-50/40 p-4">
                  <p className="text-xs font-semibold text-emerald-800 mb-2">Statistics Included:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {["Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max"].map((stat) => (
                      <span key={stat} className="px-2 py-0.5 bg-white rounded-md text-[11px] text-emerald-700 border border-emerald-200 font-medium">
                        {stat}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : node.type === "column_info" ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="rounded-xl border border-amber-200 bg-linear-to-br from-amber-50/80 to-orange-50/80 p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-amber-100 flex items-center justify-center">
                      <Info className="w-4 h-4 text-amber-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-800">Column Info</p>
                      <p className="text-xs text-slate-500">Inspect column types, missing values & unique counts</p>
                    </div>
                  </div>
                </div>

                {/* Connection Status */}
                <div className={`rounded-xl border p-3 ${connectedSourceNode ? "border-green-200 bg-green-50/60" : "border-amber-200 bg-amber-50/60"}`}>
                  <div className="flex items-center gap-2">
                    {connectedSourceNode ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
                    ) : (
                      <span className="text-amber-500 text-sm">⚠</span>
                    )}
                    <span className={`text-xs font-bold uppercase tracking-wider ${connectedSourceNode ? "text-green-800" : "text-amber-800"}`}>
                      {connectedSourceNode ? "Data Source Connected" : "No Data Source"}
                    </span>
                  </div>
                  <p className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}>
                    {connectedSourceNode
                      ? `Connected to: ${connectedSourceNode.data.label}`
                      : "Connect a data source node to inspect columns"}
                  </p>
                  {config.dataset_id && (
                    <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
                      Dataset ID: {config.dataset_id as string}
                    </div>
                  )}
                </div>

                {/* Display Options */}
                <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3">
                  <p className="text-sm font-medium text-slate-700 mb-1">Display Options</p>

                  {[
                    { field: "show_dtypes", label: "Data Types", desc: "Show column data types (int, float, object...)", color: "blue" },
                    { field: "show_missing", label: "Missing Values", desc: "Show count & percentage of missing values", color: "red" },
                    { field: "show_unique", label: "Unique Counts", desc: "Show number of unique values per column", color: "purple" },
                  ].map((opt) => (
                    <div key={opt.field} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${opt.color === "blue" ? "bg-blue-400" : opt.color === "red" ? "bg-red-400" : "bg-purple-400"}`} />
                        <div>
                          <p className="text-sm text-slate-700 font-medium">{opt.label}</p>
                          <p className="text-[11px] text-slate-400">{opt.desc}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => handleFieldChange(opt.field, !(config[opt.field] as boolean))}
                        className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
                          (config[opt.field] as boolean) !== false ? "bg-amber-500" : "bg-slate-300"
                        }`}
                      >
                        <div
                          className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform duration-200 ${
                            (config[opt.field] as boolean) !== false ? "translate-x-5.5" : "translate-x-0.5"
                          }`}
                        />
                      </button>
                    </div>
                  ))}
                </div>

                {/* Available Columns Preview */}
                {availableColumns.length > 0 && (
                  <div className="rounded-xl border border-slate-100 bg-slate-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Columns3 className="w-3.5 h-3.5 text-slate-400" />
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        {availableColumns.length} Columns Detected
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {availableColumns.slice(0, 10).map((col) => (
                        <span key={col} className="px-2 py-0.5 bg-white rounded-md text-[11px] text-slate-600 border border-slate-200 font-medium">
                          {col}
                        </span>
                      ))}
                      {availableColumns.length > 10 && (
                        <span className="px-2 py-0.5 bg-slate-100 rounded-md text-[11px] text-slate-500 font-medium">
                          +{availableColumns.length - 10} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : node.type === "chart_view" ? (
              (() => {
                const chartType = (config.chart_type as string) || "bar";
                const isPieChart = chartType === "pie";
                const chartTypes = [
                  { value: "bar", label: "Bar", Icon: BarChart },
                  { value: "line", label: "Line", Icon: LineChart },
                  { value: "scatter", label: "Scatter", Icon: ScatterChart },
                  { value: "histogram", label: "Histogram", Icon: BarChart3 },
                  { value: "pie", label: "Pie", Icon: PieChart },
                ];
                return (
                  <div className="space-y-4">
                    {/* Header */}
                    <div className="rounded-xl border border-pink-200 bg-linear-to-br from-pink-50/80 to-rose-50/80 p-4">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg bg-pink-100 flex items-center justify-center">
                          <LineChart className="w-4 h-4 text-pink-600" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-slate-800">Chart View</p>
                          <p className="text-xs text-slate-500">Visualize your data with interactive charts</p>
                        </div>
                      </div>
                    </div>

                    {/* Connection Status */}
                    <div className={`rounded-xl border p-3 ${connectedSourceNode ? "border-green-200 bg-green-50/60" : "border-amber-200 bg-amber-50/60"}`}>
                      <div className="flex items-center gap-2">
                        {connectedSourceNode ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
                        ) : (
                          <span className="text-amber-500 text-sm">⚠</span>
                        )}
                        <span className={`text-xs font-bold uppercase tracking-wider ${connectedSourceNode ? "text-green-800" : "text-amber-800"}`}>
                          {connectedSourceNode ? "Data Source Connected" : "No Data Source"}
                        </span>
                      </div>
                      <p className={`text-xs mt-1 ${connectedSourceNode ? "text-green-700" : "text-amber-700"}`}>
                        {connectedSourceNode
                          ? `Connected to: ${connectedSourceNode.data.label}`
                          : "Connect a data source node to create charts"}
                      </p>
                      {config.dataset_id && (
                        <div className="mt-1.5 text-[10px] text-slate-400 font-mono truncate">
                          Dataset ID: {config.dataset_id as string}
                        </div>
                      )}
                    </div>

                    {/* Chart Type Selector */}
                    <div className="rounded-xl border border-slate-200 bg-white p-4">
                      <p className="text-sm font-medium text-slate-700 mb-3">Chart Type</p>
                      <div className="grid grid-cols-5 gap-2">
                        {chartTypes.map((ct) => (
                          <button
                            key={ct.value}
                            onClick={() => handleFieldChange("chart_type", ct.value)}
                            className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border-2 transition-all duration-150 ${
                              chartType === ct.value
                                ? "border-pink-500 bg-pink-50 shadow-sm"
                                : "border-slate-200 bg-slate-50 hover:border-slate-300 hover:bg-slate-100"
                            }`}
                          >
                            <ct.Icon className={`w-5 h-5 ${chartType === ct.value ? "text-pink-600" : "text-slate-400"}`} />
                            <span className={`text-[11px] font-medium ${chartType === ct.value ? "text-pink-700" : "text-slate-500"}`}>
                              {ct.label}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Column Selectors */}
                    <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
                      <p className="text-sm font-medium text-slate-700">Column Configuration</p>

                      {!isPieChart ? (
                        <>
                          {/* X-Axis */}
                          <div>
                            <label className="block text-xs font-medium text-slate-600 mb-1.5">X-Axis Column</label>
                            <select
                              value={(config.x_column as string) || ""}
                              onChange={(e) => handleFieldChange("x_column", e.target.value)}
                              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-pink-500"
                            >
                              <option value="">-- Select X-Axis --</option>
                              {availableColumns.map((col) => (
                                <option key={col} value={col}>{col}</option>
                              ))}
                            </select>
                          </div>
                          {/* Y-Axis */}
                          <div>
                            <label className="block text-xs font-medium text-slate-600 mb-1.5">Y-Axis Column</label>
                            <select
                              value={(config.y_columns as string) || ""}
                              onChange={(e) => handleFieldChange("y_columns", e.target.value)}
                              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-pink-500"
                            >
                              <option value="">-- Select Y-Axis --</option>
                              {availableColumns.map((col) => (
                                <option key={col} value={col}>{col}</option>
                              ))}
                            </select>
                          </div>
                        </>
                      ) : (
                        <>
                          {/* Label Column (Pie) */}
                          <div>
                            <label className="block text-xs font-medium text-slate-600 mb-1.5">Label Column</label>
                            <select
                              value={(config.label_column as string) || ""}
                              onChange={(e) => handleFieldChange("label_column", e.target.value)}
                              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-pink-500"
                            >
                              <option value="">-- Select Label Column --</option>
                              {availableColumns.map((col) => (
                                <option key={col} value={col}>{col}</option>
                              ))}
                            </select>
                            <p className="text-[11px] text-slate-400 mt-1">Column for pie chart slice labels</p>
                          </div>
                          {/* Value Column (Pie) */}
                          <div>
                            <label className="block text-xs font-medium text-slate-600 mb-1.5">Value Column</label>
                            <select
                              value={(config.value_column as string) || ""}
                              onChange={(e) => handleFieldChange("value_column", e.target.value)}
                              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-pink-500"
                            >
                              <option value="">-- Select Value Column --</option>
                              {availableColumns.map((col) => (
                                <option key={col} value={col}>{col}</option>
                              ))}
                            </select>
                            <p className="text-[11px] text-slate-400 mt-1">Column for pie chart slice values</p>
                          </div>
                        </>
                      )}

                      {availableColumns.length === 0 && connectedSourceNode && (
                        <p className="text-xs text-amber-600">
                          ⚠ Run the pipeline first to load available columns
                        </p>
                      )}
                      {!connectedSourceNode && (
                        <p className="text-xs text-amber-600">
                          ⚠ Connect a data source node to see available columns
                        </p>
                      )}
                    </div>
                  </div>
                );
              })()
            ) : node.type === "split" ? (
              <SplitConfigPanel
                config={config}
                onFieldChange={handleFieldChange}
                connectedSourceNode={connectedSourceNode}
                availableColumns={availableColumns}
              />
            ) : [
              "linear_regression",
              "logistic_regression",
              "decision_tree",
              "random_forest",
            ].includes(node.type) ? (
              <MLAlgorithmConfigPanel
                nodeType={node.type}
                config={config}
                onFieldChange={handleFieldChange}
                connectedSourceNode={connectedSourceNode}
                availableColumns={availableColumns}
              />
            ) : [
              "r2_score",
              "mse_score",
              "rmse_score",
              "mae_score",
              "confusion_matrix",
              "classification_report",
              "accuracy_score",
              "roc_curve",
              "feature_importance",
              "residual_plot",
              "prediction_table",
            ].includes(node.type) ? (
              <ResultMetricsConfigPanel
                nodeType={node.type}
                config={config}
                onFieldChange={handleFieldChange}
                connectedSourceNode={connectedSourceNode}
              />
            ) : [
              "llm_node",
              "system_prompt",
              "chatbot_node",
              "example_node",
            ].includes(node.type) ? (
              <GenAIConfigPanel
                nodeType={node.type}
                config={config}
                onFieldChange={handleFieldChange}
                connectedSourceNode={connectedSourceNode}
              />
            ) : node.type === "camera_capture" ? (
              <CameraCaptureConfigBlock
                config={config}
                onDatasetReady={(payload) => {
                  buildCameraDataset(payload);
                }}
                isSubmitting={buildingDataset}
              />
            ) : node.type === "image_predictions" ? (
              <ImagePredictionsConfigBlock
                config={config}
                connectedSourceNode={connectedSourceNode}
                onFieldChange={handleFieldChange}
              />
            ) : nodeDef.configFields && nodeDef.configFields.length > 0 ? (
              <div className="space-y-4">
                {nodeDef.configFields.map((field) => {
                  // Check conditional display
                  if (field.conditionalDisplay) {
                    const conditionValue = config[
                      field.conditionalDisplay.field
                    ] as string;
                    if (
                      field.conditionalDisplay.equals &&
                      conditionValue !== field.conditionalDisplay.equals
                    ) {
                      return null;
                    }
                    if (
                      field.conditionalDisplay.notEquals &&
                      conditionValue === field.conditionalDisplay.notEquals
                    ) {
                      return null;
                    }
                  }

                  const autoFilledValue = field.autoFill
                    ? getAutoFilledValue(field.name)
                    : undefined;
                  const currentValue =
                    config[field.name] ?? autoFilledValue ?? field.defaultValue;

                  // Check if this is a dataset_id field for view/preprocessing nodes (should be read-only)
                  const isDatasetIdField =
                    field.name === "dataset_id" &&
                    [
                      "table_view",
                      "data_preview",
                      "statistics_view",
                      "column_info",
                      "chart_view",
                      "missing_value_handler",
                      "encoding",
                      "transformation",
                      "scaling",
                      "feature_selection",
                      "split",
                      // Image pipeline nodes — inherit from connected node
                      "image_preprocessing",
                      "image_augmentation",
                      "image_split",
                    ].includes(node.type);

                  // Check if this is a train_dataset_id field for ML nodes (should be read-only)
                  const isTrainDatasetIdField =
                    (field.name === "train_dataset_id" &&
                    [
                      "linear_regression",
                      "logistic_regression",
                      "decision_tree",
                      "random_forest",
                      "cnn_classifier",
                    ].includes(node.type)) ||
                    // Also treat image_predictions source fields as read-only
                    (["test_dataset_id", "model_path"].includes(field.name) &&
                      node.type === "image_predictions");

                  // Check if this is a model_output_id field for result nodes (should be read-only)
                  const isModelOutputIdField =
                    field.name === "model_output_id" &&
                    [
                      "r2_score",
                      "mse_score",
                      "rmse_score",
                      "mae_score",
                      "confusion_matrix",
                      "classification_report",
                      "accuracy_score",
                      "roc_curve",
                      "feature_importance",
                      "residual_plot",
                      "prediction_table",
                    ].includes(node.type);

                  return (
                    <div key={field.name}>
                      <label className="block text-sm font-medium text-slate-700 mb-2">
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
                            readOnly={
                              isDatasetIdField ||
                              isTrainDatasetIdField ||
                              isModelOutputIdField
                            }
                            className={`w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                              isDatasetIdField ||
                              isTrainDatasetIdField ||
                              isModelOutputIdField
                                ? "cursor-not-allowed opacity-75"
                                : ""
                            }`}
                            placeholder={
                              (isDatasetIdField || isTrainDatasetIdField) &&
                              !currentValue
                                ? "Connect a data source node"
                                : field.description
                            }
                          />
                          {isDatasetIdField && connectedSourceNode && (
                            <p className="text-xs text-green-600 mt-1">
                              ✓ Connected to: {connectedSourceNode.data.label}
                            </p>
                          )}
                          {isDatasetIdField && !connectedSourceNode && (
                            <p className="text-xs text-amber-600 mt-1">
                              ⚠ Connect a data source node — dataset ID is inherited automatically
                            </p>
                          )}
                          {isTrainDatasetIdField && connectedSourceNode && (
                            <p className="text-xs text-green-600 mt-1">
                              ✓ Connected to: {connectedSourceNode.data.label}
                            </p>
                          )}
                          {isTrainDatasetIdField &&
                            connectedSourceNode &&
                            !currentValue && (
                              <p className="text-xs text-amber-600 mt-1">
                                ⚠ Please execute the split node first to
                                generate training dataset
                              </p>
                            )}
                          {isTrainDatasetIdField && !connectedSourceNode && (
                            <p className="text-xs text-amber-600 mt-1">
                              ⚠ Connect a split node to auto-fill training
                              dataset
                            </p>
                          )}
                          {isModelOutputIdField && connectedSourceNode && (
                            <p className="text-xs text-green-600 mt-1">
                              ✓ Connected to: {connectedSourceNode.data.label}
                            </p>
                          )}
                          {isModelOutputIdField &&
                            connectedSourceNode &&
                            !currentValue && (
                              <p className="text-xs text-amber-600 mt-1">
                                ⚠ Please execute the ML algorithm node first to
                                generate model
                              </p>
                            )}
                          {isModelOutputIdField && !connectedSourceNode && (
                            <p className="text-xs text-amber-600 mt-1">
                              ⚠ Connect an ML algorithm node to auto-fill model
                              output
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
                          className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      )}

                      {field.type === "range" && (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between text-sm text-slate-500">
                            <span>
                              Train:{" "}
                              {((currentValue as number) * 100).toFixed(0)}%
                            </span>
                            <span>
                              Test:{" "}
                              {((1 - (currentValue as number)) * 100).toFixed(
                                0,
                              )}
                              %
                            </span>
                          </div>
                          <input
                            type="range"
                            value={(currentValue as number) || 0.8}
                            onChange={(e) =>
                              handleFieldChange(
                                field.name,
                                parseFloat(e.target.value),
                              )
                            }
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer slider-thumb"
                            style={{
                              background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${((currentValue as number) || 0.8) * 100}%, #E2E8F0 ${((currentValue as number) || 0.8) * 100}%, #E2E8F0 100%)`,
                            }}
                          />
                          <div className="flex justify-between text-xs text-slate-500">
                            <span>10%</span>
                            <span>50%</span>
                            <span>90%</span>
                          </div>
                        </div>
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
                                    dtypes: selectedDataset.dtypes,
                                  }));
                                }
                              }
                            }}
                            disabled={
                              loadingDatasets &&
                              node.type === "select_dataset" &&
                              field.name === "dataset_id"
                            }
                            className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
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
                                  field.name === "target_column" &&
                                  (availableColumns.length > 0
                                    ? availableColumns
                                    : datasetMetadata?.columns || []
                                  ).map((col: string) => (
                                    <option key={col} value={col}>
                                      {col}
                                    </option>
                                  ))}
                                {/* Auto-fill columns for chart view column selection */}
                                {field.autoFill &&
                                  node.type === "chart_view" &&
                                  (field.name === "x_column" ||
                                    field.name === "y_column" ||
                                    field.name === "y_columns" ||
                                    field.name === "label_column" ||
                                    field.name === "value_column") &&
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
                              <p className="text-xs text-green-600 mt-1">
                                ✓ Selected: {config.filename as string}
                              </p>
                            )}

                          {/* Show warning for target_column if no columns available */}
                          {field.name === "target_column" &&
                            availableColumns.length === 0 &&
                            connectedSourceNode && (
                              <p className="text-xs text-amber-600 mt-1">
                                ⚠ Execute the split node first to load available
                                columns
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
                          className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder={field.description}
                        />
                      )}

                      {field.type === "json" && (
                        <div>
                          <textarea
                            value={
                              typeof currentValue === "string"
                                ? currentValue
                                : JSON.stringify(currentValue || {}, null, 2)
                            }
                            onChange={(e) => {
                              try {
                                const parsed = JSON.parse(e.target.value);
                                handleFieldChange(field.name, parsed);
                              } catch {
                                // Allow invalid JSON while typing
                                handleFieldChange(field.name, e.target.value);
                              }
                            }}
                            rows={8}
                            className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder={field.placeholder || field.description}
                          />
                          <p className="text-xs text-slate-500 mt-1">
                            💡 Enter valid JSON configuration
                          </p>
                        </div>
                      )}

                      {field.type === "multiselect" && (
                        <div>
                          <select
                            multiple
                            value={
                              currentValue
                                ? Array.isArray(currentValue)
                                  ? currentValue
                                  : (currentValue as string)
                                      .split(",")
                                      .map((s) => s.trim())
                                : []
                            }
                            onChange={(e) => {
                              const selected = Array.from(
                                e.target.selectedOptions,
                                (option) => option.value,
                              );
                              // Send as array, not comma-separated string
                              handleFieldChange(field.name, selected);
                            }}
                            className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            size={Math.min(availableColumns.length || 5, 8)}
                          >
                            {/* Show available columns from connected source for autoFill fields */}
                            {field.autoFill && availableColumns.length > 0 ? (
                              availableColumns.map((col: string) => (
                                <option key={col} value={col}>
                                  {col}
                                </option>
                              ))
                            ) : field.autoFill ? (
                              <option disabled>
                                Connect a data source to see columns
                              </option>
                            ) : null}
                          </select>
                          {availableColumns.length > 0 && field.autoFill && (
                            <p className="text-xs text-slate-500 mt-1">
                              💡 Hold Ctrl/Cmd to select multiple columns
                            </p>
                          )}
                          {!connectedSourceNode && field.autoFill && (
                            <p className="text-xs text-amber-600 mt-1">
                              ⚠ Connect a data source node first
                            </p>
                          )}
                        </div>
                      )}

                      {field.type === "password" && (
                        <input
                          type="password"
                          value={(currentValue as string) || ""}
                          onChange={(e) =>
                            handleFieldChange(field.name, e.target.value)
                          }
                          className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder={field.placeholder || field.description}
                        />
                      )}

                      {field.type === "custom" && field.name === "examples" && (
                        <div className="space-y-3">
                          {((currentValue as any[]) || []).map(
                            (example: any, index: number) => (
                              <div
                                key={index}
                                className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-3"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-sm font-medium text-slate-600">
                                    Example {index + 1}
                                  </span>
                                  <button
                                    onClick={() => {
                                      const newExamples = [
                                        ...(currentValue as any[]),
                                      ];
                                      newExamples.splice(index, 1);
                                      handleFieldChange(
                                        field.name,
                                        newExamples,
                                      );
                                    }}
                                    className="text-red-500 hover:text-red-600 text-sm"
                                  >
                                    Remove
                                  </button>
                                </div>
                                <div>
                                  <label className="block text-xs text-slate-500 mb-1">
                                    User Input
                                  </label>
                                  <textarea
                                    value={example.userInput || ""}
                                    onChange={(e) => {
                                      const newExamples = [
                                        ...(currentValue as any[]),
                                      ];
                                      newExamples[index] = {
                                        ...newExamples[index],
                                        userInput: e.target.value,
                                      };
                                      handleFieldChange(
                                        field.name,
                                        newExamples,
                                      );
                                    }}
                                    rows={2}
                                    className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Enter user input example..."
                                  />
                                </div>
                                <div>
                                  <label className="block text-xs text-slate-500 mb-1">
                                    Expected Output
                                  </label>
                                  <textarea
                                    value={example.expectedOutput || ""}
                                    onChange={(e) => {
                                      const newExamples = [
                                        ...(currentValue as any[]),
                                      ];
                                      newExamples[index] = {
                                        ...newExamples[index],
                                        expectedOutput: e.target.value,
                                      };
                                      handleFieldChange(
                                        field.name,
                                        newExamples,
                                      );
                                    }}
                                    rows={2}
                                    className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Enter expected output..."
                                  />
                                </div>
                              </div>
                            ),
                          )}
                          <button
                            onClick={() => {
                              const newExamples = [
                                ...((currentValue as any[]) || []),
                                { userInput: "", expectedOutput: "" },
                              ];
                              handleFieldChange(field.name, newExamples);
                            }}
                            className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                          >
                            + Add Example
                          </button>
                          <p className="text-xs text-slate-500 mt-1">
                            Add examples to teach the AI how to respond in
                            specific situations
                          </p>
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
                            className="w-4 h-4 bg-white border-slate-300 rounded text-blue-500 focus:ring-2 focus:ring-blue-500"
                          />
                          <span className="text-sm text-slate-500">
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
                          className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white file:cursor-pointer hover:file:bg-blue-700"
                        />
                      )}

                      {field.description && field.type !== "checkbox" && (
                        <p className="text-xs text-slate-500 mt-1">
                          {field.description}
                        </p>
                      )}

                      {autoFilledValue && (
                        <p className="text-xs text-green-600 mt-1">
                          Auto-filled from dataset
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-slate-400">
                <p>No configuration required for this node</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-slate-200 flex items-center justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-slate-500 hover:text-slate-800 transition-colors"
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
