import { motion, AnimatePresence } from "framer-motion";
import { X, Save } from "lucide-react";
import { useState, useEffect } from "react";
import { getNodeByType } from "../../config/nodeDefinitions";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { UploadDatasetButton } from "./UploadDatasetButton";
import { FeatureEngineeringConfigPanel } from "./FeatureEngineeringConfigPanel";
import {
  listProjectDatasets,
  type DatasetMetadata,
} from "../../lib/api/datasetApi";

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

  // Reset config when nodeId changes to prevent config sharing between nodes
  // ALWAYS load the latest config from the store (no caching)
  useEffect(() => {
    if (node) {
      console.log(
        "🔄 Loading LATEST config for node:",
        node.id,
        node.data.config,
      );
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

          console.log("📊 Auto-filling result node from ML algorithm:");
          console.log("  - Source node type:", sourceNode?.type);
          console.log("  - Model ID:", modelId);

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

          if (isMLNode && sourceNode?.type === "split") {
            // ML nodes get train_dataset_id and target_column from split node
            const trainDatasetId = sourceResult?.train_dataset_id;
            const targetColumn = sourceResult?.target_column;

            console.log("🎯 Auto-filling ML node from split:");
            console.log("  - Source node type:", sourceNode?.type);
            console.log("  - Source node data:", sourceNode?.data);
            console.log("  - Source result:", sourceResult);
            console.log("  - Train dataset ID:", trainDatasetId);
            console.log("  - Target column:", targetColumn);
            console.log("  - Columns in result:", sourceResult?.columns);

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
              sourceResult?.selected_dataset_id;

            // Priority 2: Check config for dataset_id (for upload/select nodes)
            if (!datasetId) {
              datasetId = sourceConfig?.dataset_id;
            }

            console.log(
              "🔗 Auto-fill dataset_id - Source node:",
              sourceNode?.type,
              "\n  - sourceResult:",
              sourceResult,
              "\n  - sourceConfig:",
              sourceConfig,
              "\n  - Resolved dataset_id:",
              datasetId,
            );

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
        console.log(
          "  ✅ Found columns from split node:",
          sourceResult.columns,
        );
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
        console.log("  ✅ Found feature_columns from split node:", allColumns);
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
    console.log("💾 ConfigModal - Saving config for node:", node.id);
    console.log("💾 ConfigModal - Config being saved:", config);
    console.log("💾 ConfigModal - target_column value:", config.target_column);
    updateNodeConfig(node.id, config);
    onClose();
  };

  const handleFieldChange = (fieldName: string, value: unknown) => {
    console.log(`📝 ConfigModal - Field "${fieldName}" changed to:`, value);

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
        sourceConfig?.dataset_id;

      if (datasetId) {
        return datasetId;
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
                renderField={(field, label, type = "text", options) => {
                  const currentValue = config[field];

                  if (type === "select") {
                    return (
                      <div key={field} className="mb-4">
                        <label className="block text-sm font-medium text-gray-200 mb-2">
                          {label}
                        </label>
                        <select
                          value={(currentValue as string) || ""}
                          onChange={(e) =>
                            handleFieldChange(field, e.target.value)
                          }
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                            className="rounded border-gray-600 text-blue-600 focus:ring-blue-500 bg-gray-800"
                          />
                          <span className="text-sm font-medium text-gray-200">
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
                        <label className="block text-sm font-medium text-gray-200 mb-2">
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
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    );
                  }

                  // Default text input
                  return (
                    <div key={field} className="mb-4">
                      <label className="block text-sm font-medium text-gray-200 mb-2">
                        {label}
                      </label>
                      <input
                        type="text"
                        value={(currentValue as string) || ""}
                        onChange={(e) =>
                          handleFieldChange(field, e.target.value)
                        }
                        readOnly={field === "dataset_id"}
                        className={`w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 ${field === "dataset_id" ? "cursor-not-allowed opacity-75" : ""}`}
                      />
                      {field === "dataset_id" && connectedSourceNode && (
                        <p className="text-xs text-green-400 mt-1">
                          ✓ Connected to: {connectedSourceNode.data.label}
                        </p>
                      )}
                    </div>
                  );
                }}
              />
            ) : node.type === "upload_file" ? (
              <div className="space-y-4">
                <div className="p-4 bg-blue-900/20 border border-blue-700 rounded-lg">
                  <p className="text-sm text-blue-200 mb-3">
                    Upload a CSV file to use in your ML pipeline
                  </p>
                  {currentProjectId ? (
                    <UploadDatasetButton
                      nodeId={node.id}
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
                    ].includes(node.type);

                  // Check if this is a train_dataset_id field for ML nodes (should be read-only)
                  const isTrainDatasetIdField =
                    field.name === "train_dataset_id" &&
                    [
                      "linear_regression",
                      "logistic_regression",
                      "decision_tree",
                      "random_forest",
                    ].includes(node.type);

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
                            readOnly={
                              isDatasetIdField ||
                              isTrainDatasetIdField ||
                              isModelOutputIdField
                            }
                            className={`w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
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
                            <p className="text-xs text-green-400 mt-1">
                              ✓ Connected to: {connectedSourceNode.data.label}
                            </p>
                          )}
                          {isDatasetIdField && !connectedSourceNode && (
                            <p className="text-xs text-amber-400 mt-1">
                              ⚠ Connect an upload dataset node to this view node
                            </p>
                          )}
                          {isTrainDatasetIdField && connectedSourceNode && (
                            <p className="text-xs text-green-400 mt-1">
                              ✓ Connected to: {connectedSourceNode.data.label}
                            </p>
                          )}
                          {isTrainDatasetIdField &&
                            connectedSourceNode &&
                            !currentValue && (
                              <p className="text-xs text-amber-400 mt-1">
                                ⚠ Please execute the split node first to
                                generate training dataset
                              </p>
                            )}
                          {isTrainDatasetIdField && !connectedSourceNode && (
                            <p className="text-xs text-amber-400 mt-1">
                              ⚠ Connect a split node to auto-fill training
                              dataset
                            </p>
                          )}
                          {isModelOutputIdField && connectedSourceNode && (
                            <p className="text-xs text-green-400 mt-1">
                              ✓ Connected to: {connectedSourceNode.data.label}
                            </p>
                          )}
                          {isModelOutputIdField &&
                            connectedSourceNode &&
                            !currentValue && (
                              <p className="text-xs text-amber-400 mt-1">
                                ⚠ Please execute the ML algorithm node first to
                                generate model
                              </p>
                            )}
                          {isModelOutputIdField && !connectedSourceNode && (
                            <p className="text-xs text-amber-400 mt-1">
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
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      )}

                      {field.type === "range" && (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between text-sm text-gray-400">
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
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                            style={{
                              background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${((currentValue as number) || 0.8) * 100}%, #1F2937 ${((currentValue as number) || 0.8) * 100}%, #1F2937 100%)`,
                            }}
                          />
                          <div className="flex justify-between text-xs text-gray-500">
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
                              <p className="text-xs text-green-400 mt-1">
                                ✓ Selected: {config.filename as string}
                              </p>
                            )}

                          {/* Show warning for target_column if no columns available */}
                          {field.name === "target_column" &&
                            availableColumns.length === 0 &&
                            connectedSourceNode && (
                              <p className="text-xs text-amber-400 mt-1">
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
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder={field.placeholder || field.description}
                          />
                          <p className="text-xs text-gray-400 mt-1">
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
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                            <p className="text-xs text-gray-400 mt-1">
                              💡 Hold Ctrl/Cmd to select multiple columns
                            </p>
                          )}
                          {!connectedSourceNode && field.autoFill && (
                            <p className="text-xs text-amber-400 mt-1">
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
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder={field.placeholder || field.description}
                        />
                      )}

                      {field.type === "custom" && field.name === "examples" && (
                        <div className="space-y-3">
                          {((currentValue as any[]) || []).map(
                            (example: any, index: number) => (
                              <div
                                key={index}
                                className="p-4 bg-gray-800 rounded-lg border border-gray-600 space-y-3"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-sm font-medium text-gray-300">
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
                                    className="text-red-400 hover:text-red-300 text-sm"
                                  >
                                    Remove
                                  </button>
                                </div>
                                <div>
                                  <label className="block text-xs text-gray-400 mb-1">
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
                                    className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Enter user input example..."
                                  />
                                </div>
                                <div>
                                  <label className="block text-xs text-gray-400 mb-1">
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
                                    className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                          <p className="text-xs text-gray-400 mt-1">
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
