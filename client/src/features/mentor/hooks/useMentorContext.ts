import { useEffect, useRef } from "react";
import { usePlaygroundStore } from "../../../store/playgroundStore";
import { useMentorStore } from "../store/mentorStore";
import { mentorApi } from "../api/mentorApi";
import type { Node } from "@xyflow/react";
import type { BaseNodeData } from "../../../types/pipeline";
import type { MentorSuggestion } from "../store/mentorStore";

interface UseMentorContextOptions {
  /** Enable auto-triggering of mentor suggestions */
  enabled?: boolean;
  /** Debounce delay in milliseconds */
  debounceMs?: number;
}

// ─── Pipeline Stage Definitions ───────────────────────────────────

type PipelineStage =
  | "idle"
  | "dataset_added"
  | "dataset_configured"
  | "column_info_suggested"
  | "column_info_executed"
  | "missing_values_suggested"
  | "encoding_suggested"
  | "split_suggested"
  | "model_suggested"
  | "metrics_suggested"
  | "ready_to_execute"
  | "complete";

// ─── Stage Message Helpers ────────────────────────────────────────

function buildColumnInfoResultsSuggestion(
  columnInfo: Array<{ column: string; dtype: string; missing?: number; unique?: number }>,
): MentorSuggestion | null {
  const missingCols = columnInfo.filter((c) => (c.missing ?? 0) > 0);
  const categoricalCols = columnInfo.filter(
    (c) => c.dtype === "object" || c.dtype === "category" || c.dtype === "string",
  );
  const totalMissing = missingCols.reduce((sum, c) => sum + (c.missing ?? 0), 0);

  // Priority 1: Missing values
  if (missingCols.length > 0) {
    const colNames = missingCols.slice(0, 3).map((c) => c.column);
    const extras = missingCols.length > 3 ? ` and ${missingCols.length - 3} more` : "";

    return {
      id: `stage-missing-values-${Date.now()}`,
      type: "next_step",
      priority: "warning",
      title: "Missing Values Detected",
      message:
        `Column Info shows your data has **${totalMissing} missing values** in columns: **${colNames.join(", ")}**${extras}.\n\n` +
        `Missing values are empty cells that can confuse ML models. Drag a **Missing Value Handler** node and connect it to your dataset.\n\n` +
        `You can fill missing values with:\n` +
        `- **Mean**: average value (good for numbers)\n` +
        `- **Median**: middle value (good when data has outliers)\n` +
        `- **Mode**: most common value (good for categories)\n` +
        `- **Drop**: remove rows with missing data`,
      voice_text:
        `Column Info shows your data has ${totalMissing} missing values in columns: ${colNames.join(", ")}${extras}. ` +
        `Missing values are empty cells that can confuse ML models. ` +
        `Drag a Missing Value Handler node and connect it to your dataset. ` +
        `You can fill missing values with the mean, median, mode, or drop the rows entirely.`,
      actions: [
        {
          label: "Add Missing Value Handler",
          type: "add_node",
          payload: { node_type: "missing_value_handler" },
        },
      ],
      dismissible: true,
    };
  }

  // Priority 2: Categorical columns
  if (categoricalCols.length > 0) {
    const colNames = categoricalCols.slice(0, 3).map((c) => c.column);
    const extras = categoricalCols.length > 3 ? ` and ${categoricalCols.length - 3} more` : "";

    return {
      id: `stage-encoding-${Date.now()}`,
      type: "next_step",
      priority: "info",
      title: "Text Columns Found",
      message:
        `Your data has **${categoricalCols.length} text columns**: **${colNames.join(", ")}**${extras}.\n\n` +
        `ML models only understand numbers, so we need to convert text to numbers. Drag an **Encoding** node.\n\n` +
        `- **Label Encoding**: gives each category a number (Red=0, Blue=1, Green=2)\n` +
        `- **One-Hot Encoding**: creates separate yes/no columns for each category`,
      voice_text:
        `Your data has ${categoricalCols.length} text columns: ${colNames.join(", ")}${extras}. ` +
        `ML models only understand numbers, so we need to convert text to numbers. Drag an Encoding node. ` +
        `Label Encoding gives each category a number. One-Hot Encoding creates separate yes or no columns for each category.`,
      actions: [
        {
          label: "Add Encoding",
          type: "add_node",
          payload: { node_type: "encoding" },
        },
      ],
      dismissible: true,
    };
  }

  // No issues — go straight to split
  return null;
}

/**
 * Hook that watches playground state changes and automatically triggers
 * relevant mentor suggestions based on user actions.
 * Implements a pipeline stage tracker for step-by-step guidance.
 */
export function useMentorContext(options: UseMentorContextOptions = {}) {
  const { enabled = true } = options;

  const { nodes, executionResult } = usePlaygroundStore();
  const { preferences, showSuggestion } = useMentorStore();

  const previousNodesRef = useRef<Node<BaseNodeData>[]>(nodes);
  const previousExecutionResultRef = useRef(executionResult);

  // Dedup refs to avoid showing duplicate suggestions
  const notifiedDatasetNodeIds = useRef<Set<string>>(new Set());
  const notifiedUploadNodeIds = useRef<Set<string>>(new Set());
  const analyzedDatasetIds = useRef<Set<string>>(new Set());

  // Pipeline stage tracking
  const pipelineStageRef = useRef<PipelineStage>("idle");
  const suggestedStages = useRef<Set<string>>(new Set());
  const analyzedColumnInfoNodes = useRef<Set<string>>(new Set());

  // ─── Stage 1: Detect when a data source node is dragged onto the canvas ───
  useEffect(() => {
    if (!enabled) return;

    const prevIds = new Set(previousNodesRef.current.map((n) => n.id));
    const dataSourceTypes = ["upload_file", "select_dataset"];

    const newDataSourceNodes = nodes.filter(
      (n) =>
        !prevIds.has(n.id) &&
        dataSourceTypes.includes(n.data.type) &&
        !notifiedDatasetNodeIds.current.has(n.id),
    );

    if (newDataSourceNodes.length > 0) {
      const node = newDataSourceNodes[0];
      notifiedDatasetNodeIds.current.add(node.id);
      pipelineStageRef.current = "dataset_added";

      const isUpload = node.data.type === "upload_file";
      const action = isUpload ? "upload your CSV file" : "select your dataset";

      showSuggestion({
        id: `dataset-node-added-${Date.now()}`,
        type: "next_step",
        priority: "info",
        title: "Dataset Node Added",
        message: `I can see you've dragged the **${node.data.label}** node onto the canvas! Now click on the node to open its settings and ${action}.`,
        voice_text: `I can see you've dragged the ${node.data.label} node onto the canvas! Now click on the node to open its settings and ${action}.`,
        actions: [],
        dismissible: true,
      });
    }

    previousNodesRef.current = nodes;
  }, [nodes, enabled, showSuggestion]);

  // ─── Stage 2: Detect dataset configured → suggest Column Info ───
  useEffect(() => {
    if (!enabled) return;

    const dataSourceTypes = ["upload_file", "select_dataset"];

    const configuredDataNodes = nodes.filter(
      (n) =>
        dataSourceTypes.includes(n.data.type) &&
        n.data.isConfigured &&
        n.data.config?.dataset_id &&
        !notifiedUploadNodeIds.current.has(n.id),
    );

    if (configuredDataNodes.length > 0) {
      const node = configuredDataNodes[0];
      notifiedUploadNodeIds.current.add(node.id);
      pipelineStageRef.current = "dataset_configured";

      const cfg = node.data.config;
      const filename = cfg?.filename || "your dataset";
      const nRows = cfg?.n_rows || "N/A";
      const nCols = cfg?.n_columns || "N/A";

      showSuggestion({
        id: `dataset-uploaded-${Date.now()}`,
        type: "next_step",
        priority: "info",
        title: "Dataset Loaded",
        message:
          `Your dataset **${filename}** is loaded with **${nRows} rows** and **${nCols} columns**.\n\n` +
          `Now let's understand your data! Drag a **Column Info** node from the **View** section on the left ` +
          `and connect it to your dataset node, then click **Run** to see column details.`,
        voice_text:
          `Your dataset ${filename} is loaded with ${nRows} rows and ${nCols} columns. ` +
          `Now let's understand your data! Drag a Column Info node from the View section on the left ` +
          `and connect it to your dataset node, then click Run to see column details.`,
        actions: [
          {
            label: "Add Column Info",
            type: "add_node",
            payload: { node_type: "column_info" },
          },
        ],
        dismissible: true,
      });
    }
  }, [nodes, enabled, showSuggestion]);

  // ─── Stage 3: Watch for Column Info execution results ───
  useEffect(() => {
    if (!enabled || !executionResult?.success || !executionResult?.nodeResults) return;

    // Find column_info nodes that have execution results
    const columnInfoNodes = nodes.filter(
      (n) =>
        n.data.type === "column_info" &&
        !analyzedColumnInfoNodes.current.has(n.id),
    );

    for (const node of columnInfoNodes) {
      const result = executionResult.nodeResults[node.id];
      if (!result?.success || !result?.output) continue;

      const output = result.output as { column_info?: Array<{ column: string; dtype: string; missing?: number; unique?: number }> };
      if (!output.column_info || output.column_info.length === 0) continue;

      analyzedColumnInfoNodes.current.add(node.id);
      pipelineStageRef.current = "column_info_executed";

      // Analyze column info results and generate appropriate suggestion
      const suggestion = buildColumnInfoResultsSuggestion(output.column_info);

      if (suggestion) {
        showSuggestion(suggestion);
      } else {
        // No missing values, no categorical columns → suggest Split directly
        if (!suggestedStages.current.has("split")) {
          suggestedStages.current.add("split");
          pipelineStageRef.current = "split_suggested";

          showSuggestion({
            id: `stage-split-${Date.now()}`,
            type: "next_step",
            priority: "info",
            title: "Data Looks Clean!",
            message:
              `Your data looks great — no missing values and all numeric columns!\n\n` +
              `Now let's split your data. We keep **80% for training** (teaching the model) and **20% for testing** ` +
              `(checking if it learned well). Drag a **Target & Split** node.\n\n` +
              `You'll need to pick which column you want to predict (the **target column**).`,
            voice_text:
              `Your data looks great, no missing values and all numeric columns! ` +
              `Now let's split your data. We keep 80% for training and 20% for testing. ` +
              `Drag a Target and Split node. You'll need to pick which column you want to predict.`,
            actions: [
              {
                label: "Add Target & Split",
                type: "add_node",
                payload: { node_type: "split" },
              },
            ],
            dismissible: true,
          });
        }
      }
    }
  }, [executionResult, nodes, enabled, showSuggestion]);

  // ─── Stage 4: Watch for preprocessing nodes → suggest next step ───
  useEffect(() => {
    if (!enabled) return;

    const nodeTypes = nodes.map((n) => n.data.type);
    const hasMissingHandler = nodeTypes.includes("missing_value_handler");
    const hasEncoding = nodeTypes.includes("encoding");
    const hasSplit = nodeTypes.includes("split");
    const hasModel = nodeTypes.some((t) =>
      ["linear_regression", "logistic_regression", "decision_tree", "random_forest"].includes(t),
    );

    // After missing value handler is added → check if encoding is needed, then suggest split
    if (hasMissingHandler && !suggestedStages.current.has("after_missing")) {
      suggestedStages.current.add("after_missing");

      // Check if there are categorical columns that need encoding
      // We look at the analyzed column info results
      let hasCategorical = false;
      for (const node of nodes) {
        if (node.data.type !== "column_info") continue;
        const result = executionResult?.nodeResults?.[node.id];
        if (!result?.success || !result?.output) continue;
        const output = result.output as { column_info?: Array<{ column: string; dtype: string }> };
        if (output.column_info) {
          hasCategorical = output.column_info.some(
            (c) => c.dtype === "object" || c.dtype === "category" || c.dtype === "string",
          );
        }
      }

      if (hasCategorical && !hasEncoding) {
        showSuggestion({
          id: `stage-encoding-after-missing-${Date.now()}`,
          type: "next_step",
          priority: "info",
          title: "Next: Encode Text Columns",
          message:
            `Missing values handled! Now let's convert text columns to numbers.\n\n` +
            `Drag an **Encoding** node and connect it after the Missing Value Handler.\n\n` +
            `- **Label Encoding**: gives each category a number (Red=0, Blue=1)\n` +
            `- **One-Hot Encoding**: creates separate yes/no columns for each category`,
          voice_text:
            `Missing values handled! Now let's convert text columns to numbers. ` +
            `Drag an Encoding node and connect it after the Missing Value Handler. ` +
            `Label Encoding gives each category a number. One-Hot Encoding creates separate yes or no columns for each category.`,
          actions: [
            {
              label: "Add Encoding",
              type: "add_node",
              payload: { node_type: "encoding" },
            },
          ],
          dismissible: true,
        });
      } else if (!hasSplit) {
        // No encoding needed → suggest split
        if (!suggestedStages.current.has("split")) {
          suggestedStages.current.add("split");
          pipelineStageRef.current = "split_suggested";

          showSuggestion({
            id: `stage-split-after-preprocessing-${Date.now()}`,
            type: "next_step",
            priority: "info",
            title: "Next: Split Your Data",
            message:
              `Data is preprocessed! Now let's split your data.\n\n` +
              `We keep **80% for training** (teaching the model) and **20% for testing** (checking if it learned well).\n\n` +
              `Drag a **Target & Split** node. You'll need to pick which column you want to predict (the **target column**).`,
            voice_text:
              `Data is preprocessed! Now let's split your data. ` +
              `We keep 80% for training and 20% for testing. ` +
              `Drag a Target and Split node. You'll need to pick which column you want to predict.`,
            actions: [
              {
                label: "Add Target & Split",
                type: "add_node",
                payload: { node_type: "split" },
              },
            ],
            dismissible: true,
          });
        }
      }
    }

    // After encoding is added → suggest split
    if (hasEncoding && !hasSplit && !suggestedStages.current.has("split")) {
      suggestedStages.current.add("split");
      pipelineStageRef.current = "split_suggested";

      showSuggestion({
        id: `stage-split-after-encoding-${Date.now()}`,
        type: "next_step",
        priority: "info",
        title: "Next: Split Your Data",
        message:
          `Encoding done! Now let's split your data.\n\n` +
          `We keep **80% for training** (teaching the model) and **20% for testing** (checking if it learned well).\n\n` +
          `Drag a **Target & Split** node. You'll need to pick which column you want to predict (the **target column**).`,
        voice_text:
          `Encoding done! Now let's split your data. ` +
          `We keep 80% for training and 20% for testing. ` +
          `Drag a Target and Split node. You'll need to pick which column you want to predict.`,
        actions: [
          {
            label: "Add Target & Split",
            type: "add_node",
            payload: { node_type: "split" },
          },
        ],
        dismissible: true,
      });
    }

    // After split is added → suggest model
    if (hasSplit && !hasModel && !suggestedStages.current.has("model")) {
      suggestedStages.current.add("model");
      pipelineStageRef.current = "model_suggested";

      showSuggestion({
        id: `stage-model-${Date.now()}`,
        type: "next_step",
        priority: "info",
        title: "Next: Add Your Model",
        message:
          `Data is ready for training! Now drag a **Linear Regression** node and connect it to the Split node.\n\n` +
          `Linear Regression will learn patterns from your training data to predict the target column. ` +
          `It finds the best-fitting straight line through your data points.`,
        voice_text:
          `Data is ready for training! Now drag a Linear Regression node and connect it to the Split node. ` +
          `Linear Regression will learn patterns from your training data to predict the target column.`,
        actions: [
          {
            label: "Add Linear Regression",
            type: "add_node",
            payload: { node_type: "linear_regression" },
          },
        ],
        dismissible: true,
      });
    }

    // After model is added → suggest metrics
    if (hasModel && !suggestedStages.current.has("metrics")) {
      const metricTypes = ["r2_score", "mse_score", "rmse_score", "mae_score"];
      const hasAnyMetric = nodeTypes.some((t) => metricTypes.includes(t));

      if (!hasAnyMetric) {
        suggestedStages.current.add("metrics");
        pipelineStageRef.current = "metrics_suggested";

        showSuggestion({
          id: `stage-metrics-${Date.now()}`,
          type: "next_step",
          priority: "info",
          title: "Next: Check Model Performance",
          message:
            `Almost there! Let's see how well your model performs. Add metric nodes and connect them to the Linear Regression node:\n\n` +
            `- **R\u00B2 Score**: How much of the data pattern your model captures (closer to 1 = better)\n` +
            `- **RMSE**: Average prediction error in the same units as your target (lower = better)\n` +
            `- **MAE**: Average absolute error (lower = better)`,
          voice_text:
            `Almost there! Let's see how well your model performs. Add metric nodes and connect them to the Linear Regression node. ` +
            `R squared score shows how much of the data pattern your model captures, closer to 1 is better. ` +
            `RMSE shows the average prediction error, lower is better. ` +
            `MAE shows the average absolute error, lower is better.`,
          actions: [
            {
              label: "Add R\u00B2 Score",
              type: "add_node",
              payload: { node_type: "r2_score" },
            },
            {
              label: "Add RMSE",
              type: "add_node",
              payload: { node_type: "rmse_score" },
            },
            {
              label: "Add MAE",
              type: "add_node",
              payload: { node_type: "mae_score" },
            },
          ],
          dismissible: true,
        });
      }
    }

    // After metrics are added → suggest execute
    if (hasModel && !suggestedStages.current.has("execute")) {
      const metricTypes = ["r2_score", "mse_score", "rmse_score", "mae_score"];
      const hasAnyMetric = nodeTypes.some((t) => metricTypes.includes(t));

      if (hasAnyMetric && hasSplit) {
        suggestedStages.current.add("execute");
        pipelineStageRef.current = "ready_to_execute";

        showSuggestion({
          id: `stage-execute-${Date.now()}`,
          type: "next_step",
          priority: "info",
          title: "Pipeline Complete!",
          message:
            `Your pipeline is complete! Make sure all nodes are connected in order, then click the **Run** button in the toolbar to train your model and see results.\n\n` +
            `After running, click on the metric nodes to see your model's performance scores.`,
          voice_text:
            `Your pipeline is complete! Make sure all nodes are connected in order, then click the Run button in the toolbar to train your model and see results.`,
          actions: [
            {
              label: "Execute Pipeline",
              type: "execute",
              payload: {},
            },
          ],
          dismissible: true,
        });
      }
    }
  }, [nodes, executionResult, enabled, showSuggestion]);

  // ─── Auto-analyze dataset when configured ───
  useEffect(() => {
    if (!enabled || !preferences.auto_analyze) return;

    const uploadNodes = nodes.filter(
      (node) =>
        (node.data.type === "upload_file" || node.data.type === "select_dataset") &&
        node.data.isConfigured &&
        node.data.config?.dataset_id &&
        !analyzedDatasetIds.current.has(node.data.config.dataset_id as string),
    );

    uploadNodes.forEach(async (node) => {
      const cfg = node.data.config;
      const datasetId = cfg?.dataset_id as string;
      if (!datasetId) return;

      analyzedDatasetIds.current.add(datasetId);

      try {
        const insights = await mentorApi.analyzeDataset({
          nodes: [{ id: node.id, type: node.data.type, config: node.data, position: node.position }],
          edges: [],
          dataset_metadata: {
            dataset_id: datasetId,
            filename: cfg?.filename,
            n_rows: cfg?.n_rows,
            n_columns: cfg?.n_columns,
            columns: cfg?.columns,
            dtypes: cfg?.dtypes,
          },
        });

        if (insights.suggestions.length > 0) {
          insights.suggestions.forEach((suggestion) => {
            showSuggestion(suggestion);
          });
        }
      } catch (error) {
        console.error("Failed to analyze dataset:", error);
      }
    });
  }, [nodes, enabled, preferences.auto_analyze]);

  // ─── Detect execution errors and provide explanations ───
  useEffect(() => {
    if (!enabled) return;

    const currentResult = executionResult;
    const previousResult = previousExecutionResultRef.current;

    // Check if execution just completed with errors
    if (
      currentResult &&
      currentResult !== previousResult &&
      !currentResult.success &&
      currentResult.error
    ) {
      (async () => {
        try {
          // Find the failed node to get its type and config
          const failedNodeEntry = currentResult.nodeResults
            ? Object.entries(currentResult.nodeResults).find(([, r]) => !r.success)
            : undefined;
          const failedNodeId = failedNodeEntry?.[0];
          const failedNode = failedNodeId
            ? nodes.find((n) => n.id === failedNodeId)
            : undefined;

          const explanation = await mentorApi.explainError(
            currentResult.error || "Unknown error occurred",
            failedNode?.data.type || "unknown",
            (failedNode?.data.config as Record<string, unknown>) || {},
          );

          if (explanation.suggestions.length > 0) {
            showSuggestion(explanation.suggestions[0]);
          }
        } catch (error) {
          console.error("Failed to explain error:", error);
        }
      })();
    }

    previousExecutionResultRef.current = currentResult;
  }, [executionResult, enabled, nodes]);

  // ─── Detect when user might be stuck (no changes for a while) ───
  useEffect(() => {
    if (!enabled || !preferences.show_tips) return;

    const inactivityTimeout = setTimeout(() => {
      if (nodes.length > 0 && nodes.length < 3) {
        showSuggestion({
          id: `inactivity-${Date.now()}`,
          type: "learning_tip",
          priority: "info",
          title: "Need Help?",
          message:
            "Need help building your pipeline? I can suggest next steps based on what you've started!",
          voice_text:
            "Need help building your pipeline? I can suggest next steps based on what you've started!",
          timestamp: new Date().toISOString(),
          actions: [
            {
              label: "Get Suggestions",
              type: "learn_more",
            },
          ],
          dismissible: true,
        });
      }
    }, 60000); // 1 minute of inactivity

    return () => clearTimeout(inactivityTimeout);
  }, [nodes, enabled, preferences.show_tips]);
}
