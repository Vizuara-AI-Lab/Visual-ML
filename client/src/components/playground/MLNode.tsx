/**
 * Custom ML Node Component for React Flow
 * Premium flat design with clean visual hierarchy
 */

import { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { BaseNodeData } from "../../types/pipeline";
import { motion } from "framer-motion";
import {
  Eye,
  X,
  Settings,
  CheckCircle2,
  AlertCircle,
  CircleDot,
  Loader2,
  Play,
  BookOpen,
} from "lucide-react";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { hasStory } from "../../config/datasetStories";

const MLNode = ({ data, id, selected }: NodeProps<BaseNodeData>) => {
  const nodeData = data as BaseNodeData;
  const nodeId = id as string;
  const { executionResult, deleteNode, nodeExecutionStatus } =
    usePlaygroundStore();

  const viewNodeTypes = [
    "table_view",
    "data_preview",
    "statistics_view",
    "column_info",
    "chart_view",
    "missing_value_handler",
    "encoding",
    "scaling",
    "feature_selection",
    "r2_score",
    "mse_score",
    "rmse_score",
    "mae_score",
    "confusion_matrix",
    "split",
    "linear_regression",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "mlp_classifier",
    "mlp_regressor",
    // Image pipeline nodes
    "image_dataset",
    "image_split",
    "image_predictions",
  ];

  const isViewNode = viewNodeTypes.includes(nodeData.type);
  const isActivityNode = nodeData.type.startsWith("activity_");

  // Get execution status from store
  const executionStatus = nodeExecutionStatus[nodeId];
  const isRunning = executionStatus === "running";
  const isPending = executionStatus === "pending";
  const isCompleted = executionStatus === "completed";
  const isFailed = executionStatus === "failed";

  // Legacy support for old execution result format
  const nodeResult = executionResult?.nodeResults?.[nodeId];
  const hasExecutionResults = !!nodeResult;
  const executionSuccess = nodeResult?.success || isCompleted;
  const executionFailed =
    (hasExecutionResults && !nodeResult?.success) || isFailed;

  const showViewButton =
    isViewNode &&
    hasExecutionResults &&
    (nodeData.isConfigured || executionSuccess);

  // Check if this is a sample_dataset node with an available story
  const datasetName = nodeData.config?.dataset_name as string | undefined;
  const showStoryButton =
    nodeData.type === "sample_dataset" &&
    nodeData.isConfigured &&
    !!datasetName &&
    hasStory(datasetName);

  const handleOpenStory = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (datasetName) {
      window.dispatchEvent(
        new CustomEvent("openDatasetStory", {
          detail: { datasetId: datasetName },
        }),
      );
    }
  };

  const handleViewData = (e: React.MouseEvent) => {
    e.stopPropagation();
    window.dispatchEvent(
      new CustomEvent("openViewNodeModal", { detail: { nodeId } }),
    );
  };

  const handleOpenActivity = (e: React.MouseEvent) => {
    e.stopPropagation();
    window.dispatchEvent(
      new CustomEvent("openActivityModal", { detail: { nodeId } }),
    );
  };

  const handleReconfig = (e: React.MouseEvent) => {
    e.stopPropagation();
    window.dispatchEvent(
      new CustomEvent("openConfigModal", { detail: { nodeId } }),
    );
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    deleteNode(nodeId);
  };

  const accentColor = nodeData.color || "#6B7280";

  return (
    <div className="relative group" style={{ minWidth: 200 }}>
      {/* Main card */}
      <motion.div
        className={`
          relative rounded-xl overflow-hidden transition-all duration-200
          ${
            selected
              ? "ring-2 ring-offset-2 shadow-xl"
              : "shadow-md hover:shadow-lg"
          }
          ${executionFailed ? "ring-2 ring-red-400/60" : ""}
          ${isRunning ? "ring-2 ring-yellow-400/60" : ""}
          ${!nodeData.isConfigured ? "opacity-80" : ""}
        `}
        style={{
          backgroundColor: "#ffffff",
          ...(selected ? ({ ringColor: accentColor } as any) : {}),
        }}
        animate={
          isRunning
            ? {
                boxShadow: [
                  "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                  "0 10px 15px -3px rgba(251, 191, 36, 0.3)",
                  "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                ],
              }
            : {}
        }
        transition={{
          duration: 2,
          repeat: isRunning ? Infinity : 0,
          ease: "easeInOut",
        }}
      >
        {/* Top accent bar */}
        <div className="h-1" style={{ backgroundColor: accentColor }} />

        {/* Content */}
        <div className="px-4 py-3">
          {/* Header row */}
          <div className="flex items-center gap-2.5">
            {/* Icon container */}
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
              style={{ backgroundColor: `${accentColor}14` }}
            >
              {typeof nodeData.icon === "string" ? (
                <span className="text-base leading-none">
                  {nodeData.icon || "📦"}
                </span>
              ) : nodeData.icon ? (
                <nodeData.icon
                  className="w-4 h-4"
                  style={{ color: accentColor }}
                />
              ) : (
                <span className="text-base leading-none">📦</span>
              )}
            </div>

            {/* Label & status */}
            <div className="flex-1 min-w-0">
              <div className="font-semibold text-[13px] text-gray-900 leading-tight truncate">
                {nodeData.label}
              </div>
              <div className="flex items-center gap-1 mt-0.5">
                {isRunning ? (
                  <>
                    <Loader2 className="w-3 h-3 text-yellow-500 animate-spin" />
                    <span className="text-[11px] text-yellow-600 font-medium">
                      Running...
                    </span>
                  </>
                ) : isPending ? (
                  <>
                    <CircleDot className="w-3 h-3 text-gray-400" />
                    <span className="text-[11px] text-gray-500 font-medium">
                      Queued
                    </span>
                  </>
                ) : executionFailed ? (
                  <>
                    <AlertCircle className="w-3 h-3 text-red-500" />
                    <span className="text-[11px] text-red-500 font-medium">
                      Error
                    </span>
                  </>
                ) : executionSuccess ? (
                  <>
                    <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                    <span className="text-[11px] text-emerald-600 font-medium">
                      Done
                    </span>
                  </>
                ) : isActivityNode ? (
                  <>
                    <Play className="w-3 h-3 text-violet-500" />
                    <span className="text-[11px] text-violet-600 font-medium">
                      Interactive
                    </span>
                  </>
                ) : nodeData.isConfigured ? (
                  <>
                    <CircleDot className="w-3 h-3 text-blue-500" />
                    <span className="text-[11px] text-blue-600 font-medium">
                      Ready
                    </span>
                  </>
                ) : (
                  <>
                    <CircleDot className="w-3 h-3 text-gray-400" />
                    <span className="text-[11px] text-gray-400 font-medium">
                      Not configured
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Validation Errors */}
          {nodeData.validationErrors &&
            nodeData.validationErrors.length > 0 && (
              <div className="mt-2 px-2 py-1 rounded-md bg-red-50 border border-red-100">
                <span className="text-[11px] text-red-600 font-medium">
                  {nodeData.validationErrors.length} validation error
                  {nodeData.validationErrors.length > 1 ? "s" : ""}
                </span>
              </div>
            )}

          {/* Action buttons */}
          {isActivityNode ? (
            <div className="mt-2.5">
              <button
                onClick={handleOpenActivity}
                className="w-full px-2.5 py-1.5 rounded-lg text-white text-[11px] font-semibold
                           flex items-center justify-center gap-1.5 transition-all duration-150
                           hover:brightness-110 active:scale-[0.98]"
                style={{ backgroundColor: accentColor }}
              >
                <Play className="w-3 h-3" />
                Open Activity
              </button>
            </div>
          ) : showViewButton ? (
            <div className="mt-2.5 flex gap-1.5">
              <button
                onClick={handleViewData}
                className="flex-1 px-2.5 py-1.5 rounded-lg text-white text-[11px] font-semibold
                           flex items-center justify-center gap-1.5 transition-all duration-150
                           hover:brightness-110 active:scale-[0.98]"
                style={{ backgroundColor: accentColor }}
              >
                <Eye className="w-3 h-3" />
                View Data
              </button>
              <button
                onClick={handleReconfig}
                className="px-2.5 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-600
                           text-[11px] font-semibold flex items-center justify-center gap-1
                           transition-all duration-150 active:scale-[0.98]"
                title="Reconfigure"
              >
                <Settings className="w-3 h-3" />
              </button>
            </div>
          ) : null}

          {/* Story button for sample_dataset nodes with available stories */}
          {showStoryButton && (
            <div className="mt-2">
              <button
                onClick={handleOpenStory}
                className="w-full px-2.5 py-1.5 rounded-lg bg-violet-50 hover:bg-violet-100 text-violet-600
                           text-[11px] font-semibold flex items-center justify-center gap-1.5
                           transition-all duration-150 active:scale-[0.98] border border-violet-200"
              >
                <BookOpen className="w-3 h-3" />
                Story Guide
              </button>
            </div>
          )}
        </div>
      </motion.div>

      {/* Delete button */}
      <button
        onClick={handleDelete}
        className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-gray-800 hover:bg-red-500
                   text-white flex items-center justify-center
                   opacity-0 group-hover:opacity-100 transition-all duration-150 z-10
                   shadow-sm hover:shadow-md"
        title="Delete node"
      >
        <X className="w-2.5 h-2.5" strokeWidth={2.5} />
      </button>

      {/* Data preview tooltip on hover after execution */}
      {hasExecutionResults &&
        executionSuccess &&
        !isActivityNode &&
        (() => {
          const out = nodeResult?.output as Record<string, unknown> | undefined;
          return (
            <div
              className="absolute -top-14 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100
                       transition-all duration-200 pointer-events-none z-20"
            >
              <div
                className="bg-gray-900/90 backdrop-blur-sm text-white text-[10px] font-medium
                          px-2.5 py-1.5 rounded-lg shadow-lg whitespace-nowrap flex items-center gap-2"
              >
                {out && typeof out === "object" ? (
                  <>
                    {out.rows != null && <span>{String(out.rows)} rows</span>}
                    {out.columns != null && (
                      <>
                        <span className="w-px h-3 bg-gray-600" />
                        <span>
                          {Array.isArray(out.columns)
                            ? out.columns.length
                            : String(out.columns)}{" "}
                          cols
                        </span>
                      </>
                    )}
                    {out.accuracy != null && (
                      <span>
                        Acc: {(Number(out.accuracy) * 100).toFixed(1)}%
                      </span>
                    )}
                    {out.r2_score != null && (
                      <span>R²: {Number(out.r2_score).toFixed(3)}</span>
                    )}
                    {out.mse != null && out.r2_score == null && (
                      <span>MSE: {Number(out.mse).toFixed(3)}</span>
                    )}
                    {out.rows == null &&
                      out.accuracy == null &&
                      out.r2_score == null &&
                      out.mse == null && <span>Completed</span>}
                  </>
                ) : (
                  <span>Completed</span>
                )}
              </div>
              <div className="w-2 h-2 bg-gray-900/90 rotate-45 mx-auto -mt-1" />
            </div>
          );
        })()}

      {/* Input Handle — hidden for activity nodes and sample_dataset */}
      {!isActivityNode && nodeData.type !== "sample_dataset" && (
        <Handle
          type="target"
          position={Position.Top}
          className="w-3! h-3! border-2! border-white! rounded-full! -top-1.5!"
          style={{
            backgroundColor: accentColor,
            boxShadow: `0 0 0 2px ${accentColor}30`,
          }}
        />
      )}

      {/* Output Handle — hidden for activity nodes and evaluate */}
      {!isActivityNode && nodeData.type !== "evaluate" && (
        <Handle
          type="source"
          position={Position.Bottom}
          className="w-3! h-3! border-2! border-white! rounded-full! -bottom-1.5!"
          style={{
            backgroundColor: accentColor,
            boxShadow: `0 0 0 2px ${accentColor}30`,
          }}
        />
      )}
    </div>
  );
};

export default memo(MLNode);
