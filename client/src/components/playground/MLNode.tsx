/**
 * Custom ML Node Component for React Flow
 */

import { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { BaseNodeData } from "../../types/pipeline";
import { Eye, X, Settings } from "lucide-react";
import { usePlaygroundStore } from "../../store/playgroundStore";

const MLNode = ({ data, id }: NodeProps<BaseNodeData>) => {
  const nodeData = data as BaseNodeData;
  const { executionResult, deleteNode } = usePlaygroundStore();

  const viewNodeTypes = [
    "table_view",
    "data_preview",
    "statistics_view",
    "column_info",
    "chart_view",
    "missing_value_handler",
  ];

  const isViewNode = viewNodeTypes.includes(nodeData.type);
  const hasExecutionResults = executionResult?.nodeResults?.[id];
  const showViewButton =
    isViewNode && nodeData.isConfigured && hasExecutionResults;

  const handleViewData = (e: React.MouseEvent) => {
    e.stopPropagation();
    // Dispatch custom event to open view modal
    window.dispatchEvent(
      new CustomEvent("openViewNodeModal", { detail: { nodeId: id } }),
    );
  };

  const handleReconfig = (e: React.MouseEvent) => {
    e.stopPropagation();
    // Dispatch custom event to open config modal
    window.dispatchEvent(
      new CustomEvent("openConfigModal", { detail: { nodeId: id } }),
    );
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    deleteNode(id);
  };

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-lg transition-all min-w-[180px] border-gray-300 relative group ${
        !nodeData.isConfigured ? "opacity-70" : ""
      }`}
      style={{
        backgroundColor: "#fff",
        borderLeftWidth: "4px",
        borderLeftColor: nodeData.color || "#gray",
      }}
    >
      {/* Delete Button */}
      <button
        onClick={handleDelete}
        className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-10"
        title="Delete node"
      >
        <X className="w-3 h-3" />
      </button>

      {/* Input Handle */}
      {nodeData.type !== "sample_dataset" && (
        <Handle
          type="target"
          position={Position.Top}
          className="w-3 h-3 !bg-gray-400 border-2 border-white"
        />
      )}

      {/* Node Content */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          {typeof nodeData.icon === 'string' ? (
            <span className="text-xl">{nodeData.icon || "📦"}</span>
          ) : nodeData.icon ? (
            <nodeData.icon className="w-5 h-5" style={{ color: nodeData.color }} />
          ) : (
            <span className="text-xl">📦</span>
          )}
          <div className="flex-1">
            <div className="font-semibold text-sm text-gray-800">
              {nodeData.label}
            </div>
            <div className="text-xs text-gray-500">
              {nodeData.isConfigured ? "✓ Configured" : "⚠ Not configured"}
            </div>
          </div>
        </div>

        {/* Validation Errors */}
        {nodeData.validationErrors && nodeData.validationErrors.length > 0 && (
          <div className="mt-1 text-xs text-red-600">
            {nodeData.validationErrors.length} error(s)
          </div>
        )}

        {/* View Data Button */}
        {showViewButton && (
          <div className="mt-2 flex gap-1">
            <button
              onClick={handleViewData}
              className="flex-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded flex items-center justify-center gap-1.5 transition-colors"
            >
              <Eye className="w-3 h-3" />
              View Data
            </button>
            <button
              onClick={handleReconfig}
              className="px-3 py-1.5 bg-gray-600 hover:bg-gray-700 text-white text-xs rounded flex items-center justify-center gap-1.5 transition-colors"
              title="Reconfigure node"
            >
              <Settings className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Output Handle */}
      {nodeData.type !== "evaluate" && (
        <Handle
          type="source"
          position={Position.Bottom}
          className="w-3 h-3 !bg-gray-400 border-2 border-white"
        />
      )}
    </div>
  );
};

export default memo(MLNode);
