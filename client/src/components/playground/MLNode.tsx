/**
 * Custom ML Node Component for React Flow
 */

import { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { BaseNodeData } from "../../types/pipeline";

const MLNode = ({ data }: NodeProps<BaseNodeData>) => {
  const nodeData = data as BaseNodeData;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-lg transition-all min-w-[180px] border-gray-300 ${
        !nodeData.isConfigured ? "opacity-70" : ""
      }`}
      style={{
        backgroundColor: "#fff",
        borderLeftWidth: "4px",
        borderLeftColor: nodeData.color || "#gray",
      }}
    >
      {/* Input Handle */}
      {nodeData.type !== "load_url" && nodeData.type !== "sample_dataset" && (
        <Handle
          type="target"
          position={Position.Top}
          className="w-3 h-3 !bg-gray-400 border-2 border-white"
        />
      )}

      {/* Node Content */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="text-xl">{nodeData.icon || "📦"}</span>
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
