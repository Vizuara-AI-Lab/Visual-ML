import { useState } from "react";
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { type Node } from "@xyflow/react";
import type { BaseNodeData } from "../../../../types/pipeline";
import type { PipelineExecutionResult } from "../../../../store/playgroundStore";

interface NodeResultsTabProps {
  executionResult: PipelineExecutionResult;
  nodes: Node<BaseNodeData>[];
}

function getErrorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    const errorObj = error as Record<string, unknown>;
    if (errorObj.details && typeof errorObj.details === "object") {
      const details = errorObj.details as Record<string, unknown>;
      if (details.reason && typeof details.reason === "string")
        return details.reason;
    }
    if (errorObj.message && typeof errorObj.message === "string") {
      const match = errorObj.message.match(/\[.*?\]: (.+)/);
      return match ? match[1] : errorObj.message;
    }
    if (errorObj.error && typeof errorObj.error === "string")
      return errorObj.error;
    return JSON.stringify(error);
  }
  return "Unknown error";
}

function NodeResultCard({
  nodeId,
  result,
  nodes,
}: {
  nodeId: string;
  result: { success: boolean; output?: unknown; error?: string };
  nodes: Node<BaseNodeData>[];
}) {
  const [showRaw, setShowRaw] = useState(false);

  const node = nodes.find((n) => n.id === nodeId);
  const label = node?.data?.label || nodeId;
  const nodeType = node?.data?.type || "unknown";
  const color = (node?.data?.color as string) || "#6366f1";

  const output = result.output as Record<string, unknown> | undefined;
  const warnings = output?.warnings as string[] | undefined;

  // Extract displayable key-value pairs (skip warnings and large objects)
  const displayEntries = output
    ? Object.entries(output).filter(
        ([key, val]) =>
          key !== "warnings" &&
          (typeof val === "string" ||
            typeof val === "number" ||
            typeof val === "boolean")
      )
    : [];

  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-100">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: `${color}20` }}
        >
          {result.success ? (
            <CheckCircle className="w-4 h-4" style={{ color }} />
          ) : (
            <XCircle className="w-4 h-4 text-red-500" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-slate-800 truncate">
            {label}
          </div>
          <div className="text-xs text-slate-500">{nodeType.replace(/_/g, " ")}</div>
        </div>
        <span
          className={`px-2 py-0.5 text-xs font-medium rounded-full ${
            result.success
              ? "bg-green-100 text-green-700"
              : "bg-red-100 text-red-700"
          }`}
        >
          {result.success ? "Success" : "Failed"}
        </span>
      </div>

      {/* Body */}
      <div className="px-4 py-3 space-y-3">
        {/* Warnings */}
        {warnings && warnings.length > 0 && (
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-yellow-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <h5 className="text-xs font-semibold text-yellow-800 mb-1">
                  Warnings ({warnings.length})
                </h5>
                <ul className="space-y-0.5">
                  {warnings.map((warning, idx) => (
                    <li key={idx} className="text-xs text-yellow-700">
                      {warning}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Structured output badges */}
        {displayEntries.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {displayEntries.map(([key, val]) => (
              <div
                key={key}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-slate-50 border border-slate-200 rounded-lg"
              >
                <span className="text-xs text-slate-500 capitalize">
                  {key.replace(/_/g, " ")}:
                </span>
                <span className="text-xs font-medium text-slate-800">
                  {typeof val === "number" ? val.toFixed(4) : String(val)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Error */}
        {result.error && (
          <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
            <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
            <span className="text-sm text-red-700">
              {getErrorMessage(result.error)}
            </span>
          </div>
        )}

        {/* Collapsible raw output */}
        {output && (
          <div>
            <button
              onClick={() => setShowRaw(!showRaw)}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-700 transition-colors"
            >
              {showRaw ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
              Raw Output
            </button>
            {showRaw && (
              <pre className="mt-2 text-xs text-slate-700 bg-slate-50 border border-slate-200 p-3 rounded-lg overflow-x-auto max-h-48 overflow-y-auto">
                {JSON.stringify(output, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function NodeResultsTab({
  executionResult,
  nodes,
}: NodeResultsTabProps) {
  if (
    !executionResult.nodeResults ||
    Object.keys(executionResult.nodeResults).length === 0
  ) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-slate-400">
        <AlertCircle className="w-8 h-8 mb-2" />
        <p className="text-sm">No node results available</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-3">
      {Object.entries(executionResult.nodeResults).map(([nodeId, result]) => (
        <NodeResultCard
          key={nodeId}
          nodeId={nodeId}
          result={
            result as {
              success: boolean;
              output?: unknown;
              error?: string;
            }
          }
          nodes={nodes}
        />
      ))}
    </div>
  );
}
