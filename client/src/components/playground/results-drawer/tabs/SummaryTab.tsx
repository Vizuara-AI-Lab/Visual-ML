import { useMemo } from "react";
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  AlertTriangle,
  Clock,
  Boxes,
} from "lucide-react";
import type { PipelineExecutionResult } from "../../../../store/playgroundStore";
import { getFriendlyMessage } from "../../../../utils/errorFormatter";

interface SummaryTabProps {
  executionResult: PipelineExecutionResult;
}

export function SummaryTab({ executionResult }: SummaryTabProps) {
  const warnings = useMemo(() => {
    const allWarnings: Array<{ nodeId: string; warnings: string[] }> = [];
    if (executionResult.nodeResults) {
      Object.entries(executionResult.nodeResults).forEach(
        ([nodeId, result]) => {
          const nodeResult = result as { output?: Record<string, unknown> };
          const w = nodeResult.output?.warnings as string[] | undefined;
          if (w && w.length > 0) allWarnings.push({ nodeId, warnings: w });
        }
      );
    }
    return allWarnings;
  }, [executionResult.nodeResults]);

  const totalWarnings = warnings.reduce(
    (sum, item) => sum + item.warnings.length,
    0
  );

  const nodeCount = executionResult.nodeResults
    ? Object.keys(executionResult.nodeResults).length
    : 0;

  const metricCount = executionResult.metrics
    ? Object.keys(executionResult.metrics).length
    : 0;

  return (
    <div className="p-4 space-y-4">
      {/* Status Banner */}
      {executionResult.success ? (
        <div className="flex items-start gap-4 p-5 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl">
          <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center shrink-0">
            <CheckCircle className="w-5 h-5 text-green-600" />
          </div>
          <div className="flex-1">
            <h4 className="text-base font-semibold text-green-800">
              Pipeline Executed Successfully
            </h4>
            <p className="text-sm text-green-700 mt-1">
              All nodes completed without errors
            </p>
            {/* Quick stats row */}
            <div className="flex items-center gap-4 mt-3">
              <div className="flex items-center gap-1.5 text-xs text-green-700">
                <Boxes className="w-3.5 h-3.5" />
                {nodeCount} node{nodeCount !== 1 ? "s" : ""} executed
              </div>
              {metricCount > 0 && (
                <div className="flex items-center gap-1.5 text-xs text-green-700">
                  <Clock className="w-3.5 h-3.5" />
                  {metricCount} metric{metricCount !== 1 ? "s" : ""} collected
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-4 p-5 bg-gradient-to-r from-red-50 to-rose-50 border border-red-200 rounded-xl">
          <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center shrink-0">
            <XCircle className="w-5 h-5 text-red-600" />
          </div>
          <div className="flex-1">
            <h4 className="text-base font-semibold text-red-800">
              Execution Failed
            </h4>
            {executionResult.error && (
              <p className="text-sm text-red-700 mt-2 leading-relaxed">
                {getFriendlyMessage(executionResult.error)}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Warnings */}
      {totalWarnings > 0 && (
        <div className="flex items-start gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
          <AlertTriangle className="w-5 h-5 text-yellow-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="text-sm font-medium text-yellow-800">
              {totalWarnings} warning{totalWarnings !== 1 ? "s" : ""} during
              execution
            </h4>
            <p className="text-xs text-yellow-700 mt-1">
              Some operations were skipped or adjusted. Check Node Results tab
              for details.
            </p>
          </div>
        </div>
      )}

      {/* Error Details (2-column on wider screens) */}
      {!executionResult.success && executionResult.errorDetails && (
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
          <h5 className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">
            Error Details
          </h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {Object.entries(executionResult.errorDetails).map(
              ([key, value]) => {
                if (key === "input_summary") return null;
                return (
                  <div key={key} className="flex items-start gap-2">
                    <span className="text-xs font-medium text-slate-500 min-w-24 capitalize">
                      {key.replace(/_/g, " ")}:
                    </span>
                    <span className="text-xs text-slate-800">
                      {typeof value === "object"
                        ? JSON.stringify(value, null, 2)
                        : String(value)}
                    </span>
                  </div>
                );
              }
            )}
          </div>
        </div>
      )}

      {/* Suggestion */}
      {executionResult.errorSuggestion && (
        <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-200 rounded-xl">
          <AlertCircle className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
          <div>
            <h5 className="text-xs font-semibold text-blue-800 uppercase tracking-wide mb-1">
              Suggestion
            </h5>
            <p className="text-sm text-blue-700">
              {executionResult.errorSuggestion}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
