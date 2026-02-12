import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  CheckCircle,
  XCircle,
  AlertCircle,
  TrendingUp,
  AlertTriangle,
} from "lucide-react";
import { usePlaygroundStore } from "../../store/playgroundStore";

interface ResultsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ResultsPanel = ({ isOpen, onClose }: ResultsPanelProps) => {
  const { executionResult, isExecuting } = usePlaygroundStore();

  // Helper function to extract user-friendly error message
  const getErrorMessage = (error: unknown): string => {
    if (typeof error === "string") {
      return error;
    }

    if (error && typeof error === "object") {
      const errorObj = error as Record<string, unknown>;

      // Try to extract the most user-friendly message
      // Priority: details.reason > message (cleaned) > error field
      if (errorObj.details && typeof errorObj.details === "object") {
        const details = errorObj.details as Record<string, unknown>;
        if (details.reason && typeof details.reason === "string") {
          return details.reason;
        }
      }

      if (errorObj.message && typeof errorObj.message === "string") {
        // Remove the "Node execution failed [node_type]: " prefix
        const message = errorObj.message;
        const match = message.match(/\[.*?\]: (.+)/);
        return match ? match[1] : message;
      }

      if (errorObj.error && typeof errorObj.error === "string") {
        return errorObj.error;
      }

      // Fallback to stringify if we can't extract a message
      return JSON.stringify(error);
    }

    return "Unknown error";
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: "100%" }}
          animate={{ x: 0 }}
          exit={{ x: "100%" }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="fixed right-0 top-0 h-full w-96 bg-white/90 backdrop-blur-md border-l border-slate-200/60 shadow-2xl z-40 flex flex-col"
        >
          {/* Header */}
          <div className="px-4 py-3 border-b border-slate-200/60 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-800">
              Execution Results
            </h3>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-500" />
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-4">
            {isExecuting && (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="w-12 h-12 border-4 border-slate-900 border-t-transparent rounded-full animate-spin" />
                <p className="text-slate-600 mt-4">Executing pipeline...</p>
              </div>
            )}

            {!isExecuting && !executionResult && (
              <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                <TrendingUp className="w-12 h-12 mb-3" />
                <p>No results yet</p>
                <p className="text-sm mt-1">
                  Execute the pipeline to see results
                </p>
              </div>
            )}

            {!isExecuting && executionResult && (
              <div className="space-y-4">
                {/* Success/Error Banner */}
                {executionResult.success ? (
                  <>
                    <div className="flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-600 shrink-0 mt-0.5" />
                      <div>
                        <h4 className="text-sm font-medium text-green-800">
                          Execution Successful
                        </h4>
                        <p className="text-xs text-green-700 mt-1">
                          All nodes executed without errors
                        </p>
                      </div>
                    </div>

                    {/* Global warnings summary */}
                    {(() => {
                      const allWarnings: Array<{
                        nodeId: string;
                        warnings: string[];
                      }> = [];
                      if (executionResult.nodeResults) {
                        Object.entries(executionResult.nodeResults).forEach(
                          ([nodeId, result]) => {
                            const nodeResult = result as {
                              output?: Record<string, unknown>;
                            };
                            const warnings = nodeResult.output?.warnings as
                              | string[]
                              | undefined;
                            if (warnings && warnings.length > 0) {
                              allWarnings.push({ nodeId, warnings });
                            }
                          },
                        );
                      }

                      if (allWarnings.length > 0) {
                        const totalWarnings = allWarnings.reduce(
                          (sum, item) => sum + item.warnings.length,
                          0,
                        );
                        return (
                          <div className="flex items-start gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <AlertTriangle className="w-5 h-5 text-yellow-600 shrink-0 mt-0.5" />
                            <div className="flex-1">
                              <h4 className="text-sm font-medium text-yellow-800">
                                Pipeline completed with {totalWarnings} warning
                                {totalWarnings !== 1 ? "s" : ""}
                              </h4>
                              <p className="text-xs text-yellow-700 mt-1">
                                Some operations were skipped or adjusted. Check
                                individual node results below for details.
                              </p>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    })()}
                  </>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                      <XCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-red-800 mb-1">
                          Execution Failed
                        </h4>
                        {executionResult.error && (
                          <p className="text-sm text-red-700 mt-2 font-medium leading-relaxed">
                            {executionResult.error}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Error Details */}
                    {executionResult.errorDetails && (
                      <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                        <h5 className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-2">
                          Details
                        </h5>
                        <div className="space-y-1.5">
                          {Object.entries(executionResult.errorDetails).map(
                            ([key, value]) => {
                              // Skip input_summary as it's too technical
                              if (key === "input_summary") return null;

                              return (
                                <div key={key} className="flex items-start">
                                  <span className="text-xs font-medium text-slate-600 min-w-25 capitalize">
                                    {key.replace(/_/g, " ")}:
                                  </span>
                                  <span className="text-xs text-slate-800">
                                    {typeof value === "object"
                                      ? JSON.stringify(value, null, 2)
                                      : String(value)}
                                  </span>
                                </div>
                              );
                            },
                          )}
                        </div>
                      </div>
                    )}

                    {/* Suggestion */}
                    {executionResult.errorSuggestion && (
                      <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
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
                )}

                {/* Node Results */}
                {executionResult.nodeResults &&
                  Object.keys(executionResult.nodeResults).length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-800 mb-2">
                        Node Results
                      </h4>
                      <div className="space-y-2">
                        {Object.entries(executionResult.nodeResults).map(
                          ([nodeId, result]) => {
                            const nodeResult = result as {
                              success: boolean;
                              output?: Record<string, unknown>;
                              error?: string;
                            };

                            // Extract warnings from output if they exist
                            const warnings = nodeResult.output?.warnings as
                              | string[]
                              | undefined;

                            return (
                              <div
                                key={nodeId}
                                className="p-3 bg-white border border-slate-200 rounded-lg"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-xs font-mono text-slate-600">
                                    {nodeId}
                                  </span>
                                  {nodeResult.success ? (
                                    <CheckCircle className="w-4 h-4 text-green-600" />
                                  ) : (
                                    <XCircle className="w-4 h-4 text-red-600" />
                                  )}
                                </div>

                                {/* Warnings Display */}
                                {warnings && warnings.length > 0 && (
                                  <div className="mb-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                                    <div className="flex items-start gap-2">
                                      <AlertTriangle className="w-4 h-4 text-yellow-600 shrink-0 mt-0.5" />
                                      <div className="flex-1">
                                        <h5 className="text-xs font-semibold text-yellow-800 mb-1">
                                          Warnings ({warnings.length})
                                        </h5>
                                        <ul className="space-y-1">
                                          {warnings.map((warning, idx) => (
                                            <li
                                              key={idx}
                                              className="text-xs text-yellow-700"
                                            >
                                              • {warning}
                                            </li>
                                          ))}
                                        </ul>
                                      </div>
                                    </div>
                                  </div>
                                )}

                                {nodeResult.output && (
                                  <pre className="text-xs text-slate-800 bg-slate-50 p-2 rounded overflow-x-auto">
                                    {JSON.stringify(nodeResult.output, null, 2)}
                                  </pre>
                                )}
                                {nodeResult.error && (
                                  <div className="flex items-start gap-2 mt-2">
                                    <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                                    <span className="text-sm text-red-700">
                                      {getErrorMessage(nodeResult.error)}
                                    </span>
                                  </div>
                                )}
                              </div>
                            );
                          },
                        )}
                      </div>
                    </div>
                  )}

                {/* Metrics */}
                {executionResult.metrics && (
                  <div>
                    <h4 className="text-sm font-medium text-slate-800 mb-2">
                      Metrics
                    </h4>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(executionResult.metrics).map(
                        ([key, value]) => (
                          <div
                            key={key}
                            className="p-3 bg-white border border-slate-200 rounded-lg"
                          >
                            <div className="text-xs text-slate-600 uppercase tracking-wide">
                              {key}
                            </div>
                            <div className="text-lg font-semibold text-slate-800 mt-1">
                              {typeof value === "number"
                                ? value.toFixed(4)
                                : String(value)}
                            </div>
                          </div>
                        ),
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
