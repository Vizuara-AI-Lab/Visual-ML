import { motion, AnimatePresence } from "framer-motion";
import { X, CheckCircle, XCircle, AlertCircle, TrendingUp } from "lucide-react";
import { usePlaygroundStore } from "../../store/playgroundStore";

interface ResultsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ResultsPanel = ({ isOpen, onClose }: ResultsPanelProps) => {
  const { executionResult, isExecuting } = usePlaygroundStore();

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: "100%" }}
          animate={{ x: 0 }}
          exit={{ x: "100%" }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="fixed right-0 top-0 h-full w-96 bg-gray-900 border-l border-gray-700 shadow-2xl z-40 flex flex-col"
        >
          {/* Header */}
          <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">
              Execution Results
            </h3>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-4">
            {isExecuting && (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
                <p className="text-gray-400 mt-4">Executing pipeline...</p>
              </div>
            )}

            {!isExecuting && !executionResult && (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500">
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
                  <div className="flex items-start gap-3 p-4 bg-green-900/20 border border-green-700 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-green-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-sm font-medium text-green-300">
                        Execution Successful
                      </h4>
                      <p className="text-xs text-green-400/80 mt-1">
                        All nodes executed without errors
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-3 p-4 bg-red-900/20 border border-red-700 rounded-lg">
                    <XCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-sm font-medium text-red-300">
                        Execution Failed
                      </h4>
                      {executionResult.error && (
                        <p className="text-xs text-red-400/80 mt-1">
                          {executionResult.error}
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Node Results */}
                {executionResult.nodeResults &&
                  Object.keys(executionResult.nodeResults).length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-300 mb-2">
                        Node Results
                      </h4>
                      <div className="space-y-2">
                        {Object.entries(executionResult.nodeResults).map(
                          ([nodeId, result]) => {
                            const nodeResult = result as {
                              success: boolean;
                              output?: unknown;
                              error?: string;
                            };
                            return (
                              <div
                                key={nodeId}
                                className="p-3 bg-gray-800 border border-gray-700 rounded-lg"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-xs font-mono text-gray-400">
                                    {nodeId}
                                  </span>
                                  {nodeResult.success ? (
                                    <CheckCircle className="w-4 h-4 text-green-400" />
                                  ) : (
                                    <XCircle className="w-4 h-4 text-red-400" />
                                  )}
                                </div>
                                {nodeResult.output && (
                                  <pre className="text-xs text-gray-300 bg-gray-900 p-2 rounded overflow-x-auto">
                                    {JSON.stringify(nodeResult.output, null, 2)}
                                  </pre>
                                )}
                                {nodeResult.error && (
                                  <div className="flex items-start gap-2 mt-2 text-xs text-red-400">
                                    <AlertCircle className="w-3 h-3 shrink-0 mt-0.5" />
                                    <span>
                                      {typeof nodeResult.error === "string"
                                        ? nodeResult.error
                                        : JSON.stringify(nodeResult.error)}
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
                    <h4 className="text-sm font-medium text-gray-300 mb-2">
                      Metrics
                    </h4>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(executionResult.metrics).map(
                        ([key, value]) => (
                          <div
                            key={key}
                            className="p-3 bg-gray-800 border border-gray-700 rounded-lg"
                          >
                            <div className="text-xs text-gray-400 uppercase tracking-wide">
                              {key}
                            </div>
                            <div className="text-lg font-semibold text-white mt-1">
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
