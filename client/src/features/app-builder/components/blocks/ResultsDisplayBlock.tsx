/**
 * Results Display Block — Shows pipeline output in table/card/json format.
 */

import type { BlockRenderProps } from "../BlockRenderer";
import type { ResultsDisplayConfig } from "../../types/appBuilder";

export default function ResultsDisplayBlock({
  block,
  mode,
  results,
}: BlockRenderProps) {
  const config = block.config as ResultsDisplayConfig;

  // In edit/preview mode or no results yet
  if (mode !== "live" || !results) {
    return (
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">{config.title}</h3>
        <div className="bg-gray-50 rounded-lg p-8 text-center">
          <p className="text-sm text-gray-400">
            {mode === "live" ? "Results will appear here after execution" : "Results display area"}
          </p>
        </div>
      </div>
    );
  }

  if (!results.success) {
    return (
      <div className="bg-white rounded-xl border border-red-200 p-6">
        <h3 className="text-sm font-medium text-red-700 mb-2">{config.title}</h3>
        <p className="text-sm text-red-500">{results.error}</p>
      </div>
    );
  }

  // If this block is mapped to a specific node, show only that node's output
  const allResults = results.results ?? {};
  const nodeId = config.nodeId || block.nodeId;
  let data: Record<string, unknown>;
  if (nodeId && (allResults as Record<string, unknown>).node_results) {
    const nodeResults = (allResults as Record<string, unknown>).node_results as Record<string, Record<string, unknown>>;
    data = nodeResults[nodeId] ?? allResults;
  } else {
    data = allResults as Record<string, unknown>;
  }

  return (
    <div className="bg-white rounded-xl border p-6">
      <h3 className="text-sm font-medium text-gray-700 mb-3">{config.title}</h3>

      {config.displayMode === "json" ? (
        <pre className="bg-gray-50 rounded-lg p-4 text-xs overflow-x-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      ) : config.displayMode === "card" ? (
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(data).map(([key, value]) => (
            <div key={key} className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">{key}</p>
              <p className="text-sm font-medium text-gray-900">
                {typeof value === "object" ? JSON.stringify(value) : String(value)}
              </p>
            </div>
          ))}
        </div>
      ) : (
        // table mode
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 px-3 text-gray-500 font-medium">Key</th>
                <th className="text-left py-2 px-3 text-gray-500 font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data).map(([key, value]) => (
                <tr key={key} className="border-b last:border-0">
                  <td className="py-2 px-3 text-gray-700">{key}</td>
                  <td className="py-2 px-3 text-gray-900">
                    {typeof value === "object" ? JSON.stringify(value) : String(value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {results.execution_time_ms && (
        <p className="text-xs text-gray-400 mt-3">
          Executed in {results.execution_time_ms}ms
        </p>
      )}
    </div>
  );
}
