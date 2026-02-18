/**
 * Metrics Card Block — Displays specific metrics from pipeline results.
 */

import type { BlockRenderProps } from "../BlockRenderer";
import type { MetricsCardConfig } from "../../types/appBuilder";

export default function MetricsCardBlock({
  block,
  mode,
  results,
}: BlockRenderProps) {
  const config = block.config as MetricsCardConfig;

  const formatValue = (value: unknown, format: string): string => {
    if (value == null) return "—";
    const num = Number(value);
    if (format === "percentage" && !isNaN(num)) return `${(num * 100).toFixed(1)}%`;
    if (format === "number" && !isNaN(num)) return num.toFixed(4);
    return String(value);
  };

  return (
    <div className="bg-white rounded-xl border p-6">
      <h3 className="text-sm font-medium text-gray-700 mb-4">{config.title}</h3>

      <div className="grid grid-cols-2 gap-4">
        {config.metrics.map((metric) => {
          let value: unknown = null;
          if (mode === "live" && results?.success && results.results) {
            const allResults = results.results as Record<string, unknown>;
            // Try node-specific results first, then fall back to flat results
            if (metric.nodeId && allResults.node_results) {
              const nodeResults = allResults.node_results as Record<string, Record<string, unknown>>;
              const nodeData = nodeResults[metric.nodeId];
              value = nodeData?.[metric.nodeOutputKey ?? metric.key] ?? nodeData?.[metric.key];
            }
            if (value == null) {
              value = allResults[metric.key];
            }
          }

          return (
            <div
              key={metric.key}
              className="bg-gray-50 rounded-lg p-4 text-center"
            >
              <p className="text-xs text-gray-500 mb-1">{metric.label}</p>
              <p className="text-xl font-bold text-gray-900">
                {value != null ? formatValue(value, metric.format) : "—"}
              </p>
            </div>
          );
        })}
      </div>

      {config.metrics.length === 0 && (
        <p className="text-sm text-gray-400 text-center py-4">
          No metrics configured
        </p>
      )}
    </div>
  );
}
