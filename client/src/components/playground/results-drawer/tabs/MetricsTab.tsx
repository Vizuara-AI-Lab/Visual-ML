import { TrendingUp } from "lucide-react";

interface MetricsTabProps {
  metrics: Record<string, number | string>;
}

const ERROR_METRICS = new Set(["mse", "rmse", "mae", "mse_score", "rmse_score", "mae_score"]);

function getInterpretation(
  key: string,
  value: number
): { label: string; color: string } {
  const isError = ERROR_METRICS.has(key.toLowerCase());

  if (isError) {
    // Lower is better for error metrics
    if (value <= 0.01) return { label: "Excellent", color: "text-green-700 bg-green-100" };
    if (value <= 0.1) return { label: "Good", color: "text-blue-700 bg-blue-100" };
    if (value <= 0.5) return { label: "Moderate", color: "text-yellow-700 bg-yellow-100" };
    return { label: "Poor", color: "text-red-700 bg-red-100" };
  }

  // Higher is better for accuracy-like metrics (0-1 range)
  if (value >= 0.9) return { label: "Excellent", color: "text-green-700 bg-green-100" };
  if (value >= 0.7) return { label: "Good", color: "text-blue-700 bg-blue-100" };
  if (value >= 0.5) return { label: "Moderate", color: "text-yellow-700 bg-yellow-100" };
  return { label: "Poor", color: "text-red-700 bg-red-100" };
}

function getProgressColor(key: string, value: number): string {
  const isError = ERROR_METRICS.has(key.toLowerCase());
  if (isError) {
    if (value <= 0.01) return "bg-green-500";
    if (value <= 0.1) return "bg-blue-500";
    if (value <= 0.5) return "bg-yellow-500";
    return "bg-red-500";
  }
  if (value >= 0.8) return "bg-green-500";
  if (value >= 0.5) return "bg-yellow-500";
  return "bg-red-500";
}

function getProgressWidth(key: string, value: number): number {
  const isError = ERROR_METRICS.has(key.toLowerCase());
  if (isError) {
    // Inverse: lower error = fuller bar, cap at 1.0
    return Math.max(0, Math.min(100, (1 - Math.min(value, 1)) * 100));
  }
  // Direct: higher = fuller bar
  return Math.max(0, Math.min(100, value * 100));
}

function MetricCard({ metricKey, value }: { metricKey: string; value: number | string }) {
  const numValue = typeof value === "number" ? value : parseFloat(String(value));
  const isNumeric = !isNaN(numValue);
  const displayValue = isNumeric ? numValue.toFixed(4) : String(value);
  const interpretation = isNumeric ? getInterpretation(metricKey, numValue) : null;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
          {metricKey.replace(/_/g, " ")}
        </span>
        {interpretation && (
          <span
            className={`px-2 py-0.5 text-xs font-medium rounded-full ${interpretation.color}`}
          >
            {interpretation.label}
          </span>
        )}
      </div>
      <div className="text-2xl font-bold text-slate-800 mb-2">
        {displayValue}
      </div>
      {isNumeric && (
        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${getProgressColor(
              metricKey,
              numValue
            )}`}
            style={{
              width: `${getProgressWidth(metricKey, numValue)}%`,
            }}
          />
        </div>
      )}
    </div>
  );
}

export function MetricsTab({ metrics }: MetricsTabProps) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-slate-400">
        <TrendingUp className="w-8 h-8 mb-2" />
        <p className="text-sm">No metrics available</p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {Object.entries(metrics).map(([key, value]) => (
          <MetricCard key={key} metricKey={key} value={value} />
        ))}
      </div>
    </div>
  );
}
