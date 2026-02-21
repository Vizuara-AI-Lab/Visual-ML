/**
 * NodeExplorerPanel — Routes a node result to the correct interactive explorer.
 * Reuses the rich explorer components from the playground.
 */

import { useState } from "react";
import {
  Database,
  Table,
  BarChart3,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  Clock,
  FileSpreadsheet,
  Columns3,
} from "lucide-react";

// Playground explorers (reused as-is)
import { LinearRegressionExplorer } from "../../../components/playground/LinearRegressionExplorer";
import { LogisticRegressionExplorer } from "../../../components/playground/LogisticRegressionExplorer";
import { MissingValueExplorer } from "../../../components/playground/MissingValueExplorer";
import { SplitExplorer } from "../../../components/playground/SplitExplorer";
import { DataExplorer } from "../../../components/playground/DataExplorer";
import { EncodingExplorer } from "../../../components/playground/EncodingExplorer";
import { ScalingExplorer } from "../../../components/playground/ScalingExplorer";
import { FeatureSelectionExplorer } from "../../../components/playground/FeatureSelectionExplorer";
import {
  R2ScoreResult,
  MSEScoreResult,
  RMSEScoreResult,
  MAEScoreResult,
  ConfusionMatrixResult,
} from "../../../components/playground/results_and_metrics";

// ─── Types ───────────────────────────────────────────────────────

interface NodeResult {
  node_type: string;
  execution_time_ms?: number;
  success?: boolean;
  error?: string | null;
  [key: string]: unknown;
}

interface NodeExplorerPanelProps {
  result: NodeResult;
  primaryColor: string;
}

// Human-readable labels
const NODE_TYPE_LABELS: Record<string, string> = {
  select_dataset: "Dataset",
  sample_dataset: "Dataset",
  upload_file: "File Upload",
  table_view: "Data Table",
  data_preview: "Data Preview",
  statistics_view: "Statistics",
  column_info: "Column Info",
  chart_view: "Chart",
  missing_value_handler: "Missing Value Handler",
  encoding: "Encoding",
  scaling: "Scaling",
  feature_selection: "Feature Selection",
  transformation: "Data Transformation",
  split: "Train/Test Split",
  linear_regression: "Linear Regression",
  logistic_regression: "Logistic Regression",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  r2_score: "R² Score",
  mse_score: "Mean Squared Error",
  rmse_score: "Root Mean Squared Error",
  mae_score: "Mean Absolute Error",
  confusion_matrix: "Confusion Matrix",
  preprocess: "Preprocessing",
};

// Node types that have rich explorer activities
const ACTIVITY_NODES = new Set([
  "linear_regression",
  "logistic_regression",
  "missing_value_handler",
  "split",
  "chart_view",
  "encoding",
  "scaling",
  "feature_selection",
]);

// Node category colors
const NODE_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  select_dataset: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-700", badge: "bg-blue-100 text-blue-700" },
  sample_dataset: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-700", badge: "bg-blue-100 text-blue-700" },
  upload_file: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-700", badge: "bg-blue-100 text-blue-700" },
  table_view: { bg: "bg-slate-50", border: "border-slate-200", text: "text-slate-700", badge: "bg-slate-100 text-slate-700" },
  data_preview: { bg: "bg-slate-50", border: "border-slate-200", text: "text-slate-700", badge: "bg-slate-100 text-slate-700" },
  statistics_view: { bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700", badge: "bg-emerald-100 text-emerald-700" },
  column_info: { bg: "bg-sky-50", border: "border-sky-200", text: "text-sky-700", badge: "bg-sky-100 text-sky-700" },
  chart_view: { bg: "bg-purple-50", border: "border-purple-200", text: "text-purple-700", badge: "bg-purple-100 text-purple-700" },
  missing_value_handler: { bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-700", badge: "bg-amber-100 text-amber-700" },
  encoding: { bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700", badge: "bg-orange-100 text-orange-700" },
  scaling: { bg: "bg-teal-50", border: "border-teal-200", text: "text-teal-700", badge: "bg-teal-100 text-teal-700" },
  feature_selection: { bg: "bg-indigo-50", border: "border-indigo-200", text: "text-indigo-700", badge: "bg-indigo-100 text-indigo-700" },
  transformation: { bg: "bg-cyan-50", border: "border-cyan-200", text: "text-cyan-700", badge: "bg-cyan-100 text-cyan-700" },
  split: { bg: "bg-cyan-50", border: "border-cyan-200", text: "text-cyan-700", badge: "bg-cyan-100 text-cyan-700" },
  linear_regression: { bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700", badge: "bg-emerald-100 text-emerald-700" },
  logistic_regression: { bg: "bg-violet-50", border: "border-violet-200", text: "text-violet-700", badge: "bg-violet-100 text-violet-700" },
  decision_tree: { bg: "bg-green-50", border: "border-green-200", text: "text-green-700", badge: "bg-green-100 text-green-700" },
  random_forest: { bg: "bg-green-50", border: "border-green-200", text: "text-green-700", badge: "bg-green-100 text-green-700" },
  r2_score: { bg: "bg-indigo-50", border: "border-indigo-200", text: "text-indigo-700", badge: "bg-indigo-100 text-indigo-700" },
  mse_score: { bg: "bg-indigo-50", border: "border-indigo-200", text: "text-indigo-700", badge: "bg-indigo-100 text-indigo-700" },
  rmse_score: { bg: "bg-indigo-50", border: "border-indigo-200", text: "text-indigo-700", badge: "bg-indigo-100 text-indigo-700" },
  mae_score: { bg: "bg-indigo-50", border: "border-indigo-200", text: "text-indigo-700", badge: "bg-indigo-100 text-indigo-700" },
  confusion_matrix: { bg: "bg-rose-50", border: "border-rose-200", text: "text-rose-700", badge: "bg-rose-100 text-rose-700" },
  preprocess: { bg: "bg-gray-50", border: "border-gray-200", text: "text-gray-700", badge: "bg-gray-100 text-gray-700" },
};

const DEFAULT_COLOR = { bg: "bg-gray-50", border: "border-gray-200", text: "text-gray-700", badge: "bg-gray-100 text-gray-700" };

// Count available activities for a node result
function countActivities(result: NodeResult): number {
  let count = 0;
  if (result.quiz_questions && Array.isArray(result.quiz_questions) && result.quiz_questions.length > 0) count++;
  if (result.coefficient_analysis) count++;
  if (result.equation_data) count++;
  if (result.prediction_playground) count++;
  if (result.strategy_comparison) count++;
  if (result.missing_heatmap) count++;
  if (result.class_distribution) count++;
  if (result.metric_explainer) count++;
  if (result.sigmoid_data) count++;
  if (result.split_visualization) count++;
  if (result.class_balance) count++;
  if (result.ratio_explorer) count++;
  if (result.exploration_data) count++;
  if (result.encoding_before_after) count++;
  if (result.encoding_map) count++;
  if (result.scaling_before_after) count++;
  if (result.scaling_method_comparison) count++;
  if (result.feature_score_details) count++;
  if (result.feature_correlation_matrix) count++;
  return count;
}

// ─── Main Component ──────────────────────────────────────────────

export default function NodeExplorerPanel({ result, primaryColor }: NodeExplorerPanelProps) {
  const nodeType = result.node_type;
  const colors = NODE_COLORS[nodeType] || DEFAULT_COLOR;
  const label = NODE_TYPE_LABELS[nodeType] || formatNodeType(nodeType);
  const activityCount = countActivities(result);
  const hasExplorer = ACTIVITY_NODES.has(nodeType) || activityCount > 0;
  const [expanded, setExpanded] = useState(false);

  // Metric nodes render inline (no expand)
  if (["r2_score", "mse_score", "rmse_score", "mae_score"].includes(nodeType)) {
    return <MetricInlineCard result={result} primaryColor={primaryColor} label={label} colors={colors} />;
  }

  if (nodeType === "confusion_matrix") {
    return (
      <div className={`rounded-xl border ${colors.border} overflow-hidden`}>
        <div className={`px-4 py-3 ${colors.bg} flex items-center gap-2`}>
          <BarChart3 className={`h-4 w-4 ${colors.text}`} />
          <span className={`text-sm font-semibold ${colors.text}`}>{label}</span>
        </div>
        <div className="p-4 bg-white">
          <ConfusionMatrixResult result={result} />
        </div>
      </div>
    );
  }

  // Dataset nodes render as info cards
  if (["select_dataset", "sample_dataset", "upload_file"].includes(nodeType)) {
    return <DatasetCard result={result} label={label} colors={colors} />;
  }

  // Nodes without explorers render as simple expandable
  if (!hasExplorer) {
    return <SimpleNodeCard result={result} label={label} colors={colors} />;
  }

  // Nodes WITH explorers — expandable card
  return (
    <div className={`rounded-xl border ${colors.border} overflow-hidden transition-shadow ${expanded ? "shadow-lg" : "shadow-sm hover:shadow-md"}`}>
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={`w-full flex items-center justify-between px-4 py-3 ${colors.bg} transition-colors`}
      >
        <div className="flex items-center gap-3">
          <div className={`w-2.5 h-2.5 rounded-full ${result.success !== false ? "bg-green-500" : "bg-red-500"}`} />
          <span className={`text-sm font-semibold ${colors.text}`}>{label}</span>
          {result.execution_time_ms != null && (
            <span className="flex items-center gap-1 text-xs text-gray-400">
              <Clock className="h-3 w-3" />
              {result.execution_time_ms < 1000 ? `${Math.round(result.execution_time_ms)}ms` : `${(result.execution_time_ms / 1000).toFixed(1)}s`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {activityCount > 0 && (
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${colors.badge}`}>
              {activityCount} {activityCount === 1 ? "activity" : "activities"}
            </span>
          )}
          {expanded ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
        </div>
      </button>

      {/* Explorer content */}
      {expanded && (
        <div className="border-t bg-white p-4">
          <ExplorerContent result={result} primaryColor={primaryColor} />
        </div>
      )}
    </div>
  );
}

// ─── Explorer Content Router ─────────────────────────────────────

function ExplorerContent({ result, primaryColor }: { result: NodeResult; primaryColor: string }) {
  const nodeType = result.node_type;

  if (nodeType === "linear_regression") {
    return <LinearRegressionExplorer result={result} />;
  }

  if (nodeType === "logistic_regression") {
    return (
      <LogisticRegressionExplorer
        result={result}
        renderResults={() => <MLModelSummary result={result} modelType="Logistic Regression" />}
      />
    );
  }

  if (nodeType === "missing_value_handler") {
    return (
      <MissingValueExplorer
        result={result}
        renderResults={() => <MissingValueSummary result={result} />}
      />
    );
  }

  if (nodeType === "split") {
    return (
      <SplitExplorer
        result={result}
        renderResults={() => <SplitSummary result={result} />}
      />
    );
  }

  if (nodeType === "chart_view") {
    return <DataExplorer result={result} />;
  }

  if (nodeType === "encoding") {
    return (
      <EncodingExplorer
        result={result}
        renderResults={() => <GenericFieldList result={result} />}
      />
    );
  }

  if (nodeType === "scaling") {
    return (
      <ScalingExplorer
        result={result}
        renderResults={() => <GenericFieldList result={result} />}
      />
    );
  }

  if (nodeType === "feature_selection") {
    return (
      <FeatureSelectionExplorer
        result={result}
        renderResults={() => <GenericFieldList result={result} />}
      />
    );
  }

  if (nodeType === "table_view" || nodeType === "data_preview") {
    return <TableContent result={result} />;
  }

  if (nodeType === "statistics_view") {
    return <StatisticsContent result={result} />;
  }

  if (nodeType === "column_info") {
    return <ColumnInfoContent result={result} />;
  }

  // Metric nodes with explorers
  if (nodeType === "r2_score") return <R2ScoreResult result={result} />;
  if (nodeType === "mse_score") return <MSEScoreResult result={result} />;
  if (nodeType === "rmse_score") return <RMSEScoreResult result={result} />;
  if (nodeType === "mae_score") return <MAEScoreResult result={result} />;

  return <GenericFieldList result={result} />;
}

// ─── Inline Metric Card ──────────────────────────────────────────

function MetricInlineCard({
  result,
  primaryColor,
  label,
  colors,
}: {
  result: NodeResult;
  primaryColor: string;
  label: string;
  colors: typeof DEFAULT_COLOR;
}) {
  const metricKeys = ["r2_score", "mse_score", "rmse_score", "mae_score", "score", "value"];
  let value: number | null = null;
  for (const key of metricKeys) {
    if (result[key] != null && typeof result[key] === "number") {
      value = result[key] as number;
      break;
    }
  }
  const isPercentage = result.node_type === "r2_score";

  return (
    <div
      className={`rounded-xl p-4 border ${colors.border}`}
      style={{ backgroundColor: `${primaryColor}08` }}
    >
      <p className="text-xs font-semibold mb-1" style={{ color: `${primaryColor}99` }}>
        {label}
      </p>
      <p className="text-3xl font-bold" style={{ color: primaryColor }}>
        {value != null
          ? isPercentage
            ? `${(value * 100).toFixed(2)}%`
            : formatNumber(value)
          : "—"}
      </p>
      {result.execution_time_ms != null && (
        <p className="text-xs text-gray-400 mt-1 flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {result.execution_time_ms < 1000 ? `${Math.round(result.execution_time_ms)}ms` : `${(result.execution_time_ms / 1000).toFixed(1)}s`}
        </p>
      )}
    </div>
  );
}

// ─── Dataset Card ────────────────────────────────────────────────

function DatasetCard({ result, label, colors }: { result: NodeResult; label: string; colors: typeof DEFAULT_COLOR }) {
  const filename = result.filename as string | undefined;
  const nRows = result.n_rows as number | undefined;
  const nCols = result.n_columns as number | undefined;
  const columns = result.columns as string[] | undefined;

  return (
    <div className={`rounded-xl border ${colors.border} ${colors.bg} p-4`}>
      <div className="flex items-center gap-2 mb-3">
        <Database className={`h-4 w-4 ${colors.text}`} />
        <h4 className={`text-sm font-semibold ${colors.text}`}>{label}</h4>
      </div>
      {filename && (
        <div className="flex items-center gap-2 mb-2">
          <FileSpreadsheet className="h-3.5 w-3.5 text-blue-500" />
          <span className="text-sm text-blue-800 font-medium">{filename}</span>
        </div>
      )}
      <div className="flex gap-4 mt-2">
        {nRows != null && (
          <div className="bg-white/70 rounded-md px-3 py-1.5">
            <p className="text-xs text-blue-500">Rows</p>
            <p className="text-lg font-bold text-blue-900">{nRows.toLocaleString()}</p>
          </div>
        )}
        {nCols != null && (
          <div className="bg-white/70 rounded-md px-3 py-1.5">
            <p className="text-xs text-blue-500">Columns</p>
            <p className="text-lg font-bold text-blue-900">{nCols}</p>
          </div>
        )}
      </div>
      {columns && columns.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {columns.map((col) => (
            <span key={col} className="px-2 py-0.5 bg-white/80 text-xs text-blue-700 rounded-md border border-blue-200">
              {col}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Simple Node Card (no explorer) ──────────────────────────────

function SimpleNodeCard({ result, label, colors }: { result: NodeResult; label: string; colors: typeof DEFAULT_COLOR }) {
  const [expanded, setExpanded] = useState(false);

  const HIDDEN = new Set(["node_type", "execution_time_ms", "timestamp", "success", "error", "dataset_id", "file_path", "storage_backend", "memory_usage_mb", "node_id", "view_type"]);
  const visibleFields = Object.entries(result).filter(([key]) => !HIDDEN.has(key) && result[key] != null);

  if (visibleFields.length === 0) return null;

  return (
    <div className={`rounded-xl border ${colors.border} overflow-hidden`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className={`w-full flex items-center justify-between px-4 py-3 ${colors.bg} transition-colors`}
      >
        <div className="flex items-center gap-2">
          <CheckCircle className={`h-4 w-4 ${result.success !== false ? "text-green-500" : "text-red-500"}`} />
          <span className={`text-sm font-semibold ${colors.text}`}>{label}</span>
        </div>
        {expanded ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
      </button>
      {expanded && (
        <div className="p-4 bg-white space-y-2">
          {visibleFields.map(([key, value]) => (
            <div key={key}>
              <p className="text-xs text-gray-500 mb-0.5">{formatNodeType(key)}</p>
              {typeof value === "object" ? (
                <pre className="bg-gray-50 rounded-md p-2 text-xs text-gray-700 overflow-x-auto max-h-40">
                  {JSON.stringify(value, null, 2)}
                </pre>
              ) : (
                <p className="text-sm text-gray-800">
                  {typeof value === "number" ? formatNumber(value) : String(value)}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Inline Result Renderers (passed to explorers as renderResults) ──

function MLModelSummary({ result, modelType }: { result: NodeResult; modelType: string }) {
  const metrics = (result.training_metrics || {}) as Record<string, unknown>;
  return (
    <div className="space-y-4">
      <div className="bg-gradient-to-r from-violet-50 to-purple-50 border border-violet-200 rounded-lg p-4">
        <h3 className="text-base font-semibold text-violet-900">{modelType} — Training Complete</h3>
        <p className="text-sm text-violet-700">
          Trained on {(result.training_samples as number)?.toLocaleString()} samples with {result.n_features as number} features
        </p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {Object.entries(metrics)
          .filter(([, v]) => typeof v === "number")
          .map(([key, value]) => (
            <div key={key} className="bg-gray-50 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">{key.replace(/_/g, " ")}</p>
              <p className="text-xl font-bold text-gray-900">{(value as number).toFixed(4)}</p>
            </div>
          ))}
      </div>
    </div>
  );
}

function MissingValueSummary({ result }: { result: NodeResult }) {
  const beforeStats = (result.before_stats || {}) as Record<string, unknown>;
  const afterStats = (result.after_stats || {}) as Record<string, unknown>;
  return (
    <div className="space-y-4">
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <h3 className="text-base font-semibold text-amber-900">Missing Value Treatment</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Rows</p>
            <p className="text-lg font-bold text-gray-900">{(beforeStats.total_rows as number) ?? "—"}</p>
          </div>
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Missing Before</p>
            <p className="text-lg font-bold text-red-600">{(beforeStats.total_missing_values as number) ?? "—"}</p>
          </div>
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Missing After</p>
            <p className="text-lg font-bold text-green-600">{(afterStats.total_missing_values as number) ?? "—"}</p>
          </div>
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Cols Affected</p>
            <p className="text-lg font-bold text-gray-900">{(beforeStats.columns_with_missing as number) ?? "—"}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function SplitSummary({ result }: { result: NodeResult }) {
  return (
    <div className="space-y-4">
      <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-4">
        <h3 className="text-base font-semibold text-cyan-900">Train/Test Split Complete</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Train Size</p>
            <p className="text-lg font-bold text-gray-900">{(result.train_size as number)?.toLocaleString() ?? "—"}</p>
          </div>
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Test Size</p>
            <p className="text-lg font-bold text-gray-900">{(result.test_size as number)?.toLocaleString() ?? "—"}</p>
          </div>
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Target</p>
            <p className="text-lg font-bold text-gray-900">{(result.target_column as string) ?? "—"}</p>
          </div>
          <div className="bg-white/70 rounded-md px-3 py-2">
            <p className="text-xs text-gray-500">Features</p>
            <p className="text-lg font-bold text-gray-900">{(result.feature_columns as string[])?.length ?? "—"}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Table Content ───────────────────────────────────────────────

function TableContent({ result }: { result: NodeResult }) {
  const data = result.data as Record<string, unknown>[] | undefined;
  const columns = result.columns as string[] | undefined;
  const [showAll, setShowAll] = useState(false);

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <Table className="h-6 w-6 mx-auto mb-2" />
        <p className="text-sm">No data to display</p>
      </div>
    );
  }

  const cols = columns ?? Object.keys(data[0]);
  const visibleRows = showAll ? data : data.slice(0, 10);

  return (
    <div>
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50">
              <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 border-b">#</th>
              {cols.map((col) => (
                <th key={col} className="text-left py-2 px-3 text-xs font-medium text-gray-500 border-b whitespace-nowrap">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, idx) => (
              <tr key={idx} className={`border-b last:border-0 ${idx % 2 === 0 ? "bg-white" : "bg-gray-50/30"} hover:bg-indigo-50/30 transition-colors`}>
                <td className="py-1.5 px-3 text-xs text-gray-400">{idx + 1}</td>
                {cols.map((col) => (
                  <td key={col} className="py-1.5 px-3 text-gray-700 whitespace-nowrap max-w-[200px] truncate">
                    {formatCellValue(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length > 10 && (
        <button onClick={() => setShowAll(!showAll)} className="w-full py-2 text-xs text-indigo-600 hover:bg-indigo-50 transition-colors mt-1 rounded-b-lg flex items-center justify-center gap-1">
          {showAll ? "Show Less" : `Show All ${data.length} Rows`}
          <ChevronDown className={`h-3 w-3 transition-transform ${showAll ? "rotate-180" : ""}`} />
        </button>
      )}
    </div>
  );
}

// ─── Statistics Content ──────────────────────────────────────────

function StatisticsContent({ result }: { result: NodeResult }) {
  const stats = result.statistics as Record<string, Record<string, number>> | undefined;
  if (!stats) return <GenericFieldList result={result} />;

  const statCols = Object.keys(stats);
  const statRows = statCols.length > 0 ? Object.keys(stats[statCols[0]]) : [];

  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50">
            <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 border-b" />
            {statCols.map((col) => (
              <th key={col} className="text-right py-2 px-3 text-xs font-medium text-gray-500 border-b whitespace-nowrap">{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {statRows.map((stat, idx) => (
            <tr key={stat} className={`border-b last:border-0 ${idx % 2 === 0 ? "bg-white" : "bg-gray-50/30"}`}>
              <td className="py-1.5 px-3 text-xs font-medium text-gray-600">{stat}</td>
              {statCols.map((col) => (
                <td key={col} className="py-1.5 px-3 text-right text-gray-700 tabular-nums">{formatNumber(stats[col][stat])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Column Info Content ─────────────────────────────────────────

function ColumnInfoContent({ result }: { result: NodeResult }) {
  const columnInfo = result.column_info as Array<{ column: string; dtype: string; missing?: number; unique?: number }> | undefined;
  if (!columnInfo || columnInfo.length === 0) return <GenericFieldList result={result} />;

  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50">
            <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 border-b">#</th>
            <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 border-b">Column</th>
            <th className="text-left py-2 px-3 text-xs font-medium text-gray-500 border-b">Type</th>
            <th className="text-center py-2 px-3 text-xs font-medium text-gray-500 border-b">Missing</th>
            <th className="text-center py-2 px-3 text-xs font-medium text-gray-500 border-b">Unique</th>
          </tr>
        </thead>
        <tbody>
          {columnInfo.map((col, idx) => (
            <tr key={idx} className={`border-b last:border-0 ${idx % 2 === 0 ? "bg-white" : "bg-gray-50/30"}`}>
              <td className="py-1.5 px-3 text-xs text-gray-400">{idx + 1}</td>
              <td className="py-1.5 px-3 font-medium text-gray-900">{col.column}</td>
              <td className="py-1.5 px-3">
                <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs font-mono">{col.dtype}</span>
              </td>
              <td className="py-1.5 px-3 text-center">
                {col.missing != null ? (
                  col.missing > 0 ? (
                    <span className="px-2 py-0.5 bg-red-50 text-red-700 rounded text-xs font-semibold">{col.missing}</span>
                  ) : (
                    <span className="px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded text-xs">0</span>
                  )
                ) : "—"}
              </td>
              <td className="py-1.5 px-3 text-center text-gray-700">{col.unique ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Generic Field List ──────────────────────────────────────────

function GenericFieldList({ result }: { result: NodeResult }) {
  const HIDDEN = new Set(["node_type", "execution_time_ms", "timestamp", "success", "error", "dataset_id", "file_path", "storage_backend", "memory_usage_mb", "node_id", "view_type"]);
  const fields = Object.entries(result).filter(([key]) => !HIDDEN.has(key) && result[key] != null);

  if (fields.length === 0) return <p className="text-sm text-gray-400 text-center py-4">No data</p>;

  const simpleFields = fields.filter(([, v]) => typeof v !== "object" || v === null);
  const complexFields = fields.filter(([, v]) => typeof v === "object" && v !== null);

  return (
    <div className="space-y-3">
      {simpleFields.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {simpleFields.map(([key, value]) => (
            <div key={key} className="bg-gray-50 rounded-md px-3 py-2">
              <p className="text-xs text-gray-500">{formatNodeType(key)}</p>
              <p className="text-sm font-medium text-gray-800">
                {typeof value === "number" ? formatNumber(value) : String(value)}
              </p>
            </div>
          ))}
        </div>
      )}
      {complexFields.map(([key, value]) => (
        <div key={key}>
          <p className="text-xs text-gray-500 mb-1">{formatNodeType(key)}</p>
          <pre className="bg-gray-50 rounded-md p-3 text-xs text-gray-700 overflow-x-auto max-h-48">
            {JSON.stringify(value, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  );
}

// ─── Helpers ─────────────────────────────────────────────────────

function formatNodeType(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatCellValue(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(4);
  }
  return String(value);
}

function formatNumber(value: number): string {
  if (Number.isInteger(value)) return value.toLocaleString();
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(3);
  return value.toFixed(4);
}
