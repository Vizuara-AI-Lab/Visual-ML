/**
 * View Node Modal - Display data from view nodes
 */

import { motion, AnimatePresence } from "framer-motion";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { getNodeByType } from "../../config/nodeDefinitions";
import { ChartViewer } from "./ChartViewer";
import { DataExplorer } from "./DataExplorer";
import { MissingValueExplorer } from "./MissingValueExplorer";
import { EncodingExplorer } from "./EncodingExplorer";
import { ScalingExplorer } from "./ScalingExplorer";
import { FeatureSelectionExplorer } from "./FeatureSelectionExplorer";
import { SplitExplorer } from "./SplitExplorer";
import { LinearRegressionExplorer } from "./LinearRegressionExplorer";
import { LogisticRegressionExplorer } from "./LogisticRegressionExplorer";
import { TransformationResults } from "./TransformationResults";
import { ScalingResults } from "./ScalingResults";
import { FeatureEngineeringResults } from "./FeatureEngineeringResults";
import { lazy, Suspense } from "react";
import {
  X,
  Table2,
  Eye,
  BarChart3,
  Columns3,
} from "lucide-react";

const DecisionTreeAnimation = lazy(() => import("./animations/DecisionTreeAnimation"));
const RandomForestAnimation = lazy(() => import("./animations/RandomForestAnimation"));
import {
  R2ScoreResult,
  MSEScoreResult,
  RMSEScoreResult,
  MAEScoreResult,
  ConfusionMatrixResult,
} from "./results_and_metrics";

interface ViewNodeModalProps {
  nodeId: string | null;
  onClose: () => void;
}

export const ViewNodeModal = ({ nodeId, onClose }: ViewNodeModalProps) => {
  const { nodes, executionResult } = usePlaygroundStore();

  const node = nodeId ? nodes.find((n) => n.id === nodeId) : null;
  const nodeDef = node ? getNodeByType(node.data.type) : null;
  const rawNodeResult = nodeId ? executionResult?.nodeResults?.[nodeId] : null;
  // Unwrap: nodeResults stores { success, output, error } — render functions expect the output directly
  const nodeResult = rawNodeResult
    ? {
        ...(rawNodeResult.output as Record<string, unknown> || {}),
        error: rawNodeResult.error,
        success: rawNodeResult.success,
      }
    : null;

  // Early return AFTER all hooks have been called
  if (!nodeId) return null;

  if (!node) {
    return (
      <AnimatePresence>
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="relative bg-white rounded-xl shadow-2xl p-8 border border-slate-200 z-10"
          >
            <p className="text-slate-700 font-medium">Node not found</p>
            <button
              onClick={onClose}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
            >
              Close
            </button>
          </motion.div>
        </div>
      </AnimatePresence>
    );
  }

  if (!nodeResult) {
    return (
      <AnimatePresence>
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="relative bg-white rounded-xl shadow-2xl p-8 border border-slate-200 z-10"
          >
            <p className="text-slate-700 font-medium">
              No execution results available for this node
            </p>
            <button
              onClick={onClose}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
            >
              Close
            </button>
          </motion.div>
        </div>
      </AnimatePresence>
    );
  }

  const nodeType = node.data.type;
  const isViewNode = [
    "table_view",
    "data_preview",
    "statistics_view",
    "column_info",
    "chart_view",
    "missing_value_handler",
    "encoding",
    "transformation",
    "scaling",
    "feature_selection",
    // Result/Metrics Nodes
    "r2_score",
    "mse_score",
    "rmse_score",
    "mae_score",
    "confusion_matrix",
    // ML Algorithm & Split Nodes
    "split",
    "linear_regression",
    "logistic_regression",
    "decision_tree",
    "random_forest",
  ].includes(nodeType);

  if (!isViewNode) return null;

  const NodeIcon = nodeDef?.icon || Eye;
  const nodeColor = nodeDef?.color || "#3B82F6";

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/40 backdrop-blur-sm"
          onClick={onClose}
        />

        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative bg-white border border-slate-200 rounded-xl shadow-2xl w-11/12 h-5/6 flex flex-col overflow-hidden z-10"
        >
          {/* Header */}
          <div
            className="px-6 py-4 border-b border-slate-200 flex items-center justify-between"
            style={{ borderTopColor: nodeColor, borderTopWidth: "3px" }}
          >
            <div className="flex items-center gap-3">
              <div
                className="p-2 rounded-lg"
                style={{ backgroundColor: `${nodeColor}20` }}
              >
                <NodeIcon
                  className="w-5 h-5"
                  style={{ color: nodeColor }}
                />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  {node.data.label}
                </h2>
                <p className="text-sm text-slate-500">
                  {nodeDef?.description || "Data View"}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-500" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto p-6">
            {nodeResult.error ? (
              <div className="p-5 bg-red-50 border border-red-200 rounded-xl">
                <p className="text-red-800 font-semibold text-sm">Error</p>
                <p className="text-red-600 text-sm mt-2 leading-relaxed">
                  {typeof nodeResult.error === "string"
                    ? nodeResult.error
                    : nodeResult.error?.message || "Unknown error"}
                </p>
              </div>
            ) : (
              <div>
                {nodeType === "table_view" && renderTableView(nodeResult)}
                {nodeType === "data_preview" && renderDataPreview(nodeResult)}
                {nodeType === "statistics_view" &&
                  renderStatisticsView(nodeResult)}
                {nodeType === "column_info" && renderColumnInfo(nodeResult)}
                {nodeType === "chart_view" && renderChartView(nodeResult)}
                {nodeType === "missing_value_handler" &&
                  renderMissingValueHandler(nodeResult)}
                {nodeType === "encoding" && renderEncoding(nodeResult)}
                {nodeType === "transformation" && (
                  <TransformationResults result={nodeResult} />
                )}
                {nodeType === "scaling" && renderScaling(nodeResult)}
                {nodeType === "feature_selection" && renderFeatureSelection(nodeResult)}
                {nodeType === "r2_score" && <R2ScoreResult result={nodeResult} />}
                {nodeType === "mse_score" && (
                  <MSEScoreResult result={nodeResult} />
                )}
                {nodeType === "rmse_score" && (
                  <RMSEScoreResult result={nodeResult} />
                )}
                {nodeType === "mae_score" && (
                  <MAEScoreResult result={nodeResult} />
                )}
                {nodeType === "confusion_matrix" && (
                  <ConfusionMatrixResult result={nodeResult} />
                )}
                {nodeType === "split" && renderSplit(nodeResult)}
                {nodeType === "linear_regression" && renderLinearRegression(nodeResult)}
                {nodeType === "logistic_regression" && renderLogisticRegression(nodeResult)}
                {nodeType === "decision_tree" && (
                  <Suspense fallback={<div className="flex items-center justify-center py-12"><div className="w-6 h-6 border-2 border-green-500 border-t-transparent rounded-full animate-spin" /></div>}>
                    <DecisionTreeAnimation />
                  </Suspense>
                )}
                {nodeType === "random_forest" && (
                  <Suspense fallback={<div className="flex items-center justify-center py-12"><div className="w-6 h-6 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" /></div>}>
                    <RandomForestAnimation />
                  </Suspense>
                )}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-slate-200 flex items-center justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-slate-500 hover:text-slate-800 transition-colors"
            >
              Close
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

// Render functions for different node types
function renderTableView(result: any) {
  const data = result.data || [];
  const columns = result.columns || [];

  return (
    <div className="space-y-4">
      {/* Summary badge */}
      <div className="flex items-center gap-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-slate-100 rounded-full border border-slate-200/60">
          <Table2 className="w-3.5 h-3.5 text-slate-600" />
          <span className="text-xs font-semibold text-slate-700">
            {data.length} rows  ×  {columns.length} columns
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto rounded-xl border border-slate-200/60 shadow-sm">
        <table className="min-w-full">
          <thead>
            <tr className="bg-slate-50/80">
              <th className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">#</th>
              {columns.map((col: string) => (
                <th
                  key={col}
                  className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.map((row: any, idx: number) => (
              <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                <td className="px-4 py-2.5 text-xs text-slate-400 font-mono">{idx + 1}</td>
                {columns.map((col: string) => (
                  <td key={col} className="px-4 py-2.5 text-sm text-slate-700">
                    {row[col] === null ? (
                      <span className="px-2 py-0.5 bg-slate-100 text-slate-400 rounded text-xs italic">null</span>
                    ) : (
                      String(row[col])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function renderDataPreview(result: any) {
  const headData = result.head_data || [];
  const tailData = result.tail_data || [];
  const totalRows = result.total_rows || 0;

  const columns =
    headData.length > 0
      ? Object.keys(headData[0])
      : tailData.length > 0
        ? Object.keys(tailData[0])
        : [];

  const renderPreviewTable = (rows: any[], label: string) => (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-slate-800">{label}</span>
        <span className="text-xs text-slate-400 font-medium">{rows.length} rows</span>
      </div>
      <div className="overflow-auto rounded-xl border border-slate-200/60 shadow-sm">
        <table className="min-w-full">
          <thead>
            <tr className="bg-slate-50/80">
              <th className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">#</th>
              {columns.map((col: string) => (
                <th
                  key={col}
                  className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((row: any, idx: number) => (
              <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                <td className="px-4 py-2.5 text-xs text-slate-400 font-mono">{idx + 1}</td>
                {columns.map((col: string) => (
                  <td key={col} className="px-4 py-2.5 text-sm text-slate-700">
                    {row[col] === null || row[col] === undefined ? (
                      <span className="px-2 py-0.5 bg-slate-100 text-slate-400 rounded text-xs italic">null</span>
                    ) : (
                      String(row[col])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-50/80 rounded-xl border border-slate-200/60 p-4">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Total Rows</div>
          <div className="text-2xl font-bold text-slate-900">{totalRows.toLocaleString()}</div>
        </div>
        <div className="bg-slate-50/80 rounded-xl border border-slate-200/60 p-4">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Columns</div>
          <div className="text-2xl font-bold text-slate-900">{columns.length}</div>
        </div>
        <div className="bg-slate-50/80 rounded-xl border border-slate-200/60 p-4">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Preview</div>
          <div className="text-2xl font-bold text-slate-900">{headData.length + tailData.length} rows</div>
        </div>
      </div>

      {headData.length > 0 && renderPreviewTable(headData, "First Rows")}
      {tailData.length > 0 && renderPreviewTable(tailData, "Last Rows")}

      {headData.length === 0 && tailData.length === 0 && (
        <div className="text-center py-12 text-slate-400 font-medium">
          No preview data available
        </div>
      )}
    </div>
  );
}

function renderStatisticsView(result: any) {
  const stats = result.statistics || {};
  const columnNames = Object.keys(stats);

  if (columnNames.length === 0) {
    return (
      <div className="text-center py-12 text-slate-400 font-medium">
        No statistics available
      </div>
    );
  }

  const formatStat = (val: any) =>
    typeof val === "number" ? val.toFixed(2) : val;

  return (
    <div className="space-y-6">
      {/* Summary badge */}
      <div className="flex items-center gap-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-emerald-50 rounded-full border border-emerald-200/60">
          <BarChart3 className="w-3.5 h-3.5 text-emerald-600" />
          <span className="text-xs font-semibold text-emerald-700">
            {columnNames.length} numeric column{columnNames.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      {/* Stats table - compact and scannable */}
      <div className="overflow-auto rounded-xl border border-slate-200/60 shadow-sm">
        <table className="min-w-full">
          <thead>
            <tr className="bg-slate-50/80">
              <th className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60 sticky left-0 bg-slate-50/80 z-10">Column</th>
              <th className="px-4 py-3 text-right text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Mean</th>
              <th className="px-4 py-3 text-right text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Std Dev</th>
              <th className="px-4 py-3 text-right text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Min</th>
              <th className="px-4 py-3 text-right text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">25%</th>
              <th className="px-4 py-3 text-right text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Median</th>
              <th className="px-4 py-3 text-right text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">75%</th>
              <th className="px-4 py-3 text-right text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Max</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {columnNames.map((colName) => {
              const s = stats[colName];
              return (
                <tr key={colName} className="hover:bg-slate-50/60 transition-colors">
                  <td className="px-4 py-3 text-sm font-semibold text-slate-900 sticky left-0 bg-white z-10">{colName}</td>
                  <td className="px-4 py-3 text-sm text-slate-700 text-right font-mono">{formatStat(s.mean)}</td>
                  <td className="px-4 py-3 text-sm text-slate-700 text-right font-mono">{formatStat(s.std)}</td>
                  <td className="px-4 py-3 text-sm text-slate-700 text-right font-mono">{formatStat(s.min)}</td>
                  <td className="px-4 py-3 text-sm text-slate-500 text-right font-mono">{formatStat(s["25%"])}</td>
                  <td className="px-4 py-3 text-sm text-slate-700 text-right font-mono font-semibold">{formatStat(s.median ?? s["50%"])}</td>
                  <td className="px-4 py-3 text-sm text-slate-500 text-right font-mono">{formatStat(s["75%"])}</td>
                  <td className="px-4 py-3 text-sm text-slate-700 text-right font-mono">{formatStat(s.max)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function renderColumnInfo(result: any) {
  const columnInfo = result.column_info || [];

  if (columnInfo.length === 0) {
    return (
      <div className="text-center py-12 text-slate-400 font-medium">
        No column information available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary badge */}
      <div className="flex items-center gap-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-sky-50 rounded-full border border-sky-200/60">
          <Columns3 className="w-3.5 h-3.5 text-sky-600" />
          <span className="text-xs font-semibold text-sky-700">
            {columnInfo.length} column{columnInfo.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto rounded-xl border border-slate-200/60 shadow-sm">
        <table className="min-w-full">
          <thead>
            <tr className="bg-slate-50/80">
              <th className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">#</th>
              <th className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Column Name</th>
              <th className="px-4 py-3 text-left text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Data Type</th>
              <th className="px-4 py-3 text-center text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Missing</th>
              <th className="px-4 py-3 text-center text-[11px] font-bold text-slate-500 uppercase tracking-wider border-b border-slate-200/60">Unique</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {columnInfo.map((col: any, idx: number) => (
              <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                <td className="px-4 py-3 text-xs text-slate-400 font-mono">{idx + 1}</td>
                <td className="px-4 py-3 text-sm font-semibold text-slate-900">{col.column}</td>
                <td className="px-4 py-3">
                  <span className="inline-flex px-2.5 py-1 bg-slate-100 text-slate-600 rounded-lg text-xs font-mono font-medium border border-slate-200/60">
                    {col.dtype}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  {col.missing !== undefined ? (
                    col.missing > 0 ? (
                      <span className="inline-flex px-2.5 py-1 bg-red-50 text-red-700 rounded-lg text-xs font-semibold border border-red-200/60">
                        {col.missing}
                      </span>
                    ) : (
                      <span className="inline-flex px-2.5 py-1 bg-emerald-50 text-emerald-700 rounded-lg text-xs font-semibold border border-emerald-200/60">
                        0
                      </span>
                    )
                  ) : (
                    <span className="text-slate-300">-</span>
                  )}
                </td>
                <td className="px-4 py-3 text-center text-sm text-slate-700 font-medium">
                  {col.unique !== undefined ? col.unique : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function renderChartView(result: any) {
  if (result?.exploration_data) {
    return <DataExplorer result={result} />;
  }
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white p-2 shadow-sm">
      <ChartViewer result={result} />
    </div>
  );
}

function renderMissingValueHandler(result: any) {
  const hasActivityData =
    result.strategy_comparison || result.quiz_questions || result.missing_heatmap;

  if (hasActivityData) {
    return (
      <MissingValueExplorer
        result={result}
        renderResults={() => <MissingValueHandlerResults result={result} />}
      />
    );
  }

  return <MissingValueHandlerResults result={result} />;
}

function renderEncoding(result: any) {
  const hasActivityData =
    result.encoding_before_after || result.encoding_map || result.encoding_method_comparison || result.quiz_questions;

  if (hasActivityData) {
    return (
      <EncodingExplorer
        result={result}
        renderResults={() => <FeatureEngineeringResults result={result} nodeTypeName="Encoding" />}
      />
    );
  }

  return <FeatureEngineeringResults result={result} nodeTypeName="Encoding" />;
}

function renderScaling(result: any) {
  const hasActivityData =
    result.scaling_before_after || result.scaling_method_comparison || result.scaling_outlier_analysis || result.quiz_questions;

  if (hasActivityData) {
    return (
      <ScalingExplorer
        result={result}
        renderResults={() => <ScalingResults result={result} />}
      />
    );
  }

  return <ScalingResults result={result} />;
}

function renderFeatureSelection(result: any) {
  const hasActivityData =
    result.feature_score_details || result.feature_correlation_matrix || result.threshold_simulation_data || result.quiz_questions;

  if (hasActivityData) {
    return (
      <FeatureSelectionExplorer
        result={result}
        renderResults={() => <FeatureEngineeringResults result={result} nodeTypeName="Feature Selection" />}
      />
    );
  }

  return <FeatureEngineeringResults result={result} nodeTypeName="Feature Selection" />;
}

function renderSplit(result: any) {
  const hasActivityData =
    result.split_visualization || result.class_balance || result.ratio_explorer || result.quiz_questions;

  if (hasActivityData) {
    return (
      <SplitExplorer
        result={result}
        renderResults={() => <SplitResults result={result} />}
      />
    );
  }

  return <SplitResults result={result} />;
}

function renderLinearRegression(result: any) {
  return <LinearRegressionExplorer result={result} />;
}

function renderLogisticRegression(result: any) {
  const hasActivityData =
    result.class_distribution || result.metric_explainer || result.sigmoid_data || result.quiz_questions;

  if (hasActivityData) {
    return (
      <LogisticRegressionExplorer
        result={result}
        renderResults={() => <MLModelResults result={result} modelType="Logistic Regression" />}
      />
    );
  }

  return <MLModelResults result={result} modelType="Logistic Regression" />;
}

function SplitResults({ result }: { result: any }) {
  const summary = result.split_summary || {};

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-cyan-50 to-sky-50 border border-cyan-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cyan-500 rounded-full flex items-center justify-center">
            <span className="text-white text-xl">✓</span>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-cyan-900">Split Complete</h3>
            <p className="text-sm text-cyan-700">
              Dataset split into training and test sets
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Train Size</div>
          <div className="text-lg font-semibold text-gray-900">{result.train_size?.toLocaleString()}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Test Size</div>
          <div className="text-lg font-semibold text-gray-900">{result.test_size?.toLocaleString()}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Target Column</div>
          <div className="text-lg font-semibold text-gray-900">{result.target_column || "N/A"}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Features</div>
          <div className="text-lg font-semibold text-gray-900">{result.feature_columns?.length || 0}</div>
        </div>
      </div>

      {/* Split Summary */}
      {summary && Object.keys(summary).length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Split Configuration</h4>
          <div className="grid grid-cols-2 gap-3 text-sm">
            {summary.train_ratio !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-600">Train Ratio:</span>
                <span className="font-semibold">{(summary.train_ratio * 100).toFixed(0)}%</span>
              </div>
            )}
            {summary.test_ratio !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-600">Test Ratio:</span>
                <span className="font-semibold">{(summary.test_ratio * 100).toFixed(0)}%</span>
              </div>
            )}
            {summary.split_type && (
              <div className="flex justify-between">
                <span className="text-gray-600">Split Type:</span>
                <span className="font-semibold capitalize">{summary.split_type}</span>
              </div>
            )}
            {summary.random_seed !== undefined && (
              <div className="flex justify-between">
                <span className="text-gray-600">Random Seed:</span>
                <span className="font-semibold">{summary.random_seed}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Dataset IDs */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-blue-900 mb-2">Output Datasets</h4>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-700">Train Dataset:</span>
            <code className="text-xs font-mono bg-white px-2 py-1 rounded border">{result.train_dataset_id || "N/A"}</code>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-700">Test Dataset:</span>
            <code className="text-xs font-mono bg-white px-2 py-1 rounded border">{result.test_dataset_id || "N/A"}</code>
          </div>
        </div>
      </div>
    </div>
  );
}

function MLModelResults({ result, modelType }: { result: any; modelType: string }) {
  const metrics = result.training_metrics || {};
  const isClassification = modelType === "Logistic Regression";

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className={`bg-gradient-to-r ${isClassification ? "from-violet-50 to-purple-50 border-violet-200" : "from-emerald-50 to-green-50 border-emerald-200"} border rounded-lg p-4`}>
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 ${isClassification ? "bg-violet-500" : "bg-emerald-500"} rounded-full flex items-center justify-center`}>
            <span className="text-white text-xl">✓</span>
          </div>
          <div>
            <h3 className={`text-lg font-semibold ${isClassification ? "text-violet-900" : "text-emerald-900"}`}>
              {modelType} Training Complete
            </h3>
            <p className={`text-sm ${isClassification ? "text-violet-700" : "text-emerald-700"}`}>
              Model trained on {result.training_samples?.toLocaleString()} samples with {result.n_features} features
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-4">Training Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(metrics)
            .filter(([, value]) => typeof value === "number")
            .map(([key, value]: [string, number]) => (
            <div key={key} className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                {key.replace(/_/g, " ")}
              </div>
              <div className="text-xl font-bold text-gray-900">
                {value.toFixed(4)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Per-Class Metrics (classification only) */}
      {isClassification && metrics.per_class_metrics && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-4">Per-Class Metrics</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">Class</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">Precision</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">Recall</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">F1 Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {Object.entries(metrics.per_class_metrics).map(([cls, m]: [string, Record<string, number>]) => (
                  <tr key={cls} className="hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-900">{cls}</td>
                    <td className="px-4 py-2 text-gray-700">{typeof m.precision === "number" ? m.precision.toFixed(4) : "N/A"}</td>
                    <td className="px-4 py-2 text-gray-700">{typeof m.recall === "number" ? m.recall.toFixed(4) : "N/A"}</td>
                    <td className="px-4 py-2 text-gray-700">{typeof m.f1 === "number" ? m.f1.toFixed(4) : "N/A"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Confusion Matrix (classification only) */}
      {isClassification && metrics.confusion_matrix && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-4">Confusion Matrix</h4>
          <div className="overflow-x-auto">
            <table className="text-sm border-collapse">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-xs text-gray-500"></th>
                  {(result.class_names || []).map((cls: string, i: number) => (
                    <th key={i} className="px-3 py-2 text-xs font-semibold text-gray-700 text-center">
                      {cls}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.confusion_matrix.map((row: number[], i: number) => (
                  <tr key={i}>
                    <td className="px-3 py-2 text-xs font-semibold text-gray-700">
                      {result.class_names?.[i] ?? `Class ${i}`}
                    </td>
                    {row.map((val: number, j: number) => (
                      <td
                        key={j}
                        className={`px-3 py-2 text-center font-mono text-sm border border-gray-200 ${
                          i === j ? "bg-violet-100 font-bold text-violet-900" : "bg-gray-50 text-gray-700"
                        }`}
                      >
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Info */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Training Samples</div>
          <div className="text-lg font-semibold text-gray-900">{result.training_samples?.toLocaleString()}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Features</div>
          <div className="text-lg font-semibold text-gray-900">{result.n_features}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Training Time</div>
          <div className="text-lg font-semibold text-gray-900">
            {result.training_time_seconds !== undefined
              ? `${result.training_time_seconds.toFixed(2)}s`
              : "N/A"}
          </div>
        </div>
      </div>

      {/* Classification-specific: class names */}
      {isClassification && result.class_names && result.class_names.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Classes ({result.n_classes})
          </h4>
          <div className="flex flex-wrap gap-2">
            {result.class_names.map((cls: string, idx: number) => (
              <span
                key={idx}
                className="px-3 py-1 bg-violet-50 text-violet-800 rounded-md text-sm font-medium border border-violet-200"
              >
                {cls}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Regression-specific: coefficients preview */}
      {!isClassification && result.coefficients && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Model Coefficients
          </h4>
          <div className="text-sm">
            <div className="flex justify-between items-center mb-2 p-2 bg-emerald-50 rounded">
              <span className="font-medium">Intercept</span>
              <span className="font-mono">{typeof result.intercept === "number" ? result.intercept.toFixed(4) : result.intercept}</span>
            </div>
            {Object.entries(result.coefficients).slice(0, 10).map(([feat, coef]: [string, any]) => (
              <div key={feat} className="flex justify-between items-center p-2 border-b border-gray-100">
                <span className="text-gray-700">{feat}</span>
                <span className="font-mono text-gray-900">{typeof coef === "number" ? coef.toFixed(4) : coef}</span>
              </div>
            ))}
            {Object.keys(result.coefficients).length > 10 && (
              <p className="text-xs text-gray-500 mt-2">
                + {Object.keys(result.coefficients).length - 10} more features...
              </p>
            )}
          </div>
        </div>
      )}

      {/* Model ID */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex justify-between items-center">
          <div>
            <div className="text-xs text-blue-700 uppercase tracking-wide mb-1">Model ID</div>
            <code className="text-sm font-mono text-blue-900">{result.model_id || "N/A"}</code>
          </div>
        </div>
      </div>
    </div>
  );
}

function MissingValueHandlerResults({ result }: { result: any }) {
  const beforeStats = result.before_stats || {};
  const afterStats = result.after_stats || {};
  const operationLog = result.operation_log || [];
  const previewData = result.preview_data;

  // Extract missing value info from before/after stats
  const beforeMissing = beforeStats.missing_by_column || {};
  const afterMissing = afterStats.missing_by_column || {};

  // Summary statistics
  const totalRows = beforeStats.total_rows || 0;
  const columnsWithMissing = beforeStats.columns_with_missing || 0;
  const totalMissingBefore = beforeStats.total_missing_values || 0;
  const totalMissingAfter = afterStats.total_missing_values || 0;
  const rowsDropped =
    (beforeStats.total_rows || 0) - (afterStats.total_rows || 0);

  return (
    <div className="space-y-6">
      {/* Summary Section */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <span className="text-sm text-gray-600">Total Rows:</span>
            <span className="ml-2 font-semibold text-gray-900">
              {totalRows}
            </span>
          </div>
          <div>
            <span className="text-sm text-gray-600">Columns with Missing:</span>
            <span className="ml-2 font-semibold text-gray-900">
              {columnsWithMissing}
            </span>
          </div>
          <div>
            <span className="text-sm text-gray-600">
              Missing Values (Before):
            </span>
            <span className="ml-2 font-semibold text-red-600">
              {totalMissingBefore}
            </span>
          </div>
          <div>
            <span className="text-sm text-gray-600">
              Missing Values (After):
            </span>
            <span className="ml-2 font-semibold text-green-600">
              {totalMissingAfter}
            </span>
          </div>
          {rowsDropped > 0 && (
            <div className="col-span-2">
              <span className="text-sm text-gray-600">Rows Dropped:</span>
              <span className="ml-2 font-semibold text-orange-600">
                {rowsDropped}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Before/After Stats */}
      {Object.keys(beforeMissing).length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Before Stats */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Before Handling
            </h3>
            <div className="space-y-2">
              {Object.entries(beforeMissing).map(
                ([column, stats]: [string, any]) => (
                  <div
                    key={column}
                    className="flex justify-between items-center p-2 bg-red-50 rounded"
                  >
                    <span className="font-medium text-gray-800">{column}</span>
                    <div className="text-sm">
                      <span className="text-red-600 font-semibold">
                        {stats.count} ({stats.percentage}%)
                      </span>
                    </div>
                  </div>
                ),
              )}
            </div>
          </div>

          {/* After Stats */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              After Handling
            </h3>
            {Object.keys(afterMissing).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(afterMissing).map(
                  ([column, stats]: [string, any]) => (
                    <div
                      key={column}
                      className="flex justify-between items-center p-2 bg-yellow-50 rounded"
                    >
                      <span className="font-medium text-gray-800">
                        {column}
                      </span>
                      <div className="text-sm">
                        <span className="text-yellow-600 font-semibold">
                          {stats.count} ({stats.percentage}%)
                        </span>
                      </div>
                    </div>
                  ),
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-green-600 font-semibold">
                ✓ All missing values handled!
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
          <p className="text-green-800 font-semibold text-lg">
            ✓ No missing values found in the dataset
          </p>
        </div>
      )}

      {/* Operation Log */}
      {operationLog && Object.keys(operationLog).length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Operations Applied
          </h3>
          <div className="space-y-2">
            {Object.entries(operationLog).map(
              ([column, operation]: [string, any]) => (
                <div
                  key={column}
                  className="flex justify-between items-center p-3 bg-gray-50 rounded border border-gray-200"
                >
                  <span className="font-medium text-gray-800">{column}</span>
                  <div className="text-sm">
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">
                      {operation.strategy}
                    </span>
                    {operation.filled_count !== undefined && (
                      <span className="ml-2 text-gray-600">
                        ({operation.filled_count} values filled)
                      </span>
                    )}
                  </div>
                </div>
              ),
            )}
          </div>
        </div>
      )}

      {/* Preview Data */}
      {previewData && previewData.after_sample && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Preview (After Handling)
          </h3>
          {renderPreviewTable(
            previewData.after_sample,
            previewData.columns ||
              Object.keys(previewData.after_sample[0] || {}),
            previewData.changes || [],
          )}
        </div>
      )}
    </div>
  );
}

// Helper function to render preview table with highlighting
function renderPreviewTable(data: any[], columns: string[], changes: any[]) {
  return (
    <div className="overflow-auto">
      <table className="min-w-full border-collapse border border-gray-300">
        <thead className="bg-gray-100">
          <tr>
            {columns.map((col: string) => (
              <th
                key={col}
                className="border border-gray-300 px-4 py-2 text-left font-semibold text-sm"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row: any, idx: number) => (
            <tr key={idx} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
              {columns.map((col: string) => {
                const isChanged = changes[idx] && changes[idx][col];
                return (
                  <td
                    key={col}
                    className={`border border-gray-300 px-4 py-2 ${isChanged ? "bg-yellow-100 font-medium text-yellow-900 border-yellow-300" : ""}`}
                  >
                    {row[col] === null ? (
                      <span className="text-gray-400 italic">null</span>
                    ) : (
                      String(row[col])
                    )}
                    {isChanged && (
                      <span
                        className="ml-1 text-yellow-600 text-xs"
                        title="Value modified"
                      >
                        ●
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 text-sm text-gray-600 flex justify-between items-center">
        <span>Showing {data.length} preview rows</span>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-yellow-100 border border-yellow-300 rounded-sm"></span>
          <span className="text-xs text-gray-500">Modified Values</span>
        </div>
      </div>
    </div>
  );
}

