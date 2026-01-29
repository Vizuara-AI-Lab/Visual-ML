/**
 * View Node Modal - Display data from view nodes
 */

import { usePlaygroundStore } from "../../store/playgroundStore";
import { ChartViewer } from "./ChartViewer";

interface ViewNodeModalProps {
  nodeId: string | null;
  onClose: () => void;
}

export const ViewNodeModal = ({ nodeId, onClose }: ViewNodeModalProps) => {
  const { nodes, executionResult } = usePlaygroundStore();

  const node = nodeId ? nodes.find((n) => n.id === nodeId) : null;
  const nodeResult = nodeId ? executionResult?.nodeResults?.[nodeId] : null;

  // Early return AFTER all hooks have been called
  if (!nodeId) return null;

  if (!node) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl p-6">
          <p className="text-gray-800">Node not found</p>
          <button
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  if (!nodeResult) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl p-6">
          <p className="text-gray-800">
            No execution results available for this node
          </p>
          <button
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Close
          </button>
        </div>
      </div>
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
  ].includes(nodeType);

  if (!isViewNode) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-11/12 h-5/6 flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-bold text-gray-800">
            {node.data.label} - Data View
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {nodeResult.error ? (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800 font-semibold">Error:</p>
              <p className="text-red-600 text-sm mt-2">
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
              {nodeType === "encoding" &&
                renderFeatureEngineering(nodeResult, "Encoding")}
              {nodeType === "transformation" &&
                renderFeatureEngineering(nodeResult, "Transformation")}
              {nodeType === "scaling" &&
                renderFeatureEngineering(nodeResult, "Scaling")}
              {nodeType === "feature_selection" &&
                renderFeatureEngineering(nodeResult, "Feature Selection")}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// Render functions for different node types
function renderTableView(result: any) {
  const data = result.data || [];
  const columns = result.columns || [];

  return (
    <div className="overflow-auto">
      <table className="min-w-full border-collapse border border-gray-300">
        <thead className="bg-gray-100">
          <tr>
            {columns.map((col: string) => (
              <th
                key={col}
                className="border border-gray-300 px-4 py-2 text-left font-semibold"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row: any, idx: number) => (
            <tr key={idx} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
              {columns.map((col: string) => (
                <td key={col} className="border border-gray-300 px-4 py-2">
                  {row[col] === null ? (
                    <span className="text-gray-400 italic">null</span>
                  ) : (
                    String(row[col])
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 text-sm text-gray-600">
        Showing {data.length} rows × {columns.length} columns
      </div>
    </div>
  );
}

function renderDataPreview(result: any) {
  return renderTableView(result);
}

function renderStatisticsView(result: any) {
  const stats = result.statistics || {};

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">
        Dataset Statistics
      </h3>
      <div className="grid grid-cols-2 gap-4">
        {Object.entries(stats).map(([key, value]) => (
          <div
            key={key}
            className="p-4 bg-gray-50 rounded-lg border border-gray-200"
          >
            <div className="text-sm text-gray-600 mb-1">{key}</div>
            <div className="text-xl font-semibold text-gray-900">
              {typeof value === "number" ? value.toFixed(2) : String(value)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function renderColumnInfo(result: any) {
  const columns = result.columns || [];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">
        Column Information
      </h3>
      <div className="overflow-auto">
        <table className="min-w-full border-collapse border border-gray-300">
          <thead className="bg-gray-100">
            <tr>
              <th className="border border-gray-300 px-4 py-2 text-left">
                Column Name
              </th>
              <th className="border border-gray-300 px-4 py-2 text-left">
                Data Type
              </th>
              <th className="border border-gray-300 px-4 py-2 text-left">
                Non-Null Count
              </th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col: any, idx: number) => (
              <tr
                key={idx}
                className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
              >
                <td className="border border-gray-300 px-4 py-2">{col.name}</td>
                <td className="border border-gray-300 px-4 py-2">
                  {col.dtype}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {col.non_null_count}
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
  return <ChartViewer result={result} />;
}

function renderMissingValueHandler(result: any) {
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

// Render Feature Engineering node results
function renderFeatureEngineering(result: any, nodeTypeName: string) {
  return (
    <div className="space-y-6">
      {/* Summary Section */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">
          {nodeTypeName} Summary
        </h3>
        <div className="grid grid-cols-2 gap-4">
          {result.original_columns && (
            <div>
              <span className="text-sm text-gray-600">Original Columns:</span>
              <span className="ml-2 font-semibold text-gray-900">
                {result.original_columns}
              </span>
            </div>
          )}
          {result.final_columns && (
            <div>
              <span className="text-sm text-gray-600">Final Columns:</span>
              <span className="ml-2 font-semibold text-gray-900">
                {result.final_columns}
              </span>
            </div>
          )}
          {result.original_features && (
            <div>
              <span className="text-sm text-gray-600">Original Features:</span>
              <span className="ml-2 font-semibold text-gray-900">
                {result.original_features}
              </span>
            </div>
          )}
          {result.selected_features && (
            <div>
              <span className="text-sm text-gray-600">Selected Features:</span>
              <span className="ml-2 font-semibold text-gray-900">
                {result.selected_features}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Dataset ID */}
      {(result.encoded_dataset_id ||
        result.transformed_dataset_id ||
        result.scaled_dataset_id ||
        result.selected_dataset_id) && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-green-900 mb-2">
            ✓ Output Dataset ID
          </h3>
          <code className="text-sm bg-white px-3 py-1 rounded border border-green-300">
            {result.encoded_dataset_id ||
              result.transformed_dataset_id ||
              result.scaled_dataset_id ||
              result.selected_dataset_id}
          </code>
        </div>
      )}

      {/* Detailed Summary */}
      {(result.encoding_summary ||
        result.transformation_summary ||
        result.scaling_summary ||
        result.selection_summary) && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Details</h3>
          <pre className="text-sm bg-gray-50 p-4 rounded border border-gray-200 overflow-auto">
            {JSON.stringify(
              result.encoding_summary ||
                result.transformation_summary ||
                result.scaling_summary ||
                result.selection_summary,
              null,
              2,
            )}
          </pre>
        </div>
      )}

      {/* New Columns (for encoding/transformation) */}
      {result.new_columns && result.new_columns.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            New Columns Created ({result.new_columns.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {result.new_columns.map((col: string, idx: number) => (
              <span
                key={idx}
                className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
              >
                {col}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Selected Features (for feature selection) */}
      {result.selected_feature_names &&
        result.selected_feature_names.length > 0 && (
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Selected Features ({result.selected_feature_names.length})
            </h3>
            <div className="flex flex-wrap gap-2">
              {result.selected_feature_names.map(
                (feature: string, idx: number) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm"
                  >
                    {feature}
                  </span>
                ),
              )}
            </div>
          </div>
        )}

      {/* Feature Scores (for feature selection) */}
      {result.feature_scores &&
        Object.keys(result.feature_scores).length > 0 && (
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Feature Importance Scores
            </h3>
            <div className="overflow-auto max-h-96">
              <table className="min-w-full border-collapse border border-gray-300">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="border border-gray-300 px-4 py-2 text-left">
                      Feature
                    </th>
                    <th className="border border-gray-300 px-4 py-2 text-left">
                      Score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.feature_scores)
                    .sort(([, a]: any, [, b]: any) => b - a)
                    .map(([feature, score]: any, idx: number) => (
                      <tr
                        key={idx}
                        className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                      >
                        <td className="border border-gray-300 px-4 py-2">
                          {feature}
                        </td>
                        <td className="border border-gray-300 px-4 py-2">
                          {typeof score === "number" ? score.toFixed(4) : score}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
    </div>
  );
}
