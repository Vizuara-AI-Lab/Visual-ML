/**
 * View Node Modal - Display data from view nodes
 */

import { useState } from "react";
import { usePlaygroundStore } from "../../store/playgroundStore";
import { ChartViewer } from "./ChartViewer";
import {
  Download,
  Copy,
  ChevronDown,
  ChevronRight,
  Search,
} from "lucide-react";

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
              {nodeType === "encoding" && (
                <FeatureEngineeringResults
                  result={nodeResult}
                  nodeTypeName="Encoding"
                />
              )}
              {nodeType === "transformation" && (
                <TransformationResults result={nodeResult} />
              )}
              {nodeType === "scaling" && <ScalingResults result={nodeResult} />}
              {nodeType === "feature_selection" && (
                <FeatureEngineeringResults
                  result={nodeResult}
                  nodeTypeName="Feature Selection"
                />
              )}
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
  const headData = result.head_data || [];
  const tailData = result.tail_data || [];
  const totalRows = result.total_rows || 0;

  // Extract columns from first row of head_data or tail_data
  const columns =
    headData.length > 0
      ? Object.keys(headData[0])
      : tailData.length > 0
        ? Object.keys(tailData[0])
        : [];

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-purple-900">
              Dataset Preview
            </h3>
            <p className="text-sm text-purple-700 mt-1">
              Total rows:{" "}
              <span className="font-semibold">
                {totalRows.toLocaleString()}
              </span>{" "}
              | Columns: <span className="font-semibold">{columns.length}</span>
            </p>
          </div>
        </div>
      </div>

      {/* Head Data */}
      {headData.length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-gray-800 mb-3">
            First {headData.length} Rows
          </h4>
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
                {headData.map((row: any, idx: number) => (
                  <tr
                    key={idx}
                    className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                  >
                    {columns.map((col: string) => (
                      <td
                        key={col}
                        className="border border-gray-300 px-4 py-2 text-sm"
                      >
                        {row[col] === null || row[col] === undefined ? (
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
          </div>
        </div>
      )}

      {/* Tail Data */}
      {tailData.length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-gray-800 mb-3">
            Last {tailData.length} Rows
          </h4>
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
                {tailData.map((row: any, idx: number) => (
                  <tr
                    key={idx}
                    className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                  >
                    {columns.map((col: string) => (
                      <td
                        key={col}
                        className="border border-gray-300 px-4 py-2 text-sm"
                      >
                        {row[col] === null || row[col] === undefined ? (
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
          </div>
        </div>
      )}

      {/* Empty state */}
      {headData.length === 0 && tailData.length === 0 && (
        <div className="text-center py-8 text-gray-500">
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
      <div className="text-center py-8 text-gray-500">
        No statistics available
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-green-900">
          Statistical Summary
        </h3>
        <p className="text-sm text-green-700 mt-1">
          Statistics for {columnNames.length} numeric column
          {columnNames.length !== 1 ? "s" : ""}
        </p>
      </div>

      <div className="space-y-4">
        {columnNames.map((colName) => {
          const colStats = stats[colName];
          return (
            <div
              key={colName}
              className="bg-white border border-gray-200 rounded-lg p-4"
            >
              <h4 className="text-md font-semibold text-gray-800 mb-3">
                {colName}
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-xs text-gray-600 mb-1">Mean</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {typeof colStats.mean === "number"
                      ? colStats.mean.toFixed(2)
                      : colStats.mean}
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-xs text-gray-600 mb-1">Std Dev</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {typeof colStats.std === "number"
                      ? colStats.std.toFixed(2)
                      : colStats.std}
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-xs text-gray-600 mb-1">Min</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {typeof colStats.min === "number"
                      ? colStats.min.toFixed(2)
                      : colStats.min}
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-xs text-gray-600 mb-1">Max</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {typeof colStats.max === "number"
                      ? colStats.max.toFixed(2)
                      : colStats.max}
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-3">
                  <div className="text-xs text-gray-600 mb-1">Median</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {typeof colStats.median === "number"
                      ? colStats.median.toFixed(2)
                      : colStats.median}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function renderColumnInfo(result: any) {
  const columnInfo = result.column_info || [];

  if (columnInfo.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No column information available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-900">
          Column Information
        </h3>
        <p className="text-sm text-blue-700 mt-1">
          Detailed information for {columnInfo.length} column
          {columnInfo.length !== 1 ? "s" : ""}
        </p>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full border-collapse border border-gray-300">
          <thead className="bg-gray-100">
            <tr>
              <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                Column Name
              </th>
              <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                Data Type
              </th>
              <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                Missing Values
              </th>
              <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                Unique Values
              </th>
            </tr>
          </thead>
          <tbody>
            {columnInfo.map((col: any, idx: number) => (
              <tr
                key={idx}
                className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
              >
                <td className="border border-gray-300 px-4 py-2 font-medium">
                  {col.column}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  <code className="text-xs bg-gray-100 px-2 py-1 rounded">
                    {col.dtype}
                  </code>
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {col.missing !== undefined ? (
                    <span
                      className={
                        col.missing > 0
                          ? "text-red-600 font-medium"
                          : "text-green-600"
                      }
                    >
                      {col.missing}
                    </span>
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </td>
                <td className="border border-gray-300 px-4 py-2">
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

// Transformation Results Component
function TransformationResults({ result }: { result: any }) {
  const [downloading, setDownloading] = useState(false);

  const transformationType =
    result.transformation_summary?.polynomial?.method ||
    Object.values(result.transformation_summary || {}).find(
      (v: any) => v?.method,
    )?.method ||
    "N/A";

  const transformedColumns = result.transformed_columns || [];
  const originalColumns = result.original_columns || 0;
  const finalColumns = result.final_columns || 0;
  const newColumns = result.new_columns || [];
  const transformedDatasetId = result.transformed_dataset_id;
  const warnings = result.transformation_summary?.warnings || [];
  const executionTime = result.execution_time_ms
    ? `${(result.execution_time_ms / 1000).toFixed(2)}s`
    : "N/A";

  // Download transformed CSV
  const downloadCSV = async () => {
    if (!transformedDatasetId) {
      alert("No dataset ID available for download");
      return;
    }

    setDownloading(true);
    try {
      const { downloadFileFromUploads } =
        await import("../../lib/api/datasetApi");
      await downloadFileFromUploads(transformedDatasetId);
    } catch (error) {
      console.error("Download failed:", error);
      alert("Failed to download dataset. Please try again.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center">
            <span className="text-white text-xl">✓</span>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-purple-900">
              Transformation Complete
            </h3>
            <p className="text-sm text-purple-700">
              Data successfully transformed using {transformationType}
            </p>
          </div>
        </div>
      </div>

      {/* Warnings Section */}
      {warnings.length > 0 && (
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <div className="text-orange-500 text-xl">⚠</div>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-orange-900 mb-2">
                Warnings
              </h4>
              <ul className="space-y-1">
                {warnings.map((warning: string, idx: number) => (
                  <li key={idx} className="text-sm text-orange-800">
                    • {warning}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Transformation Summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-4">
          Transformation Summary
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
              Method
            </div>
            <div className="text-lg font-semibold text-gray-900 capitalize">
              {transformationType}
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
              Columns Transformed
            </div>
            <div className="text-lg font-semibold text-gray-900">
              {transformedColumns.length}
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
              Total Columns
            </div>
            <div className="text-lg font-semibold text-gray-900">
              {originalColumns} → {finalColumns}
            </div>
            {newColumns.length > 0 && (
              <div className="text-xs text-green-600 mt-1">
                +{newColumns.length} new
              </div>
            )}
          </div>

          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
              Execution Time
            </div>
            <div className="text-lg font-semibold text-gray-900">
              {executionTime}
            </div>
          </div>
        </div>
      </div>

      {/* Output Dataset */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-blue-700 uppercase tracking-wide mb-1">
              Output Dataset ID
            </div>
            <code className="text-sm font-mono text-blue-900">
              {transformedDatasetId || "N/A"}
            </code>
          </div>
          <button
            onClick={downloadCSV}
            disabled={downloading || !transformedDatasetId}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            {downloading ? "Downloading..." : "Download CSV"}
          </button>
        </div>
      </div>

      {/* Transformed Columns List */}
      {transformedColumns.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Transformed Columns ({transformedColumns.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {transformedColumns.map((col: string, idx: number) => (
              <span
                key={idx}
                className="px-3 py-1.5 bg-purple-50 text-purple-800 rounded-md text-sm font-medium border border-purple-200"
              >
                {col}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* New Columns (for polynomial) */}
      {newColumns.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            New Columns Created ({newColumns.length})
          </h4>
          <div className="max-h-60 overflow-y-auto">
            <div className="flex flex-wrap gap-2">
              {newColumns.slice(0, 20).map((col: string, idx: number) => (
                <span
                  key={idx}
                  className="px-3 py-1.5 bg-green-50 text-green-800 rounded-md text-sm font-medium border border-green-200"
                >
                  {col}
                </span>
              ))}
              {newColumns.length > 20 && (
                <span className="px-3 py-1.5 bg-gray-50 text-gray-600 rounded-md text-sm font-medium border border-gray-200">
                  +{newColumns.length - 20} more...
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Before/After Preview (if available) */}
      {result.preview_data && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Before → After Preview
          </h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">
                    Column
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">
                    Before
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">
                    After
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {result.preview_data.map((row: any, idx: number) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-900">
                      {row.column}
                    </td>
                    <td className="px-4 py-2 text-gray-700">{row.before}</td>
                    <td className="px-4 py-2 text-gray-700">{row.after}</td>
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

// Scaling Results Component
function ScalingResults({ result }: { result: any }) {
  const [downloading, setDownloading] = useState(false);

  const scalingMethod =
    result.scaling_method || result.scaling_summary?.method || "N/A";
  const scaledColumns = result.scaled_columns || [];
  const originalRows = result.scaling_summary?.original_rows || 0;
  const finalRows = result.scaling_summary?.final_rows || 0;
  const rowsDropped = originalRows - finalRows;
  const executionTime = result.execution_time_ms
    ? `${(result.execution_time_ms / 1000).toFixed(2)}s`
    : "N/A";
  const scaledDatasetId = result.scaled_dataset_id;

  // Download actual scaled CSV
  const downloadCSV = async () => {
    if (!scaledDatasetId) {
      alert("No dataset ID available for download");
      return;
    }

    setDownloading(true);
    try {
      // Import the download function for uploads folder
      const { downloadFileFromUploads } =
        await import("../../lib/api/datasetApi");
      await downloadFileFromUploads(scaledDatasetId);
    } catch (error) {
      console.error("Download failed:", error);
      alert("Failed to download dataset. Please try again.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
            <span className="text-white text-xl">✓</span>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-green-900">
              Scaling Complete
            </h3>
            <p className="text-sm text-green-700">
              Data successfully scaled using {scalingMethod}
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
            Method
          </div>
          <div className="text-lg font-semibold text-gray-900 capitalize">
            {scalingMethod}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
            Columns Scaled
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {scaledColumns.length}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
            Row Count
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {finalRows.toLocaleString()}
          </div>
          {rowsDropped > 0 && (
            <div className="text-xs text-orange-600 mt-1">
              -{rowsDropped} cleaned
            </div>
          )}
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
            Execution Time
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {executionTime}
          </div>
        </div>
      </div>

      {/* Output Dataset */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-blue-700 uppercase tracking-wide mb-1">
              Output Dataset ID
            </div>
            <code className="text-sm font-mono text-blue-900">
              {scaledDatasetId || "N/A"}
            </code>
          </div>
          <button
            onClick={downloadCSV}
            disabled={downloading || !scaledDatasetId}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            {downloading ? "Downloading..." : "Download CSV"}
          </button>
        </div>
      </div>

      {/* Scaled Columns List */}
      {scaledColumns.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Scaled Columns ({scaledColumns.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {scaledColumns.map((col: string, idx: number) => (
              <span
                key={idx}
                className="px-3 py-1.5 bg-teal-50 text-teal-800 rounded-md text-sm font-medium border border-teal-200"
              >
                {col}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Before/After Preview (if available) */}
      {result.preview_data && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Before → After Preview
          </h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">
                    Column
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">
                    Before
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">
                    After
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {result.preview_data.map((row: any, idx: number) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-900">
                      {row.column}
                    </td>
                    <td className="px-4 py-2 text-gray-700">{row.before}</td>
                    <td className="px-4 py-2 text-gray-700">{row.after}</td>
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

// Feature Engineering Results Component
function FeatureEngineeringResults({
  result,
  nodeTypeName,
}: {
  result: any;
  nodeTypeName: string;
}) {
  const [showRawConfig, setShowRawConfig] = useState(false);
  const [showAllColumns, setShowAllColumns] = useState(false);
  const [columnSearch, setColumnSearch] = useState("");

  const isEncodingNode = nodeTypeName === "Encoding";

  // Filter columns based on search
  const newColumns = result.new_columns || [];
  const filteredColumns = columnSearch
    ? newColumns.filter((col: string) =>
        col.toLowerCase().includes(columnSearch.toLowerCase()),
      )
    : newColumns;

  const columnsToShow = showAllColumns
    ? filteredColumns
    : filteredColumns.slice(0, 10);
  const hasMoreColumns = filteredColumns.length > 10;

  // Download columns as CSV
  const downloadColumns = () => {
    const csv = newColumns.join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${nodeTypeName.toLowerCase()}_columns.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Copy config JSON
  const copyConfig = () => {
    const config = JSON.stringify(
      result.encoding_summary ||
        result.transformation_summary ||
        result.scaling_summary ||
        result.selection_summary,
      null,
      2,
    );
    navigator.clipboard.writeText(config);
  };

  return (
    <div className="space-y-6">
      {/* Compact Summary Cards */}
      <div className="grid grid-cols-2 gap-4">
        {/* Stats Card */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-blue-900 mb-3 flex items-center gap-2">
            📊 Column Statistics
          </h3>
          <div className="space-y-2">
            {result.original_columns !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">Original:</span>
                <span className="font-semibold text-gray-900 text-lg">
                  {result.original_columns}
                </span>
              </div>
            )}
            {result.final_columns !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">Final:</span>
                <span className="font-semibold text-gray-900 text-lg">
                  {result.final_columns}
                </span>
              </div>
            )}
            {newColumns.length > 0 && (
              <div className="flex justify-between items-center pt-2 border-t border-blue-200">
                <span className="text-sm text-gray-700">New Columns:</span>
                <span className="font-semibold text-blue-600 text-lg">
                  +{newColumns.length}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Output Dataset Card */}
        <div className="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-green-900 mb-3 flex items-center gap-2">
            ✅ Output Dataset
          </h3>
          <code className="text-xs bg-white px-3 py-2 rounded border border-green-300 block break-all">
            {result.encoded_dataset_id ||
              result.transformed_dataset_id ||
              result.scaled_dataset_id ||
              result.selected_dataset_id ||
              "N/A"}
          </code>
        </div>
      </div>

      {/* Encoding-specific Summary */}
      {isEncodingNode && result.encoding_summary && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Encoding Configuration Summary
          </h3>
          <div className="space-y-4">
            {Object.entries(result.encoding_summary).map(
              ([column, config]: [string, any]) => (
                <div
                  key={column}
                  className="bg-gray-50 rounded-lg p-4 border border-gray-200"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h4 className="font-semibold text-gray-900">{column}</h4>
                    <span className="px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-xs font-medium">
                      {config.method || "N/A"}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    {config.unique_values !== undefined && (
                      <div>
                        <span className="text-gray-600">
                          Unique Categories:
                        </span>
                        <span className="ml-2 font-semibold text-gray-900">
                          {config.unique_values}
                        </span>
                      </div>
                    )}
                    {config.handle_unknown && (
                      <div>
                        <span className="text-gray-600">Handle Unknown:</span>
                        <span className="ml-2 font-semibold text-gray-900">
                          {config.handle_unknown}
                        </span>
                      </div>
                    )}
                    {config.handle_missing && (
                      <div>
                        <span className="text-gray-600">Handle Missing:</span>
                        <span className="ml-2 font-semibold text-gray-900">
                          {config.handle_missing}
                        </span>
                      </div>
                    )}
                    {config.new_columns && (
                      <div className="col-span-2">
                        <span className="text-gray-600">
                          Generated Columns:
                        </span>
                        <span className="ml-2 font-semibold text-blue-600">
                          {Array.isArray(config.new_columns)
                            ? config.new_columns.length
                            : 0}{" "}
                          columns
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ),
            )}
          </div>
        </div>
      )}

      {/* New Columns Section */}
      {newColumns.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Generated Columns ({newColumns.length})
            </h3>
            <div className="flex gap-2">
              <button
                onClick={downloadColumns}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
              >
                <Download className="w-4 h-4" />
                Download CSV
              </button>
            </div>
          </div>

          {/* Search */}
          {newColumns.length > 10 && (
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search columns..."
                  value={columnSearch}
                  onChange={(e) => setColumnSearch(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              {columnSearch && (
                <p className="text-xs text-gray-500 mt-1">
                  Found {filteredColumns.length} columns
                </p>
              )}
            </div>
          )}

          {/* Column List */}
          <div className="flex flex-wrap gap-2">
            {columnsToShow.map((col: string, idx: number) => (
              <span
                key={idx}
                className="px-3 py-1 bg-blue-50 text-blue-800 rounded-md text-sm border border-blue-200"
              >
                {col}
              </span>
            ))}
          </div>

          {/* Show More/Less */}
          {hasMoreColumns && !columnSearch && (
            <button
              onClick={() => setShowAllColumns(!showAllColumns)}
              className="mt-4 text-sm text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
            >
              {showAllColumns ? (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show Less
                </>
              ) : (
                <>
                  <ChevronRight className="w-4 h-4" />+
                  {filteredColumns.length - 10} more columns...
                </>
              )}
            </button>
          )}
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

      {/* Advanced / Debug Section */}
      {(result.encoding_summary ||
        result.transformation_summary ||
        result.scaling_summary ||
        result.selection_summary) && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <button
            onClick={() => setShowRawConfig(!showRawConfig)}
            className="flex items-center justify-between w-full text-left"
          >
            <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              {showRawConfig ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              Advanced / Debug: Raw Configuration
            </h3>
            <button
              onClick={(e) => {
                e.stopPropagation();
                copyConfig();
              }}
              className="flex items-center gap-1 px-2 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-700"
            >
              <Copy className="w-3 h-3" />
              Copy JSON
            </button>
          </button>

          {showRawConfig && (
            <pre className="text-xs bg-gray-900 text-green-400 p-4 rounded mt-3 overflow-auto max-h-96 border border-gray-700">
              {JSON.stringify(
                result.encoding_summary ||
                  result.transformation_summary ||
                  result.scaling_summary ||
                  result.selection_summary,
                null,
                2,
              )}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
