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

  console.log("🔍 ViewNodeModal opened for nodeId:", nodeId);
  console.log("📋 Execution result:", executionResult);
  console.log("🎯 Found node:", node);
  console.log("📊 Node result:", nodeResult);

  if (!node) {
    console.error("❌ Node not found for ID:", nodeId);
    return null;
  }

  if (!nodeResult) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-11/12 max-w-2xl flex flex-col p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">
            {node.data.label} - No Data
          </h2>
          <p className="text-gray-600 mb-4">
            This node hasn't been executed yet. Please run the pipeline first.
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
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

function renderTableView(result: any) {
  const data = result.data || [];
  const columns = result.columns || [];

  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">No data to display</div>
    );
  }

  return (
    <div className="overflow-auto">
      <table className="min-w-full border-collapse border border-gray-300">
        <thead>
          <tr className="bg-gray-100">
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
                  {String(row[col] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 text-sm text-gray-600">
        Showing {data.length} of {result.total_rows || data.length} rows
      </div>
    </div>
  );
}

function renderDataPreview(result: any) {
  const headData = result.head_data || [];
  const tailData = result.tail_data || [];

  // Extract columns from first row if not provided
  const columns =
    result.columns || (headData.length > 0 ? Object.keys(headData[0]) : []);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-3">First Rows</h3>
        {renderTableView({ ...result, data: headData, columns })}
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-3">Last Rows</h3>
        {renderTableView({ ...result, data: tailData, columns })}
      </div>
      <div className="mt-4 text-sm text-gray-600">
        Total rows: {result.total_rows || 0}
      </div>
    </div>
  );
}

function renderStatisticsView(result: any) {
  const statistics = result.statistics || {};

  return (
    <div className="overflow-auto">
      <table className="min-w-full border-collapse border border-gray-300">
        <thead>
          <tr className="bg-gray-100">
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Column
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Mean
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Std
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Min
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Max
            </th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(statistics).map(
            ([col, stats]: [string, any], idx) => (
              <tr
                key={col}
                className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
              >
                <td className="border border-gray-300 px-4 py-2 font-medium">
                  {col}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {stats.mean?.toFixed(2) || "N/A"}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {stats.std?.toFixed(2) || "N/A"}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {stats.min || "N/A"}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {stats.max || "N/A"}
                </td>
              </tr>
            ),
          )}
        </tbody>
      </table>
    </div>
  );
}

function renderColumnInfo(result: any) {
  const columnInfo = result.column_info || [];

  return (
    <div className="overflow-auto">
      <table className="min-w-full border-collapse border border-gray-300">
        <thead>
          <tr className="bg-gray-100">
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Column
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Type
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Missing
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
              Unique
            </th>
          </tr>
        </thead>
        <tbody>
          {columnInfo.map((info: any, idx: number) => (
            <tr key={idx} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
              <td className="border border-gray-300 px-4 py-2 font-medium">
                {info.column}
              </td>
              <td className="border border-gray-300 px-4 py-2">{info.dtype}</td>
              <td className="border border-gray-300 px-4 py-2">
                {info.missing || 0}
              </td>
              <td className="border border-gray-300 px-4 py-2">
                {info.unique || 0}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function renderChartView(result: any) {
  return (
    <div className="w-full h-full">
      <ChartViewer
        chartType={result.chart_type}
        chartData={result.chart_data}
      />
    </div>
  );
}

function renderMissingValueHandler(result: any) {
  const beforeStats = result.before_stats?.missing_by_column || {};
  const afterStats = result.after_stats?.missing_by_column || {};
  const operationLog = result.operation_log || {};
  const previewData = result.preview_data;

  // Convert operation_log object to array for rendering
  const operationLogArray = Object.entries(operationLog).map(([column, log]: [string, any]) => ({
    column,
    ...log,
  }));

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="font-semibold text-red-900 mb-2">Before Processing</h4>
          <div className="text-sm text-red-800 space-y-1">
            <p><strong>Total Rows:</strong> {result.before_stats?.total_rows || result.original_rows || 0}</p>
            <p><strong>Columns with Missing:</strong> {result.before_stats?.columns_with_missing || 0}</p>
            <p><strong>Total Missing Values:</strong> {result.before_stats?.total_missing_values || 0}</p>
          </div>
        </div>
        <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
          <h4 className="font-semibold text-green-900 mb-2">After Processing</h4>
          <div className="text-sm text-green-800 space-y-1">
            <p><strong>Total Rows:</strong> {result.after_stats?.total_rows || result.final_rows || 0}</p>
            <p><strong>Rows Dropped:</strong> {result.rows_dropped || 0}</p>
            <p><strong>Columns with Missing:</strong> {result.after_stats?.columns_with_missing || 0}</p>
            <p><strong>Total Missing Values:</strong> {result.after_stats?.total_missing_values || 0}</p>
          </div>
        </div>
      </div>

      {/* Operation Log */}
      {operationLogArray.length > 0 && (
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="text-lg font-semibold mb-3 text-blue-900">
            Operations Applied
          </h3>
          <ul className="space-y-2">
            {operationLogArray.map((log: any, idx: number) => (
              <li key={idx} className="text-sm text-blue-800">
                <span className="font-medium">{log.column}:</span>{" "}
                {log.strategy} {log.fill_value && `(value: ${log.fill_value})`}
                {log.filled_count !== undefined && (
                  <span className="text-blue-600 ml-2">
                    ({log.filled_count} values filled)
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Before/After Statistics Comparison - Only show columns with missing values */}
      {(Object.keys(beforeStats).length > 0 || Object.keys(afterStats).length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="text-lg font-semibold mb-3 text-gray-800">
              Columns with Missing Values (Before)
            </h3>
            <div className="overflow-auto border border-gray-300 rounded-lg">
              <table className="min-w-full border-collapse">
                <thead>
                  <tr className="bg-red-100">
                    <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                      Column
                    </th>
                    <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                      Missing
                    </th>
                    <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                      %
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(beforeStats).length > 0 ? (
                    Object.entries(beforeStats).map(
                      ([col, stats]: [string, any], idx) => (
                        <tr
                          key={col}
                          className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                        >
                          <td className="border border-gray-300 px-4 py-2 font-medium">
                            {col}
                          </td>
                          <td className="border border-gray-300 px-4 py-2">
                            {stats.count || 0}
                          </td>
                          <td className="border border-gray-300 px-4 py-2">
                            {stats.percentage?.toFixed(2) || "0.00"}%
                          </td>
                        </tr>
                      ),
                    )
                  ) : (
                    <tr>
                      <td colSpan={3} className="border border-gray-300 px-4 py-2 text-center text-gray-500">
                        No missing values found
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-3 text-gray-800">
              Columns with Missing Values (After)
            </h3>
            <div className="overflow-auto border border-gray-300 rounded-lg">
              <table className="min-w-full border-collapse">
                <thead>
                  <tr className="bg-green-100">
                    <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                      Column
                    </th>
                    <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                      Missing
                    </th>
                    <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
                      %
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(afterStats).length > 0 ? (
                    Object.entries(afterStats).map(
                      ([col, stats]: [string, any], idx) => (
                        <tr
                          key={col}
                          className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                        >
                          <td className="border border-gray-300 px-4 py-2 font-medium">
                            {col}
                          </td>
                          <td className="border border-gray-300 px-4 py-2 text-green-700 font-semibold">
                            {stats.count || 0}
                          </td>
                          <td className="border border-gray-300 px-4 py-2 text-green-700 font-semibold">
                            {stats.percentage?.toFixed(2) || "0.00"}%
                          </td>
                        </tr>
                      ),
                    )
                  ) : (
                    <tr>
                      <td colSpan={3} className="border border-gray-300 px-4 py-2 text-center text-green-700 font-semibold">
                        ✓ All missing values handled!
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Preview Data (if available) */}
      {previewData && (
        <div>
          <h3 className="text-lg font-semibold mb-3 text-gray-800">
            Preview (First {previewData.rows_shown || 10} Rows)
          </h3>
          {renderPreviewTable({
            data: previewData.after_sample || [],
            columns: Object.keys(previewData.after_sample?.[0] || {}),
            changes: previewData.changes || [],
            total_rows: previewData.total_rows_after || 0,
          })}
        </div>
      )}

      {/* Summary */}
      <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
        <h3 className="text-lg font-semibold mb-2 text-green-900">
          ✓ Processing Complete
        </h3>
        <p className="text-sm text-green-800">
          Dataset ID: <span className="font-mono">{result.preprocessed_dataset_id}</span>
        </p>
        {result.reversible_operation_id && (
          <p className="text-sm text-green-800 mt-1">
            Operation ID: <span className="font-mono">{result.reversible_operation_id}</span>
          </p>
        )}
      </div>
    </div>
  );
}

function renderPreviewTable(result: any) {
  const data = result.data || [];
  const columns = result.columns || [];
  const changes = result.changes || [];

  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">No data to display</div>
    );
  }

  return (
    <div className="overflow-auto border border-gray-300 rounded-lg">
      <table className="min-w-full border-collapse">
        <thead>
          <tr className="bg-gray-100">
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
                    {isChanged && <span className="ml-1 text-yellow-600 text-xs" title="Value modified">●</span>}
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
