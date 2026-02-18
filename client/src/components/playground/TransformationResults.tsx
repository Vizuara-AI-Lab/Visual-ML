import { useState } from "react";
import { Download } from "lucide-react";

// Transformation Results Component
export function TransformationResults({ result }: { result: any }) {
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
      <div className="bg-linear-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-4">
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
