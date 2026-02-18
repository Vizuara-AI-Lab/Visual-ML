import { useState } from "react";
import { Download } from "lucide-react";

// Scaling Results Component
export function ScalingResults({ result }: { result: any }) {
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
      <div className="bg-linear-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg p-4">
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
