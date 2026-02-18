import { useState } from "react";
import { Download, Copy, ChevronDown, ChevronRight, Search } from "lucide-react";

// Feature Engineering Results Component
export function FeatureEngineeringResults({
  result,
  nodeTypeName,
}: {
  result: any;
  nodeTypeName: string;
}) {
  const [showRawConfig, setShowRawConfig] = useState(false);
  const [showAllColumns, setShowAllColumns] = useState(false);
  const [columnSearch, setColumnSearch] = useState("");
  const [downloading, setDownloading] = useState(false);

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

  // Get the dataset ID
  const datasetId =
    result.encoded_dataset_id ||
    result.transformed_dataset_id ||
    result.scaled_dataset_id ||
    result.selected_dataset_id;

  // Download complete dataset with all columns as CSV
  const downloadDataset = async () => {
    if (!datasetId) {
      alert("No dataset ID available for download");
      return;
    }

    setDownloading(true);
    try {
      const { downloadFileFromUploads } =
        await import("../../lib/api/datasetApi");
      await downloadFileFromUploads(datasetId);
    } catch (error) {
      console.error("Download failed:", error);
      alert("Failed to download dataset. Please try again.");
    } finally {
      setDownloading(false);
    }
  };

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
          <code className="text-xs bg-white px-3 py-2 rounded border border-green-300 block break-all mb-3">
            {datasetId || "N/A"}
          </code>
          <button
            onClick={downloadDataset}
            disabled={downloading || !datasetId}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed text-sm font-medium"
          >
            <Download className="w-4 h-4" />
            {downloading ? "Downloading..." : "Download Complete Dataset"}
          </button>
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
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-600 text-white rounded-md hover:bg-slate-700 text-sm"
              >
                <Download className="w-4 h-4" />
                Column Names
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

      {/* Enhanced Feature Selection Results */}
      {result.selected_feature_names &&
        result.selected_feature_names.length > 0 && (
          <div className="space-y-4">
            {/* Method Summary Card */}
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-purple-900 mb-3 flex items-center gap-2">
                🎯 Selection Method
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-700">Method:</span>
                  <span className="font-semibold text-gray-900">
                    {result.selection_summary?.method?.toUpperCase() || "N/A"}
                  </span>
                </div>
                {result.selection_summary?.variance_threshold !== undefined && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">
                      Variance Threshold:
                    </span>
                    <span className="font-semibold text-purple-600">
                      {result.selection_summary.variance_threshold}
                    </span>
                  </div>
                )}
                {result.selection_summary?.correlation_threshold !==
                  undefined && (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-700">
                        Correlation Threshold:
                      </span>
                      <span className="font-semibold text-purple-600">
                        {result.selection_summary.correlation_threshold}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-700">Mode:</span>
                      <span className="font-semibold text-gray-900">
                        {result.selection_summary.correlation_mode || "N/A"}
                      </span>
                    </div>
                  </>
                )}
                {result.selection_summary?.n_features && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">Target K:</span>
                    <span className="font-semibold text-purple-600">
                      {result.selection_summary.n_features}
                    </span>
                  </div>
                )}
                <div className="flex justify-between items-center pt-2 border-t border-purple-200">
                  <span className="text-sm text-gray-700">Reduction:</span>
                  <span className="font-semibold text-red-600">
                    {result.selection_summary?.reduction_percentage || 0}%
                  </span>
                </div>
              </div>
            </div>

            {/* Dataset Shape Comparison */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                📊 Dataset Shape Comparison
              </h3>
              <div className="flex items-center justify-center gap-6">
                <div className="text-center">
                  <div className="text-sm text-gray-600 mb-1">Before</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {result.selection_summary?.original_features || 0}
                  </div>
                  <div className="text-xs text-gray-500">features</div>
                </div>
                <div className="text-3xl text-gray-400">→</div>
                <div className="text-center">
                  <div className="text-sm text-gray-600 mb-1">After</div>
                  <div className="text-2xl font-bold text-green-600">
                    {result.selected_feature_names.length}
                  </div>
                  <div className="text-xs text-gray-500">features</div>
                </div>
              </div>
            </div>

            {/* Selected Features */}
            <div className="bg-white border border-green-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-green-900 mb-3">
                ✅ Selected Features ({result.selected_feature_names.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {result.selected_feature_names.map(
                  (feature: string, idx: number) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium border border-green-300"
                    >
                      {feature}
                    </span>
                  ),
                )}
              </div>
            </div>

            {/* Removed Features */}
            {result.removed_feature_names &&
              result.removed_feature_names.length > 0 && (
                <div className="bg-white border border-red-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-red-900 mb-3">
                    ❌ Removed Features ({result.removed_feature_names.length})
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {result.removed_feature_names.map(
                      (feature: string, idx: number) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium border border-red-300"
                        >
                          {feature}
                        </span>
                      ),
                    )}
                  </div>
                </div>
              )}
          </div>
        )}

      {/* Enhanced Feature Importance/Score Ranking Table */}
      {result.feature_scores &&
        Object.keys(result.feature_scores).length > 0 && (
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                📈 Feature Ranking (
                {result.selection_summary?.method === "variance"
                  ? "Variance"
                  : "Max Correlation"}{" "}
                Scores)
              </h3>
              <span className="text-sm text-gray-600">
                {Object.keys(result.feature_scores).length} features
              </span>
            </div>
            <div className="overflow-auto max-h-96 border border-gray-300 rounded">
              <table className="min-w-full border-collapse">
                <thead className="bg-gray-100 sticky top-0">
                  <tr>
                    <th className="border-b border-gray-300 px-4 py-3 text-left text-xs font-semibold text-gray-700">
                      Rank
                    </th>
                    <th className="border-b border-gray-300 px-4 py-3 text-left text-xs font-semibold text-gray-700">
                      Feature Name
                    </th>
                    <th className="border-b border-gray-300 px-4 py-3 text-right text-xs font-semibold text-gray-700">
                      Score
                    </th>
                    <th className="border-b border-gray-300 px-4 py-3 text-center text-xs font-semibold text-gray-700">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.feature_scores)
                    .sort(([, a]: any, [, b]: any) => {
                      // For variance: higher is better (sort descending)
                      // For correlation: lower is better (sort ascending)
                      if (result.selection_summary?.method === "correlation") {
                        return a - b; // ascending
                      }
                      return b - a; // descending (variance)
                    })
                    .map(([feature, score]: any, idx: number) => {
                      const isSelected =
                        result.selected_feature_names?.includes(feature);
                      return (
                        <tr
                          key={idx}
                          className={`${
                            idx % 2 === 0 ? "bg-white" : "bg-gray-50"
                          } hover:bg-gray-100 transition-colors`}
                        >
                          <td className="border-b border-gray-200 px-4 py-2 text-sm text-gray-600">
                            #{idx + 1}
                          </td>
                          <td className="border-b border-gray-200 px-4 py-2 text-sm font-medium text-gray-900">
                            {feature}
                          </td>
                          <td className="border-b border-gray-200 px-4 py-2 text-sm text-right font-mono">
                            {typeof score === "number"
                              ? score.toFixed(6)
                              : score}
                          </td>
                          <td className="border-b border-gray-200 px-4 py-2 text-center">
                            {isSelected ? (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-green-100 text-green-800 border border-green-300">
                                ✓ Selected
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-red-100 text-red-800 border border-red-300">
                                ✗ Removed
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
            {result.selection_summary?.method === "variance" && (
              <p className="text-xs text-gray-600 mt-2">
                💡 Higher variance = more information retained. Features below
                threshold {result.selection_summary.variance_threshold} were
                removed.
              </p>
            )}
            {result.selection_summary?.method === "correlation" && (
              <p className="text-xs text-gray-600 mt-2">
                💡 Lower correlation = less redundancy. Features with max
                correlation above{" "}
                {result.selection_summary.correlation_threshold} were removed.
              </p>
            )}
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
