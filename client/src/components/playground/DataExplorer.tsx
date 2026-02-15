/**
 * DataExplorer Component - Interactive data exploration with tabs
 * Wraps ChartViewer and adds Distributions, Correlations, Summary tabs
 */

import { useState } from "react";
import { ChartViewer } from "./ChartViewer";
import { Bar } from "react-chartjs-2";

interface DataExplorerProps {
  result: any;
}

type ExplorerTab = "chart" | "distributions" | "correlations" | "summary";

export const DataExplorer = ({ result }: DataExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("chart");

  const explorationData = result?.exploration_data;
  const hasExplorationData = !!explorationData;

  const tabs: { id: ExplorerTab; label: string; available: boolean }[] = [
    { id: "chart", label: "Chart", available: true },
    {
      id: "distributions",
      label: "Distributions",
      available:
        !!explorationData?.distributions &&
        Object.keys(explorationData.distributions).length > 0,
    },
    {
      id: "correlations",
      label: "Correlations",
      available:
        !!explorationData?.correlations?.columns?.length &&
        explorationData.correlations.columns.length >= 2,
    },
    {
      id: "summary",
      label: "Summary",
      available: !!explorationData?.summary,
    },
  ];

  return (
    <div className="space-y-4">
      {/* Tab navigation */}
      {hasExplorationData && (
        <div className="flex border-b border-gray-200">
          {tabs
            .filter((t) => t.available)
            .map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                {tab.label}
              </button>
            ))}
        </div>
      )}

      {/* Tab content */}
      {activeTab === "chart" && <ChartViewer result={result} />}
      {activeTab === "distributions" && explorationData?.distributions && (
        <DistributionsTab data={explorationData.distributions} />
      )}
      {activeTab === "correlations" && explorationData?.correlations && (
        <CorrelationsTab data={explorationData.correlations} />
      )}
      {activeTab === "summary" && explorationData?.summary && (
        <SummaryTab data={explorationData.summary} />
      )}
    </div>
  );
};

// --- Distributions Tab ---

function DistributionsTab({ data }: { data: Record<string, any> }) {
  const columns = Object.keys(data);

  if (columns.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No numeric columns available for distribution analysis
      </div>
    );
  }

  const colorPalette = [
    "rgba(59, 130, 246, 0.7)",
    "rgba(16, 185, 129, 0.7)",
    "rgba(239, 68, 68, 0.7)",
    "rgba(245, 158, 11, 0.7)",
    "rgba(139, 92, 246, 0.7)",
    "rgba(236, 72, 153, 0.7)",
    "rgba(14, 165, 233, 0.7)",
    "rgba(249, 115, 22, 0.7)",
    "rgba(168, 85, 247, 0.7)",
    "rgba(34, 197, 94, 0.7)",
  ];

  const borderPalette = colorPalette.map((c) => c.replace("0.7", "1"));

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">
          Column Distributions
        </h3>
        <p className="text-xs text-indigo-700 mt-1">
          Histograms showing the value distribution for each numeric column
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {columns.map((colName, idx) => {
          const dist = data[colName];
          const chartData = {
            labels: dist.labels,
            datasets: [
              {
                label: colName,
                data: dist.counts,
                backgroundColor: colorPalette[idx % colorPalette.length],
                borderColor: borderPalette[idx % borderPalette.length],
                borderWidth: 1,
              },
            ],
          };
          const options = {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { display: false },
              title: {
                display: true,
                text: colName,
                font: { size: 13, weight: "bold" as const },
              },
            },
            scales: {
              x: {
                ticks: { maxRotation: 45, font: { size: 9 } },
                title: { display: false },
              },
              y: {
                beginAtZero: true,
                title: {
                  display: true,
                  text: "Count",
                  font: { size: 10 },
                },
              },
            },
          };

          return (
            <div
              key={colName}
              className="bg-white border border-gray-200 rounded-lg p-3"
            >
              <div style={{ height: "220px" }}>
                <Bar data={chartData} options={options} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Correlations Tab ---

function CorrelationsTab({
  data,
}: {
  data: { columns: string[]; matrix: number[][] };
}) {
  const { columns, matrix } = data;

  if (columns.length < 2) {
    return (
      <div className="text-center py-8 text-gray-500">
        Need at least 2 numeric columns for correlation analysis
      </div>
    );
  }

  const getCorrelationColor = (value: number): string => {
    if (value >= 0.7) return "bg-blue-500 text-white";
    if (value >= 0.4) return "bg-blue-300 text-blue-900";
    if (value >= 0.2) return "bg-blue-100 text-blue-800";
    if (value > -0.2) return "bg-gray-50 text-gray-700";
    if (value > -0.4) return "bg-red-100 text-red-800";
    if (value > -0.7) return "bg-red-300 text-red-900";
    return "bg-red-500 text-white";
  };

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">
          Correlation Matrix
        </h3>
        <p className="text-xs text-purple-700 mt-1">
          Pearson correlation between numeric columns. Blue = positive, Red =
          negative.
        </p>
      </div>

      {/* Color legend */}
      <div className="flex items-center gap-2 text-xs text-gray-600">
        <span>Strong negative</span>
        <div className="flex gap-0.5">
          <div className="w-6 h-4 bg-red-500 rounded-sm" />
          <div className="w-6 h-4 bg-red-300 rounded-sm" />
          <div className="w-6 h-4 bg-red-100 rounded-sm" />
          <div className="w-6 h-4 bg-gray-50 border border-gray-200 rounded-sm" />
          <div className="w-6 h-4 bg-blue-100 rounded-sm" />
          <div className="w-6 h-4 bg-blue-300 rounded-sm" />
          <div className="w-6 h-4 bg-blue-500 rounded-sm" />
        </div>
        <span>Strong positive</span>
      </div>

      <div className="overflow-auto">
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              <th className="p-2 border border-gray-300 bg-gray-100 font-semibold sticky left-0 z-10"></th>
              {columns.map((col) => (
                <th
                  key={col}
                  className="p-2 border border-gray-300 bg-gray-100 font-semibold text-center min-w-[70px]"
                >
                  <div className="truncate max-w-[80px]" title={col}>
                    {col}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.map((rowCol, i) => (
              <tr key={rowCol}>
                <th className="p-2 border border-gray-300 bg-gray-100 font-semibold text-left sticky left-0 z-10">
                  <div className="truncate max-w-[100px]" title={rowCol}>
                    {rowCol}
                  </div>
                </th>
                {columns.map((colCol, j) => {
                  const value = matrix[i][j];
                  return (
                    <td
                      key={j}
                      className={`p-2 border border-gray-300 text-center font-mono ${getCorrelationColor(value)}`}
                      title={`${rowCol} vs ${colCol}: ${value.toFixed(4)}`}
                    >
                      {value.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Educational note */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
        <h4 className="text-xs font-semibold text-gray-700 mb-1">
          Understanding Correlations
        </h4>
        <ul className="text-xs text-gray-600 space-y-1 list-disc list-inside">
          <li>
            <strong>+1.0</strong>: Perfect positive correlation (both increase
            together)
          </li>
          <li>
            <strong>0.0</strong>: No linear relationship
          </li>
          <li>
            <strong>-1.0</strong>: Perfect negative correlation (one increases as
            other decreases)
          </li>
          <li>
            Values above <strong>0.7</strong> or below <strong>-0.7</strong>{" "}
            indicate strong relationships
          </li>
        </ul>
      </div>
    </div>
  );
}

// --- Summary Tab ---

function SummaryTab({ data }: { data: any }) {
  const {
    total_rows,
    total_columns,
    numeric_columns,
    categorical_columns,
    columns,
  } = data;

  return (
    <div className="space-y-4">
      {/* Overview banner */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-green-900">
          Dataset Summary
        </h3>
        <div className="flex gap-6 mt-2 text-xs text-green-700">
          <span>
            Rows: <strong>{total_rows?.toLocaleString()}</strong>
          </span>
          <span>
            Columns: <strong>{total_columns}</strong>
          </span>
          <span>
            Numeric: <strong>{numeric_columns?.length || 0}</strong>
          </span>
          <span>
            Categorical: <strong>{categorical_columns?.length || 0}</strong>
          </span>
        </div>
      </div>

      {/* Stats table */}
      <div className="overflow-auto">
        <table className="min-w-full border-collapse border border-gray-300 text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="border border-gray-300 px-3 py-2 text-left font-semibold">
                Column
              </th>
              <th className="border border-gray-300 px-3 py-2 text-left font-semibold">
                Type
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Count
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Missing
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Unique
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Mean
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Std
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Min
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Q25
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Median
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Q75
              </th>
              <th className="border border-gray-300 px-3 py-2 text-right font-semibold">
                Max
              </th>
            </tr>
          </thead>
          <tbody>
            {(columns || []).map((col: any, idx: number) => (
              <tr
                key={col.name}
                className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
              >
                <td className="border border-gray-300 px-3 py-2 font-medium">
                  {col.name}
                </td>
                <td className="border border-gray-300 px-3 py-2">
                  <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">
                    {col.dtype}
                  </code>
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right">
                  {col.count}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right">
                  <span
                    className={
                      col.missing > 0
                        ? "text-red-600 font-medium"
                        : "text-green-600"
                    }
                  >
                    {col.missing}
                  </span>
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right">
                  {col.unique}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                  {col.mean !== undefined ? col.mean.toFixed(2) : "-"}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                  {col.std !== undefined ? col.std.toFixed(2) : "-"}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                  {col.min !== undefined ? col.min.toFixed(2) : "-"}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                  {col.q25 !== undefined ? col.q25.toFixed(2) : "-"}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                  {col.median !== undefined ? col.median.toFixed(2) : "-"}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                  {col.q75 !== undefined ? col.q75.toFixed(2) : "-"}
                </td>
                <td className="border border-gray-300 px-3 py-2 text-right font-mono">
                  {col.max !== undefined ? col.max.toFixed(2) : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Educational note */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
        <h4 className="text-xs font-semibold text-gray-700 mb-1">
          Understanding Summary Statistics
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs text-gray-600">
          <div>
            <strong>Count:</strong> Non-null values
          </div>
          <div>
            <strong>Mean:</strong> Average value
          </div>
          <div>
            <strong>Std:</strong> Standard deviation (spread)
          </div>
          <div>
            <strong>Q25/Median/Q75:</strong> Quartile boundaries
          </div>
          <div>
            <strong>Missing:</strong> Null/NaN values
          </div>
          <div>
            <strong>Unique:</strong> Distinct values
          </div>
        </div>
      </div>
    </div>
  );
}
