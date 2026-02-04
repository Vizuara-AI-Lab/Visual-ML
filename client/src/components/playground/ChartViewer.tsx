/**
 * ChartViewer Component - Render various chart types using Chart.js
 */

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Line, Scatter, Pie } from "react-chartjs-2";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
);

interface ChartViewerProps {
  result?: any;
  chartType?: string;
  chartData?: any;
}

export const ChartViewer = ({
  result,
  chartType,
  chartData,
}: ChartViewerProps) => {
  // Extract from result if provided (for view modal)
  const actualChartType = result?.chart_type || chartType;
  const actualChartData = result?.chart_data || chartData;

  // Handle error in chart data
  if (actualChartData?.error) {
    return (
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-yellow-800 font-semibold">Configuration needed:</p>
        <p className="text-yellow-600 text-sm mt-2">{actualChartData.error}</p>
      </div>
    );
  }

  // Check if we have valid data
  if (!actualChartData?.datasets && !actualChartData?.labels) {
    return (
      <div className="text-center py-8 text-gray-500">
        No data available for visualization
      </div>
    );
  }

  // Color palettes for charts
  const colorPalette = [
    "rgba(59, 130, 246, 0.8)", // blue
    "rgba(16, 185, 129, 0.8)", // green
    "rgba(239, 68, 68, 0.8)", // red
    "rgba(245, 158, 11, 0.8)", // amber
    "rgba(139, 92, 246, 0.8)", // purple
    "rgba(236, 72, 153, 0.8)", // pink
    "rgba(14, 165, 233, 0.8)", // sky
    "rgba(249, 115, 22, 0.8)", // orange
    "rgba(168, 85, 247, 0.8)", // violet
    "rgba(34, 197, 94, 0.8)", // emerald
  ];

  const borderColorPalette = [
    "rgba(59, 130, 246, 1)",
    "rgba(16, 185, 129, 1)",
    "rgba(239, 68, 68, 1)",
    "rgba(245, 158, 11, 1)",
    "rgba(139, 92, 246, 1)",
    "rgba(236, 72, 153, 1)",
    "rgba(14, 165, 233, 1)",
    "rgba(249, 115, 22, 1)",
    "rgba(168, 85, 247, 1)",
    "rgba(34, 197, 94, 1)",
  ];

  // Prepare data based on chart type
  const prepareChartData = () => {
    if (actualChartType === "pie") {
      return {
        ...actualChartData,
        datasets: actualChartData.datasets.map((dataset: any, idx: number) => ({
          ...dataset,
          backgroundColor: colorPalette,
          borderColor: borderColorPalette,
          borderWidth: 1,
        })),
      };
    } else if (actualChartType === "scatter") {
      return {
        ...actualChartData,
        datasets: actualChartData.datasets.map((dataset: any, idx: number) => ({
          ...dataset,
          backgroundColor: colorPalette[idx % colorPalette.length],
          borderColor: borderColorPalette[idx % borderColorPalette.length],
          pointRadius: 5,
          pointHoverRadius: 7,
        })),
      };
    } else {
      // Bar, Line, Histogram
      return {
        ...actualChartData,
        datasets: actualChartData.datasets.map((dataset: any, idx: number) => ({
          ...dataset,
          backgroundColor: colorPalette[idx % colorPalette.length],
          borderColor: borderColorPalette[idx % borderColorPalette.length],
          borderWidth: 2,
        })),
      };
    }
  };

  // Chart options
  const getChartOptions = () => {
    // Get column names from chart data
    const xColumnName =
      actualChartData?.x_column || actualChartData?.column_name || "Category";
    const yColumnName = actualChartData?.y_columns?.[0] || "Value";

    const baseOptions = {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: "top" as const,
          labels: {
            font: {
              size: 12,
            },
          },
        },
        title: {
          display: true,
          text: `${actualChartType.charAt(0).toUpperCase() + actualChartType.slice(1)} Chart`,
          font: {
            size: 16,
            weight: "bold",
          },
        },
      },
    };

    if (actualChartType === "scatter") {
      return {
        ...baseOptions,
        scales: {
          x: {
            type: "linear" as const,
            position: "bottom" as const,
            title: {
              display: true,
              text: xColumnName,
            },
          },
          y: {
            title: {
              display: true,
              text: yColumnName,
            },
          },
        },
      };
    }

    if (actualChartType === "pie") {
      return {
        ...baseOptions,
        plugins: {
          ...baseOptions.plugins,
          legend: {
            position: "right" as const,
            labels: {
              font: {
                size: 11,
              },
              boxWidth: 15,
            },
          },
          tooltip: {
            callbacks: {
              label: function (context: any) {
                return context.label || "";
              },
            },
          },
        },
      };
    }

    // For bar, line, histogram
    return {
      ...baseOptions,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: yColumnName,
          },
        },
        x: {
          title: {
            display: true,
            text: xColumnName,
          },
        },
      },
    };
  };

  const data = prepareChartData();
  const options = getChartOptions();

  return (
    <div className="w-full h-full flex items-center justify-center p-4">
      <div className="w-full max-w-4xl space-y-4">
        {/* Pie chart info banner */}
        {actualChartType === "pie" && actualChartData?.column_name && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="flex items-center justify-between text-sm">
              <div>
                <span className="font-semibold text-blue-900">Column: </span>
                <span className="text-blue-700">
                  {actualChartData.column_name}
                </span>
              </div>
              <div>
                <span className="font-semibold text-blue-900">
                  Total Count:{" "}
                </span>
                <span className="text-blue-700">
                  {actualChartData.total_count?.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="font-semibold text-blue-900">
                  Categories:{" "}
                </span>
                <span className="text-blue-700">
                  {actualChartData.categories_shown}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Chart container */}
        <div style={{ height: "500px" }}>
          {actualChartType === "bar" && <Bar data={data} options={options} />}
          {actualChartType === "line" && <Line data={data} options={options} />}
          {actualChartType === "scatter" && (
            <Scatter data={data} options={options} />
          )}
          {actualChartType === "histogram" && (
            <Bar data={data} options={options} />
          )}
          {actualChartType === "pie" && (
            <div className="flex items-center justify-center h-full">
              <div style={{ width: "450px", height: "450px" }}>
                <Pie data={data} options={options} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
