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
  chartType: string;
  chartData: any;
}

export const ChartViewer = ({ chartType, chartData }: ChartViewerProps) => {
  // Handle error in chart data
  if (chartData.error) {
    return (
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-yellow-800 font-semibold">Configuration needed:</p>
        <p className="text-yellow-600 text-sm mt-2">{chartData.error}</p>
      </div>
    );
  }

  // Check if we have valid data
  if (!chartData.datasets && !chartData.labels) {
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
    if (chartType === "pie") {
      return {
        ...chartData,
        datasets: chartData.datasets.map((dataset: any, idx: number) => ({
          ...dataset,
          backgroundColor: colorPalette,
          borderColor: borderColorPalette,
          borderWidth: 1,
        })),
      };
    } else if (chartType === "scatter") {
      return {
        ...chartData,
        datasets: chartData.datasets.map((dataset: any, idx: number) => ({
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
        ...chartData,
        datasets: chartData.datasets.map((dataset: any, idx: number) => ({
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
          text: `${chartType.charAt(0).toUpperCase() + chartType.slice(1)} Chart`,
          font: {
            size: 16,
            weight: "bold",
          },
        },
      },
    };

    if (chartType === "scatter") {
      return {
        ...baseOptions,
        scales: {
          x: {
            type: "linear" as const,
            position: "bottom" as const,
            title: {
              display: true,
              text: "X Axis",
            },
          },
          y: {
            title: {
              display: true,
              text: "Y Axis",
            },
          },
        },
      };
    }

    if (chartType === "pie") {
      return baseOptions;
    }

    // For bar, line, histogram
    return {
      ...baseOptions,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Value",
          },
        },
        x: {
          title: {
            display: true,
            text: "Category",
          },
        },
      },
    };
  };

  const data = prepareChartData();
  const options = getChartOptions();

  return (
    <div className="w-full h-full flex items-center justify-center p-4">
      <div className="w-full max-w-4xl" style={{ height: "500px" }}>
        {chartType === "bar" && <Bar data={data} options={options} />}
        {chartType === "line" && <Line data={data} options={options} />}
        {chartType === "scatter" && <Scatter data={data} options={options} />}
        {chartType === "histogram" && <Bar data={data} options={options} />}
        {chartType === "pie" && (
          <div className="flex items-center justify-center h-full">
            <div style={{ width: "400px", height: "400px" }}>
              <Pie data={data} options={options} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
