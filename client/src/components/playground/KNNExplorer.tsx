/**
 * KNNExplorer - Interactive learning activities for K-Nearest Neighbors
 * Tabbed component: Results | K Selection | Decision Boundary | Nearest Neighbors | Metrics Explained | Quiz | How It Works
 */

import { useState, useMemo, useCallback } from "react";
import {
  ClipboardList,
  SlidersHorizontal,
  Grid3X3,

  Gauge,
  HelpCircle,
  Cog,
  CheckCircle,
  XCircle,
  Trophy,
  Timer,
  Hash,
  Database,
  RotateCcw,

  Diamond,
  CircleDot,
} from "lucide-react";

interface KNNExplorerProps {
  result: any;
}

type ExplorerTab =
  | "results"
  | "k_selection"
  | "decision_boundary"

  | "metric_explainer"
  | "quiz"
  | "how_it_works";

export const KNNExplorer = ({ result }: KNNExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const isRegression = result.task_type === "regression";
  const metrics = result.training_metrics || {};

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "k_selection",
      label: "K Selection",
      icon: SlidersHorizontal,
      available: !!result.k_sweep_data?.length,
    },
    {
      id: "decision_boundary",
      label: "Decision Boundary",
      icon: Grid3X3,
      available: !isRegression && !!result.decision_boundary_grid,
    },

    {
      id: "metric_explainer",
      label: "Metrics Explained",
      icon: Gauge,
      available: !!result.metric_explainer?.metrics?.length,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available: !!result.quiz_questions && result.quiz_questions.length > 0,
    },
    {
      id: "how_it_works",
      label: "How It Works",
      icon: Cog,
      available: true,
    },
  ];

  return (
    <div className="space-y-4">
      {/* Tab navigation */}
      <div className="flex border-b border-gray-200 overflow-x-auto">
        {tabs
          .filter((t) => t.available)
          .map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-teal-500 text-teal-700"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
      </div>

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === "results" && (
          <ResultsTab result={result} isRegression={isRegression} metrics={metrics} />
        )}
        {activeTab === "k_selection" && (
          <KSelectionTab data={result.k_sweep_data} />
        )}
        {activeTab === "decision_boundary" && (
          <DecisionBoundaryTab data={result.decision_boundary_grid} />
        )}

        {activeTab === "metric_explainer" && (
          <MetricExplainerTab data={result.metric_explainer} />
        )}
        {activeTab === "quiz" && (
          <QuizTab questions={result.quiz_questions || []} />
        )}
        {activeTab === "how_it_works" && (
          <HowItWorksTab
            nNeighbors={result.n_neighbors}
            sampleData={result.sample_neighbors}
          />
        )}
      </div>
    </div>
  );
};

// ======================== Results Tab ========================

function ResultsTab({
  result,
  isRegression,
  metrics,
}: {
  result: any;
  isRegression: boolean;
  metrics: any;
}) {
  const primaryMetric = isRegression ? metrics.r2 : metrics.accuracy;
  const primaryLabel = isRegression ? "R\u00B2 Score" : "Accuracy";
  const primaryPct = primaryMetric != null ? primaryMetric * 100 : null;

  const confusionMatrix = !isRegression ? metrics.confusion_matrix : null;
  const predictions = isRegression ? result.regression_predictions : null;

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 border border-teal-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-teal-500 rounded-full flex items-center justify-center">
            <CheckCircle className="text-white w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-teal-900">
              KNN Training Complete
            </h3>
            <p className="text-sm text-teal-700">
              K={result.n_neighbors} model trained on{" "}
              {result.training_samples?.toLocaleString()} samples with{" "}
              {result.n_features} features ({isRegression ? "regression" : "classification"})
            </p>
          </div>
        </div>
      </div>

      {/* Hero Metric Ring Gauge */}
      {primaryPct != null && (
        <div className="flex justify-center">
          <div className="relative w-36 h-36">
            <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#e5e7eb" strokeWidth="8" />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke={primaryPct >= 80 ? "#14b8a6" : primaryPct >= 60 ? "#f59e0b" : "#ef4444"}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${(primaryPct / 100) * 327} 327`}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-bold text-gray-900">
                {primaryPct.toFixed(1)}%
              </span>
              <span className="text-xs text-gray-500">{primaryLabel}</span>
            </div>
          </div>
        </div>
      )}

      {/* Summary Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Hash className="w-3.5 h-3.5" /> K Value
          </div>
          <div className="text-lg font-semibold text-gray-900">{result.n_neighbors}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <SlidersHorizontal className="w-3.5 h-3.5" /> Weights
          </div>
          <div className="text-lg font-semibold text-gray-900 capitalize">
            {result.weights || "uniform"}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <CircleDot className="w-3.5 h-3.5" /> Distance
          </div>
          <div className="text-lg font-semibold text-gray-900 capitalize">
            {result.distance_metric || "minkowski"}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Timer className="w-3.5 h-3.5" /> Training Time
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {result.training_time_seconds?.toFixed(2)}s
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <ClipboardList className="w-3.5 h-3.5" /> Task Type
          </div>
          <div className="text-lg font-semibold text-gray-900 capitalize">
            {result.task_type}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Database className="w-3.5 h-3.5" /> Samples
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {result.training_samples?.toLocaleString()}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Hash className="w-3.5 h-3.5" /> Features
          </div>
          <div className="text-lg font-semibold text-gray-900">{result.n_features}</div>
        </div>
      </div>

      {/* All Metrics */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <h4 className="text-sm font-semibold text-gray-900">
            {result.metadata?.evaluated_on === "test" ? "Test Metrics" : "Training Metrics"}
          </h4>
          {result.metadata?.evaluated_on === "test" ? (
            <span className="text-[10px] font-medium bg-teal-100 text-teal-700 px-2 py-0.5 rounded-full">
              on unseen data
            </span>
          ) : (
            <span className="text-[10px] font-medium bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
              on training data
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(metrics)
            .filter(([key, value]) => typeof value === "number" && key !== "confusion_matrix")
            .map(([key, value]) => (
              <div key={key} className="bg-gray-50 rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  {key.replace(/_/g, " ")}
                </div>
                <div className="text-xl font-bold text-gray-900">
                  {(value as number).toFixed(4)}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Confusion Matrix (classification only) */}
      {confusionMatrix && Array.isArray(confusionMatrix) && confusionMatrix.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Confusion Matrix</h4>
          <div className="flex justify-center">
            <ConfusionMatrixGrid matrix={confusionMatrix} labels={result.class_labels} />
          </div>
        </div>
      )}

      {/* Regression: Actual vs Predicted Scatter */}
      {isRegression && predictions && Array.isArray(predictions) && predictions.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Actual vs Predicted</h4>
          <ActualVsPredictedScatter predictions={predictions} />
        </div>
      )}

      {/* Model Details */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Model Details</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Task Type:</span>
            <span className="font-semibold capitalize">{result.task_type}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Target Column:</span>
            <span className="font-semibold">{result.target_column || "N/A"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">K Neighbors:</span>
            <span className="font-semibold">{result.n_neighbors}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Weights:</span>
            <span className="font-semibold capitalize">{result.weights || "uniform"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Distance Metric:</span>
            <span className="font-semibold capitalize">{result.distance_metric || "minkowski"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Model ID:</span>
            <code className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">
              {result.model_id?.slice(0, 20)}...
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}

// ======================== Confusion Matrix Grid ========================

function ConfusionMatrixGrid({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels?: string[];
}) {
  const maxVal = useMemo(() => {
    let max = 0;
    for (const row of matrix) {
      for (const val of row) {
        if (val > max) max = val;
      }
    }
    return max || 1;
  }, [matrix]);

  const n = matrix.length;
  const cellSize = Math.min(48, Math.floor(240 / n));

  return (
    <div className="inline-block">
      <div className="text-xs text-gray-500 text-center mb-1">Predicted</div>
      <div className="flex">
        <div className="flex flex-col justify-center mr-1">
          <span
            className="text-xs text-gray-500"
            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          >
            Actual
          </span>
        </div>
        <div>
          {/* Header row */}
          {labels && (
            <div className="flex" style={{ marginLeft: cellSize }}>
              {labels.map((label, i) => (
                <div
                  key={i}
                  className="text-[10px] text-gray-500 text-center truncate"
                  style={{ width: cellSize }}
                  title={label}
                >
                  {label.length > 4 ? label.slice(0, 4) : label}
                </div>
              ))}
            </div>
          )}
          {matrix.map((row, ri) => (
            <div key={ri} className="flex items-center">
              {labels && (
                <div
                  className="text-[10px] text-gray-500 text-right pr-1 truncate"
                  style={{ width: cellSize }}
                  title={labels[ri]}
                >
                  {labels[ri]?.length > 4 ? labels[ri].slice(0, 4) : labels[ri]}
                </div>
              )}
              {row.map((val, ci) => {
                const intensity = val / maxVal;
                const isDiag = ri === ci;
                const bgColor = isDiag
                  ? `rgba(20, 184, 166, ${0.15 + intensity * 0.7})`
                  : `rgba(239, 68, 68, ${intensity * 0.5})`;
                return (
                  <div
                    key={ci}
                    className="flex items-center justify-center border border-white text-xs font-semibold"
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: bgColor,
                      color: intensity > 0.5 ? "#fff" : "#374151",
                    }}
                    title={`Actual: ${labels?.[ri] || ri}, Predicted: ${labels?.[ci] || ci}, Count: ${val}`}
                  >
                    {val}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ======================== Actual vs Predicted Scatter ========================

function ActualVsPredictedScatter({
  predictions,
}: {
  predictions: { actual: number; predicted: number }[];
}) {
  const { minVal, maxVal, scaleX, scaleY } = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;
    for (const p of predictions) {
      if (p.actual < min) min = p.actual;
      if (p.actual > max) max = p.actual;
      if (p.predicted < min) min = p.predicted;
      if (p.predicted > max) max = p.predicted;
    }
    const padding = (max - min) * 0.1 || 1;
    const lo = min - padding;
    const hi = max + padding;
    const sx = (v: number) => 50 + ((v - lo) / (hi - lo)) * 400;
    const sy = (v: number) => 270 - ((v - lo) / (hi - lo)) * 240;
    return { minVal: lo, maxVal: hi, scaleX: sx, scaleY: sy };
  }, [predictions]);

  return (
    <svg viewBox="0 0 500 300" className="w-full max-w-lg mx-auto">
      {/* Axes */}
      <line x1="50" y1="270" x2="450" y2="270" stroke="#d1d5db" strokeWidth="1" />
      <line x1="50" y1="30" x2="50" y2="270" stroke="#d1d5db" strokeWidth="1" />
      {/* Perfect prediction line */}
      <line
        x1={scaleX(minVal)}
        y1={scaleY(minVal)}
        x2={scaleX(maxVal)}
        y2={scaleY(maxVal)}
        stroke="#d1d5db"
        strokeWidth="1"
        strokeDasharray="4 4"
      />
      {/* Points */}
      {predictions.slice(0, 200).map((p, i) => (
        <circle
          key={i}
          cx={scaleX(p.actual)}
          cy={scaleY(p.predicted)}
          r="3"
          fill="#14b8a6"
          fillOpacity="0.6"
          stroke="#0d9488"
          strokeWidth="0.5"
        />
      ))}
      {/* Labels */}
      <text x="250" y="295" textAnchor="middle" className="text-[11px]" fill="#6b7280">
        Actual
      </text>
      <text
        x="15"
        y="150"
        textAnchor="middle"
        transform="rotate(-90, 15, 150)"
        className="text-[11px]"
        fill="#6b7280"
      >
        Predicted
      </text>
    </svg>
  );
}

// ======================== K Selection Tab ========================

function KSelectionTab({ data }: { data: any[] | undefined }) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-400">
        <SlidersHorizontal className="w-12 h-12 mb-3" />
        <p className="text-sm">Run pipeline to see K selection analysis</p>
      </div>
    );
  }

  const kValues = data.map((d: any) => d.k);
  const trainScores = data.map((d: any) => d.train_score);
  const testScores = data.map((d: any) => d.test_score);
  const allScores = [...trainScores, ...testScores].filter((s) => s != null);
  const minScore = Math.min(...allScores);
  const maxScore = Math.max(...allScores);
  const scorePad = (maxScore - minScore) * 0.1 || 0.05;
  const yLo = Math.max(0, minScore - scorePad);
  const yHi = Math.min(1, maxScore + scorePad);

  const kMin = Math.min(...kValues);
  const kMax = Math.max(...kValues);

  const chartLeft = 60;
  const chartRight = 460;
  const chartTop = 30;
  const chartBottom = 260;
  const chartW = chartRight - chartLeft;
  const chartH = chartBottom - chartTop;

  const sx = (k: number) =>
    chartLeft + ((k - kMin) / (kMax - kMin || 1)) * chartW;
  const sy = (s: number) =>
    chartBottom - ((s - yLo) / (yHi - yLo || 1)) * chartH;

  const buildPath = (scores: number[]) => {
    return scores
      .map((s, i) => {
        const x = sx(kValues[i]);
        const y = sy(s);
        return `${i === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ");
  };

  const trainPath = buildPath(trainScores);
  const testPath = buildPath(testScores);

  const currentK = data.find((d: any) => d.is_current_k);
  const bestK = data.find((d: any) => d.is_best_k);

  // Overfitting regions: where train_score - test_score > 0.05
  const overfitThreshold = 0.05;
  const overfitRegions: string[] = [];
  for (let i = 0; i < data.length; i++) {
    const gap = trainScores[i] - testScores[i];
    if (gap > overfitThreshold) {
      const x = sx(kValues[i]);
      const yTop = sy(trainScores[i]);
      const yBot = sy(testScores[i]);
      overfitRegions.push(`${x},${yTop},${x},${yBot}`);
    }
  }

  // Build overfit shading polygon segments
  const overfitPolygon = useMemo(() => {
    const segments: { startIdx: number; endIdx: number }[] = [];
    let inRegion = false;
    let start = 0;
    for (let i = 0; i < data.length; i++) {
      const gap = trainScores[i] - testScores[i];
      if (gap > overfitThreshold && !inRegion) {
        inRegion = true;
        start = i;
      } else if ((gap <= overfitThreshold || i === data.length - 1) && inRegion) {
        segments.push({ startIdx: start, endIdx: gap > overfitThreshold ? i : i - 1 });
        inRegion = false;
      }
    }
    return segments.map((seg) => {
      const topPoints = [];
      const bottomPoints = [];
      for (let i = seg.startIdx; i <= seg.endIdx; i++) {
        topPoints.push(`${sx(kValues[i])},${sy(trainScores[i])}`);
        bottomPoints.unshift(`${sx(kValues[i])},${sy(testScores[i])}`);
      }
      return [...topPoints, ...bottomPoints].join(" ");
    });
  }, [data]);

  // Y-axis ticks
  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const step = (yHi - yLo) / 5;
    for (let i = 0; i <= 5; i++) {
      ticks.push(yLo + step * i);
    }
    return ticks;
  }, [yLo, yHi]);

  const hoverData = hoverIdx != null ? data[hoverIdx] : null;

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
        <SlidersHorizontal className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />
        <p className="text-sm text-teal-800">
          <span className="font-semibold">K Selection Analysis</span> shows how
          different values of K affect model performance. Lower K values fit the
          training data more closely (risk of overfitting), while higher K values
          create smoother decision boundaries.
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <svg viewBox="0 0 500 300" className="w-full">
          {/* Grid lines */}
          {yTicks.map((tick, i) => (
            <g key={i}>
              <line
                x1={chartLeft}
                y1={sy(tick)}
                x2={chartRight}
                y2={sy(tick)}
                stroke="#f3f4f6"
                strokeWidth="1"
              />
              <text
                x={chartLeft - 8}
                y={sy(tick) + 4}
                textAnchor="end"
                fill="#9ca3af"
                fontSize="10"
              >
                {tick.toFixed(2)}
              </text>
            </g>
          ))}

          {/* Overfit shading */}
          {overfitPolygon.map((points, i) => (
            <polygon
              key={i}
              points={points}
              fill="#ef4444"
              fillOpacity="0.1"
              stroke="none"
            />
          ))}

          {/* Axes */}
          <line
            x1={chartLeft}
            y1={chartBottom}
            x2={chartRight}
            y2={chartBottom}
            stroke="#d1d5db"
            strokeWidth="1"
          />
          <line
            x1={chartLeft}
            y1={chartTop}
            x2={chartLeft}
            y2={chartBottom}
            stroke="#d1d5db"
            strokeWidth="1"
          />

          {/* Train score line (dashed gray) */}
          <path
            d={trainPath}
            fill="none"
            stroke="#9ca3af"
            strokeWidth="2"
            strokeDasharray="6 3"
          />

          {/* Test score line (solid teal) */}
          <path d={testPath} fill="none" stroke="#14b8a6" strokeWidth="2" />

          {/* Current K vertical line */}
          {currentK && (
            <g>
              <line
                x1={sx(currentK.k)}
                y1={chartTop}
                x2={sx(currentK.k)}
                y2={chartBottom}
                stroke="#6366f1"
                strokeWidth="1.5"
                strokeDasharray="4 4"
              />
              <text
                x={sx(currentK.k)}
                y={chartTop - 5}
                textAnchor="middle"
                fill="#6366f1"
                fontSize="10"
                fontWeight="600"
              >
                Current K={currentK.k}
              </text>
            </g>
          )}

          {/* Best K star */}
          {bestK && (
            <g>
              <polygon
                points={starPoints(sx(bestK.k), sy(bestK.test_score), 8)}
                fill="#f59e0b"
                stroke="#d97706"
                strokeWidth="1"
              />
              <text
                x={sx(bestK.k) + 12}
                y={sy(bestK.test_score) - 4}
                fill="#d97706"
                fontSize="10"
                fontWeight="600"
              >
                Best K={bestK.k}
              </text>
            </g>
          )}

          {/* Clickable data points - train */}
          {data.map((d: any, i: number) => (
            <circle
              key={`train-${i}`}
              cx={sx(d.k)}
              cy={sy(d.train_score)}
              r={hoverIdx === i ? 5 : 3}
              fill="#9ca3af"
              stroke="#fff"
              strokeWidth="1"
              className="cursor-pointer"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
            />
          ))}

          {/* Clickable data points - test */}
          {data.map((d: any, i: number) => (
            <circle
              key={`test-${i}`}
              cx={sx(d.k)}
              cy={sy(d.test_score)}
              r={hoverIdx === i ? 5 : 3}
              fill="#14b8a6"
              stroke="#fff"
              strokeWidth="1"
              className="cursor-pointer"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
            />
          ))}

          {/* Hover tooltip */}
          {hoverData && hoverIdx != null && (
            <g>
              <rect
                x={sx(hoverData.k) - 50}
                y={sy(hoverData.test_score) - 50}
                width="100"
                height="40"
                rx="4"
                fill="white"
                stroke="#d1d5db"
                strokeWidth="1"
              />
              <text
                x={sx(hoverData.k)}
                y={sy(hoverData.test_score) - 34}
                textAnchor="middle"
                fill="#374151"
                fontSize="10"
                fontWeight="600"
              >
                K = {hoverData.k}
              </text>
              <text
                x={sx(hoverData.k)}
                y={sy(hoverData.test_score) - 20}
                textAnchor="middle"
                fill="#6b7280"
                fontSize="9"
              >
                Train: {hoverData.train_score?.toFixed(3)} | Test: {hoverData.test_score?.toFixed(3)}
              </text>
            </g>
          )}

          {/* X-axis labels */}
          {data.map((d: any, i: number) => {
            // Show every label if few, else every other
            if (data.length > 15 && i % 2 !== 0) return null;
            return (
              <text
                key={`xlabel-${i}`}
                x={sx(d.k)}
                y={chartBottom + 15}
                textAnchor="middle"
                fill="#9ca3af"
                fontSize="10"
              >
                {d.k}
              </text>
            );
          })}

          {/* Axis titles */}
          <text x={250} y={295} textAnchor="middle" fill="#6b7280" fontSize="11">
            K (Number of Neighbors)
          </text>
          <text
            x="15"
            y="150"
            textAnchor="middle"
            transform="rotate(-90, 15, 150)"
            fill="#6b7280"
            fontSize="11"
          >
            Score
          </text>
        </svg>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-2">
          <div className="flex items-center gap-2">
            <svg width="20" height="3">
              <line
                x1="0"
                y1="1.5"
                x2="20"
                y2="1.5"
                stroke="#9ca3af"
                strokeWidth="2"
                strokeDasharray="4 2"
              />
            </svg>
            <span className="text-xs text-gray-500">Train Score</span>
          </div>
          <div className="flex items-center gap-2">
            <svg width="20" height="3">
              <line x1="0" y1="1.5" x2="20" y2="1.5" stroke="#14b8a6" strokeWidth="2" />
            </svg>
            <span className="text-xs text-gray-500">Test Score</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-100 border border-red-300 rounded-sm" />
            <span className="text-xs text-gray-500">Overfitting Region</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Star points helper for K Selection
function starPoints(cx: number, cy: number, r: number): string {
  const points: string[] = [];
  for (let i = 0; i < 5; i++) {
    const outerAngle = (Math.PI / 2) * -1 + (2 * Math.PI * i) / 5;
    const innerAngle = outerAngle + Math.PI / 5;
    points.push(`${cx + r * Math.cos(outerAngle)},${cy + r * Math.sin(outerAngle)}`);
    points.push(
      `${cx + (r * 0.4) * Math.cos(innerAngle)},${cy + (r * 0.4) * Math.sin(innerAngle)}`
    );
  }
  return points.join(" ");
}

// ======================== Decision Boundary Tab ========================

function DecisionBoundaryTab({ data }: { data: any }) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-400">
        <Grid3X3 className="w-12 h-12 mb-3" />
        <p className="text-sm">Decision boundary data not available</p>
      </div>
    );
  }

  const rawGrid = data.grid;
  const points = data.points || [];
  const classes = data.classes || [];
  const pcaVariance = data.pca_explained_variance || data.pca_variance || [0, 0];

  // Backend sends grid as compact object {x_range, y_range, nx, ny, predictions, confidence}
  // Expand into array of {x, y, prediction, confidence}
  const { grid, xRange, yRange, nx, ny } = useMemo(() => {
    if (!rawGrid) return { grid: [], xRange: [0, 1] as [number, number], yRange: [0, 1] as [number, number], nx: 50, ny: 50 };
    if (Array.isArray(rawGrid)) {
      // Already an array (legacy format)
      const xs = rawGrid.map((g: any) => g.x);
      const ys = rawGrid.map((g: any) => g.y);
      return {
        grid: rawGrid,
        xRange: [Math.min(...xs), Math.max(...xs)] as [number, number],
        yRange: [Math.min(...ys), Math.max(...ys)] as [number, number],
        nx: Math.round(Math.sqrt(rawGrid.length)) || 50,
        ny: Math.round(Math.sqrt(rawGrid.length)) || 50,
      };
    }
    const { x_range, y_range, nx: gnx, ny: gny, predictions, confidence } = rawGrid;
    const [xMin, xMax] = x_range || [0, 1];
    const [yMin, yMax] = y_range || [0, 1];
    const cells: any[] = [];
    for (let j = 0; j < gny; j++) {
      for (let i = 0; i < gnx; i++) {
        const idx = j * gnx + i;
        cells.push({
          x: xMin + (i / (gnx - 1)) * (xMax - xMin),
          y: yMin + (j / (gny - 1)) * (yMax - yMin),
          prediction: predictions[idx],
          confidence: confidence ? confidence[idx] : 0.5,
        });
      }
    }
    return { grid: cells, xRange: [xMin, xMax] as [number, number], yRange: [yMin, yMax] as [number, number], nx: gnx, ny: gny };
  }, [rawGrid]);

  const classColors = useMemo(() => {
    const palette = [
      "#14b8a6", "#6366f1", "#f59e0b", "#ef4444", "#8b5cf6",
      "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#64748b",
    ];
    const map: Record<string, string> = {};
    classes.forEach((cls: string, i: number) => {
      map[cls] = palette[i % palette.length];
    });
    return map;
  }, [classes]);

  const chartLeft = 50;
  const chartRight = 450;
  const chartTop = 20;
  const chartBottom = 280;
  const chartW = chartRight - chartLeft;
  const chartH = chartBottom - chartTop;

  const sx = (x: number) => chartLeft + ((x - xRange[0]) / (xRange[1] - xRange[0] || 1)) * chartW;
  const sy = (y: number) => chartBottom - ((y - yRange[0]) / (yRange[1] - yRange[0] || 1)) * chartH;

  const cellW = chartW / nx;
  const cellH = chartH / ny;

  if (!grid.length && !points.length) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-400">
        <Grid3X3 className="w-12 h-12 mb-3" />
        <p className="text-sm">No decision boundary data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
        <Grid3X3 className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />
        <p className="text-sm text-teal-800">
          <span className="font-semibold">Decision Boundary</span> visualizes how
          KNN partitions the feature space. Each region is colored by the predicted
          class. The data is projected to 2D using PCA. Brighter colors indicate
          higher confidence.
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <svg viewBox="0 0 500 320" className="w-full">
          {/* Grid cells */}
          {grid.map((cell: any, i: number) => {
            const color = classColors[String(cell.prediction)] || "#d1d5db";
            const opacity = 0.2 + (cell.confidence || 0.5) * 0.6;
            return (
              <rect
                key={i}
                x={sx(cell.x) - cellW / 2}
                y={sy(cell.y) - cellH / 2}
                width={cellW + 0.5}
                height={cellH + 0.5}
                fill={color}
                fillOpacity={opacity}
              />
            );
          })}

          {/* Data points */}
          {points.map((pt: any, i: number) => {
            const color = classColors[String(pt.label)] || "#374151";
            return (
              <circle
                key={i}
                cx={sx(pt.x)}
                cy={sy(pt.y)}
                r="3.5"
                fill={color}
                stroke="#fff"
                strokeWidth="1"
              />
            );
          })}

          {/* Axes */}
          <line x1={chartLeft} y1={chartBottom} x2={chartRight} y2={chartBottom} stroke="#d1d5db" strokeWidth="1" />
          <line x1={chartLeft} y1={chartTop} x2={chartLeft} y2={chartBottom} stroke="#d1d5db" strokeWidth="1" />

          {/* Axis labels */}
          <text x={250} y={305} textAnchor="middle" fill="#6b7280" fontSize="11">
            PC1 ({(pcaVariance[0] * 100).toFixed(1)}% variance)
          </text>
          <text
            x="12"
            y="150"
            textAnchor="middle"
            transform="rotate(-90, 12, 150)"
            fill="#6b7280"
            fontSize="11"
          >
            PC2 ({(pcaVariance[1] * 100).toFixed(1)}% variance)
          </text>
        </svg>

        {/* Class Legend */}
        <div className="flex flex-wrap items-center justify-center gap-4 mt-3">
          {classes.map((cls: string) => (
            <div key={cls} className="flex items-center gap-1.5">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: classColors[cls] }}
              />
              <span className="text-xs text-gray-600">{cls}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


function diamondPoints(cx: number, cy: number, r: number): string {
  return `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
}

// ======================== Metric Explainer Tab ========================

function MetricExplainerTab({ data }: { data: any }) {
  const metrics = data?.metrics || [];
  const colorMap: Record<string, string> = {
    green: "bg-green-50 border-green-200",
    blue: "bg-blue-50 border-blue-200",
    orange: "bg-orange-50 border-orange-200",
    purple: "bg-purple-50 border-purple-200",
    yellow: "bg-yellow-50 border-yellow-200",
    red: "bg-red-50 border-red-200",
  };
  const barColorMap: Record<string, string> = {
    green: "bg-green-500",
    blue: "bg-blue-500",
    orange: "bg-orange-500",
    purple: "bg-purple-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-purple-200 bg-purple-50 px-4 py-3">
        <Gauge className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
        <div className="text-sm text-purple-800">
          <p>
            <span className="font-semibold">Understanding your metrics</span> — Each
            metric tells a different part of the story about your KNN model's
            performance.
          </p>
          {data?.evaluated_on && (
            <p className="mt-1 text-xs font-medium">
              {data.evaluated_on === "test"
                ? "Evaluated on test data (unseen by the model during training)"
                : "Evaluated on training data \u2014 these may look inflated since the model has seen this data before"}
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {metrics.map((m: any) => (
          <div
            key={m.metric}
            className={`rounded-lg border p-4 space-y-2 ${
              colorMap[m.color] || "bg-gray-50 border-gray-200"
            }`}
          >
            <div className="flex justify-between items-center">
              <h4 className="font-semibold text-gray-900">{m.metric}</h4>
              {m.value_pct != null ? (
                <span className="text-lg font-bold text-gray-900">{m.value_pct}%</span>
              ) : (
                <span className="text-lg font-bold text-gray-900">{m.value}</span>
              )}
            </div>
            {m.value_pct != null && (
              <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
                <div
                  className={`h-full rounded-full ${barColorMap[m.color] || "bg-gray-500"}`}
                  style={{ width: `${Math.min(m.value_pct, 100)}%` }}
                />
              </div>
            )}
            <p className="text-sm text-gray-700">{m.analogy}</p>
            <p className="text-xs text-gray-500 italic">{m.when_useful}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ======================== Quiz Tab ========================

function QuizTab({ questions }: { questions: any[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  if (!questions.length) return null;

  const question = questions[currentQ];
  const isCorrect = selectedAnswer === question.correct_answer;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelectedAnswer(idx);
    setAnswered(true);
    const correct = idx === question.correct_answer;
    if (correct) setScore((s) => s + 1);
    setAnswers((a) => [...a, idx]);
  };

  const handleNext = () => {
    if (currentQ + 1 >= questions.length) {
      setShowResults(true);
    } else {
      setCurrentQ((q) => q + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    }
  };

  const handleRetry = () => {
    setCurrentQ(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setScore(0);
    setAnswers([]);
    setShowResults(false);
  };

  if (showResults) {
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          <Trophy
            className={`w-16 h-16 mx-auto mb-4 ${
              pct >= 80 ? "text-teal-500" : pct >= 50 ? "text-yellow-500" : "text-red-500"
            }`}
          />
          <h3 className="text-2xl font-bold text-gray-900">
            {score} / {questions.length}
          </h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80
              ? "Excellent! You understand K-Nearest Neighbors well!"
              : pct >= 50
              ? "Good job! Review the topics you missed."
              : "Keep learning! KNN has many concepts to master."}
          </p>
          <button
            onClick={handleRetry}
            className="mt-4 px-6 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors text-sm font-medium"
          >
            Try Again
          </button>
        </div>

        {/* Review */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700">Review</h4>
          {questions.map((q: any, i: number) => {
            const userAns = answers[i];
            const correct = userAns === q.correct_answer;
            return (
              <div
                key={i}
                className={`rounded-lg border p-3 ${
                  correct
                    ? "border-green-200 bg-green-50"
                    : "border-red-200 bg-red-50"
                }`}
              >
                <div className="flex items-start gap-2">
                  {correct ? (
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{q.question}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      Your answer: {q.options[userAns ?? 0]}{" "}
                      {!correct && `| Correct: ${q.options[q.correct_answer]}`}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">{q.explanation}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />
        <p className="text-sm text-teal-800">
          <span className="font-semibold">Test your knowledge</span> about K-Nearest
          Neighbors! Answer {questions.length} questions to check your understanding.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex items-center justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-2.5 h-2.5 rounded-full transition-all ${
              i === currentQ
                ? "bg-teal-500 scale-125"
                : i < answers.length
                ? answers[i] === questions[i].correct_answer
                  ? "bg-green-400"
                  : "bg-red-400"
                : "bg-gray-300"
            }`}
          />
        ))}
      </div>

      {/* Question */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <p className="text-xs text-gray-400 mb-2">
          Question {currentQ + 1} of {questions.length}
        </p>
        <p className="text-base font-medium text-gray-900 mb-4">{question.question}</p>

        <div className="space-y-2">
          {question.options.map((opt: string, idx: number) => {
            let style = "border-gray-200 bg-white hover:bg-gray-50 text-gray-700";
            if (answered) {
              if (idx === question.correct_answer) {
                style = "border-green-300 bg-green-50 text-green-800";
              } else if (idx === selectedAnswer && !isCorrect) {
                style = "border-red-300 bg-red-50 text-red-800";
              } else {
                style = "border-gray-200 bg-gray-50 text-gray-400";
              }
            } else if (idx === selectedAnswer) {
              style = "border-teal-400 bg-teal-50 text-teal-700";
            }

            return (
              <button
                key={idx}
                onClick={() => handleSelect(idx)}
                disabled={answered}
                className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-colors ${style} disabled:cursor-default`}
              >
                <span className="font-medium mr-2">
                  {String.fromCharCode(65 + idx)}.
                </span>
                {opt}
              </button>
            );
          })}
        </div>

        {answered && (
          <div
            className={`mt-4 p-3 rounded-lg text-sm ${
              isCorrect ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
            }`}
          >
            <p className="font-semibold mb-1">{isCorrect ? "Correct!" : "Not quite."}</p>
            <p>{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors text-sm font-medium"
          >
            {currentQ + 1 >= questions.length ? "See Results" : "Next Question"}
          </button>
        )}
      </div>
    </div>
  );
}

// ======================== How It Works Tab ========================

const INITIAL_SAMPLE_POINTS = [
  { x: 80, y: 80, label: "A" },
  { x: 120, y: 60, label: "A" },
  { x: 90, y: 130, label: "A" },
  { x: 140, y: 100, label: "A" },
  { x: 60, y: 110, label: "A" },
  { x: 100, y: 50, label: "A" },
  { x: 300, y: 200, label: "B" },
  { x: 340, y: 220, label: "B" },
  { x: 280, y: 250, label: "B" },
  { x: 320, y: 180, label: "B" },
  { x: 360, y: 240, label: "B" },
  { x: 310, y: 260, label: "B" },
  { x: 200, y: 150, label: "A" },
  { x: 220, y: 180, label: "B" },
  { x: 180, y: 200, label: "B" },
  { x: 250, y: 130, label: "A" },
  { x: 160, y: 160, label: "A" },
  { x: 240, y: 210, label: "B" },
];

const LABEL_COLORS: Record<string, string> = {
  A: "#14b8a6",
  B: "#6366f1",
};

function HowItWorksTab({
  nNeighbors,
  sampleData,
}: {
  nNeighbors?: number;
  sampleData?: any[];
}) {
  const [k, setK] = useState(nNeighbors || 5);
  const [queryPoint, setQueryPoint] = useState<{ x: number; y: number } | null>(null);
  const [animationStep, setAnimationStep] = useState(0);
  const [animRadius, setAnimRadius] = useState(0);

  const points = useMemo(() => {
    return INITIAL_SAMPLE_POINTS;
  }, []);

  const sortedNeighbors = useMemo(() => {
    if (!queryPoint) return [];
    return points
      .map((p) => ({
        ...p,
        distance: Math.sqrt(
          (p.x - queryPoint.x) ** 2 + (p.y - queryPoint.y) ** 2
        ),
      }))
      .sort((a, b) => a.distance - b.distance);
  }, [queryPoint, points]);

  const kNearest = useMemo(() => sortedNeighbors.slice(0, k), [sortedNeighbors, k]);

  const voteCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const n of kNearest) {
      counts[n.label] = (counts[n.label] || 0) + 1;
    }
    return counts;
  }, [kNearest]);

  const prediction = useMemo(() => {
    let best = "";
    let bestCount = 0;
    for (const [label, count] of Object.entries(voteCounts)) {
      if (count > bestCount) {
        bestCount = count;
        best = label;
      }
    }
    return best;
  }, [voteCounts]);

  const maxDistance = useMemo(() => {
    if (kNearest.length === 0) return 0;
    return kNearest[kNearest.length - 1]?.distance || 0;
  }, [kNearest]);

  const handleSvgClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = e.currentTarget;
      const rect = svg.getBoundingClientRect();
      const scaleX = 420 / rect.width;
      const scaleY = 300 / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      setQueryPoint({ x, y });
      setAnimationStep(0);
      setAnimRadius(0);

      // Animate expanding circle
      let step = 0;
      const totalSteps = 30;
      const distToK = (() => {
        const sorted = points
          .map((p) => Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2))
          .sort((a, b) => a - b);
        return sorted[k - 1] || 100;
      })();

      const interval = setInterval(() => {
        step++;
        setAnimRadius((step / totalSteps) * (distToK + 10));
        if (step >= totalSteps) {
          clearInterval(interval);
          setAnimationStep(1);
          // After a beat, show lines
          setTimeout(() => setAnimationStep(2), 400);
          // Then show votes
          setTimeout(() => setAnimationStep(3), 800);
          // Then show prediction
          setTimeout(() => setAnimationStep(4), 1200);
        }
      }, 30);
    },
    [points, k]
  );

  const handleReset = useCallback(() => {
    setQueryPoint(null);
    setAnimationStep(0);
    setAnimRadius(0);
  }, []);

  const handleKChange = useCallback((newK: number) => {
    setK(newK);
    setQueryPoint(null);
    setAnimationStep(0);
    setAnimRadius(0);
  }, []);

  const maxVote = Math.max(...Object.values(voteCounts).map(Number), 1);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />
        <div className="text-sm text-teal-800 space-y-2">
          <p className="font-semibold">How K-Nearest Neighbors Works</p>
          <p className="text-teal-700">
            KNN is one of the simplest machine learning algorithms. It makes
            predictions by looking at the K closest training examples and going with
            the majority vote. Click anywhere on the plot below to place a query
            point and watch KNN in action!
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 bg-white border border-gray-200 rounded-lg p-3">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700">K =</label>
          <input
            type="range"
            min="1"
            max="15"
            value={k}
            onChange={(e) => handleKChange(Number(e.target.value))}
            className="w-32 accent-teal-500"
          />
          <span className="text-sm font-bold text-teal-700 w-6 text-center">{k}</span>
        </div>
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors text-sm"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>
        {!queryPoint && (
          <span className="text-xs text-gray-400 italic">Click on the plot to place a query point</span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Interactive SVG scatter */}
        <div className="md:col-span-2 bg-white border border-gray-200 rounded-lg p-4">
          <svg
            viewBox="0 0 420 300"
            className="w-full cursor-crosshair"
            onClick={handleSvgClick}
          >
            {/* Background data points (dimmed when query exists) */}
            {points.map((p, i) => {
              const isNeighbor =
                queryPoint && animationStep >= 2 && kNearest.some((n) => n.x === p.x && n.y === p.y);
              return (
                <circle
                  key={i}
                  cx={p.x}
                  cy={p.y}
                  r={isNeighbor ? 7 : 5}
                  fill={LABEL_COLORS[p.label] || "#6b7280"}
                  fillOpacity={queryPoint && !isNeighbor ? 0.25 : 1}
                  stroke={isNeighbor ? "#fff" : "none"}
                  strokeWidth={isNeighbor ? 2 : 0}
                  className="transition-all duration-300"
                />
              );
            })}

            {/* Expanding search circle animation */}
            {queryPoint && animationStep === 0 && (
              <circle
                cx={queryPoint.x}
                cy={queryPoint.y}
                r={animRadius}
                fill="none"
                stroke="#14b8a6"
                strokeWidth="1.5"
                strokeDasharray="4 3"
                strokeOpacity="0.6"
              />
            )}

            {/* Final search radius circle */}
            {queryPoint && animationStep >= 1 && (
              <circle
                cx={queryPoint.x}
                cy={queryPoint.y}
                r={maxDistance + 5}
                fill="none"
                stroke="#14b8a6"
                strokeWidth="1.5"
                strokeDasharray="4 3"
                strokeOpacity="0.4"
              />
            )}

            {/* Lines to K nearest */}
            {queryPoint &&
              animationStep >= 2 &&
              kNearest.map((n, i) => (
                <g key={`kline-${i}`}>
                  <line
                    x1={queryPoint.x}
                    y1={queryPoint.y}
                    x2={n.x}
                    y2={n.y}
                    stroke="#14b8a6"
                    strokeWidth="1"
                    strokeDasharray="3 2"
                    strokeOpacity="0.6"
                  />
                  <text
                    x={(queryPoint.x + n.x) / 2}
                    y={(queryPoint.y + n.y) / 2 - 5}
                    textAnchor="middle"
                    fill="#6b7280"
                    fontSize="8"
                  >
                    {n.distance.toFixed(1)}
                  </text>
                </g>
              ))}

            {/* Query point diamond */}
            {queryPoint && (
              <polygon
                points={diamondPoints(queryPoint.x, queryPoint.y, 10)}
                fill="#1f2937"
                stroke="#fff"
                strokeWidth="2"
              />
            )}

            {/* Prediction label */}
            {queryPoint && animationStep >= 4 && prediction && (
              <g>
                <rect
                  x={queryPoint.x + 14}
                  y={queryPoint.y - 20}
                  width="60"
                  height="22"
                  rx="4"
                  fill={LABEL_COLORS[prediction] || "#6b7280"}
                  stroke="#fff"
                  strokeWidth="1"
                />
                <text
                  x={queryPoint.x + 44}
                  y={queryPoint.y - 5}
                  textAnchor="middle"
                  fill="#fff"
                  fontSize="11"
                  fontWeight="600"
                >
                  {prediction}
                </text>
              </g>
            )}

            {/* Legend */}
            <g transform="translate(10, 275)">
              <circle cx="0" cy="0" r="5" fill={LABEL_COLORS["A"]} />
              <text x="10" y="4" fill="#6b7280" fontSize="10">
                Class A
              </text>
              <circle cx="70" cy="0" r="5" fill={LABEL_COLORS["B"]} />
              <text x="80" y="4" fill="#6b7280" fontSize="10">
                Class B
              </text>
              <polygon points={diamondPoints(150, 0, 6)} fill="#1f2937" />
              <text x="160" y="4" fill="#6b7280" fontSize="10">
                Query
              </text>
            </g>
          </svg>
        </div>

        {/* Vote counting panel */}
        <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
          {!queryPoint ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-400 py-8">
              <Diamond className="w-10 h-10 mb-3" />
              <p className="text-sm text-center">
                Click on the plot to place a query point
              </p>
            </div>
          ) : animationStep < 3 ? (
            <div className="flex flex-col items-center justify-center h-full py-8">
              <div className="w-6 h-6 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-gray-500 mt-3">Finding neighbors...</p>
            </div>
          ) : (
            <>
              <h5 className="text-sm font-semibold text-gray-900">
                Vote Counting (K={k})
              </h5>
              <div className="space-y-3">
                {Object.entries(voteCounts)
                  .sort(([, a], [, b]) => Number(b) - Number(a))
                  .map(([label, count]) => (
                    <div key={label} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span
                          className="font-semibold"
                          style={{ color: LABEL_COLORS[label] || "#374151" }}
                        >
                          Class {label}
                        </span>
                        <span className="font-bold text-gray-800">
                          {String(count)} vote{Number(count) !== 1 ? "s" : ""}
                        </span>
                      </div>
                      <div className="h-4 rounded-full bg-gray-100 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${(Number(count) / maxVote) * 100}%`,
                            backgroundColor: LABEL_COLORS[label] || "#6b7280",
                          }}
                        />
                      </div>
                    </div>
                  ))}
              </div>

              {animationStep >= 4 && prediction && (
                <div
                  className="rounded-lg p-3 text-center"
                  style={{
                    backgroundColor: `${LABEL_COLORS[prediction]}15`,
                    border: `1px solid ${LABEL_COLORS[prediction]}40`,
                  }}
                >
                  <p className="text-xs text-gray-500 mb-1">Prediction</p>
                  <p
                    className="text-xl font-bold"
                    style={{ color: LABEL_COLORS[prediction] }}
                  >
                    Class {prediction}
                  </p>
                </div>
              )}

              <div className="text-xs text-gray-500 space-y-1 pt-2 border-t border-gray-100">
                <p>
                  <span className="font-medium">K neighbors found:</span> {kNearest.length}
                </p>
                <p>
                  <span className="font-medium">Max distance:</span>{" "}
                  {maxDistance.toFixed(1)}
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Step-by-step cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          {
            step: 1,
            title: "Place Query",
            desc: "Choose a new data point to classify",
            active: !queryPoint,
          },
          {
            step: 2,
            title: "Measure Distances",
            desc: "Calculate distance to all training points",
            active: queryPoint != null && animationStep < 1,
          },
          {
            step: 3,
            title: "Find K Nearest",
            desc: `Select the ${k} closest neighbors`,
            active: animationStep >= 1 && animationStep < 3,
          },
          {
            step: 4,
            title: "Vote",
            desc: "Count labels among the neighbors",
            active: animationStep >= 3 && animationStep < 4,
          },
          {
            step: 5,
            title: "Predict",
            desc: "Majority label wins!",
            active: animationStep >= 4,
          },
        ].map((s) => (
          <div
            key={s.step}
            className={`rounded-lg border p-3 text-center transition-all ${
              s.active
                ? "border-teal-400 bg-teal-50 shadow-sm"
                : "border-gray-200 bg-white"
            }`}
          >
            <div
              className={`w-7 h-7 rounded-full flex items-center justify-center mx-auto mb-1.5 text-xs font-bold ${
                s.active
                  ? "bg-teal-500 text-white"
                  : "bg-gray-100 text-gray-400"
              }`}
            >
              {s.step}
            </div>
            <p
              className={`text-xs font-semibold ${
                s.active ? "text-teal-700" : "text-gray-500"
              }`}
            >
              {s.title}
            </p>
            <p className="text-[10px] text-gray-400 mt-0.5">{s.desc}</p>
          </div>
        ))}
      </div>

      {/* Explanatory cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Search className="w-4 h-4 text-teal-600" />
            Instance-Based Learning
          </h4>
          <p className="text-sm text-gray-600">
            Unlike most algorithms, KNN doesn't build a model during training. It
            memorizes all training data and makes predictions by comparing new
            points to stored examples. This is called "lazy learning."
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-blue-600" />
            Choosing K
          </h4>
          <p className="text-sm text-gray-600">
            Small K (like 1) follows the data very closely but is sensitive to
            noise. Large K creates smoother boundaries but may miss local patterns.
            The sweet spot depends on your data — the K Selection tab helps find it.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <CircleDot className="w-4 h-4 text-orange-600" />
            Distance Matters
          </h4>
          <p className="text-sm text-gray-600">
            KNN uses distance (usually Euclidean) to find neighbors. This means
            feature scaling is crucial — a feature with range 0-1000 would dominate
            one with range 0-1. Always normalize your data before using KNN.
          </p>
        </div>
      </div>

      {/* Settings reference */}
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Settings You Can Tune</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">n_neighbors (K):</span>
            <span className="text-gray-600">
              How many nearest points to consider. Currently set to{" "}
              {nNeighbors || "5"}. Odd values avoid ties in binary classification.
            </span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">weights:</span>
            <span className="text-gray-600">
              "uniform" treats all neighbors equally. "distance" gives closer
              neighbors more influence, which often improves accuracy.
            </span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">metric:</span>
            <span className="text-gray-600">
              Distance function: Euclidean (straight line), Manhattan (city blocks),
              or Minkowski (generalizes both). Euclidean is the default.
            </span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">algorithm:</span>
            <span className="text-gray-600">
              How neighbors are found: "ball_tree" and "kd_tree" are fast for large
              datasets. "brute" checks every point — simple but slow.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default KNNExplorer;
