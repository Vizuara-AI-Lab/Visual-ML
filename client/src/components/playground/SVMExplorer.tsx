/**
 * SVMExplorer - Interactive learning activities for Support Vector Machines
 * Tabbed component: Results | Decision Boundary | Support Vectors | Kernel Comparison | C Sensitivity | Quiz | How It Works
 */

import { useState, useMemo, useCallback } from "react";
import {
  ClipboardList,
  Crosshair,
  CircleDot,
  Columns3,
  SlidersHorizontal,
  HelpCircle,
  Cog,
  CheckCircle,
  XCircle,
  Trophy,
  Timer,
  Hash,
  Database,
  Shield,
  Target,
  Zap,
  Info,
} from "lucide-react";

interface SVMExplorerProps {
  result: any;
}

type ExplorerTab =
  | "results"
  | "decision_boundary"
  | "support_vectors"
  | "kernel_comparison"
  | "c_sensitivity"
  | "quiz"
  | "how_it_works";

/* ───────── colour palette (class-indexed) ───────── */
const CLASS_COLORS = [
  "#6366f1", // indigo
  "#f59e0b", // amber
  "#10b981", // emerald
  "#ef4444", // red
  "#8b5cf6", // violet
  "#06b6d4", // cyan
  "#f97316", // orange
  "#ec4899", // pink
];
const classColor = (i: number) => CLASS_COLORS[i % CLASS_COLORS.length];

/* ───────── linear scale helper ───────── */
function linScale(
  domain: [number, number],
  range: [number, number]
): (v: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v: number) => r0 + ((v - d0) / span) * (r1 - r0);
}

/* ================================================================== */
/*  Root Component                                                     */
/* ================================================================== */

export const SVMExplorer = ({ result }: SVMExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "decision_boundary",
      label: "Decision Boundary",
      icon: Crosshair,
      available: !!result.decision_boundary_grid,
    },
    {
      id: "support_vectors",
      label: "Support Vectors",
      icon: CircleDot,
      available: !!result.support_vectors_2d,
    },
    {
      id: "kernel_comparison",
      label: "Kernel Comparison",
      icon: Columns3,
      available:
        !!result.kernel_comparison && result.kernel_comparison.length > 0,
    },
    {
      id: "c_sensitivity",
      label: "C Sensitivity",
      icon: SlidersHorizontal,
      available: !!result.c_sensitivity_data,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available: !!result.quiz_questions && result.quiz_questions.length > 0,
    },
    { id: "how_it_works", label: "How It Works", icon: Cog, available: true },
  ];

  const isRegression = result.task_type === "regression";
  const metrics = result.training_metrics || {};

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
                    ? "border-indigo-500 text-indigo-700"
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
          <ResultsTab
            result={result}
            isRegression={isRegression}
            metrics={metrics}
          />
        )}
        {activeTab === "decision_boundary" && (
          <DecisionBoundaryTab data={result.decision_boundary_grid} svData={result.support_vectors_2d} />
        )}
        {activeTab === "support_vectors" && (
          <SupportVectorsTab data={result.support_vectors_2d} />
        )}
        {activeTab === "kernel_comparison" && (
          <KernelComparisonTab
            data={result.kernel_comparison}
            currentKernel={result.kernel}
          />
        )}
        {activeTab === "c_sensitivity" && (
          <CSensitivityTab data={result.c_sensitivity_data} />
        )}
        {activeTab === "quiz" && (
          <QuizTab questions={result.quiz_questions || []} />
        )}
        {activeTab === "how_it_works" && (
          <HowItWorksTab
            result={result}
          />
        )}
      </div>
    </div>
  );
};

/* ================================================================== */
/*  Tab 1 - Results                                                    */
/* ================================================================== */

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
  const svRatio =
    result.support_vector_ratio != null
      ? result.support_vector_ratio * 100
      : null;

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-indigo-50 to-violet-50 border border-indigo-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-500 rounded-full flex items-center justify-center">
            <CheckCircle className="text-white w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-indigo-900">
              SVM Training Complete
            </h3>
            <p className="text-sm text-indigo-700">
              {result.kernel?.toUpperCase()} kernel with C={result.C ?? "auto"}{" "}
              trained on {result.training_samples?.toLocaleString()} samples with{" "}
              {result.n_features} features (
              {isRegression ? "regression" : "classification"})
            </p>
          </div>
        </div>
      </div>

      {/* Hero Metric - Ring Gauge */}
      {primaryPct != null && (
        <div className="flex justify-center">
          <div className="relative w-36 h-36">
            <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke="#e5e7eb"
                strokeWidth="8"
              />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke={
                  primaryPct >= 80
                    ? "#6366f1"
                    : primaryPct >= 60
                    ? "#f59e0b"
                    : "#ef4444"
                }
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${(primaryPct / 100) * 327} 327`}
                style={{
                  transition: "stroke-dasharray 0.8s ease-out",
                }}
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
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        <MetricCard icon={Shield} label="Kernel" value={result.kernel?.toUpperCase() ?? "N/A"} />
        <MetricCard icon={SlidersHorizontal} label="C" value={result.C ?? "N/A"} />
        <MetricCard icon={Target} label="Gamma" value={result.gamma ?? "N/A"} />
        <MetricCard
          icon={CircleDot}
          label="Support Vectors"
          value={result.n_support_vectors?.toLocaleString() ?? "N/A"}
        />
        <MetricCard
          icon={Hash}
          label="SV Ratio"
          value={svRatio != null ? `${svRatio.toFixed(1)}%` : "N/A"}
        />
        <MetricCard
          icon={Zap}
          label="Task Type"
          value={result.task_type ?? "N/A"}
        />
        <MetricCard
          icon={Database}
          label="Training Samples"
          value={result.training_samples?.toLocaleString() ?? "N/A"}
        />
        <MetricCard
          icon={Hash}
          label="Features"
          value={result.n_features ?? "N/A"}
        />
        <MetricCard
          icon={Timer}
          label="Training Time"
          value={
            result.training_time_seconds != null
              ? `${result.training_time_seconds.toFixed(2)}s`
              : "N/A"
          }
        />
      </div>

      {/* Support Vector Ratio Bar */}
      {svRatio != null && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            Support Vector Ratio
          </h4>
          <div className="flex items-center gap-3">
            <div className="flex-1 h-4 rounded-full bg-gray-100 overflow-hidden">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all duration-700"
                style={{ width: `${Math.min(svRatio, 100)}%` }}
              />
            </div>
            <span className="text-sm font-semibold text-gray-700 w-16 text-right">
              {svRatio.toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {result.n_support_vectors} of {result.training_samples} training
            points are support vectors
          </p>
        </div>
      )}

      {/* All Metrics */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <h4 className="text-sm font-semibold text-gray-900">
            {result.metadata?.evaluated_on === "test"
              ? "Test Metrics"
              : "Training Metrics"}
          </h4>
          {result.metadata?.evaluated_on === "test" ? (
            <span className="text-[10px] font-medium bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full">
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
            .filter(
              ([key, value]) =>
                typeof value === "number" && key !== "confusion_matrix"
            )
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

      {/* Model Details */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">
          Model Details
        </h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Task Type:</span>
            <span className="font-semibold capitalize">
              {result.task_type}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Kernel:</span>
            <span className="font-semibold">{result.kernel}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">C:</span>
            <span className="font-semibold">{result.C}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Gamma:</span>
            <span className="font-semibold">{result.gamma}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Support Vectors:</span>
            <span className="font-semibold">{result.n_support_vectors}</span>
          </div>
          {result.model_id && (
            <div className="flex justify-between">
              <span className="text-gray-600">Model ID:</span>
              <code className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                {result.model_id?.slice(0, 20)}...
              </code>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: any;
  label: string;
  value: string | number;
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
        <Icon className="w-3.5 h-3.5" /> {label}
      </div>
      <div className="text-lg font-semibold text-gray-900">{value}</div>
    </div>
  );
}

/* ================================================================== */
/*  Tab 2 - Decision Boundary                                          */
/* ================================================================== */

function DecisionBoundaryTab({ data, svData }: { data: any; svData?: any }) {
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);

  const {
    grid: rawGrid,
    points = [],
    classes = [],
    pca_explained_variance: pca_variance = [],
  } = data || {};

  // Backend sends grid as compact object {x_min,x_max,y_min,y_max,nx,ny,predictions,decision_values}
  // Expand into array of {x, y, prediction, decision_value}
  const grid: any[] = useMemo(() => {
    if (!rawGrid) return [];
    if (Array.isArray(rawGrid)) return rawGrid; // already expanded
    const { x_min, x_max, y_min, y_max, nx, ny, predictions, decision_values } = rawGrid;
    if (nx == null || ny == null || !predictions) return [];
    const cells: any[] = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        cells.push({
          x: x_min + (i / (nx - 1)) * (x_max - x_min),
          y: y_min + (j / (ny - 1)) * (y_max - y_min),
          prediction: predictions[idx],
          decision_value: decision_values ? decision_values[idx] : null,
        });
      }
    }
    return cells;
  }, [rawGrid]);

  const svPoints = svData?.sv_points || [];

  const VB_W = 500;
  const VB_H = 340;
  const PAD = 50;

  const xRange: [number, number] = useMemo(() => {
    if (rawGrid?.x_min != null && rawGrid?.x_max != null) return [rawGrid.x_min, rawGrid.x_max];
    const xs = [...grid.map((g: any) => g.x), ...points.map((p: any) => p.x)];
    return xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1];
  }, [grid, points, rawGrid]);

  const yRange: [number, number] = useMemo(() => {
    if (rawGrid?.y_min != null && rawGrid?.y_max != null) return [rawGrid.y_min, rawGrid.y_max];
    const ys = [...grid.map((g: any) => g.y), ...points.map((p: any) => p.y)];
    return ys.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
  }, [grid, points, rawGrid]);

  const scaleX = useMemo(
    () => linScale(xRange, [PAD, VB_W - PAD]),
    [xRange]
  );
  const scaleY = useMemo(
    () => linScale(yRange, [VB_H - PAD, PAD]),
    [yRange]
  );

  // Build class->color map
  const classColorMap = useMemo(() => {
    const m: Record<string, string> = {};
    classes.forEach((c: string, i: number) => {
      m[c] = classColor(i);
    });
    return m;
  }, [classes]);

  // SVG grid cells: each cell is a small rect
  const gridSize = useMemo(() => {
    const cols = rawGrid?.nx || [...new Set(grid.map((g: any) => g.x))].length || 50;
    const rows = rawGrid?.ny || [...new Set(grid.map((g: any) => g.y))].length || 50;
    const cellW = (VB_W - 2 * PAD) / cols;
    const cellH = (VB_H - 2 * PAD) / rows;
    return { cols, rows, cellW, cellH };
  }, [grid, rawGrid]);

  // Build set of SV coordinates for quick lookup
  const svSet = useMemo(() => {
    const s = new Set<string>();
    svPoints.forEach((sv: any) => s.add(`${sv.x.toFixed(6)},${sv.y.toFixed(6)}`));
    return s;
  }, [svPoints]);

  if (!grid.length && !points.length) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400 text-sm">
        No decision boundary data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3">
        <Crosshair className="mt-0.5 h-5 w-5 shrink-0 text-indigo-600" />
        <p className="text-sm text-indigo-800">
          <span className="font-semibold">Decision Boundary</span> -- The
          background color shows which class the SVM predicts in each region.
          Support vectors (squares) are the critical points defining the
          boundary. The margin band shows the zone where the model is least
          certain.
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full h-auto">
          {/* Grid heatmap */}
          {grid.map((cell: any, i: number) => {
            const ci = classes.indexOf(String(cell.prediction));
            const col = classColor(ci >= 0 ? ci : 0);
            // Margin band: where |decision_value| is close to 0
            const isMargin =
              cell.decision_value != null && Math.abs(cell.decision_value) < 0.3;
            return (
              <rect
                key={i}
                x={scaleX(cell.x) - gridSize.cellW / 2}
                y={scaleY(cell.y) - gridSize.cellH / 2}
                width={gridSize.cellW + 0.5}
                height={gridSize.cellH + 0.5}
                fill={col}
                opacity={isMargin ? 0.25 : 0.15}
              />
            );
          })}

          {/* Margin band overlay (semi-transparent strip) */}
          {grid
            .filter(
              (cell: any) =>
                cell.decision_value != null &&
                Math.abs(cell.decision_value) < 0.15
            )
            .map((cell: any, i: number) => (
              <rect
                key={`margin-${i}`}
                x={scaleX(cell.x) - gridSize.cellW / 2}
                y={scaleY(cell.y) - gridSize.cellH / 2}
                width={gridSize.cellW + 0.5}
                height={gridSize.cellH + 0.5}
                fill="#fbbf24"
                opacity={0.25}
              />
            ))}

          {/* Regular points (dimmed circles) */}
          {points.map((pt: any, i: number) => {
            const ci = classes.indexOf(String(pt.label));
            const isSV = svSet.has(`${pt.x.toFixed(6)},${pt.y.toFixed(6)}`);
            if (isSV) return null; // draw SVs separately on top
            return (
              <circle
                key={`pt-${i}`}
                cx={scaleX(pt.x)}
                cy={scaleY(pt.y)}
                r={3}
                fill={classColor(ci >= 0 ? ci : 0)}
                opacity={0.5}
                stroke="#fff"
                strokeWidth={0.5}
              />
            );
          })}

          {/* Support vectors (squares with thick outlines) */}
          {svPoints.map((sv: any, i: number) => {
            const ci = classes.indexOf(String(sv.label));
            return (
              <rect
                key={`sv-${i}`}
                x={scaleX(sv.x) - 5}
                y={scaleY(sv.y) - 5}
                width={10}
                height={10}
                fill={classColor(ci >= 0 ? ci : 0)}
                stroke="#1e1b4b"
                strokeWidth={2}
                opacity={0.9}
                onMouseEnter={() => setHoveredPoint(i)}
                onMouseLeave={() => setHoveredPoint(null)}
                className="cursor-pointer"
              />
            );
          })}

          {/* Tooltip for hovered SV */}
          {hoveredPoint != null && svPoints[hoveredPoint] && (
            <g>
              <rect
                x={scaleX(svPoints[hoveredPoint].x) + 10}
                y={scaleY(svPoints[hoveredPoint].y) - 20}
                width={90}
                height={24}
                rx={4}
                fill="#1e1b4b"
                opacity={0.9}
              />
              <text
                x={scaleX(svPoints[hoveredPoint].x) + 14}
                y={scaleY(svPoints[hoveredPoint].y) - 4}
                fill="white"
                fontSize={10}
                fontFamily="monospace"
              >
                SV class: {svPoints[hoveredPoint].label}
              </text>
            </g>
          )}

          {/* PCA axis labels */}
          <text
            x={VB_W / 2}
            y={VB_H - 5}
            textAnchor="middle"
            fontSize={11}
            fill="#6b7280"
          >
            PC1{pca_variance[0] != null ? ` (${(pca_variance[0] * 100).toFixed(1)}% var)` : ""}
          </text>
          <text
            x={12}
            y={VB_H / 2}
            textAnchor="middle"
            fontSize={11}
            fill="#6b7280"
            transform={`rotate(-90, 12, ${VB_H / 2})`}
          >
            PC2{pca_variance[1] != null ? ` (${(pca_variance[1] * 100).toFixed(1)}% var)` : ""}
          </text>
        </svg>
      </div>

      {/* Class Legend */}
      {classes.length > 0 && (
        <div className="flex items-center gap-4 flex-wrap">
          {classes.map((cls: string, i: number) => (
            <div key={cls} className="flex items-center gap-2 text-sm text-gray-700">
              <div
                className="w-3 h-3 rounded-sm"
                style={{ backgroundColor: classColor(i) }}
              />
              <span>{cls}</span>
            </div>
          ))}
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <div className="w-3 h-3 border-2 border-gray-800 bg-gray-300 rounded-sm" />
            <span>Support Vector</span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ================================================================== */
/*  Tab 3 - Support Vectors                                            */
/* ================================================================== */

function SupportVectorsTab({ data }: { data: any }) {
  const [hoveredSV, setHoveredSV] = useState<number | null>(null);

  const {
    all_points = [],
    sv_points = [],
    n_total = 0,
    n_sv = 0,
    sv_ratio: ratio = 0,
    class_labels: classes = [],
  } = data || {};

  const VB_W = 500;
  const VB_H = 340;
  const PAD = 50;

  const xRange: [number, number] = useMemo(() => {
    const xs = all_points.map((p: any) => p.x);
    return xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1];
  }, [all_points]);

  const yRange: [number, number] = useMemo(() => {
    const ys = all_points.map((p: any) => p.y);
    return ys.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
  }, [all_points]);

  const scaleX = useMemo(
    () => linScale(xRange, [PAD, VB_W - PAD]),
    [xRange]
  );
  const scaleY = useMemo(
    () => linScale(yRange, [VB_H - PAD, PAD]),
    [yRange]
  );

  const ratioPct = (ratio * 100).toFixed(1);

  const handleMouseEnter = useCallback((i: number) => setHoveredSV(i), []);
  const handleMouseLeave = useCallback(() => setHoveredSV(null), []);

  if (!all_points.length) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400 text-sm">
        No support vector data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3">
        <CircleDot className="mt-0.5 h-5 w-5 shrink-0 text-indigo-600" />
        <p className="text-sm text-indigo-800">
          <span className="font-semibold">Support Vectors</span> are the
          training points closest to the decision boundary. They alone define
          where the boundary lies -- all other points could be removed without
          changing the model.
        </p>
      </div>

      {/* Counter bar */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">
            {n_sv} of {n_total} training points are support vectors ({ratioPct}
            %)
          </span>
        </div>
        <div className="h-4 rounded-full bg-gray-100 overflow-hidden">
          <div
            className="h-full rounded-full bg-indigo-500 transition-all duration-700"
            style={{ width: `${Math.min(ratio * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Scatter plot */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full h-auto">
          {/* Non-support vectors first (behind) */}
          {all_points
            .filter((p: any) => !p.is_sv)
            .map((pt: any, i: number) => {
              const ci = classes.indexOf(String(pt.label));
              return (
                <circle
                  key={`non-sv-${i}`}
                  cx={scaleX(pt.x)}
                  cy={scaleY(pt.y)}
                  r={3}
                  fill={classColor(ci >= 0 ? ci : 0)}
                  opacity={0.3}
                  stroke={classColor(ci >= 0 ? ci : 0)}
                  strokeWidth={0.5}
                />
              );
            })}

          {/* Support vectors on top */}
          {all_points
            .filter((p: any) => p.is_sv)
            .map((pt: any, i: number) => {
              const ci = classes.indexOf(String(pt.label));
              return (
                <circle
                  key={`sv-${i}`}
                  cx={scaleX(pt.x)}
                  cy={scaleY(pt.y)}
                  r={6}
                  fill={classColor(ci >= 0 ? ci : 0)}
                  opacity={1}
                  stroke="#1e1b4b"
                  strokeWidth={2}
                  onMouseEnter={() => handleMouseEnter(i)}
                  onMouseLeave={handleMouseLeave}
                  className="cursor-pointer"
                />
              );
            })}

          {/* Tooltip */}
          {hoveredSV != null && (() => {
            const svPts = all_points.filter((p: any) => p.is_sv);
            const pt = svPts[hoveredSV];
            if (!pt) return null;
            const tx = scaleX(pt.x);
            const ty = scaleY(pt.y);
            // Flip tooltip if near right edge
            const flipX = tx > VB_W - 120;
            return (
              <g>
                <rect
                  x={flipX ? tx - 100 : tx + 10}
                  y={ty - 20}
                  width={90}
                  height={24}
                  rx={4}
                  fill="#1e1b4b"
                  opacity={0.9}
                />
                <text
                  x={flipX ? tx - 96 : tx + 14}
                  y={ty - 4}
                  fill="white"
                  fontSize={10}
                  fontFamily="monospace"
                >
                  Class: {pt.label}
                </text>
              </g>
            );
          })()}

          {/* Axis labels */}
          <text
            x={VB_W / 2}
            y={VB_H - 5}
            textAnchor="middle"
            fontSize={11}
            fill="#6b7280"
          >
            PC1
          </text>
          <text
            x={12}
            y={VB_H / 2}
            textAnchor="middle"
            fontSize={11}
            fill="#6b7280"
            transform={`rotate(-90, 12, ${VB_H / 2})`}
          >
            PC2
          </text>
        </svg>
      </div>

      {/* Class legend */}
      {classes.length > 0 && (
        <div className="flex items-center gap-4 flex-wrap">
          {classes.map((cls: string, i: number) => (
            <div key={cls} className="flex items-center gap-2 text-sm text-gray-700">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: classColor(i) }}
              />
              <span>{cls}</span>
            </div>
          ))}
          <div className="flex items-center gap-2 text-sm text-gray-500 ml-2">
            <div className="w-4 h-4 rounded-full border-2 border-gray-800 bg-gray-300" />
            <span className="text-xs">= Support Vector (large, outlined)</span>
          </div>
        </div>
      )}

      {/* Explanation cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-indigo-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <CircleDot className="w-4 h-4 text-indigo-600" />
            What are Support Vectors?
          </h4>
          <p className="text-sm text-gray-600">
            The critical data points closest to the decision boundary. These are
            the training examples that are hardest to classify -- they sit right
            on the edge between classes.
          </p>
        </div>
        <div className="rounded-lg border border-indigo-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Shield className="w-4 h-4 text-indigo-600" />
            Why do they matter?
          </h4>
          <p className="text-sm text-gray-600">
            Only support vectors define the decision boundary. All other
            training points can be removed without changing the model at all.
            The SVM only "remembers" these critical points.
          </p>
        </div>
        <div className="rounded-lg border border-indigo-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Target className="w-4 h-4 text-indigo-600" />
            Fewer is better?
          </h4>
          <p className="text-sm text-gray-600">
            A lower support vector ratio generally means a simpler, more
            generalizable model. If too many points are SVs, the model may be
            struggling to find a clear separation.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  Tab 4 - Kernel Comparison                                          */
/* ================================================================== */

function KernelComparisonTab({
  data,
  currentKernel,
}: {
  data: any[];
  currentKernel?: string;
}) {
  if (!data || !data.length) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400 text-sm">
        No kernel comparison data available.
      </div>
    );
  }

  const hasGridData = data.some((k: any) => k.grid && k.grid.length > 0);

  const kernelDescriptions: Record<string, string> = {
    linear: "Straight line boundary. Best for linearly separable data.",
    rbf: "Flexible curved boundary using Gaussian similarity. The most common default kernel.",
    poly: "Polynomial curved boundary. Can capture interactions between features.",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3">
        <Columns3 className="mt-0.5 h-5 w-5 shrink-0 text-indigo-600" />
        <p className="text-sm text-indigo-800">
          <span className="font-semibold">Kernel Comparison</span> -- The
          kernel function determines the shape of the decision boundary. Each
          kernel maps data into a different feature space, allowing SVM to find
          non-linear boundaries.
        </p>
      </div>

      <div className={`grid grid-cols-1 ${data.length >= 3 ? "md:grid-cols-3" : data.length === 2 ? "md:grid-cols-2" : ""} gap-4`}>
        {data.map((k: any, idx: number) => {
          const isCurrent = k.is_current || k.kernel === currentKernel;
          return (
            <div
              key={k.kernel}
              className={`rounded-lg border-2 p-4 space-y-3 ${
                isCurrent
                  ? "border-indigo-500 bg-indigo-50"
                  : "border-gray-200 bg-white"
              }`}
            >
              <div className="flex items-center justify-between">
                <h4 className="font-semibold text-gray-900 uppercase text-sm">
                  {k.kernel}
                </h4>
                {isCurrent && (
                  <span className="text-[10px] font-medium bg-indigo-500 text-white px-2 py-0.5 rounded-full">
                    Current
                  </span>
                )}
              </div>

              {/* Mini grid heatmap if available */}
              {hasGridData && k.grid && k.grid.length > 0 ? (
                <KernelMiniPlot
                  grid={k.grid}
                  points={k.points || []}
                  classes={k.classes || []}
                />
              ) : null}

              {/* Stats */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">{k.score_metric === "r2" ? "R² Score" : "Accuracy"}</span>
                  <span className="font-bold text-gray-900">
                    {k.score != null
                      ? `${(k.score * 100).toFixed(1)}%`
                      : "N/A"}
                  </span>
                </div>
                {k.score != null && (
                  <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-indigo-500"
                      style={{ width: `${Math.max(0, k.score) * 100}%` }}
                    />
                  </div>
                )}
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Support Vectors</span>
                  <span className="font-medium text-gray-700">
                    {k.n_support_vectors ?? "N/A"}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Training Time</span>
                  <span className="font-medium text-gray-700">
                    {k.training_time != null
                      ? `${k.training_time.toFixed(3)}s`
                      : "N/A"}
                  </span>
                </div>
              </div>

              {/* Kernel description */}
              <p className="text-xs text-gray-500 italic">
                {kernelDescriptions[k.kernel] || "Custom kernel function."}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function KernelMiniPlot({
  grid,
  points,
  classes,
}: {
  grid: any[];
  points: any[];
  classes: string[];
}) {
  const VB = 160;
  const PAD = 8;

  const xRange: [number, number] = useMemo(() => {
    const xs = [...grid.map((g: any) => g.x), ...points.map((p: any) => p.x)];
    return xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1];
  }, [grid, points]);

  const yRange: [number, number] = useMemo(() => {
    const ys = [...grid.map((g: any) => g.y), ...points.map((p: any) => p.y)];
    return ys.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
  }, [grid, points]);

  const sx = useMemo(() => linScale(xRange, [PAD, VB - PAD]), [xRange]);
  const sy = useMemo(() => linScale(yRange, [VB - PAD, PAD]), [yRange]);

  const uniqueXs = [...new Set(grid.map((g: any) => g.x))];
  const uniqueYs = [...new Set(grid.map((g: any) => g.y))];
  const cols = uniqueXs.length || 25;
  const rows = uniqueYs.length || 25;
  const cellW = (VB - 2 * PAD) / cols;
  const cellH = (VB - 2 * PAD) / rows;

  return (
    <svg
      viewBox={`0 0 ${VB} ${VB}`}
      className="w-full h-auto rounded bg-gray-50"
    >
      {grid.map((cell: any, i: number) => {
        const ci = classes.indexOf(String(cell.prediction));
        return (
          <rect
            key={i}
            x={sx(cell.x) - cellW / 2}
            y={sy(cell.y) - cellH / 2}
            width={cellW + 0.5}
            height={cellH + 0.5}
            fill={classColor(ci >= 0 ? ci : 0)}
            opacity={0.2}
          />
        );
      })}
      {points.map((pt: any, i: number) => {
        const ci = classes.indexOf(String(pt.label));
        return (
          <circle
            key={i}
            cx={sx(pt.x)}
            cy={sy(pt.y)}
            r={2.5}
            fill={classColor(ci >= 0 ? ci : 0)}
            stroke="#fff"
            strokeWidth={0.5}
            opacity={0.8}
          />
        );
      })}
    </svg>
  );
}

/* ================================================================== */
/*  Tab 5 - C Sensitivity                                              */
/* ================================================================== */

function CSensitivityTab({ data }: { data: any }) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const {
    c_values = [],
    scores = [],
    n_support_vectors = [],
    current_c,
  } = data || {};

  const VB_W = 500;
  const VB_H = 300;
  const PAD_L = 55;
  const PAD_R = 55;
  const PAD_T = 30;
  const PAD_B = 50;

  // Log scale for C values on x-axis
  const logCValues = useMemo(
    () => c_values.map((c: number) => Math.log10(c)),
    [c_values]
  );

  const xRange: [number, number] = useMemo(() => {
    return logCValues.length
      ? [Math.min(...logCValues), Math.max(...logCValues)]
      : [-2, 2];
  }, [logCValues]);

  const scoreRange: [number, number] = useMemo(() => {
    if (!scores.length) return [0, 1];
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const pad = (max - min) * 0.1 || 0.05;
    return [Math.max(0, min - pad), Math.min(1, max + pad)];
  }, [scores]);

  const svRange: [number, number] = useMemo(() => {
    if (!n_support_vectors.length) return [0, 100];
    const min = Math.min(...n_support_vectors);
    const max = Math.max(...n_support_vectors);
    const pad = (max - min) * 0.1 || 5;
    return [Math.max(0, min - pad), max + pad];
  }, [n_support_vectors]);

  const scaleX = useMemo(
    () => linScale(xRange, [PAD_L, VB_W - PAD_R]),
    [xRange]
  );
  const scaleYScore = useMemo(
    () => linScale(scoreRange, [VB_H - PAD_B, PAD_T]),
    [scoreRange]
  );
  const scaleYSV = useMemo(
    () => linScale(svRange, [VB_H - PAD_B, PAD_T]),
    [svRange]
  );

  // Build polyline paths
  const scorePath = useMemo(() => {
    return logCValues
      .map(
        (logC: number, i: number) =>
          `${i === 0 ? "M" : "L"}${scaleX(logC)},${scaleYScore(scores[i])}`
      )
      .join(" ");
  }, [logCValues, scores, scaleX, scaleYScore]);

  const svPath = useMemo(() => {
    return logCValues
      .map(
        (logC: number, i: number) =>
          `${i === 0 ? "M" : "L"}${scaleX(logC)},${scaleYSV(n_support_vectors[i])}`
      )
      .join(" ");
  }, [logCValues, n_support_vectors, scaleX, scaleYSV]);

  const currentCLogX =
    current_c != null ? scaleX(Math.log10(current_c)) : null;

  if (!c_values.length) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400 text-sm">
        No C sensitivity data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3">
        <SlidersHorizontal className="mt-0.5 h-5 w-5 shrink-0 text-indigo-600" />
        <p className="text-sm text-indigo-800">
          <span className="font-semibold">C Parameter Sensitivity</span> --
          The regularization parameter C controls the trade-off between a smooth
          decision boundary and classifying training points correctly. A larger
          C means fewer misclassifications but a narrower margin.
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full h-auto">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
            const y = PAD_T + frac * (VB_H - PAD_T - PAD_B);
            return (
              <line
                key={frac}
                x1={PAD_L}
                y1={y}
                x2={VB_W - PAD_R}
                y2={y}
                stroke="#f3f4f6"
                strokeWidth={1}
              />
            );
          })}

          {/* Current C vertical line */}
          {currentCLogX != null && (
            <>
              <line
                x1={currentCLogX}
                y1={PAD_T}
                x2={currentCLogX}
                y2={VB_H - PAD_B}
                stroke="#6366f1"
                strokeWidth={1.5}
                strokeDasharray="6 3"
                opacity={0.6}
              />
              <text
                x={currentCLogX}
                y={PAD_T - 6}
                textAnchor="middle"
                fontSize={10}
                fill="#6366f1"
                fontWeight="bold"
              >
                C={current_c}
              </text>
            </>
          )}

          {/* SV count line (dashed gray) */}
          <path
            d={svPath}
            fill="none"
            stroke="#9ca3af"
            strokeWidth={2}
            strokeDasharray="5 3"
          />

          {/* Accuracy line (solid indigo) */}
          <path
            d={scorePath}
            fill="none"
            stroke="#6366f1"
            strokeWidth={2.5}
          />

          {/* Data points - accuracy */}
          {logCValues.map((logC: number, i: number) => (
            <circle
              key={`score-${i}`}
              cx={scaleX(logC)}
              cy={scaleYScore(scores[i])}
              r={hoveredIdx === i ? 5 : 3.5}
              fill="#6366f1"
              stroke="#fff"
              strokeWidth={1.5}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
              className="cursor-pointer"
            />
          ))}

          {/* Data points - SV count */}
          {logCValues.map((logC: number, i: number) => (
            <circle
              key={`sv-${i}`}
              cx={scaleX(logC)}
              cy={scaleYSV(n_support_vectors[i])}
              r={hoveredIdx === i ? 4 : 2.5}
              fill="#9ca3af"
              stroke="#fff"
              strokeWidth={1}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
              className="cursor-pointer"
            />
          ))}

          {/* Hovered tooltip */}
          {hoveredIdx != null && (
            <g>
              <rect
                x={scaleX(logCValues[hoveredIdx]) - 55}
                y={scaleYScore(scores[hoveredIdx]) - 38}
                width={110}
                height={30}
                rx={4}
                fill="#1e1b4b"
                opacity={0.9}
              />
              <text
                x={scaleX(logCValues[hoveredIdx])}
                y={scaleYScore(scores[hoveredIdx]) - 22}
                textAnchor="middle"
                fill="white"
                fontSize={9}
                fontFamily="monospace"
              >
                C={c_values[hoveredIdx]} | Acc:{(scores[hoveredIdx] * 100).toFixed(1)}% | SVs:{n_support_vectors[hoveredIdx]}
              </text>
            </g>
          )}

          {/* Left Y-axis label (Accuracy) */}
          <text
            x={10}
            y={VB_H / 2}
            textAnchor="middle"
            fontSize={10}
            fill="#6366f1"
            fontWeight="bold"
            transform={`rotate(-90, 10, ${VB_H / 2})`}
          >
            Accuracy
          </text>

          {/* Left Y-axis ticks */}
          {[scoreRange[0], (scoreRange[0] + scoreRange[1]) / 2, scoreRange[1]].map(
            (v, i) => (
              <text
                key={`ly-${i}`}
                x={PAD_L - 5}
                y={scaleYScore(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="#6366f1"
              >
                {(v * 100).toFixed(0)}%
              </text>
            )
          )}

          {/* Right Y-axis label (Support Vectors) */}
          <text
            x={VB_W - 8}
            y={VB_H / 2}
            textAnchor="middle"
            fontSize={10}
            fill="#9ca3af"
            fontWeight="bold"
            transform={`rotate(90, ${VB_W - 8}, ${VB_H / 2})`}
          >
            Support Vectors
          </text>

          {/* Right Y-axis ticks */}
          {[svRange[0], (svRange[0] + svRange[1]) / 2, svRange[1]].map(
            (v, i) => (
              <text
                key={`ry-${i}`}
                x={VB_W - PAD_R + 5}
                y={scaleYSV(v) + 3}
                textAnchor="start"
                fontSize={9}
                fill="#9ca3af"
              >
                {Math.round(v)}
              </text>
            )
          )}

          {/* X-axis labels */}
          {c_values.map((c: number, i: number) => (
            <text
              key={`xlab-${i}`}
              x={scaleX(logCValues[i])}
              y={VB_H - PAD_B + 16}
              textAnchor="middle"
              fontSize={9}
              fill="#6b7280"
            >
              {c}
            </text>
          ))}
          <text
            x={VB_W / 2}
            y={VB_H - 4}
            textAnchor="middle"
            fontSize={11}
            fill="#6b7280"
          >
            C (regularization parameter)
          </text>
        </svg>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-2">
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <div className="w-6 h-0.5 bg-indigo-500" />
            <span>Accuracy</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <div className="w-6 h-0.5 border-t-2 border-dashed border-gray-400" />
            <span>Support Vectors</span>
          </div>
        </div>
      </div>

      {/* Annotation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <h4 className="font-semibold text-amber-900 mb-1 text-sm">
            Low C -- Wider Margin
          </h4>
          <p className="text-sm text-amber-800">
            With a small C, the SVM tolerates more misclassifications to achieve
            a wider margin. This often means more support vectors and a simpler,
            more generalizable boundary.
          </p>
        </div>
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <h4 className="font-semibold text-red-900 mb-1 text-sm">
            High C -- Narrow Margin
          </h4>
          <p className="text-sm text-red-800">
            With a large C, the SVM tries to classify every training point
            correctly, resulting in a narrow margin and fewer support vectors.
            This can lead to overfitting on noisy data.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  Tab 6 - Quiz                                                       */
/* ================================================================== */

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
              pct >= 80
                ? "text-indigo-500"
                : pct >= 50
                ? "text-yellow-500"
                : "text-red-500"
            }`}
          />
          <h3 className="text-2xl font-bold text-gray-900">
            {score} / {questions.length}
          </h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80
              ? "Excellent! You understand SVMs well!"
              : pct >= 50
              ? "Good job! Review the topics you missed."
              : "Keep learning! SVMs have many concepts to master."}
          </p>
          <button
            onClick={handleRetry}
            className="mt-4 px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
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
                    <p className="text-sm font-medium text-gray-900">
                      {q.question}
                    </p>
                    <p className="text-xs text-gray-600 mt-1">
                      Your answer: {q.options[userAns ?? 0]}{" "}
                      {!correct &&
                        `| Correct: ${q.options[q.correct_answer]}`}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {q.explanation}
                    </p>
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
      <div className="flex items-start gap-3 rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-indigo-600" />
        <p className="text-sm text-indigo-800">
          <span className="font-semibold">Test your knowledge</span> about
          Support Vector Machines! Answer {questions.length} questions to check
          your understanding.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex items-center justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-2.5 h-2.5 rounded-full transition-all ${
              i === currentQ
                ? "bg-indigo-500 scale-125"
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
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs text-gray-400">
            Question {currentQ + 1} of {questions.length}
          </p>
          {question.difficulty && (
            <span
              className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${
                question.difficulty === "easy"
                  ? "bg-green-100 text-green-700"
                  : question.difficulty === "medium"
                  ? "bg-amber-100 text-amber-700"
                  : "bg-red-100 text-red-700"
              }`}
            >
              {question.difficulty}
            </span>
          )}
        </div>
        <p className="text-base font-medium text-gray-900 mb-4">
          {question.question}
        </p>

        <div className="space-y-2">
          {question.options.map((opt: string, idx: number) => {
            let style =
              "border-gray-200 bg-white hover:bg-gray-50 text-gray-700";
            if (answered) {
              if (idx === question.correct_answer) {
                style = "border-green-300 bg-green-50 text-green-800";
              } else if (idx === selectedAnswer && !isCorrect) {
                style = "border-red-300 bg-red-50 text-red-800";
              } else {
                style = "border-gray-200 bg-gray-50 text-gray-400";
              }
            } else if (idx === selectedAnswer) {
              style = "border-indigo-400 bg-indigo-50 text-indigo-700";
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
              isCorrect
                ? "bg-green-50 text-green-800"
                : "bg-red-50 text-red-800"
            }`}
          >
            <p className="font-semibold mb-1">
              {isCorrect ? "Correct!" : "Not quite."}
            </p>
            <p>{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
          >
            {currentQ + 1 >= questions.length ? "See Results" : "Next Question"}
          </button>
        )}
      </div>
    </div>
  );
}

/* ================================================================== */
/*  Tab 7 - How It Works                                               */
/* ================================================================== */

function HowItWorksTab({ result }: { result: any }) {
  const [boundaryY, setBoundaryY] = useState(150);
  const [optimized, setOptimized] = useState(false);
  const [kernelDemo, setKernelDemo] = useState<"linear" | "rbf">("linear");
  const [dragging, setDragging] = useState(false);

  const VB_W = 500;
  const VB_H = 300;

  // Generate mock 2D data or use decision_boundary_grid points
  const mockPoints = useMemo(() => {
    const dbPoints = result?.decision_boundary_grid?.points;
    if (dbPoints && dbPoints.length > 0) {
      // Use real data, limit to 40 points for clarity
      return dbPoints.slice(0, 40).map((p: any) => ({
        x: p.x,
        y: p.y,
        label: p.label,
      }));
    }
    // Generate mock data: two clusters
    const pts: { x: number; y: number; label: string }[] = [];
    const rng = (seed: number) => {
      let s = seed;
      return () => {
        s = (s * 16807) % 2147483647;
        return s / 2147483647;
      };
    };
    const r = rng(42);
    for (let i = 0; i < 20; i++) {
      pts.push({ x: r() * 180 + 30, y: r() * 100 + 20, label: "A" });
    }
    for (let i = 0; i < 20; i++) {
      pts.push({ x: r() * 180 + 30, y: r() * 100 + 170, label: "B" });
    }
    return pts;
  }, [result]);

  // Scale for mock points
  const xRange: [number, number] = useMemo(() => {
    const xs = mockPoints.map((p: any) => p.x);
    return [Math.min(...xs) - 10, Math.max(...xs) + 10];
  }, [mockPoints]);
  const yRange: [number, number] = useMemo(() => {
    const ys = mockPoints.map((p: any) => p.y);
    return [Math.min(...ys) - 10, Math.max(...ys) + 10];
  }, [mockPoints]);

  const scaleX = useMemo(() => linScale(xRange, [40, VB_W - 20]), [xRange]);
  const scaleY = useMemo(() => linScale(yRange, [VB_H - 30, 20]), [yRange]);

  const classes = useMemo(() => {
    const set = new Set(mockPoints.map((p: any) => String(p.label)));
    return [...set];
  }, [mockPoints]);

  // Compute optimal boundary Y (midpoint between class means)
  const optimalY = useMemo(() => {
    const classA = mockPoints.filter(
      (p: any) => String(p.label) === classes[0]
    );
    const classB = mockPoints.filter(
      (p: any) => String(p.label) === classes[1]
    );
    if (!classA.length || !classB.length) return VB_H / 2;
    const meanA =
      classA.reduce((s: number, p: any) => s + scaleY(p.y), 0) /
      classA.length;
    const meanB =
      classB.reduce((s: number, p: any) => s + scaleY(p.y), 0) /
      classB.length;
    return (meanA + meanB) / 2;
  }, [mockPoints, classes, scaleY]);

  // Margin width
  const marginWidth = 30;

  const currentBoundaryY = optimized ? optimalY : boundaryY;

  // Points within margin
  const marginPoints = useMemo(() => {
    return mockPoints.filter((p: any) => {
      const py = scaleY(p.y);
      return Math.abs(py - currentBoundaryY) < marginWidth;
    });
  }, [mockPoints, currentBoundaryY, scaleY]);

  const handleDrag = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!dragging || optimized) return;
      const svg = e.currentTarget;
      const rect = svg.getBoundingClientRect();
      const y = ((e.clientY - rect.top) / rect.height) * VB_H;
      setBoundaryY(Math.max(30, Math.min(VB_H - 30, y)));
    },
    [dragging, optimized]
  );

  const handleOptimize = useCallback(() => {
    setOptimized(true);
    setDragging(false);
  }, []);

  const handleReset = useCallback(() => {
    setOptimized(false);
    setBoundaryY(150);
  }, []);

  // RBF kernel demo: curved boundary (simple quadratic approximation)
  const rbfPath = useMemo(() => {
    const cx = VB_W / 2;
    const amplitude = 40;
    const pts: string[] = [];
    for (let x = 40; x <= VB_W - 20; x += 5) {
      const offset =
        amplitude * Math.sin(((x - cx) / (VB_W - 60)) * Math.PI * 2);
      pts.push(`${x},${optimalY + offset}`);
    }
    return `M${pts.join(" L")}`;
  }, [optimalY]);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-indigo-600" />
        <div className="text-sm text-indigo-800 space-y-2">
          <p className="font-semibold">How SVM Works</p>
          <p className="text-indigo-700">
            Support Vector Machine finds the boundary that maximizes the gap
            (margin) between classes. Try dragging the boundary line below, then
            click "Optimize" to see where SVM would place it.
          </p>
        </div>
      </div>

      {/* Interactive scatter */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-500">
              Kernel demo:
            </span>
            <button
              onClick={() => setKernelDemo("linear")}
              className={`text-xs px-3 py-1 rounded-full ${
                kernelDemo === "linear"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              Linear
            </button>
            <button
              onClick={() => setKernelDemo("rbf")}
              className={`text-xs px-3 py-1 rounded-full ${
                kernelDemo === "rbf"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              RBF
            </button>
          </div>
          <div className="flex items-center gap-2">
            {!optimized ? (
              <button
                onClick={handleOptimize}
                className="text-xs px-4 py-1.5 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors font-medium"
              >
                Optimize
              </button>
            ) : (
              <button
                onClick={handleReset}
                className="text-xs px-4 py-1.5 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
              >
                Reset
              </button>
            )}
          </div>
        </div>

        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          className="w-full h-auto cursor-ns-resize"
          onMouseMove={handleDrag}
          onMouseDown={() => !optimized && setDragging(true)}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => setDragging(false)}
        >
          {/* Background split */}
          <rect
            x={0}
            y={0}
            width={VB_W}
            height={currentBoundaryY - marginWidth / 2}
            fill={classColor(0)}
            opacity={0.05}
          />
          <rect
            x={0}
            y={currentBoundaryY + marginWidth / 2}
            width={VB_W}
            height={VB_H - currentBoundaryY - marginWidth / 2}
            fill={classColor(1)}
            opacity={0.05}
          />

          {/* Margin band */}
          {kernelDemo === "linear" && (
            <rect
              x={0}
              y={currentBoundaryY - marginWidth}
              width={VB_W}
              height={marginWidth * 2}
              fill="#a5b4fc"
              opacity={0.15}
              rx={2}
            />
          )}

          {/* Decision boundary */}
          {kernelDemo === "linear" ? (
            <line
              x1={0}
              y1={currentBoundaryY}
              x2={VB_W}
              y2={currentBoundaryY}
              stroke="#6366f1"
              strokeWidth={2.5}
              strokeDasharray={optimized ? "0" : "8 4"}
              style={{ transition: optimized ? "all 0.6s ease-out" : "none" }}
            />
          ) : (
            <path
              d={rbfPath}
              fill="none"
              stroke="#6366f1"
              strokeWidth={2.5}
            />
          )}

          {/* Margin boundary lines (linear only) */}
          {kernelDemo === "linear" && (
            <>
              <line
                x1={0}
                y1={currentBoundaryY - marginWidth}
                x2={VB_W}
                y2={currentBoundaryY - marginWidth}
                stroke="#6366f1"
                strokeWidth={1}
                strokeDasharray="4 3"
                opacity={0.4}
              />
              <line
                x1={0}
                y1={currentBoundaryY + marginWidth}
                x2={VB_W}
                y2={currentBoundaryY + marginWidth}
                stroke="#6366f1"
                strokeWidth={1}
                strokeDasharray="4 3"
                opacity={0.4}
              />
            </>
          )}

          {/* Points */}
          {mockPoints.map((pt: any, i: number) => {
            const ci = classes.indexOf(String(pt.label));
            const px = scaleX(pt.x);
            const py = scaleY(pt.y);
            const isInMargin =
              kernelDemo === "linear" &&
              Math.abs(py - currentBoundaryY) < marginWidth;
            return (
              <circle
                key={i}
                cx={px}
                cy={py}
                r={isInMargin ? 5 : 4}
                fill={classColor(ci >= 0 ? ci : 0)}
                stroke={isInMargin ? "#1e1b4b" : "#fff"}
                strokeWidth={isInMargin ? 2 : 1}
                opacity={isInMargin ? 1 : 0.7}
              />
            );
          })}

          {/* Labels */}
          {optimized && kernelDemo === "linear" && (
            <>
              <text
                x={VB_W - 10}
                y={currentBoundaryY - marginWidth - 6}
                textAnchor="end"
                fontSize={9}
                fill="#6366f1"
                opacity={0.7}
              >
                margin
              </text>
              <text
                x={VB_W - 10}
                y={currentBoundaryY + 5}
                textAnchor="end"
                fontSize={10}
                fill="#4338ca"
                fontWeight="bold"
              >
                Maximum Margin Boundary
              </text>
            </>
          )}
          {!optimized && kernelDemo === "linear" && (
            <text
              x={VB_W / 2}
              y={20}
              textAnchor="middle"
              fontSize={11}
              fill="#6b7280"
            >
              Drag the boundary line up/down, then click Optimize
            </text>
          )}
        </svg>

        {/* Margin stats */}
        {kernelDemo === "linear" && (
          <div className="flex items-center justify-center gap-4 mt-2 text-xs text-gray-500">
            <span>
              Points in margin:{" "}
              <strong className="text-indigo-700">{marginPoints.length}</strong>
            </span>
            <span>
              {optimized ? "Boundary optimized for maximum margin" : "Move the boundary to explore"}
            </span>
          </div>
        )}
      </div>

      {/* Info cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Target className="w-4 h-4 text-indigo-600" />
            Maximum Margin
          </h4>
          <p className="text-sm text-gray-600">
            SVM finds the boundary that maximizes the gap between classes. A
            wider margin means the model is more confident and likely to
            generalize better to unseen data.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <CircleDot className="w-4 h-4 text-indigo-600" />
            Support Vectors
          </h4>
          <p className="text-sm text-gray-600">
            Only the closest points to the boundary (support vectors) define it.
            If you remove any non-support-vector point, the boundary stays the
            same. This makes SVMs memory-efficient.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Zap className="w-4 h-4 text-indigo-600" />
            Kernel Trick
          </h4>
          <p className="text-sm text-gray-600">
            For non-linear data, kernels implicitly map points to a
            higher-dimensional space where a linear boundary exists. The RBF
            kernel can create complex curved boundaries without explicitly
            computing the transformation.
          </p>
        </div>
      </div>

      {/* Settings You Can Tune */}
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <h4 className="font-semibold text-gray-900 mb-2">
          Settings You Can Tune
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="flex gap-2">
            <span className="font-semibold text-indigo-700 shrink-0">
              kernel:
            </span>
            <span className="text-gray-600">
              The function used to transform data (linear, rbf, poly).
              Determines the shape of the decision boundary.
            </span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-indigo-700 shrink-0">C:</span>
            <span className="text-gray-600">
              Regularization parameter. Higher C = stricter classification
              (narrower margin). Lower C = more tolerance (wider margin).
            </span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-indigo-700 shrink-0">
              gamma:
            </span>
            <span className="text-gray-600">
              Controls reach of each training example (RBF/poly). High gamma =
              points must be very close to influence each other.
            </span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-indigo-700 shrink-0">
              degree:
            </span>
            <span className="text-gray-600">
              Polynomial degree (only for poly kernel). Higher degree = more
              complex curves.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SVMExplorer;
