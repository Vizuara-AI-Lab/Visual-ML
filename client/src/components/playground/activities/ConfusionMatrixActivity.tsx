/**
 * Confusion Matrix Activity — comprehensive interactive explorer
 * 5 tabs: Threshold Explorer, Metric Deep Dive, Real-World Scenarios,
 *         Multi-Class Matrix, Build Your Own Classifier
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Info,
  Table,
  SlidersHorizontal,
  BarChart3,
  Activity,
  Target,
  Brain,
  AlertTriangle,
  Grid3X3,
  MousePointerClick,
  RotateCcw,
  ChevronRight,
  Zap,
  Shield,
  Heart,
  Mail,
  DollarSign,
  Scale,
  TrendingUp,
  Eye,
  Award,
  Crosshair,
} from "lucide-react";

// =========================================================================
// Utility: Seeded random
// =========================================================================

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function gaussianSample(rand: () => number, mean: number, std: number): number {
  const u1 = rand();
  const u2 = rand();
  const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

// =========================================================================
// Types
// =========================================================================

interface Sample {
  id: number;
  trueLabel: 0 | 1;
  predictedProb: number;
}

interface ConfusionCounts {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
}

interface MultiClassSample {
  id: number;
  trueClass: number;
  predictedClass: number;
  probs: number[];
}

interface ClassifyPoint {
  id: number;
  x: number;
  y: number;
  trueLabel: 0 | 1;
  userPrediction: 0 | 1 | null;
  aiPrediction: 0 | 1;
}

type TabId =
  | "threshold"
  | "deepdive"
  | "scenarios"
  | "multiclass"
  | "buildyourown";

// =========================================================================
// Data generation
// =========================================================================

function generateSamples(
  count: number,
  seed: number,
  negMean = 0.35,
  posMean = 0.65,
  negStd = 0.15,
  posStd = 0.15,
  negFraction = 0.5
): Sample[] {
  const rand = seededRandom(seed);
  const samples: Sample[] = [];
  const negCount = Math.floor(count * negFraction);
  const posCount = count - negCount;

  for (let i = 0; i < negCount; i++) {
    const prob = Math.max(0, Math.min(1, gaussianSample(rand, negMean, negStd)));
    samples.push({ id: i, trueLabel: 0, predictedProb: prob });
  }
  for (let i = 0; i < posCount; i++) {
    const prob = Math.max(0, Math.min(1, gaussianSample(rand, posMean, posStd)));
    samples.push({ id: negCount + i, trueLabel: 1, predictedProb: prob });
  }
  return samples;
}

function generateMultiClassSamples(
  count: number,
  numClasses: number,
  seed: number
): MultiClassSample[] {
  const rand = seededRandom(seed);
  const samples: MultiClassSample[] = [];
  const perClass = Math.floor(count / numClasses);

  for (let c = 0; c < numClasses; c++) {
    for (let i = 0; i < perClass; i++) {
      const probs: number[] = [];
      let sum = 0;
      for (let k = 0; k < numClasses; k++) {
        const base = k === c ? 2.5 + rand() * 1.5 : 0.3 + rand() * 0.8;
        probs.push(base);
        sum += base;
      }
      for (let k = 0; k < numClasses; k++) probs[k] /= sum;

      // Add confusion between adjacent classes
      if (rand() < 0.15) {
        const confusedWith = (c + 1) % numClasses;
        const swap = probs[c] * 0.6;
        probs[c] -= swap;
        probs[confusedWith] += swap;
      }

      const predictedClass = probs.indexOf(Math.max(...probs));
      samples.push({
        id: c * perClass + i,
        trueClass: c,
        predictedClass,
        probs,
      });
    }
  }
  return samples;
}

function generateClassifyPoints(seed: number): ClassifyPoint[] {
  const rand = seededRandom(seed);
  const points: ClassifyPoint[] = [];

  for (let i = 0; i < 60; i++) {
    const trueLabel: 0 | 1 = i < 30 ? 0 : 1;
    let x: number, y: number;
    if (trueLabel === 0) {
      x = 30 + gaussianSample(rand, 0, 18);
      y = 50 + gaussianSample(rand, 0, 22);
    } else {
      x = 70 + gaussianSample(rand, 0, 18);
      y = 50 + gaussianSample(rand, 0, 22);
    }
    x = Math.max(5, Math.min(95, x));
    y = Math.max(5, Math.min(95, y));

    const boundary = 50;
    const aiPrediction: 0 | 1 = x >= boundary ? 1 : 0;

    points.push({
      id: i,
      x,
      y,
      trueLabel,
      userPrediction: null,
      aiPrediction,
    });
  }
  return points;
}

// =========================================================================
// Computation helpers
// =========================================================================

function computeConfusion(samples: Sample[], threshold: number): ConfusionCounts {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const s of samples) {
    const predicted = s.predictedProb >= threshold ? 1 : 0;
    if (s.trueLabel === 1 && predicted === 1) tp++;
    else if (s.trueLabel === 0 && predicted === 1) fp++;
    else if (s.trueLabel === 0 && predicted === 0) tn++;
    else fn++;
  }
  return { tp, fp, tn, fn };
}

function computeMetrics(c: ConfusionCounts): Metrics {
  const total = c.tp + c.fp + c.tn + c.fn;
  const accuracy = total > 0 ? (c.tp + c.tn) / total : 0;
  const precision = c.tp + c.fp > 0 ? c.tp / (c.tp + c.fp) : 0;
  const recall = c.tp + c.fn > 0 ? c.tp / (c.tp + c.fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const specificity = c.tn + c.fp > 0 ? c.tn / (c.tn + c.fp) : 0;
  return { accuracy, precision, recall, f1, specificity };
}

function cellColor(
  count: number,
  maxCount: number,
  baseColor: [number, number, number]
): string {
  const t = maxCount > 0 ? count / maxCount : 0;
  const r = Math.round(255 - t * (255 - baseColor[0]));
  const g = Math.round(255 - t * (255 - baseColor[1]));
  const b = Math.round(255 - t * (255 - baseColor[2]));
  return `rgb(${r}, ${g}, ${b})`;
}

function textColorForIntensity(count: number, maxCount: number): string {
  const t = maxCount > 0 ? count / maxCount : 0;
  return t > 0.45 ? "#ffffff" : "#1e293b";
}

// =========================================================================
// Density curve for distributions
// =========================================================================

function buildDensityCurve(
  values: number[],
  numPoints: number,
  bandwidth: number
): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = [];
  for (let i = 0; i <= numPoints; i++) {
    const x = i / numPoints;
    let density = 0;
    for (const v of values) {
      const u = (x - v) / bandwidth;
      density += Math.exp(-0.5 * u * u);
    }
    density /= values.length * bandwidth * Math.sqrt(2 * Math.PI);
    points.push({ x, y: density });
  }
  return points;
}

// =========================================================================
// Distribution SVG constants
// =========================================================================

const DIST_W = 440;
const DIST_H = 120;
const DIST_PAD = { top: 14, right: 15, bottom: 24, left: 15 };
const DIST_PLOT_W = DIST_W - DIST_PAD.left - DIST_PAD.right;
const DIST_PLOT_H = DIST_H - DIST_PAD.top - DIST_PAD.bottom;

// =========================================================================
// Tab definitions
// =========================================================================

const TAB_DEFS: { id: TabId; label: string; icon: typeof Info }[] = [
  { id: "threshold", label: "Threshold Explorer", icon: SlidersHorizontal },
  { id: "deepdive", label: "Metric Deep Dive", icon: TrendingUp },
  { id: "scenarios", label: "Real-World Scenarios", icon: AlertTriangle },
  { id: "multiclass", label: "Multi-Class Matrix", icon: Grid3X3 },
  { id: "buildyourown", label: "Build Your Own", icon: MousePointerClick },
];

// =========================================================================
// Main component
// =========================================================================

export default function ConfusionMatrixActivity() {
  const [activeTab, setActiveTab] = useState<TabId>("threshold");

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-100 p-1 rounded-xl overflow-x-auto">
        {TAB_DEFS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                isActive
                  ? "bg-white text-indigo-700 shadow-sm"
                  : "text-slate-500 hover:text-slate-700 hover:bg-white/50"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      {activeTab === "threshold" && <ThresholdExplorerTab />}
      {activeTab === "deepdive" && <MetricDeepDiveTab />}
      {activeTab === "scenarios" && <RealWorldScenariosTab />}
      {activeTab === "multiclass" && <MultiClassMatrixTab />}
      {activeTab === "buildyourown" && <BuildYourOwnTab />}
    </div>
  );
}

// =========================================================================
// TAB 1: Threshold Explorer
// =========================================================================

function ThresholdExplorerTab() {
  const [threshold, setThreshold] = useState(0.5);
  const [prevThreshold, setPrevThreshold] = useState(0.5);
  const [animatingCells, setAnimatingCells] = useState<Record<string, boolean>>({});

  const samples = useMemo(() => generateSamples(100, 42), []);
  const confusion = useMemo(() => computeConfusion(samples, threshold), [samples, threshold]);
  const metrics = useMemo(() => computeMetrics(confusion), [confusion]);
  const prevConfusion = useMemo(
    () => computeConfusion(samples, prevThreshold),
    [samples, prevThreshold]
  );

  const cellChanged = useMemo(
    () => ({
      tp: confusion.tp !== prevConfusion.tp,
      fp: confusion.fp !== prevConfusion.fp,
      tn: confusion.tn !== prevConfusion.tn,
      fn: confusion.fn !== prevConfusion.fn,
    }),
    [confusion, prevConfusion]
  );

  const maxCount = Math.max(confusion.tp, confusion.fp, confusion.tn, confusion.fn, 1);

  const switchedIds = useMemo(() => {
    const ids = new Set<number>();
    for (const s of samples) {
      const oldPred = s.predictedProb >= prevThreshold ? 1 : 0;
      const newPred = s.predictedProb >= threshold ? 1 : 0;
      if (oldPred !== newPred) ids.add(s.id);
    }
    return ids;
  }, [samples, threshold, prevThreshold]);

  const { negCurve, posCurve, maxDensity } = useMemo(() => {
    const neg = samples.filter((s) => s.trueLabel === 0).map((s) => s.predictedProb);
    const pos = samples.filter((s) => s.trueLabel === 1).map((s) => s.predictedProb);
    const bandwidth = 0.06;
    const nCurve = buildDensityCurve(neg, 100, bandwidth);
    const pCurve = buildDensityCurve(pos, 100, bandwidth);
    const maxD = Math.max(...nCurve.map((p) => p.y), ...pCurve.map((p) => p.y));
    return { negCurve: nCurve, posCurve: pCurve, maxDensity: maxD };
  }, [samples]);

  const toDistX = (x: number) => DIST_PAD.left + x * DIST_PLOT_W;
  const toDistY = (y: number) => {
    const t = maxDensity > 0 ? y / maxDensity : 0;
    return DIST_PAD.top + DIST_PLOT_H * (1 - t);
  };

  const negPath = negCurve
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toDistX(p.x).toFixed(2)} ${toDistY(p.y).toFixed(2)}`)
    .join(" ");
  const posPath = posCurve
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toDistX(p.x).toFixed(2)} ${toDistY(p.y).toFixed(2)}`)
    .join(" ");

  const negFillPath = negPath + ` L ${toDistX(1).toFixed(2)} ${toDistY(0).toFixed(2)} L ${toDistX(0).toFixed(2)} ${toDistY(0).toFixed(2)} Z`;
  const posFillPath = posPath + ` L ${toDistX(1).toFixed(2)} ${toDistY(0).toFixed(2)} L ${toDistX(0).toFixed(2)} ${toDistY(0).toFixed(2)} Z`;
  const thresholdX = toDistX(threshold);

  const handleThresholdChange = useCallback(
    (newThreshold: number) => {
      setPrevThreshold(threshold);
      setThreshold(newThreshold);
      setAnimatingCells({ tp: true, fp: true, tn: true, fn: true });
      setTimeout(() => setAnimatingCells({}), 400);
    },
    [threshold]
  );

  const matrixCells = [
    { key: "tp", label: "True Positive", abbr: "TP", count: confusion.tp, row: 0, col: 0, baseColor: [34, 197, 94] as [number, number, number], description: "Correctly predicted positive" },
    { key: "fp", label: "False Positive", abbr: "FP", count: confusion.fp, row: 0, col: 1, baseColor: [239, 68, 68] as [number, number, number], description: "Incorrectly predicted positive" },
    { key: "fn", label: "False Negative", abbr: "FN", count: confusion.fn, row: 1, col: 0, baseColor: [249, 115, 22] as [number, number, number], description: "Incorrectly predicted negative" },
    { key: "tn", label: "True Negative", abbr: "TN", count: confusion.tn, row: 1, col: 1, baseColor: [59, 130, 246] as [number, number, number], description: "Correctly predicted negative" },
  ];

  const metricDefs = [
    { name: "Accuracy", value: metrics.accuracy, color: "#6366f1", tip: "(TP+TN) / Total" },
    { name: "Precision", value: metrics.precision, color: "#22c55e", tip: "TP / (TP+FP)" },
    { name: "Recall", value: metrics.recall, color: "#f59e0b", tip: "TP / (TP+FN)" },
    { name: "F1 Score", value: metrics.f1, color: "#ef4444", tip: "2PR / (P+R)" },
    { name: "Specificity", value: metrics.specificity, color: "#3b82f6", tip: "TN / (TN+FP)" },
  ];

  return (
    <div className="space-y-5">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">
            Understanding the Confusion Matrix
          </h3>
          <p className="text-xs text-indigo-700 mt-1">
            A confusion matrix shows all possible prediction outcomes. <strong>True Positives (TP)</strong> and{" "}
            <strong>True Negatives (TN)</strong> are correct; <strong>False Positives (FP)</strong> and{" "}
            <strong>False Negatives (FN)</strong> are errors. Adjust the threshold to see how metrics trade off.
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-5">
          {/* Threshold slider */}
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                <SlidersHorizontal className="w-4 h-4 text-indigo-500" />
                Classification Threshold
              </label>
              <span className="text-sm font-mono font-bold text-indigo-600 bg-indigo-50 px-3 py-1 rounded-lg">
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
              className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>0.0 (predict all positive)</span>
              <span>1.0 (predict all negative)</span>
            </div>

            {/* Quick presets */}
            <div className="flex gap-1.5 flex-wrap mt-3">
              <span className="text-[10px] text-slate-400 self-center mr-1">Presets:</span>
              {[0.2, 0.3, 0.5, 0.7, 0.8].map((t) => (
                <button
                  key={t}
                  onClick={() => handleThresholdChange(t)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                    Math.abs(threshold - t) < 0.005
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {t.toFixed(1)}
                </button>
              ))}
            </div>
          </div>

          {/* 2x2 Confusion Matrix */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Table className="w-4 h-4 text-indigo-500" />
              <h4 className="text-sm font-semibold text-slate-700">Confusion Matrix</h4>
              {switchedIds.size > 0 && (
                <span className="text-[10px] text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full font-medium animate-pulse">
                  {switchedIds.size} sample{switchedIds.size !== 1 ? "s" : ""} reclassified
                </span>
              )}
            </div>

            <div className="max-w-md mx-auto">
              <div className="text-center mb-2">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Predicted
                </span>
              </div>
              <div className="flex">
                <div className="flex items-center mr-2">
                  <span
                    className="text-xs font-semibold text-slate-500 uppercase tracking-wider"
                    style={{ writingMode: "vertical-lr", transform: "rotate(180deg)" }}
                  >
                    Actual
                  </span>
                </div>
                <div className="flex-1">
                  <div className="grid grid-cols-2 gap-1 mb-1">
                    <div className="text-center">
                      <span className="text-[11px] font-semibold text-slate-600">Positive</span>
                    </div>
                    <div className="text-center">
                      <span className="text-[11px] font-semibold text-slate-600">Negative</span>
                    </div>
                  </div>

                  {[0, 1].map((row) => (
                    <div key={row} className="flex items-center gap-1 mb-1">
                      <div className="grid grid-cols-2 gap-1 flex-1">
                        {matrixCells
                          .filter((c) => c.row === row)
                          .map((cell) => (
                            <div
                              key={cell.key}
                              className={`relative rounded-xl p-4 text-center transition-all duration-300 shadow-sm ${
                                animatingCells[cell.key] && cellChanged[cell.key as keyof typeof cellChanged]
                                  ? "scale-105"
                                  : "scale-100"
                              }`}
                              style={{
                                backgroundColor: cellColor(cell.count, maxCount, cell.baseColor),
                                boxShadow: cellChanged[cell.key as keyof typeof cellChanged]
                                  ? `0 0 16px 2px rgba(${cell.baseColor.join(",")}, 0.5)`
                                  : "0 1px 3px rgba(0,0,0,0.08)",
                              }}
                            >
                              <div
                                className="text-[11px] font-bold uppercase tracking-wider mb-1 transition-colors duration-300"
                                style={{
                                  color: textColorForIntensity(cell.count, maxCount),
                                  opacity: 0.8,
                                }}
                              >
                                {cell.abbr}
                              </div>
                              <div
                                className="text-3xl font-extrabold tabular-nums transition-colors duration-300"
                                style={{ color: textColorForIntensity(cell.count, maxCount) }}
                              >
                                {cell.count}
                              </div>
                              <div
                                className="text-[9px] mt-1 transition-colors duration-300"
                                style={{
                                  color: textColorForIntensity(cell.count, maxCount),
                                  opacity: 0.7,
                                }}
                              >
                                {cell.description}
                              </div>
                            </div>
                          ))}
                      </div>
                      <span className="text-[11px] font-semibold text-slate-500 w-10 text-center">
                        {row === 0 ? "Pos" : "Neg"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Probability distribution */}
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-4 h-4 text-indigo-500" />
              <h4 className="text-sm font-semibold text-slate-700">
                Predicted Probability Distributions
              </h4>
            </div>
            <svg
              viewBox={`0 0 ${DIST_W} ${DIST_H}`}
              className="w-full max-w-[460px] mx-auto"
              style={{ aspectRatio: `${DIST_W}/${DIST_H}` }}
            >
              <rect
                x={DIST_PAD.left} y={DIST_PAD.top}
                width={DIST_PLOT_W} height={DIST_PLOT_H}
                fill="#f8fafc" stroke="#e2e8f0" strokeWidth={0.5} rx={3}
              />
              <path d={negFillPath} fill="#ef4444" opacity={0.15} />
              <path d={posFillPath} fill="#22c55e" opacity={0.15} />
              <path d={negPath} fill="none" stroke="#ef4444" strokeWidth={2} opacity={0.8} />
              <path d={posPath} fill="none" stroke="#22c55e" strokeWidth={2} opacity={0.8} />

              <line
                x1={thresholdX} y1={DIST_PAD.top}
                x2={thresholdX} y2={DIST_PAD.top + DIST_PLOT_H}
                stroke="#4f46e5" strokeWidth={2} strokeDasharray="4,3"
              />
              <text
                x={thresholdX} y={DIST_PAD.top - 2}
                fontSize={9} fill="#4f46e5" textAnchor="middle" fontWeight={700}
              >
                T={threshold.toFixed(2)}
              </text>

              <text
                x={DIST_PAD.left + (thresholdX - DIST_PAD.left) / 2}
                y={DIST_PAD.top + DIST_PLOT_H - 4}
                fontSize={8} fill="#64748b" textAnchor="middle" fontWeight={600}
              >
                Predict Negative
              </text>
              <text
                x={thresholdX + (DIST_PAD.left + DIST_PLOT_W - thresholdX) / 2}
                y={DIST_PAD.top + DIST_PLOT_H - 4}
                fontSize={8} fill="#64748b" textAnchor="middle" fontWeight={600}
              >
                Predict Positive
              </text>

              {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
                <g key={tick}>
                  <line
                    x1={toDistX(tick)} y1={DIST_PAD.top + DIST_PLOT_H}
                    x2={toDistX(tick)} y2={DIST_PAD.top + DIST_PLOT_H + 4}
                    stroke="#94a3b8" strokeWidth={0.5}
                  />
                  <text
                    x={toDistX(tick)} y={DIST_PAD.top + DIST_PLOT_H + 14}
                    fontSize={8} fill="#94a3b8" textAnchor="middle"
                  >
                    {tick.toFixed(2)}
                  </text>
                </g>
              ))}

              <g transform={`translate(${DIST_PAD.left + 6}, ${DIST_PAD.top + 4})`}>
                <rect x={0} y={0} width={102} height={28} rx={3} fill="white" fillOpacity={0.92} stroke="#e2e8f0" strokeWidth={0.5} />
                <line x1={6} y1={9} x2={18} y2={9} stroke="#ef4444" strokeWidth={2} />
                <text x={22} y={12} fontSize={8} fill="#64748b">Actual Negative</text>
                <line x1={6} y1={21} x2={18} y2={21} stroke="#22c55e" strokeWidth={2} />
                <text x={22} y={24} fontSize={8} fill="#64748b">Actual Positive</text>
              </g>
            </svg>

            {/* Sample dots below the distribution */}
            <div className="mt-2 flex flex-wrap gap-[3px] justify-center">
              {samples.map((s) => {
                const predicted = s.predictedProb >= threshold ? 1 : 0;
                const correct = s.trueLabel === predicted;
                const switched = switchedIds.has(s.id);
                let bg = correct
                  ? s.trueLabel === 1 ? "#22c55e" : "#3b82f6"
                  : s.trueLabel === 1 ? "#f97316" : "#ef4444";
                return (
                  <div
                    key={s.id}
                    className={`w-2 h-2 rounded-full transition-all duration-300 ${
                      switched ? "scale-150 ring-2 ring-yellow-400" : ""
                    }`}
                    style={{ backgroundColor: bg }}
                    title={`ID:${s.id} True:${s.trueLabel} Prob:${s.predictedProb.toFixed(2)} Pred:${predicted}`}
                  />
                );
              })}
            </div>
            <div className="flex gap-4 justify-center mt-2 text-[9px] text-slate-400">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#22c55e] inline-block" /> TP</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#3b82f6] inline-block" /> TN</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#ef4444] inline-block" /> FP</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#f97316] inline-block" /> FN</span>
            </div>
          </div>
        </div>

        {/* Side panel */}
        <div className="w-full lg:w-72 space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="w-4 h-4 text-indigo-500" />
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">
                Classification Metrics
              </p>
            </div>
            <div className="space-y-2.5">
              {metricDefs.map((metric) => (
                <div key={metric.name}>
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-xs text-slate-600 font-medium">{metric.name}</span>
                    <span className="text-xs font-mono font-bold text-slate-800">
                      {(metric.value * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{ width: `${metric.value * 100}%`, backgroundColor: metric.color }}
                    />
                  </div>
                  <p className="text-[9px] text-slate-400 mt-0.5">{metric.tip}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Counts Summary
            </p>
            <div className="grid grid-cols-2 gap-2">
              {matrixCells.map((cell) => (
                <div
                  key={cell.key}
                  className="flex items-center gap-2 rounded-lg p-2 transition-all duration-200"
                  style={{
                    backgroundColor: `rgba(${cell.baseColor.join(",")}, 0.08)`,
                    borderLeft: `3px solid rgb(${cell.baseColor.join(",")})`,
                  }}
                >
                  <span className="text-lg font-extrabold tabular-nums" style={{ color: `rgb(${cell.baseColor.join(",")})` }}>
                    {cell.count}
                  </span>
                  <span className="text-[10px] text-slate-600 font-semibold">{cell.abbr}</span>
                </div>
              ))}
            </div>
            <div className="mt-2 pt-2 border-t border-slate-100 flex justify-between text-[10px] text-slate-400">
              <span>Total samples: {samples.length}</span>
              <span>Correct: {confusion.tp + confusion.tn}</span>
            </div>
          </div>

          {/* Insight card */}
          <div
            className={`rounded-lg p-3 border transition-all duration-300 ${
              metrics.precision > 0.7 && metrics.recall > 0.7
                ? "bg-green-50 border-green-200"
                : metrics.precision > metrics.recall + 0.2
                  ? "bg-blue-50 border-blue-200"
                  : metrics.recall > metrics.precision + 0.2
                    ? "bg-amber-50 border-amber-200"
                    : "bg-slate-50 border-slate-200"
            }`}
          >
            <p className="text-xs font-semibold text-slate-700 mb-1">
              {metrics.precision > 0.7 && metrics.recall > 0.7
                ? "Good Balance"
                : metrics.precision > metrics.recall + 0.2
                  ? "High Precision, Low Recall"
                  : metrics.recall > metrics.precision + 0.2
                    ? "High Recall, Low Precision"
                    : "Moderate Performance"}
            </p>
            <p className="text-[10px] text-slate-500">
              {metrics.precision > 0.7 && metrics.recall > 0.7
                ? "Both precision and recall are above 70%. The model finds most positives without too many false alarms."
                : metrics.precision > metrics.recall + 0.2
                  ? "Conservative model: when it says positive it is usually right, but misses many actual positives."
                  : metrics.recall > metrics.precision + 0.2
                    ? "Aggressive model: catches most positives, but raises many false alarms."
                    : "Adjust the threshold to find the best tradeoff for your use case."}
            </p>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Try these experiments:</p>
            <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
              <li>Set threshold to 0.2 to maximize recall</li>
              <li>Set threshold to 0.8 to maximize precision</li>
              <li>Find the threshold that maximizes F1 score</li>
              <li>Watch how lowering the threshold trades FN for FP</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// TAB 2: Metric Deep Dive
// =========================================================================

function MetricDeepDiveTab() {
  const [threshold, setThreshold] = useState(0.5);
  const samples = useMemo(() => generateSamples(100, 42), []);
  const confusion = useMemo(() => computeConfusion(samples, threshold), [samples, threshold]);
  const metrics = useMemo(() => computeMetrics(confusion), [confusion]);

  const handleThresholdChange = useCallback((val: number) => setThreshold(val), []);

  const metricDetails = useMemo(
    () => [
      {
        name: "Accuracy",
        value: metrics.accuracy,
        color: "#6366f1",
        formula: "(TP + TN) / (TP + TN + FP + FN)",
        numeratorParts: [
          { label: "TP", value: confusion.tp, color: "#22c55e" },
          { label: "TN", value: confusion.tn, color: "#3b82f6" },
        ],
        denominatorParts: [
          { label: "TP", value: confusion.tp, color: "#22c55e" },
          { label: "TN", value: confusion.tn, color: "#3b82f6" },
          { label: "FP", value: confusion.fp, color: "#ef4444" },
          { label: "FN", value: confusion.fn, color: "#f97316" },
        ],
        bestFor: "General overview of model correctness. Works best with balanced classes.",
        icon: Target,
      },
      {
        name: "Precision",
        value: metrics.precision,
        color: "#22c55e",
        formula: "TP / (TP + FP)",
        numeratorParts: [{ label: "TP", value: confusion.tp, color: "#22c55e" }],
        denominatorParts: [
          { label: "TP", value: confusion.tp, color: "#22c55e" },
          { label: "FP", value: confusion.fp, color: "#ef4444" },
        ],
        bestFor: "Use when false positives are costly (e.g., spam filter: do not lose real mail).",
        icon: Crosshair,
      },
      {
        name: "Recall",
        value: metrics.recall,
        color: "#f59e0b",
        formula: "TP / (TP + FN)",
        numeratorParts: [{ label: "TP", value: confusion.tp, color: "#22c55e" }],
        denominatorParts: [
          { label: "TP", value: confusion.tp, color: "#22c55e" },
          { label: "FN", value: confusion.fn, color: "#f97316" },
        ],
        bestFor: "Use when false negatives are costly (e.g., medical diagnosis: do not miss a disease).",
        icon: Eye,
      },
      {
        name: "F1 Score",
        value: metrics.f1,
        color: "#ef4444",
        formula: "2 * (Precision * Recall) / (Precision + Recall)",
        numeratorParts: [
          { label: "Prec", value: metrics.precision, color: "#22c55e" },
          { label: "Rec", value: metrics.recall, color: "#f59e0b" },
        ],
        denominatorParts: [
          { label: "Prec", value: metrics.precision, color: "#22c55e" },
          { label: "Rec", value: metrics.recall, color: "#f59e0b" },
        ],
        bestFor: "Harmonic mean of precision and recall. Balances both concerns in a single number.",
        icon: Scale,
      },
      {
        name: "Specificity",
        value: metrics.specificity,
        color: "#3b82f6",
        formula: "TN / (TN + FP)",
        numeratorParts: [{ label: "TN", value: confusion.tn, color: "#3b82f6" }],
        denominatorParts: [
          { label: "TN", value: confusion.tn, color: "#3b82f6" },
          { label: "FP", value: confusion.fp, color: "#ef4444" },
        ],
        bestFor: "How well the model identifies negatives. Important in screening tests.",
        icon: Shield,
      },
    ],
    [metrics, confusion]
  );

  // Radar chart
  const radarSize = 200;
  const radarCenter = radarSize / 2;
  const radarRadius = 75;
  const radarMetrics = [
    { name: "Acc", value: metrics.accuracy },
    { name: "Prec", value: metrics.precision },
    { name: "Rec", value: metrics.recall },
    { name: "F1", value: metrics.f1 },
    { name: "Spec", value: metrics.specificity },
  ];

  const radarPoints = radarMetrics.map((m, i) => {
    const angle = (2 * Math.PI * i) / radarMetrics.length - Math.PI / 2;
    return {
      x: radarCenter + Math.cos(angle) * radarRadius * m.value,
      y: radarCenter + Math.sin(angle) * radarRadius * m.value,
      labelX: radarCenter + Math.cos(angle) * (radarRadius + 18),
      labelY: radarCenter + Math.sin(angle) * (radarRadius + 18),
      name: m.name,
      value: m.value,
    };
  });

  const radarPath = radarPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ") + " Z";

  return (
    <div className="space-y-5">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 flex gap-3">
        <Brain className="w-5 h-5 text-purple-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-purple-900">Metric Deep Dive</h3>
          <p className="text-xs text-purple-700 mt-1">
            Each metric tells a different story about model performance. Explore their formulas,
            see which confusion matrix cells contribute to each, and understand when to use each metric.
            The radar chart gives you a holistic view at a glance.
          </p>
        </div>
      </div>

      {/* Synced threshold slider */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-purple-500" />
            Threshold (synced across all metrics)
          </label>
          <span className="text-sm font-mono font-bold text-purple-600 bg-purple-50 px-3 py-1 rounded-lg">
            {threshold.toFixed(2)}
          </span>
        </div>
        <input
          type="range" min={0} max={1} step={0.01} value={threshold}
          onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
          className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
        />
      </div>

      <div className="flex gap-6 flex-col xl:flex-row">
        {/* Metric cards */}
        <div className="flex-1 space-y-4">
          {metricDetails.map((md) => {
            const Icon = md.icon;
            return (
              <div key={md.name} className="bg-white border border-slate-200 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Icon className="w-4 h-4" style={{ color: md.color }} />
                  <h4 className="text-sm font-bold text-slate-700">{md.name}</h4>
                  <span
                    className="ml-auto text-lg font-extrabold font-mono tabular-nums"
                    style={{ color: md.color }}
                  >
                    {(md.value * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Formula with colored highlights */}
                <div className="bg-slate-50 rounded-lg p-3 mb-3 font-mono text-xs text-slate-600">
                  <span className="text-slate-400">Formula: </span>
                  {md.formula}
                </div>

                {/* Building blocks diagram */}
                <div className="mb-3">
                  <p className="text-[10px] text-slate-400 uppercase font-medium mb-1.5">Building Blocks</p>
                  <div className="flex items-center gap-2 flex-wrap">
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-slate-400">Numerator:</span>
                      {md.numeratorParts.map((part, i) => (
                        <span key={i} className="flex items-center gap-0.5">
                          {i > 0 && <span className="text-slate-400 text-[10px]">+</span>}
                          <span
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold text-white"
                            style={{ backgroundColor: part.color }}
                          >
                            {part.label}={typeof part.value === "number" && part.value < 1
                              ? (part.value * 100).toFixed(0) + "%"
                              : part.value}
                          </span>
                        </span>
                      ))}
                    </div>
                    <span className="text-slate-300">/</span>
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-slate-400">Denominator:</span>
                      {md.denominatorParts.map((part, i) => (
                        <span key={i} className="flex items-center gap-0.5">
                          {i > 0 && <span className="text-slate-400 text-[10px]">+</span>}
                          <span
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold border"
                            style={{ borderColor: part.color, color: part.color }}
                          >
                            {part.label}={typeof part.value === "number" && part.value < 1
                              ? (part.value * 100).toFixed(0) + "%"
                              : part.value}
                          </span>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Animated bar */}
                <div className="w-full h-4 bg-slate-100 rounded-full overflow-hidden mb-2">
                  <div
                    className="h-full rounded-full transition-all duration-500 ease-out flex items-center justify-end pr-1"
                    style={{ width: `${Math.max(md.value * 100, 2)}%`, backgroundColor: md.color }}
                  >
                    {md.value > 0.15 && (
                      <span className="text-[8px] text-white font-bold">
                        {(md.value * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>

                {/* Best for */}
                <div className="flex items-start gap-2 bg-amber-50 rounded-lg p-2.5">
                  <Zap className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
                  <p className="text-[11px] text-amber-700">{md.bestFor}</p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Radar chart */}
        <div className="w-full xl:w-64 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4 sticky top-4">
            <h4 className="text-sm font-bold text-slate-700 mb-3 text-center">All Metrics at a Glance</h4>
            <svg viewBox={`0 0 ${radarSize} ${radarSize}`} className="w-full mx-auto max-w-[220px]">
              {/* Grid rings */}
              {[0.25, 0.5, 0.75, 1.0].map((ring) => {
                const pts = radarMetrics.map((_, i) => {
                  const angle = (2 * Math.PI * i) / radarMetrics.length - Math.PI / 2;
                  return `${radarCenter + Math.cos(angle) * radarRadius * ring},${radarCenter + Math.sin(angle) * radarRadius * ring}`;
                });
                return (
                  <polygon
                    key={ring}
                    points={pts.join(" ")}
                    fill="none"
                    stroke="#e2e8f0"
                    strokeWidth={0.5}
                  />
                );
              })}

              {/* Axis lines */}
              {radarMetrics.map((_, i) => {
                const angle = (2 * Math.PI * i) / radarMetrics.length - Math.PI / 2;
                const ex = radarCenter + Math.cos(angle) * radarRadius;
                const ey = radarCenter + Math.sin(angle) * radarRadius;
                return (
                  <line
                    key={i}
                    x1={radarCenter} y1={radarCenter} x2={ex} y2={ey}
                    stroke="#e2e8f0" strokeWidth={0.5}
                  />
                );
              })}

              {/* Filled area */}
              <path d={radarPath} fill="#6366f1" fillOpacity={0.15} stroke="#6366f1" strokeWidth={1.5} />

              {/* Points and labels */}
              {radarPoints.map((p, i) => (
                <g key={i}>
                  <circle cx={p.x} cy={p.y} r={3.5} fill="#6366f1" stroke="white" strokeWidth={1.5} />
                  <text
                    x={p.labelX} y={p.labelY}
                    fontSize={9} fill="#475569" textAnchor="middle" dominantBaseline="middle"
                    fontWeight={600}
                  >
                    {p.name}
                  </text>
                  <text
                    x={p.labelX} y={p.labelY + 11}
                    fontSize={8} fill="#6366f1" textAnchor="middle" fontWeight={700}
                  >
                    {(p.value * 100).toFixed(0)}%
                  </text>
                </g>
              ))}
            </svg>

            {/* Compact confusion matrix summary */}
            <div className="mt-4 grid grid-cols-2 gap-1.5 text-center">
              {[
                { l: "TP", v: confusion.tp, c: "#22c55e" },
                { l: "FP", v: confusion.fp, c: "#ef4444" },
                { l: "FN", v: confusion.fn, c: "#f97316" },
                { l: "TN", v: confusion.tn, c: "#3b82f6" },
              ].map((item) => (
                <div
                  key={item.l}
                  className="rounded-lg p-2"
                  style={{ backgroundColor: `${item.c}10`, border: `1px solid ${item.c}30` }}
                >
                  <div className="text-[10px] font-semibold" style={{ color: item.c }}>{item.l}</div>
                  <div className="text-lg font-extrabold tabular-nums" style={{ color: item.c }}>{item.v}</div>
                </div>
              ))}
            </div>

            <div className="mt-4 bg-slate-50 rounded-lg p-2.5">
              <p className="text-[10px] text-slate-500 text-center">
                Each metric highlights different cells. Watch the radar shape change as you adjust the threshold.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// TAB 3: Real-World Scenarios
// =========================================================================

interface Scenario {
  id: string;
  name: string;
  icon: typeof Heart;
  description: string;
  negFraction: number;
  negMean: number;
  posMean: number;
  negStd: number;
  posStd: number;
  fpCost: number;
  fnCost: number;
  keyMetric: string;
  lesson: string;
}

const SCENARIOS: Scenario[] = [
  {
    id: "medical",
    name: "Medical Diagnosis",
    icon: Heart,
    description: "99% healthy, 1% sick. Missing a disease (FN) can be fatal.",
    negFraction: 0.99,
    negMean: 0.3,
    posMean: 0.7,
    negStd: 0.15,
    posStd: 0.15,
    fpCost: 10,
    fnCost: 1000,
    keyMetric: "Recall",
    lesson: "In medical screening, recall is critical. A false negative means a sick patient goes home untreated.",
  },
  {
    id: "spam",
    name: "Spam Filter",
    icon: Mail,
    description: "70% legit, 30% spam. Losing real email (FP) is very annoying.",
    negFraction: 0.7,
    negMean: 0.35,
    posMean: 0.7,
    negStd: 0.12,
    posStd: 0.12,
    fpCost: 100,
    fnCost: 5,
    keyMetric: "Precision",
    lesson: "For spam filtering, precision matters most. A false positive means a real email gets buried in spam.",
  },
  {
    id: "fraud",
    name: "Fraud Detection",
    icon: DollarSign,
    description: "99.5% legit, 0.5% fraud. Missing fraud (FN) is expensive.",
    negFraction: 0.995,
    negMean: 0.25,
    posMean: 0.75,
    negStd: 0.18,
    posStd: 0.12,
    fpCost: 5,
    fnCost: 500,
    keyMetric: "Recall",
    lesson: "Fraud detection uses recall-heavy thresholds. It is better to flag some legit transactions than let fraud slip through.",
  },
  {
    id: "balanced",
    name: "Balanced Test",
    icon: Scale,
    description: "50/50 class split. A balanced baseline for comparison.",
    negFraction: 0.5,
    negMean: 0.35,
    posMean: 0.65,
    negStd: 0.15,
    posStd: 0.15,
    fpCost: 10,
    fnCost: 10,
    keyMetric: "F1 Score",
    lesson: "With balanced classes and equal costs, F1 score or accuracy give a good picture of performance.",
  },
];

function RealWorldScenariosTab() {
  const [selectedScenario, setSelectedScenario] = useState<string>("medical");
  const [threshold, setThreshold] = useState(0.5);
  const [customFpCost, setCustomFpCost] = useState<number | null>(null);
  const [customFnCost, setCustomFnCost] = useState<number | null>(null);

  const scenario = SCENARIOS.find((s) => s.id === selectedScenario)!;
  const fpCost = customFpCost ?? scenario.fpCost;
  const fnCost = customFnCost ?? scenario.fnCost;

  const samples = useMemo(
    () =>
      generateSamples(
        200,
        123 + selectedScenario.charCodeAt(0),
        scenario.negMean,
        scenario.posMean,
        scenario.negStd,
        scenario.posStd,
        scenario.negFraction
      ),
    [selectedScenario, scenario]
  );

  const confusion = useMemo(() => computeConfusion(samples, threshold), [samples, threshold]);
  const metrics = useMemo(() => computeMetrics(confusion), [confusion]);
  const totalCost = confusion.fp * fpCost + confusion.fn * fnCost;

  // Find optimal threshold by sweeping
  const optimalThreshold = useMemo(() => {
    let bestT = 0.5;
    let bestCost = Infinity;
    for (let t = 0; t <= 1; t += 0.01) {
      const c = computeConfusion(samples, t);
      const cost = c.fp * fpCost + c.fn * fnCost;
      if (cost < bestCost) {
        bestCost = cost;
        bestT = t;
      }
    }
    return { threshold: bestT, cost: bestCost };
  }, [samples, fpCost, fnCost]);

  // Cost curve data
  const costCurve = useMemo(() => {
    const points: { t: number; cost: number }[] = [];
    for (let t = 0; t <= 1; t += 0.02) {
      const c = computeConfusion(samples, t);
      points.push({ t, cost: c.fp * fpCost + c.fn * fnCost });
    }
    return points;
  }, [samples, fpCost, fnCost]);

  const maxCostCurve = Math.max(...costCurve.map((p) => p.cost), 1);
  const costSvgW = 360;
  const costSvgH = 100;
  const costPad = { top: 10, right: 10, bottom: 22, left: 40 };
  const costPlotW = costSvgW - costPad.left - costPad.right;
  const costPlotH = costSvgH - costPad.top - costPad.bottom;

  const costPath = costCurve
    .map((p, i) => {
      const x = costPad.left + p.t * costPlotW;
      const y = costPad.top + costPlotH * (1 - p.cost / maxCostCurve);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  const handleScenarioChange = useCallback((id: string) => {
    setSelectedScenario(id);
    setThreshold(0.5);
    setCustomFpCost(null);
    setCustomFnCost(null);
  }, []);

  const maxCount = Math.max(confusion.tp, confusion.fp, confusion.tn, confusion.fn, 1);

  const ScenarioIcon = scenario.icon;

  return (
    <div className="space-y-5">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 flex gap-3">
        <AlertTriangle className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-orange-900">Real-World Scenarios</h3>
          <p className="text-xs text-orange-700 mt-1">
            The "best" threshold depends on the real-world costs of errors. In medical diagnosis,
            missing a disease is catastrophic. In spam filtering, losing a real email is unacceptable.
            Explore how different data distributions and cost structures change the optimal threshold.
          </p>
        </div>
      </div>

      {/* Scenario selector */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {SCENARIOS.map((s) => {
          const SIcon = s.icon;
          const isSelected = s.id === selectedScenario;
          return (
            <button
              key={s.id}
              onClick={() => handleScenarioChange(s.id)}
              className={`p-3 rounded-xl border text-left transition-all ${
                isSelected
                  ? "bg-indigo-50 border-indigo-300 shadow-sm"
                  : "bg-white border-slate-200 hover:border-slate-300 hover:shadow-sm"
              }`}
            >
              <div className="flex items-center gap-2 mb-1.5">
                <SIcon className={`w-4 h-4 ${isSelected ? "text-indigo-600" : "text-slate-400"}`} />
                <span className={`text-xs font-bold ${isSelected ? "text-indigo-700" : "text-slate-700"}`}>
                  {s.name}
                </span>
              </div>
              <p className="text-[10px] text-slate-500 leading-tight">{s.description}</p>
              <div className="mt-2 flex gap-1">
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">
                  FP cost: {s.fpCost}
                </span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">
                  FN cost: {s.fnCost}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-4">
          {/* Threshold + optimal hint */}
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                <SlidersHorizontal className="w-4 h-4 text-orange-500" />
                Threshold
              </label>
              <div className="flex items-center gap-2">
                <span className="text-sm font-mono font-bold text-orange-600 bg-orange-50 px-3 py-1 rounded-lg">
                  {threshold.toFixed(2)}
                </span>
                <button
                  onClick={() => setThreshold(optimalThreshold.threshold)}
                  className="text-[10px] px-2 py-1 rounded bg-green-100 text-green-700 font-semibold hover:bg-green-200 transition-colors"
                >
                  Jump to optimal ({optimalThreshold.threshold.toFixed(2)})
                </button>
              </div>
            </div>
            <input
              type="range" min={0} max={1} step={0.01} value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
            />
          </div>

          {/* Confusion matrix for this scenario */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <ScenarioIcon className="w-4 h-4 text-orange-500" />
              <h4 className="text-sm font-semibold text-slate-700">{scenario.name} — Confusion Matrix</h4>
            </div>
            <div className="grid grid-cols-2 gap-2 max-w-xs mx-auto">
              {[
                { k: "tp", l: "TP", v: confusion.tp, c: [34, 197, 94] as [number, number, number] },
                { k: "fp", l: "FP", v: confusion.fp, c: [239, 68, 68] as [number, number, number] },
                { k: "fn", l: "FN", v: confusion.fn, c: [249, 115, 22] as [number, number, number] },
                { k: "tn", l: "TN", v: confusion.tn, c: [59, 130, 246] as [number, number, number] },
              ].map((cell) => (
                <div
                  key={cell.k}
                  className="rounded-xl p-3 text-center transition-all duration-300"
                  style={{
                    backgroundColor: cellColor(cell.v, maxCount, cell.c),
                  }}
                >
                  <div
                    className="text-[10px] font-bold uppercase"
                    style={{ color: textColorForIntensity(cell.v, maxCount), opacity: 0.8 }}
                  >
                    {cell.l}
                  </div>
                  <div
                    className="text-2xl font-extrabold tabular-nums"
                    style={{ color: textColorForIntensity(cell.v, maxCount) }}
                  >
                    {cell.v}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Cost curve SVG */}
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
              <DollarSign className="w-4 h-4 text-orange-500" />
              Total Cost vs. Threshold
            </h4>
            <svg viewBox={`0 0 ${costSvgW} ${costSvgH}`} className="w-full max-w-[380px] mx-auto">
              <rect
                x={costPad.left} y={costPad.top}
                width={costPlotW} height={costPlotH}
                fill="#fefce8" stroke="#e2e8f0" strokeWidth={0.5} rx={3}
              />
              <path d={costPath} fill="none" stroke="#f97316" strokeWidth={2} />
              {/* Current threshold marker */}
              <circle
                cx={costPad.left + threshold * costPlotW}
                cy={costPad.top + costPlotH * (1 - totalCost / maxCostCurve)}
                r={4} fill="#f97316" stroke="white" strokeWidth={1.5}
              />
              {/* Optimal marker */}
              <circle
                cx={costPad.left + optimalThreshold.threshold * costPlotW}
                cy={costPad.top + costPlotH * (1 - optimalThreshold.cost / maxCostCurve)}
                r={4} fill="#22c55e" stroke="white" strokeWidth={1.5}
              />
              <text
                x={costPad.left + optimalThreshold.threshold * costPlotW}
                y={costPad.top - 2}
                fontSize={8} fill="#22c55e" textAnchor="middle" fontWeight={700}
              >
                Optimal
              </text>
              {/* X-axis labels */}
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <text key={t} x={costPad.left + t * costPlotW} y={costSvgH - 4} fontSize={8} fill="#94a3b8" textAnchor="middle">
                  {t.toFixed(2)}
                </text>
              ))}
              {/* Y-axis labels */}
              <text x={costPad.left - 4} y={costPad.top + 4} fontSize={7} fill="#94a3b8" textAnchor="end">
                {maxCostCurve.toFixed(0)}
              </text>
              <text x={costPad.left - 4} y={costPad.top + costPlotH} fontSize={7} fill="#94a3b8" textAnchor="end">
                0
              </text>
            </svg>
          </div>
        </div>

        {/* Right panel: metrics + cost matrix */}
        <div className="w-full lg:w-72 space-y-4">
          {/* Metrics */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase font-medium mb-2">Metrics</p>
            {[
              { n: "Accuracy", v: metrics.accuracy, c: "#6366f1" },
              { n: "Precision", v: metrics.precision, c: "#22c55e" },
              { n: "Recall", v: metrics.recall, c: "#f59e0b" },
              { n: "F1 Score", v: metrics.f1, c: "#ef4444" },
              { n: "Specificity", v: metrics.specificity, c: "#3b82f6" },
            ].map((m) => (
              <div key={m.n} className="mb-2">
                <div className="flex justify-between text-xs">
                  <span className={`font-medium ${m.n === scenario.keyMetric ? "text-orange-600 font-bold" : "text-slate-600"}`}>
                    {m.n}
                    {m.n === scenario.keyMetric && (
                      <span className="ml-1 text-[9px] bg-orange-100 text-orange-600 px-1.5 py-0.5 rounded-full">KEY</span>
                    )}
                  </span>
                  <span className="font-mono font-bold text-slate-800">{(m.v * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden mt-0.5">
                  <div className="h-full rounded-full transition-all duration-300" style={{ width: `${m.v * 100}%`, backgroundColor: m.c }} />
                </div>
              </div>
            ))}
          </div>

          {/* Cost matrix */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase font-medium mb-2">Cost Matrix</p>
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-slate-600">FP Cost:</span>
                <input
                  type="number"
                  min={0}
                  max={10000}
                  value={fpCost}
                  onChange={(e) => setCustomFpCost(Math.max(0, parseInt(e.target.value) || 0))}
                  className="w-20 text-xs font-mono border border-slate-200 rounded px-2 py-1 text-right"
                />
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-slate-600">FN Cost:</span>
                <input
                  type="number"
                  min={0}
                  max={10000}
                  value={fnCost}
                  onChange={(e) => setCustomFnCost(Math.max(0, parseInt(e.target.value) || 0))}
                  className="w-20 text-xs font-mono border border-slate-200 rounded px-2 py-1 text-right"
                />
              </div>
              <div className="border-t border-slate-100 pt-2 mt-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">FP * FP Cost:</span>
                  <span className="font-mono font-bold text-red-600">{(confusion.fp * fpCost).toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-xs mt-1">
                  <span className="text-slate-500">FN * FN Cost:</span>
                  <span className="font-mono font-bold text-orange-600">{(confusion.fn * fnCost).toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm mt-2 pt-2 border-t border-slate-100">
                  <span className="font-semibold text-slate-700">Total Cost:</span>
                  <span className="font-mono font-extrabold text-slate-900">{totalCost.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-xs mt-1">
                  <span className="text-green-600 font-medium">Optimal Cost:</span>
                  <span className="font-mono font-bold text-green-600">{optimalThreshold.cost.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Lesson */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <Award className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-semibold text-green-800">Key Lesson</p>
            </div>
            <p className="text-[11px] text-green-700">{scenario.lesson}</p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-[10px] text-slate-500">
              <strong>Class balance:</strong> {((1 - scenario.negFraction) * 100).toFixed(1)}% positive,{" "}
              {(scenario.negFraction * 100).toFixed(1)}% negative ({samples.length} total samples)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// TAB 4: Multi-Class Matrix
// =========================================================================

const CLASS_LABELS = ["A", "B", "C"];
const CLASS_COLORS = ["#6366f1", "#22c55e", "#f59e0b"];

function MultiClassMatrixTab() {
  const samples = useMemo(() => generateMultiClassSamples(150, 3, 77), []);

  // Compute 3x3 confusion matrix
  const matrix = useMemo(() => {
    const m: number[][] = Array.from({ length: 3 }, () => [0, 0, 0]);
    for (const s of samples) {
      m[s.trueClass][s.predictedClass]++;
    }
    return m;
  }, [samples]);

  const maxCell = useMemo(() => Math.max(...matrix.flat(), 1), [matrix]);

  // Per-class metrics
  const perClassMetrics = useMemo(() => {
    return CLASS_LABELS.map((_, c) => {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      for (let t = 0; t < 3; t++) {
        for (let p = 0; p < 3; p++) {
          if (t === c && p === c) tp += matrix[t][p];
          else if (t !== c && p === c) fp += matrix[t][p];
          else if (t === c && p !== c) fn += matrix[t][p];
          else tn += matrix[t][p];
        }
      }
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      return { label: CLASS_LABELS[c], color: CLASS_COLORS[c], tp, fp, fn, tn, precision, recall, f1 };
    });
  }, [matrix]);

  // Macro and micro averages
  const macroAvg = useMemo(() => {
    const avgP = perClassMetrics.reduce((s, m) => s + m.precision, 0) / 3;
    const avgR = perClassMetrics.reduce((s, m) => s + m.recall, 0) / 3;
    const avgF1 = perClassMetrics.reduce((s, m) => s + m.f1, 0) / 3;
    return { precision: avgP, recall: avgR, f1: avgF1 };
  }, [perClassMetrics]);

  const microAvg = useMemo(() => {
    const tpSum = perClassMetrics.reduce((s, m) => s + m.tp, 0);
    const fpSum = perClassMetrics.reduce((s, m) => s + m.fp, 0);
    const fnSum = perClassMetrics.reduce((s, m) => s + m.fn, 0);
    const precision = tpSum + fpSum > 0 ? tpSum / (tpSum + fpSum) : 0;
    const recall = tpSum + fnSum > 0 ? tpSum / (tpSum + fnSum) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1 };
  }, [perClassMetrics]);

  // Most confused pair
  const mostConfused = useMemo(() => {
    let maxOff = 0;
    let pair = [0, 0];
    for (let t = 0; t < 3; t++) {
      for (let p = 0; p < 3; p++) {
        if (t !== p && matrix[t][p] > maxOff) {
          maxOff = matrix[t][p];
          pair = [t, p];
        }
      }
    }
    return { true_: CLASS_LABELS[pair[0]], pred: CLASS_LABELS[pair[1]], count: maxOff };
  }, [matrix]);

  // Heatmap color
  const heatColor = (val: number, isDiag: boolean) => {
    const t = maxCell > 0 ? val / maxCell : 0;
    if (isDiag) {
      // green scale
      const r = Math.round(240 - t * 190);
      const g = Math.round(250 - t * 55);
      const b = Math.round(240 - t * 170);
      return `rgb(${r},${g},${b})`;
    } else {
      // red scale
      const r = Math.round(254 - t * 15);
      const g = Math.round(240 - t * 175);
      const b = Math.round(240 - t * 175);
      return `rgb(${r},${g},${b})`;
    }
  };

  const heatTextColor = (val: number) => {
    const t = maxCell > 0 ? val / maxCell : 0;
    return t > 0.5 ? "#ffffff" : "#1e293b";
  };

  return (
    <div className="space-y-5">
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-4 flex gap-3">
        <Grid3X3 className="w-5 h-5 text-violet-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-violet-900">Multi-Class Confusion Matrix</h3>
          <p className="text-xs text-violet-700 mt-1">
            Confusion matrices extend to any number of classes. With 3 classes (A, B, C), diagonal cells
            show correct predictions; off-diagonal cells show misclassifications. Each class gets its own
            precision, recall, and F1 score. Macro averaging treats all classes equally; micro averaging
            weights by support.
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-5">
          {/* 3x3 Matrix Heatmap */}
          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
              <Table className="w-4 h-4 text-violet-500" />
              3-Class Confusion Matrix (Heatmap)
            </h4>

            <div className="max-w-sm mx-auto">
              <div className="text-center mb-2">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Predicted Class</span>
              </div>
              <div className="flex">
                <div className="flex items-center mr-2">
                  <span
                    className="text-xs font-semibold text-slate-500 uppercase tracking-wider"
                    style={{ writingMode: "vertical-lr", transform: "rotate(180deg)" }}
                  >
                    True Class
                  </span>
                </div>
                <div className="flex-1">
                  {/* Column headers */}
                  <div className="grid grid-cols-3 gap-1 mb-1">
                    {CLASS_LABELS.map((l, i) => (
                      <div key={i} className="text-center">
                        <span className="text-xs font-bold" style={{ color: CLASS_COLORS[i] }}>
                          {l}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Matrix rows */}
                  {matrix.map((row, t) => (
                    <div key={t} className="flex items-center gap-1 mb-1">
                      <div className="grid grid-cols-3 gap-1 flex-1">
                        {row.map((val, p) => {
                          const isDiag = t === p;
                          return (
                            <div
                              key={p}
                              className={`rounded-lg p-3 text-center transition-all duration-300 ${
                                isDiag ? "ring-2 ring-green-300" : ""
                              }`}
                              style={{ backgroundColor: heatColor(val, isDiag) }}
                            >
                              <div className="text-2xl font-extrabold tabular-nums" style={{ color: heatTextColor(val) }}>
                                {val}
                              </div>
                              <div className="text-[8px] font-medium mt-0.5" style={{ color: heatTextColor(val), opacity: 0.7 }}>
                                {isDiag ? "correct" : `${CLASS_LABELS[t]} as ${CLASS_LABELS[p]}`}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <span className="text-xs font-bold w-6 text-center" style={{ color: CLASS_COLORS[t] }}>
                        {CLASS_LABELS[t]}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Most confused indicator */}
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3 flex items-center gap-3">
              <AlertTriangle className="w-4 h-4 text-red-500 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-red-800">Most Confused Pair</p>
                <p className="text-[11px] text-red-600">
                  Class <strong>{mostConfused.true_}</strong> misclassified as{" "}
                  <strong>{mostConfused.pred}</strong>: {mostConfused.count} times
                </p>
              </div>
            </div>
          </div>

          {/* Heatmap bar visualization */}
          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <h4 className="text-sm font-bold text-slate-700 mb-3">Normalized Heatmap (Row %)</h4>
            <div className="space-y-3">
              {matrix.map((row, t) => {
                const rowTotal = row.reduce((a, b) => a + b, 0) || 1;
                return (
                  <div key={t}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-bold w-16" style={{ color: CLASS_COLORS[t] }}>
                        True {CLASS_LABELS[t]}
                      </span>
                      <span className="text-[10px] text-slate-400">({rowTotal} samples)</span>
                    </div>
                    <div className="flex h-6 rounded-lg overflow-hidden gap-0.5">
                      {row.map((val, p) => {
                        const pct = (val / rowTotal) * 100;
                        return (
                          <div
                            key={p}
                            className="flex items-center justify-center transition-all duration-300"
                            style={{
                              width: `${Math.max(pct, 2)}%`,
                              backgroundColor: t === p ? CLASS_COLORS[p] : "#ef4444",
                              opacity: t === p ? 1 : 0.5,
                            }}
                          >
                            {pct > 8 && (
                              <span className="text-[9px] font-bold text-white">
                                {CLASS_LABELS[p]} {pct.toFixed(0)}%
                              </span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right panel: per-class metrics */}
        <div className="w-full lg:w-80 space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase font-medium mb-3">Per-Class Metrics</p>
            {perClassMetrics.map((m) => (
              <div key={m.label} className="mb-4 last:mb-0">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: m.color }} />
                  <span className="text-sm font-bold" style={{ color: m.color }}>Class {m.label}</span>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { n: "Precision", v: m.precision },
                    { n: "Recall", v: m.recall },
                    { n: "F1", v: m.f1 },
                  ].map((metric) => (
                    <div key={metric.n} className="bg-slate-50 rounded-lg p-2 text-center">
                      <div className="text-[10px] text-slate-400 font-medium">{metric.n}</div>
                      <div className="text-sm font-extrabold tabular-nums text-slate-800">
                        {(metric.v * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
                {/* Mini confusion counts */}
                <div className="flex gap-2 mt-1.5 text-[9px] text-slate-400">
                  <span>TP:{m.tp}</span>
                  <span>FP:{m.fp}</span>
                  <span>FN:{m.fn}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Macro vs Micro */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase font-medium mb-3">Averaging Methods</p>
            <div className="space-y-3">
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-xs font-bold text-indigo-700">Macro Average</span>
                  <span className="text-[9px] bg-indigo-50 text-indigo-500 px-1.5 py-0.5 rounded">
                    treats classes equally
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { n: "Prec", v: macroAvg.precision },
                    { n: "Rec", v: macroAvg.recall },
                    { n: "F1", v: macroAvg.f1 },
                  ].map((m) => (
                    <div key={m.n} className="bg-indigo-50 rounded-lg p-2 text-center">
                      <div className="text-[10px] text-indigo-400">{m.n}</div>
                      <div className="text-sm font-bold text-indigo-700 tabular-nums">{(m.v * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-xs font-bold text-emerald-700">Micro Average</span>
                  <span className="text-[9px] bg-emerald-50 text-emerald-500 px-1.5 py-0.5 rounded">
                    weights by support
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { n: "Prec", v: microAvg.precision },
                    { n: "Rec", v: microAvg.recall },
                    { n: "F1", v: microAvg.f1 },
                  ].map((m) => (
                    <div key={m.n} className="bg-emerald-50 rounded-lg p-2 text-center">
                      <div className="text-[10px] text-emerald-400">{m.n}</div>
                      <div className="text-sm font-bold text-emerald-700 tabular-nums">{(m.v * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Lesson */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <Award className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-semibold text-green-800">Key Lesson</p>
            </div>
            <p className="text-[11px] text-green-700">
              Confusion matrices extend to any number of classes. The diagonal shows correct predictions.
              Macro average treats all classes equally (good for imbalanced data awareness), while micro
              average aggregates contributions from all classes (good for overall performance).
            </p>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Things to notice:</p>
            <ul className="text-[11px] text-amber-700 space-y-1 list-disc list-inside">
              <li>Which class has the highest recall? Why?</li>
              <li>Look at off-diagonal cells for systematic confusions</li>
              <li>Compare macro vs. micro averages</li>
              <li>The row-normalized view shows error distribution per class</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// TAB 5: Build Your Own Classifier
// =========================================================================

function BuildYourOwnTab() {
  const [points, setPoints] = useState<ClassifyPoint[]>(() => generateClassifyPoints(99));
  const [selectedPrediction, setSelectedPrediction] = useState<0 | 1>(1);
  const svgRef = useRef<SVGSVGElement>(null);

  const classifiedCount = points.filter((p) => p.userPrediction !== null).length;
  const totalPoints = points.length;

  // User confusion matrix
  const userConfusion = useMemo(() => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const p of points) {
      if (p.userPrediction === null) continue;
      if (p.trueLabel === 1 && p.userPrediction === 1) tp++;
      else if (p.trueLabel === 0 && p.userPrediction === 1) fp++;
      else if (p.trueLabel === 0 && p.userPrediction === 0) tn++;
      else fn++;
    }
    return { tp, fp, tn, fn };
  }, [points]);

  const userMetrics = useMemo(() => computeMetrics(userConfusion), [userConfusion]);

  // AI confusion matrix
  const aiConfusion = useMemo(() => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const p of points) {
      if (p.trueLabel === 1 && p.aiPrediction === 1) tp++;
      else if (p.trueLabel === 0 && p.aiPrediction === 1) fp++;
      else if (p.trueLabel === 0 && p.aiPrediction === 0) tn++;
      else fn++;
    }
    return { tp, fp, tn, fn };
  }, [points]);

  const aiMetrics = useMemo(() => computeMetrics(aiConfusion), [aiConfusion]);

  const handlePointClick = useCallback(
    (id: number) => {
      setPoints((prev) =>
        prev.map((p) =>
          p.id === id ? { ...p, userPrediction: selectedPrediction } : p
        )
      );
    },
    [selectedPrediction]
  );

  const handleReset = useCallback(() => {
    setPoints(generateClassifyPoints(99));
  }, []);

  const handleClassifyAll = useCallback(
    (prediction: 0 | 1) => {
      setPoints((prev) =>
        prev.map((p) =>
          p.userPrediction === null ? { ...p, userPrediction: prediction } : p
        )
      );
    },
    []
  );

  const [showTrueLabels, setShowTrueLabels] = useState(false);
  const [showAiBoundary, setShowAiBoundary] = useState(false);

  // SVG dimensions
  const svgW = 400;
  const svgH = 400;

  return (
    <div className="space-y-5">
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 flex gap-3">
        <MousePointerClick className="w-5 h-5 text-teal-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-teal-900">Build Your Own Classifier</h3>
          <p className="text-xs text-teal-700 mt-1">
            Classification is about making decisions under uncertainty. Click on each point to classify
            it as class 0 (blue) or class 1 (green). Watch your confusion matrix build up in real-time,
            then compare your performance against the AI classifier. Can you beat the machine?
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-4">
          {/* Controls bar */}
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-slate-600">Classify as:</span>
                <button
                  onClick={() => setSelectedPrediction(0)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    selectedPrediction === 0
                      ? "bg-blue-600 text-white shadow-sm"
                      : "bg-blue-50 text-blue-600 hover:bg-blue-100"
                  }`}
                >
                  Class 0 (Negative)
                </button>
                <button
                  onClick={() => setSelectedPrediction(1)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    selectedPrediction === 1
                      ? "bg-green-600 text-white shadow-sm"
                      : "bg-green-50 text-green-600 hover:bg-green-100"
                  }`}
                >
                  Class 1 (Positive)
                </button>
              </div>

              <div className="flex items-center gap-2 ml-auto">
                <button
                  onClick={() => handleClassifyAll(0)}
                  className="px-2 py-1 rounded text-[10px] font-semibold bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors"
                >
                  All remaining as 0
                </button>
                <button
                  onClick={() => handleClassifyAll(1)}
                  className="px-2 py-1 rounded text-[10px] font-semibold bg-green-50 text-green-600 hover:bg-green-100 transition-colors"
                >
                  All remaining as 1
                </button>
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors"
                >
                  <RotateCcw className="w-3 h-3" />
                  Reset
                </button>
              </div>
            </div>

            {/* Toggle options */}
            <div className="flex items-center gap-4 mt-3 pt-3 border-t border-slate-100">
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showTrueLabels}
                  onChange={(e) => setShowTrueLabels(e.target.checked)}
                  className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                />
                <span className="text-xs text-slate-600">Show true labels (cheat mode)</span>
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showAiBoundary}
                  onChange={(e) => setShowAiBoundary(e.target.checked)}
                  className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                />
                <span className="text-xs text-slate-600">Show AI decision boundary</span>
              </label>
            </div>

            <div className="mt-2 text-[10px] text-slate-400">
              Progress: {classifiedCount}/{totalPoints} points classified
              <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden mt-1">
                <div
                  className="h-full bg-teal-500 rounded-full transition-all duration-300"
                  style={{ width: `${(classifiedCount / totalPoints) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* SVG Canvas */}
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <h4 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
              <Crosshair className="w-4 h-4 text-teal-500" />
              Click points to classify them
            </h4>
            <svg
              ref={svgRef}
              viewBox={`0 0 ${svgW} ${svgH}`}
              className="w-full max-w-[420px] mx-auto border border-slate-100 rounded-lg bg-slate-50 cursor-crosshair"
              style={{ aspectRatio: "1/1" }}
            >
              {/* Background grid */}
              {[0, 1, 2, 3, 4].map((i) => (
                <g key={i}>
                  <line
                    x1={i * 100} y1={0} x2={i * 100} y2={svgH}
                    stroke="#e2e8f0" strokeWidth={0.5}
                  />
                  <line
                    x1={0} y1={i * 100} x2={svgW} y2={i * 100}
                    stroke="#e2e8f0" strokeWidth={0.5}
                  />
                </g>
              ))}

              {/* AI boundary */}
              {showAiBoundary && (
                <>
                  <rect x={0} y={0} width={svgW / 2} height={svgH} fill="#3b82f6" opacity={0.04} />
                  <rect x={svgW / 2} y={0} width={svgW / 2} height={svgH} fill="#22c55e" opacity={0.04} />
                  <line
                    x1={svgW / 2} y1={0} x2={svgW / 2} y2={svgH}
                    stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="6,4"
                  />
                  <text x={svgW / 2 + 4} y={16} fontSize={10} fill="#94a3b8" fontWeight={600}>
                    AI boundary
                  </text>
                </>
              )}

              {/* Points */}
              {points.map((p) => {
                const cx = (p.x / 100) * svgW;
                const cy = (p.y / 100) * svgH;
                const isClassified = p.userPrediction !== null;

                // Outer ring color: shows user prediction
                let fillColor = "#94a3b8"; // gray for unclassified
                if (isClassified) {
                  fillColor = p.userPrediction === 1 ? "#22c55e" : "#3b82f6";
                }

                // Inner color: shows true label if cheat mode
                let innerColor: string | null = null;
                if (showTrueLabels) {
                  innerColor = p.trueLabel === 1 ? "#22c55e" : "#3b82f6";
                }

                // Correctness ring
                let ringColor: string | null = null;
                if (isClassified) {
                  ringColor = p.userPrediction === p.trueLabel ? "#22c55e" : "#ef4444";
                }

                return (
                  <g
                    key={p.id}
                    onClick={() => handlePointClick(p.id)}
                    className="cursor-pointer"
                    style={{ transition: "transform 0.15s" }}
                  >
                    {/* Correctness ring */}
                    {ringColor && (
                      <circle
                        cx={cx} cy={cy} r={9}
                        fill="none" stroke={ringColor} strokeWidth={2}
                        opacity={0.6}
                      />
                    )}
                    {/* Main point */}
                    <circle
                      cx={cx} cy={cy}
                      r={isClassified ? 6 : 5}
                      fill={fillColor}
                      stroke="white" strokeWidth={1.5}
                      opacity={isClassified ? 1 : 0.6}
                    />
                    {/* True label inner dot */}
                    {innerColor && (
                      <circle cx={cx} cy={cy} r={2.5} fill={innerColor} stroke="none" />
                    )}
                  </g>
                );
              })}

              {/* Legend */}
              <g transform="translate(8, 370)">
                <rect x={0} y={0} width={155} height={24} rx={4} fill="white" fillOpacity={0.9} stroke="#e2e8f0" strokeWidth={0.5} />
                <circle cx={10} cy={12} r={4} fill="#94a3b8" />
                <text x={18} y={15} fontSize={8} fill="#64748b">Unclassified</text>
                <circle cx={70} cy={12} r={4} fill="#3b82f6" />
                <text x={78} y={15} fontSize={8} fill="#64748b">Class 0</text>
                <circle cx={116} cy={12} r={4} fill="#22c55e" />
                <text x={124} y={15} fontSize={8} fill="#64748b">Class 1</text>
              </g>
            </svg>
          </div>
        </div>

        {/* Right panel */}
        <div className="w-full lg:w-80 space-y-4">
          {/* Your confusion matrix */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase font-medium mb-2 flex items-center gap-1.5">
              <MousePointerClick className="w-3.5 h-3.5" />
              Your Confusion Matrix
            </p>
            {classifiedCount === 0 ? (
              <div className="text-center py-4 text-xs text-slate-400">
                Click points on the canvas to start classifying
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-1.5">
                  {[
                    { l: "TP", v: userConfusion.tp, c: "#22c55e" },
                    { l: "FP", v: userConfusion.fp, c: "#ef4444" },
                    { l: "FN", v: userConfusion.fn, c: "#f97316" },
                    { l: "TN", v: userConfusion.tn, c: "#3b82f6" },
                  ].map((cell) => (
                    <div
                      key={cell.l}
                      className="rounded-lg p-2 text-center"
                      style={{ backgroundColor: `${cell.c}15`, border: `1px solid ${cell.c}30` }}
                    >
                      <div className="text-[10px] font-semibold" style={{ color: cell.c }}>{cell.l}</div>
                      <div className="text-xl font-extrabold tabular-nums" style={{ color: cell.c }}>{cell.v}</div>
                    </div>
                  ))}
                </div>

                {/* Your metrics */}
                <div className="mt-3 space-y-1.5">
                  {[
                    { n: "Accuracy", v: userMetrics.accuracy, c: "#6366f1" },
                    { n: "Precision", v: userMetrics.precision, c: "#22c55e" },
                    { n: "Recall", v: userMetrics.recall, c: "#f59e0b" },
                    { n: "F1 Score", v: userMetrics.f1, c: "#ef4444" },
                  ].map((m) => (
                    <div key={m.n}>
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-600 font-medium">{m.n}</span>
                        <span className="font-mono font-bold text-slate-800">{(m.v * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden mt-0.5">
                        <div className="h-full rounded-full transition-all duration-300" style={{ width: `${m.v * 100}%`, backgroundColor: m.c }} />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* AI comparison */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase font-medium mb-2 flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5" />
              AI Classifier Comparison
            </p>
            <div className="grid grid-cols-2 gap-1.5 mb-3">
              {[
                { l: "TP", v: aiConfusion.tp, c: "#22c55e" },
                { l: "FP", v: aiConfusion.fp, c: "#ef4444" },
                { l: "FN", v: aiConfusion.fn, c: "#f97316" },
                { l: "TN", v: aiConfusion.tn, c: "#3b82f6" },
              ].map((cell) => (
                <div
                  key={cell.l}
                  className="rounded-lg p-2 text-center"
                  style={{ backgroundColor: `${cell.c}10`, border: `1px solid ${cell.c}20` }}
                >
                  <div className="text-[9px] font-semibold" style={{ color: cell.c }}>{cell.l}</div>
                  <div className="text-lg font-extrabold tabular-nums" style={{ color: cell.c }}>{cell.v}</div>
                </div>
              ))}
            </div>
            <div className="space-y-1.5">
              {[
                { n: "Accuracy", v: aiMetrics.accuracy, c: "#6366f1" },
                { n: "Precision", v: aiMetrics.precision, c: "#22c55e" },
                { n: "Recall", v: aiMetrics.recall, c: "#f59e0b" },
                { n: "F1 Score", v: aiMetrics.f1, c: "#ef4444" },
              ].map((m) => (
                <div key={m.n}>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-600 font-medium">{m.n}</span>
                    <span className="font-mono font-bold text-slate-800">{(m.v * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden mt-0.5">
                    <div className="h-full rounded-full transition-all duration-300" style={{ width: `${m.v * 100}%`, backgroundColor: m.c }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Comparison result */}
          {classifiedCount >= 10 && (
            <div
              className={`rounded-lg p-3 border ${
                userMetrics.f1 > aiMetrics.f1
                  ? "bg-green-50 border-green-200"
                  : userMetrics.f1 === aiMetrics.f1
                    ? "bg-yellow-50 border-yellow-200"
                    : "bg-red-50 border-red-200"
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                {userMetrics.f1 > aiMetrics.f1 ? (
                  <Award className="w-4 h-4 text-green-600" />
                ) : (
                  <Brain className="w-4 h-4 text-red-600" />
                )}
                <p className="text-xs font-bold">
                  {userMetrics.f1 > aiMetrics.f1
                    ? "You are beating the AI!"
                    : userMetrics.f1 === aiMetrics.f1
                      ? "Tied with the AI!"
                      : "AI is ahead (for now)"}
                </p>
              </div>
              <p className="text-[10px] text-slate-500">
                Your F1: {(userMetrics.f1 * 100).toFixed(1)}% vs AI F1: {(aiMetrics.f1 * 100).toFixed(1)}%
                {classifiedCount < totalPoints &&
                  ` (${totalPoints - classifiedCount} points remaining)`}
              </p>
            </div>
          )}

          {/* Lesson */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center gap-1.5 mb-1">
              <Award className="w-3.5 h-3.5 text-green-600" />
              <p className="text-xs font-semibold text-green-800">Key Lesson</p>
            </div>
            <p className="text-[11px] text-green-700">
              Classification is about making decisions under uncertainty. In the overlapping region
              between classes, every classifier (human or machine) will make mistakes.
              The confusion matrix quantifies exactly what kinds of mistakes are being made.
            </p>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Tips:</p>
            <ul className="text-[11px] text-amber-700 space-y-1 list-disc list-inside">
              <li>Points on the left tend to be class 0, right tend to be class 1</li>
              <li>Toggle "Show true labels" to see where you went wrong</li>
              <li>The AI uses a simple vertical boundary at x=50</li>
              <li>Can you find a better strategy than the AI?</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
