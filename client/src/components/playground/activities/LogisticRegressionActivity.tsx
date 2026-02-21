/**
 * Logistic Regression Activity — comprehensive interactive SVG activity
 * Three tabbed exploration sections covering sigmoid, threshold metrics,
 * and decision boundaries.
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Info,
  Binary,
  SlidersHorizontal,
  Target,
  TrendingUp,
  Crosshair,
  Play,
  RotateCcw,
} from "lucide-react";

// ══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function gaussianPair(rand: () => number): number {
  const u1 = rand();
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

function sigmoid(z: number): number {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Interpolate between two RGB colors. t in [0,1]. */
function lerpColor(
  r1: number,
  g1: number,
  b1: number,
  r2: number,
  g2: number,
  b2: number,
  t: number
): string {
  const r = Math.round(lerp(r1, r2, t));
  const g = Math.round(lerp(g1, g2, t));
  const b = Math.round(lerp(b1, b2, t));
  return `rgb(${r},${g},${b})`;
}

// ══════════════════════════════════════════════════════════════════════════
// TYPES
// ══════════════════════════════════════════════════════════════════════════

interface DataPoint2D {
  x1: number;
  x2: number;
  label: number;
  probability: number;
}

interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

type TabId =
  | "sigmoid"
  | "threshold"
  | "boundary";

interface TabDef {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

// ══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ══════════════════════════════════════════════════════════════════════════

const SVG_W = 520;
const SVG_H = 340;
const PAD = { top: 30, right: 25, bottom: 40, left: 50 };
const PLOT_W = SVG_W - PAD.left - PAD.right;
const PLOT_H = SVG_H - PAD.top - PAD.bottom;

// Smaller dimensions for the Sigmoid Explorer chart (~35% reduction)
const SIG_SVG_W = 340;
const SIG_SVG_H = 220;
const SIG_PAD = { top: 24, right: 20, bottom: 32, left: 42 };
const SIG_PLOT_W = SIG_SVG_W - SIG_PAD.left - SIG_PAD.right;
const SIG_PLOT_H = SIG_SVG_H - SIG_PAD.top - SIG_PAD.bottom;

const TABS: TabDef[] = [
  { id: "sigmoid", label: "Sigmoid Explorer", icon: <TrendingUp className="w-3.5 h-3.5" /> },
  { id: "threshold", label: "Threshold & Metrics", icon: <SlidersHorizontal className="w-3.5 h-3.5" /> },
  { id: "boundary", label: "Decision Boundary", icon: <Crosshair className="w-3.5 h-3.5" /> },
];

const INFO_BANNERS: Record<TabId, { title: string; body: string }> = {
  sigmoid: {
    title: "The Sigmoid Function",
    body: "The sigmoid function maps any real number to a probability between 0 and 1: P(y=1) = 1/(1+e^(-z)) where z = w*x + b. Adjust the weight (w) and bias (b) to see how the curve shifts horizontally and changes steepness. Hover over the plot to read probabilities at any x position.",
  },
  threshold: {
    title: "Threshold & Classification Metrics",
    body: "Logistic regression outputs probabilities. A decision threshold determines the cutoff: points above become class 1, below become class 0. Adjusting the threshold trades off precision (how many predicted positives are correct) vs recall (how many actual positives are detected). Watch the confusion matrix and all five metrics update live.",
  },
  boundary: {
    title: "Decision Boundary in Feature Space",
    body: "In 2D feature space, logistic regression learns a linear decision boundary: w1*x1 + w2*x2 + b = 0. The probability heatmap shows how confident the model is across the space. Adjust weights and bias to see the boundary rotate and shift, or press Auto Fit. Try different dataset patterns to see how a linear boundary handles them.",
  },
};

// ══════════════════════════════════════════════════════════════════════════
// DATA GENERATION
// ══════════════════════════════════════════════════════════════════════════

function generateData50(seed: number): DataPoint2D[] {
  const rand = seededRandom(seed);
  const points: DataPoint2D[] = [];
  const c0 = { x1: -0.8, x2: -0.6 };
  const c1 = { x1: 0.8, x2: 0.6 };
  const spread = 0.85;

  for (let i = 0; i < 50; i++) {
    const label = i < 25 ? 0 : 1;
    const center = label === 0 ? c0 : c1;
    const x1 = center.x1 + gaussianPair(rand) * spread;
    const x2 = center.x2 + gaussianPair(rand) * spread;
    const z = 1.5 * x1 + 1.2 * x2 + 0.1;
    const probability = sigmoid(z);
    points.push({ x1, x2, label, probability });
  }
  return points;
}

type DatasetPattern = "linear" | "overlapping" | "clustered";

function generatePatternData(
  pattern: DatasetPattern,
  seed: number
): DataPoint2D[] {
  const rand = seededRandom(seed);
  const pts: DataPoint2D[] = [];
  const n = 60;

  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? 0 : 1;
    let x1: number, x2: number;

    if (pattern === "linear") {
      if (label === 0) {
        x1 = -1.5 + gaussianPair(rand) * 0.5;
        x2 = gaussianPair(rand) * 1.2;
      } else {
        x1 = 1.5 + gaussianPair(rand) * 0.5;
        x2 = gaussianPair(rand) * 1.2;
      }
    } else if (pattern === "overlapping") {
      if (label === 0) {
        x1 = -0.3 + gaussianPair(rand) * 1.0;
        x2 = -0.3 + gaussianPair(rand) * 1.0;
      } else {
        x1 = 0.3 + gaussianPair(rand) * 1.0;
        x2 = 0.3 + gaussianPair(rand) * 1.0;
      }
    } else {
      // clustered
      if (label === 0) {
        x1 = -1.5 + gaussianPair(rand) * 0.35;
        x2 = -1.0 + gaussianPair(rand) * 0.35;
      } else {
        x1 = 1.5 + gaussianPair(rand) * 0.35;
        x2 = 1.0 + gaussianPair(rand) * 0.35;
      }
    }

    pts.push({ x1, x2, label, probability: 0 });
  }

  return pts;
}

// ══════════════════════════════════════════════════════════════════════════
// METRICS COMPUTATION
// ══════════════════════════════════════════════════════════════════════════

function computeMetrics(
  data: { label: number; probability: number }[],
  threshold: number
): Metrics {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const p of data) {
    const pred = p.probability >= threshold ? 1 : 0;
    if (pred === 1 && p.label === 1) tp++;
    else if (pred === 1 && p.label === 0) fp++;
    else if (pred === 0 && p.label === 0) tn++;
    else fn++;
  }
  const accuracy = (tp + tn) / (tp + fp + tn + fn || 1);
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
  return { accuracy, precision, recall, f1, specificity, tp, fp, tn, fn };
}

// ══════════════════════════════════════════════════════════════════════════
// COORDINATE HELPERS
// ══════════════════════════════════════════════════════════════════════════

function toSvgX(
  v: number,
  vMin: number,
  vMax: number,
  w = SVG_W,
  pad = PAD
): number {
  const pw = w - pad.left - pad.right;
  return pad.left + ((v - vMin) / (vMax - vMin)) * pw;
}

function toSvgY(
  v: number,
  vMin: number,
  vMax: number,
  h = SVG_H,
  pad = PAD
): number {
  const ph = h - pad.top - pad.bottom;
  return pad.top + ph * (1 - (v - vMin) / (vMax - vMin));
}

// ══════════════════════════════════════════════════════════════════════════
// TAB 1 — SIGMOID EXPLORER
// ══════════════════════════════════════════════════════════════════════════

function SigmoidExplorerTab() {
  const [weight, setWeight] = useState(1.0);
  const [bias, setBias] = useState(0.0);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const xMin = -6;
  const xMax = 6;

  // Sample data points
  const dataPoints = useMemo(() => {
    const rand = seededRandom(77);
    const pts: { x: number; label: number }[] = [];
    for (let i = 0; i < 20; i++) {
      const label = i < 10 ? 0 : 1;
      const cx = label === 0 ? -2.0 : 2.0;
      const x = cx + gaussianPair(rand) * 1.2;
      pts.push({ x: clamp(x, xMin + 0.3, xMax - 0.3), label });
    }
    return pts;
  }, []);

  const sigmoidCurve = useMemo(() => {
    const parts: string[] = [];
    const steps = 300;
    for (let i = 0; i <= steps; i++) {
      const x = xMin + (i / steps) * (xMax - xMin);
      const z = weight * x + bias;
      const p = sigmoid(z);
      const sx = toSvgX(x, xMin, xMax, SIG_SVG_W, SIG_PAD);
      const sy = toSvgY(p, 0, 1, SIG_SVG_H, SIG_PAD);
      parts.push(`${i === 0 ? "M" : "L"} ${sx.toFixed(2)} ${sy.toFixed(2)}`);
    }
    return parts.join(" ");
  }, [weight, bias]);

  const hoverProb = useMemo(() => {
    if (hoverX === null) return null;
    return sigmoid(weight * hoverX + bias);
  }, [hoverX, weight, bias]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const scaleX = SIG_SVG_W / rect.width;
      const svgX = (e.clientX - rect.left) * scaleX;
      const plotLeft = SIG_PAD.left;
      const plotRight = SIG_SVG_W - SIG_PAD.right;
      if (svgX < plotLeft || svgX > plotRight) {
        setHoverX(null);
        return;
      }
      const t = (svgX - plotLeft) / (plotRight - plotLeft);
      const x = xMin + t * (xMax - xMin);
      setHoverX(x);
    },
    []
  );

  return (
    <div className="space-y-4">
      {/* SVG Sigmoid Plot */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
          <TrendingUp className="w-3.5 h-3.5 text-indigo-500" />
          <span className="text-xs font-semibold text-slate-700">
            Sigmoid Curve: P = 1 / (1 + e^(-(w*x + b)))
          </span>
          {hoverProb !== null && hoverX !== null && (
            <span className="ml-auto text-xs font-mono text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
              x={hoverX.toFixed(2)} | z={((weight * hoverX) + bias).toFixed(2)} | P={hoverProb.toFixed(4)}
            </span>
          )}
        </div>
        <div className="p-3 flex justify-center">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${SIG_SVG_W} ${SIG_SVG_H}`}
            className="w-full"
            style={{ maxWidth: SIG_SVG_W, aspectRatio: `${SIG_SVG_W}/${SIG_SVG_H}` }}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoverX(null)}
          >
            {/* Background */}
            <rect
              x={SIG_PAD.left}
              y={SIG_PAD.top}
              width={SIG_PLOT_W}
              height={SIG_PLOT_H}
              fill="#fafbfc"
              stroke="#e2e8f0"
              strokeWidth={1}
            />

            {/* Y-axis grid */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
              <g key={`yg-${v}`}>
                <line
                  x1={SIG_PAD.left}
                  y1={toSvgY(v, 0, 1, SIG_SVG_H, SIG_PAD)}
                  x2={SIG_SVG_W - SIG_PAD.right}
                  y2={toSvgY(v, 0, 1, SIG_SVG_H, SIG_PAD)}
                  stroke="#f1f5f9"
                  strokeWidth={1}
                />
                <text
                  x={SIG_PAD.left - 8}
                  y={toSvgY(v, 0, 1, SIG_SVG_H, SIG_PAD) + 3.5}
                  fontSize={9}
                  fill="#94a3b8"
                  textAnchor="end"
                >
                  {v.toFixed(2)}
                </text>
              </g>
            ))}

            {/* X-axis labels */}
            {[-4, -2, 0, 2, 4].map((v) => (
              <text
                key={`xl-${v}`}
                x={toSvgX(v, xMin, xMax, SIG_SVG_W, SIG_PAD)}
                y={SIG_SVG_H - SIG_PAD.bottom + 16}
                fontSize={9}
                fill="#94a3b8"
                textAnchor="middle"
              >
                {v}
              </text>
            ))}

            {/* Axis labels */}
            <text
              x={SIG_PAD.left + SIG_PLOT_W / 2}
              y={SIG_SVG_H - 4}
              fontSize={10}
              fill="#64748b"
              textAnchor="middle"
              fontWeight={600}
            >
              x (input)
            </text>
            <text
              x={10}
              y={SIG_PAD.top + SIG_PLOT_H / 2}
              fontSize={10}
              fill="#64748b"
              textAnchor="middle"
              fontWeight={600}
              transform={`rotate(-90, 10, ${SIG_PAD.top + SIG_PLOT_H / 2})`}
            >
              P(y=1)
            </text>

            {/* 0.5 reference line */}
            <line
              x1={SIG_PAD.left}
              y1={toSvgY(0.5, 0, 1, SIG_SVG_H, SIG_PAD)}
              x2={SIG_SVG_W - SIG_PAD.right}
              y2={toSvgY(0.5, 0, 1, SIG_SVG_H, SIG_PAD)}
              stroke="#cbd5e1"
              strokeWidth={1}
              strokeDasharray="4,4"
            />

            {/* The sigmoid curve */}
            <path
              d={sigmoidCurve}
              fill="none"
              stroke="#6366f1"
              strokeWidth={2.5}
            />

            {/* Data points on the curve */}
            {dataPoints.map((pt, i) => {
              const z = weight * pt.x + bias;
              const p = sigmoid(z);
              const cx = toSvgX(pt.x, xMin, xMax, SIG_SVG_W, SIG_PAD);
              const cy = toSvgY(p, 0, 1, SIG_SVG_H, SIG_PAD);
              return (
                <circle
                  key={`dp-${i}`}
                  cx={cx}
                  cy={cy}
                  r={4}
                  fill={pt.label === 1 ? "#3b82f6" : "#ef4444"}
                  stroke="#fff"
                  strokeWidth={1.5}
                  opacity={0.9}
                />
              );
            })}

            {/* Hover vertical line */}
            {hoverX !== null && (
              <>
                <line
                  x1={toSvgX(hoverX, xMin, xMax, SIG_SVG_W, SIG_PAD)}
                  y1={SIG_PAD.top}
                  x2={toSvgX(hoverX, xMin, xMax, SIG_SVG_W, SIG_PAD)}
                  y2={SIG_SVG_H - SIG_PAD.bottom}
                  stroke="#a5b4fc"
                  strokeWidth={1}
                  strokeDasharray="3,3"
                />
                {hoverProb !== null && (
                  <circle
                    cx={toSvgX(hoverX, xMin, xMax, SIG_SVG_W, SIG_PAD)}
                    cy={toSvgY(hoverProb, 0, 1, SIG_SVG_H, SIG_PAD)}
                    r={5}
                    fill="#6366f1"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                )}
              </>
            )}

            {/* Bias marker on x-axis */}
            {bias !== 0 && weight !== 0 && (
              <g>
                <line
                  x1={toSvgX(-bias / weight, xMin, xMax, SIG_SVG_W, SIG_PAD)}
                  y1={toSvgY(0.5, 0, 1, SIG_SVG_H, SIG_PAD) - 8}
                  x2={toSvgX(-bias / weight, xMin, xMax, SIG_SVG_W, SIG_PAD)}
                  y2={toSvgY(0.5, 0, 1, SIG_SVG_H, SIG_PAD) + 8}
                  stroke="#f59e0b"
                  strokeWidth={2}
                />
                <text
                  x={toSvgX(-bias / weight, xMin, xMax, SIG_SVG_W, SIG_PAD)}
                  y={toSvgY(0.5, 0, 1, SIG_SVG_H, SIG_PAD) - 12}
                  fontSize={8}
                  fill="#d97706"
                  textAnchor="middle"
                  fontWeight={600}
                >
                  midpoint
                </text>
              </g>
            )}

            {/* Legend */}
            <g transform={`translate(${SIG_SVG_W - SIG_PAD.right - 110}, ${SIG_PAD.top + 6})`}>
              <rect
                x={0}
                y={0}
                width={102}
                height={34}
                rx={4}
                fill="white"
                fillOpacity={0.92}
                stroke="#e2e8f0"
                strokeWidth={0.5}
              />
              <circle cx={10} cy={11} r={3.5} fill="#ef4444" />
              <text x={19} y={14} fontSize={8} fill="#64748b">
                Class 0 (negative)
              </text>
              <circle cx={10} cy={24} r={3.5} fill="#3b82f6" />
              <text x={19} y={27} fontSize={8} fill="#64748b">
                Class 1 (positive)
              </text>
            </g>
          </svg>
        </div>
      </div>

      {/* Sliders */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-700">
              Weight (w) — controls steepness
            </label>
            <span className="text-sm font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
              {weight.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.05}
            value={weight}
            onChange={(e) => setWeight(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>-5 (inverted)</span>
            <span>0 (flat)</span>
            <span>+5 (steep)</span>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-700">
              Bias (b) — shifts curve left/right
            </label>
            <span className="text-sm font-mono font-bold text-amber-600 bg-amber-50 px-2 py-0.5 rounded">
              {bias.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.05}
            value={bias}
            onChange={(e) => setBias(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>-5 (shift right)</span>
            <span>0</span>
            <span>+5 (shift left)</span>
          </div>
        </div>
      </div>

      {/* Formula display */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-center">
        <p className="text-xs text-slate-500 mb-1">Current model</p>
        <p className="text-sm font-mono font-semibold text-slate-800">
          z = {weight.toFixed(2)} * x + ({bias.toFixed(2)}) &nbsp;&nbsp;|&nbsp;&nbsp;
          P(y=1) = 1 / (1 + e^(-z))
        </p>
        <p className="text-xs text-slate-500 mt-1">
          Midpoint (P=0.5) at x = {weight !== 0 ? (-bias / weight).toFixed(2) : "undefined"}
        </p>
      </div>

      {/* Experiment tips */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Try these experiments:</p>
        <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
          <li>Set w=1, b=0 for the standard sigmoid, then increase w to see it steepen</li>
          <li>Set a negative weight to flip the curve (class meanings swap)</li>
          <li>Adjust bias to shift where the 50% probability point falls</li>
          <li>Hover over the curve to read exact probabilities at any x</li>
        </ul>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════
// TAB 2 — THRESHOLD & METRICS
// ══════════════════════════════════════════════════════════════════════════

function ThresholdMetricsTab() {
  const [threshold, setThreshold] = useState(0.5);
  const [prevThreshold, setPrevThreshold] = useState(0.5);
  const [flashIds, setFlashIds] = useState<Set<number>>(new Set());

  const data = useMemo(() => generateData50(42), []);

  const metrics = useMemo(
    () => computeMetrics(data, threshold),
    [data, threshold]
  );

  const classifiedData = useMemo(
    () =>
      data.map((p, i) => {
        const predicted = p.probability >= threshold ? 1 : 0;
        const correct = predicted === p.label;
        return { ...p, predicted, correct, idx: i };
      }),
    [data, threshold]
  );

  // Detect which points changed classification
  useEffect(() => {
    if (threshold === prevThreshold) return;
    const changed = new Set<number>();
    data.forEach((p, i) => {
      const oldPred = p.probability >= prevThreshold ? 1 : 0;
      const newPred = p.probability >= threshold ? 1 : 0;
      if (oldPred !== newPred) changed.add(i);
    });
    if (changed.size > 0) {
      setFlashIds(changed);
      const timer = setTimeout(() => setFlashIds(new Set()), 600);
      return () => clearTimeout(timer);
    }
    setPrevThreshold(threshold);
  }, [threshold, data, prevThreshold]);

  const handleThresholdChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setPrevThreshold(threshold);
      setThreshold(parseFloat(e.target.value));
    },
    [threshold]
  );

  // Scatter plot coordinate helpers
  const sMin = -3, sMax = 3;

  // Decision boundary: 1.5*x1 + 1.2*x2 + 0.1 = logit(threshold)
  const thresholdZ = useMemo(() => {
    if (threshold <= 0.001) return -7;
    if (threshold >= 0.999) return 7;
    return -Math.log(1 / threshold - 1);
  }, [threshold]);

  return (
    <div className="space-y-4">
      <div className="flex gap-4 flex-col lg:flex-row">
        {/* Scatter plot */}
        <div className="flex-1 min-w-0">
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
            <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
              <Binary className="w-3.5 h-3.5 text-indigo-500" />
              <span className="text-xs font-semibold text-slate-700">
                2D Feature Space &amp; Decision Boundary
              </span>
            </div>
            <div className="p-3">
              <svg
                viewBox={`0 0 ${SVG_W} ${SVG_H}`}
                className="w-full"
                style={{ aspectRatio: `${SVG_W}/${SVG_H}` }}
              >
                <rect
                  x={PAD.left}
                  y={PAD.top}
                  width={PLOT_W}
                  height={PLOT_H}
                  fill="#fafbfc"
                  stroke="#e2e8f0"
                  strokeWidth={1}
                />

                {/* Grid */}
                {[-2, -1, 0, 1, 2].map((v) => (
                  <g key={`tg-${v}`}>
                    <line
                      x1={toSvgX(v, sMin, sMax)}
                      y1={PAD.top}
                      x2={toSvgX(v, sMin, sMax)}
                      y2={SVG_H - PAD.bottom}
                      stroke="#f1f5f9"
                      strokeWidth={1}
                    />
                    <line
                      x1={PAD.left}
                      y1={toSvgY(v, sMin, sMax)}
                      x2={SVG_W - PAD.right}
                      y2={toSvgY(v, sMin, sMax)}
                      stroke="#f1f5f9"
                      strokeWidth={1}
                    />
                    <text
                      x={toSvgX(v, sMin, sMax)}
                      y={SVG_H - PAD.bottom + 18}
                      fontSize={9}
                      fill="#94a3b8"
                      textAnchor="middle"
                    >
                      {v}
                    </text>
                    <text
                      x={PAD.left - 8}
                      y={toSvgY(v, sMin, sMax) + 3.5}
                      fontSize={9}
                      fill="#94a3b8"
                      textAnchor="end"
                    >
                      {v}
                    </text>
                  </g>
                ))}

                {/* Axis labels */}
                <text
                  x={PAD.left + PLOT_W / 2}
                  y={SVG_H - 4}
                  fontSize={10}
                  fill="#64748b"
                  textAnchor="middle"
                  fontWeight={600}
                >
                  Feature x1
                </text>
                <text
                  x={12}
                  y={PAD.top + PLOT_H / 2}
                  fontSize={10}
                  fill="#64748b"
                  textAnchor="middle"
                  fontWeight={600}
                  transform={`rotate(-90, 12, ${PAD.top + PLOT_H / 2})`}
                >
                  Feature x2
                </text>

                {/* Clip */}
                <defs>
                  <clipPath id="t2Clip">
                    <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} />
                  </clipPath>
                </defs>

                {/* Decision boundary */}
                <line
                  x1={toSvgX(sMin, sMin, sMax)}
                  y1={toSvgY(
                    (thresholdZ - 0.1 - 1.5 * sMin) / 1.2,
                    sMin,
                    sMax
                  )}
                  x2={toSvgX(sMax, sMin, sMax)}
                  y2={toSvgY(
                    (thresholdZ - 0.1 - 1.5 * sMax) / 1.2,
                    sMin,
                    sMax
                  )}
                  stroke="#f59e0b"
                  strokeWidth={2}
                  strokeDasharray="8,4"
                  opacity={0.9}
                  clipPath="url(#t2Clip)"
                />

                {/* Points */}
                {classifiedData.map((p, i) => {
                  const cx = toSvgX(clamp(p.x1, sMin, sMax), sMin, sMax);
                  const cy = toSvgY(clamp(p.x2, sMin, sMax), sMin, sMax);
                  const fill = p.label === 1 ? "#3b82f6" : "#ef4444";
                  const isFlashing = flashIds.has(i);
                  return (
                    <g key={`t2p-${i}`}>
                      {/* Flash ring for changed points */}
                      {isFlashing && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r={12}
                          fill="none"
                          stroke="#fbbf24"
                          strokeWidth={3}
                          opacity={0.9}
                        >
                          <animate
                            attributeName="r"
                            from="6"
                            to="14"
                            dur="0.5s"
                            repeatCount="2"
                          />
                          <animate
                            attributeName="opacity"
                            from="1"
                            to="0"
                            dur="0.5s"
                            repeatCount="2"
                          />
                        </circle>
                      )}
                      {/* Correct/incorrect ring */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={8}
                        fill="none"
                        stroke={p.correct ? "#22c55e" : "#ef4444"}
                        strokeWidth={2}
                        opacity={0.7}
                      />
                      <circle
                        cx={cx}
                        cy={cy}
                        r={4.5}
                        fill={fill}
                        stroke="#fff"
                        strokeWidth={1}
                        opacity={0.9}
                      />
                    </g>
                  );
                })}

                {/* Legend */}
                <g transform={`translate(${PAD.left + 6}, ${SVG_H - PAD.bottom - 55})`}>
                  <rect
                    x={0}
                    y={0}
                    width={130}
                    height={48}
                    rx={4}
                    fill="white"
                    fillOpacity={0.92}
                    stroke="#e2e8f0"
                    strokeWidth={0.5}
                  />
                  <circle cx={12} cy={13} r={4} fill="#ef4444" />
                  <text x={22} y={16} fontSize={9} fill="#64748b">Class 0 (negative)</text>
                  <circle cx={12} cy={29} r={4} fill="#3b82f6" />
                  <text x={22} y={32} fontSize={9} fill="#64748b">Class 1 (positive)</text>
                  <circle cx={12} cy={42} r={5} fill="none" stroke="#22c55e" strokeWidth={1.5} />
                  <text x={22} y={45} fontSize={8} fill="#64748b">Green = correct</text>
                </g>
              </svg>
            </div>
          </div>
        </div>

        {/* Confusion matrix + metrics sidebar */}
        <div className="w-full lg:w-64 space-y-3">
          {/* Confusion matrix */}
          <div className="bg-white border border-slate-200 rounded-lg p-4">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2 flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-indigo-500" />
              Confusion Matrix
            </p>
            {/* Header labels */}
            <div className="grid grid-cols-3 gap-1 text-center text-[9px] text-slate-500 mb-1">
              <div />
              <div className="font-semibold">Pred 1</div>
              <div className="font-semibold">Pred 0</div>
            </div>
            <div className="grid grid-cols-3 gap-1.5 text-center text-xs">
              <div className="flex items-center justify-end pr-1 text-[9px] text-slate-500 font-semibold">
                True 1
              </div>
              <div className="bg-green-50 border border-green-200 rounded p-2">
                <p className="text-[9px] text-green-600 font-medium">TP</p>
                <p className="text-lg font-bold text-green-700">{metrics.tp}</p>
              </div>
              <div className="bg-red-50 border border-red-200 rounded p-2">
                <p className="text-[9px] text-red-600 font-medium">FN</p>
                <p className="text-lg font-bold text-red-700">{metrics.fn}</p>
              </div>
              <div className="flex items-center justify-end pr-1 text-[9px] text-slate-500 font-semibold">
                True 0
              </div>
              <div className="bg-red-50 border border-red-200 rounded p-2">
                <p className="text-[9px] text-red-600 font-medium">FP</p>
                <p className="text-lg font-bold text-red-700">{metrics.fp}</p>
              </div>
              <div className="bg-green-50 border border-green-200 rounded p-2">
                <p className="text-[9px] text-green-600 font-medium">TN</p>
                <p className="text-lg font-bold text-green-700">{metrics.tn}</p>
              </div>
            </div>
          </div>

          {/* Metrics cards */}
          <div className="space-y-2">
            {[
              { label: "Accuracy", value: metrics.accuracy, color: "indigo" },
              { label: "Precision", value: metrics.precision, color: "emerald" },
              { label: "Recall", value: metrics.recall, color: "sky" },
              { label: "F1-Score", value: metrics.f1, color: "violet" },
              { label: "Specificity", value: metrics.specificity, color: "amber" },
            ].map((m) => (
              <div
                key={m.label}
                className="bg-white border border-slate-200 rounded-lg px-3 py-2 flex items-center justify-between"
              >
                <span className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">
                  {m.label}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-20 bg-slate-100 rounded-full h-1.5">
                    <div
                      className={`bg-${m.color}-500 h-1.5 rounded-full transition-all duration-200`}
                      style={{ width: `${m.value * 100}%`, backgroundColor: m.color === "indigo" ? "#6366f1" : m.color === "emerald" ? "#10b981" : m.color === "sky" ? "#0ea5e9" : m.color === "violet" ? "#8b5cf6" : "#f59e0b" }}
                    />
                  </div>
                  <span className="text-sm font-bold font-mono w-14 text-right" style={{ color: m.color === "indigo" ? "#4f46e5" : m.color === "emerald" ? "#059669" : m.color === "sky" ? "#0284c7" : m.color === "violet" ? "#7c3aed" : "#d97706" }}>
                    {(m.value * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Threshold slider */}
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-amber-500" />
            Decision Threshold
          </label>
          <span className="text-sm font-mono font-bold text-amber-600 bg-amber-50 px-2.5 py-0.5 rounded">
            {threshold.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min={0.01}
          max={0.99}
          step={0.01}
          value={threshold}
          onChange={handleThresholdChange}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
        />
        <div className="flex justify-between text-[10px] text-slate-400 mt-1">
          <span>0.01 (classify most as positive)</span>
          <span>0.99 (classify most as negative)</span>
        </div>
      </div>

      {/* Tips */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Try these experiments:</p>
        <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
          <li>Set threshold to 0.5 and observe the default decision boundary</li>
          <li>Lower to 0.2 -- recall goes up but precision drops (more false positives)</li>
          <li>Raise to 0.8 -- precision goes up but recall drops (more false negatives)</li>
          <li>Watch points flash when they change classification</li>
        </ul>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════
// TAB 3 — DECISION BOUNDARY
// ══════════════════════════════════════════════════════════════════════════

function DecisionBoundaryTab() {
  const [w1, setW1] = useState(1.5);
  const [w2, setW2] = useState(1.2);
  const [bias, setBias] = useState(0.1);
  const [pattern, setPattern] = useState<DatasetPattern>("linear");
  const [autoFitting, setAutoFitting] = useState(false);

  const data = useMemo(() => generatePatternData(pattern, 123), [pattern]);

  // Recompute probabilities based on current weights
  const dataWithProbs = useMemo(
    () =>
      data.map((p) => ({
        ...p,
        probability: sigmoid(w1 * p.x1 + w2 * p.x2 + bias),
      })),
    [data, w1, w2, bias]
  );

  // Accuracy
  const accuracy = useMemo(() => {
    let correct = 0;
    for (const p of dataWithProbs) {
      const pred = p.probability >= 0.5 ? 1 : 0;
      if (pred === p.label) correct++;
    }
    return correct / dataWithProbs.length;
  }, [dataWithProbs]);

  const sMin = -3.5, sMax = 3.5;
  const heatmapRes = 30;

  // Probability heatmap
  const heatmapCells = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; color: string }[] = [];
    const cellW = PLOT_W / heatmapRes;
    const cellH = PLOT_H / heatmapRes;

    for (let row = 0; row < heatmapRes; row++) {
      for (let col = 0; col < heatmapRes; col++) {
        const x1 = sMin + (col / heatmapRes) * (sMax - sMin);
        const x2 = sMax - (row / heatmapRes) * (sMax - sMin);
        const p = sigmoid(w1 * x1 + w2 * x2 + bias);
        // Red (class 0) to Blue (class 1)
        const color = lerpColor(239, 68, 68, 59, 130, 246, p);
        cells.push({
          x: PAD.left + col * cellW,
          y: PAD.top + row * cellH,
          w: cellW + 0.5,
          h: cellH + 0.5,
          color,
        });
      }
    }
    return cells;
  }, [w1, w2, bias, sMin, sMax]);

  // Auto fit using gradient descent
  const autoFit = useCallback(() => {
    setAutoFitting(true);
    let cw1 = 0.1, cw2 = 0.1, cb = 0.0;
    const lr = 0.05;
    const steps = 300;

    for (let step = 0; step < steps; step++) {
      let dw1 = 0, dw2 = 0, db = 0;
      for (const p of data) {
        const z = cw1 * p.x1 + cw2 * p.x2 + cb;
        const pred = sigmoid(z);
        const err = pred - p.label;
        dw1 += err * p.x1;
        dw2 += err * p.x2;
        db += err;
      }
      cw1 -= (lr * dw1) / data.length;
      cw2 -= (lr * dw2) / data.length;
      cb -= (lr * db) / data.length;
    }

    // Animate sliders toward the optimal values
    const animSteps = 20;
    const startW1 = w1, startW2 = w2, startB = bias;
    let frame = 0;
    const animate = () => {
      frame++;
      const t = frame / animSteps;
      setW1(lerp(startW1, cw1, t));
      setW2(lerp(startW2, cw2, t));
      setBias(lerp(startB, cb, t));
      if (frame < animSteps) {
        requestAnimationFrame(animate);
      } else {
        setAutoFitting(false);
      }
    };
    requestAnimationFrame(animate);
  }, [data, w1, w2, bias]);

  return (
    <div className="space-y-4">
      {/* Pattern selector */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-semibold text-slate-600">Dataset:</span>
        {(["linear", "overlapping", "clustered"] as DatasetPattern[]).map((p) => (
          <button
            key={p}
            onClick={() => setPattern(p)}
            className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-colors ${
              pattern === p
                ? "bg-indigo-500 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {p.charAt(0).toUpperCase() + p.slice(1)}
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={autoFit}
          disabled={autoFitting}
          className="px-3 py-1.5 text-xs rounded-lg font-medium bg-emerald-500 text-white hover:bg-emerald-600 disabled:opacity-50 flex items-center gap-1.5 transition-colors"
        >
          <Play className="w-3 h-3" />
          {autoFitting ? "Fitting..." : "Auto Fit"}
        </button>
        <button
          onClick={() => { setW1(0.1); setW2(0.1); setBias(0); }}
          className="px-3 py-1.5 text-xs rounded-lg font-medium bg-slate-100 text-slate-600 hover:bg-slate-200 flex items-center gap-1.5 transition-colors"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </button>
      </div>

      {/* Scatter plot with heatmap */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
          <Crosshair className="w-3.5 h-3.5 text-indigo-500" />
          <span className="text-xs font-semibold text-slate-700">
            Decision Boundary with Probability Heatmap
          </span>
          <span className="ml-auto text-xs font-mono text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
            Accuracy: {(accuracy * 100).toFixed(1)}%
          </span>
        </div>
        <div className="p-3">
          <svg
            viewBox={`0 0 ${SVG_W} ${SVG_H}`}
            className="w-full"
            style={{ aspectRatio: `${SVG_W}/${SVG_H}` }}
          >
            <defs>
              <clipPath id="dbClip">
                <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} />
              </clipPath>
            </defs>

            {/* Heatmap */}
            <g clipPath="url(#dbClip)">
              {heatmapCells.map((c, i) => (
                <rect
                  key={`hm-${i}`}
                  x={c.x}
                  y={c.y}
                  width={c.w}
                  height={c.h}
                  fill={c.color}
                  opacity={0.35}
                />
              ))}
            </g>

            {/* Plot border */}
            <rect
              x={PAD.left}
              y={PAD.top}
              width={PLOT_W}
              height={PLOT_H}
              fill="none"
              stroke="#e2e8f0"
              strokeWidth={1}
            />

            {/* Grid */}
            {[-3, -2, -1, 0, 1, 2, 3].map((v) => (
              <g key={`dbg-${v}`}>
                <line
                  x1={toSvgX(v, sMin, sMax)}
                  y1={PAD.top}
                  x2={toSvgX(v, sMin, sMax)}
                  y2={SVG_H - PAD.bottom}
                  stroke="rgba(255,255,255,0.4)"
                  strokeWidth={0.5}
                />
                <line
                  x1={PAD.left}
                  y1={toSvgY(v, sMin, sMax)}
                  x2={SVG_W - PAD.right}
                  y2={toSvgY(v, sMin, sMax)}
                  stroke="rgba(255,255,255,0.4)"
                  strokeWidth={0.5}
                />
                <text
                  x={toSvgX(v, sMin, sMax)}
                  y={SVG_H - PAD.bottom + 16}
                  fontSize={9}
                  fill="#94a3b8"
                  textAnchor="middle"
                >
                  {v}
                </text>
                <text
                  x={PAD.left - 8}
                  y={toSvgY(v, sMin, sMax) + 3.5}
                  fontSize={9}
                  fill="#94a3b8"
                  textAnchor="end"
                >
                  {v}
                </text>
              </g>
            ))}

            {/* Decision boundary line: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b) / w2 */}
            {Math.abs(w2) > 0.01 && (
              <line
                x1={toSvgX(sMin, sMin, sMax)}
                y1={toSvgY((-w1 * sMin - bias) / w2, sMin, sMax)}
                x2={toSvgX(sMax, sMin, sMax)}
                y2={toSvgY((-w1 * sMax - bias) / w2, sMin, sMax)}
                stroke="#fbbf24"
                strokeWidth={2.5}
                clipPath="url(#dbClip)"
              />
            )}

            {/* Data points */}
            {dataWithProbs.map((p, i) => {
              const cx = toSvgX(clamp(p.x1, sMin, sMax), sMin, sMax);
              const cy = toSvgY(clamp(p.x2, sMin, sMax), sMin, sMax);
              const pred = p.probability >= 0.5 ? 1 : 0;
              const correct = pred === p.label;
              return (
                <g key={`db-pt-${i}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={7}
                    fill="none"
                    stroke={correct ? "#22c55e" : "#ef4444"}
                    strokeWidth={2}
                    opacity={0.8}
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r={4}
                    fill={p.label === 1 ? "#3b82f6" : "#ef4444"}
                    stroke="#fff"
                    strokeWidth={1.5}
                  />
                </g>
              );
            })}

            {/* Axis labels */}
            <text
              x={PAD.left + PLOT_W / 2}
              y={SVG_H - 4}
              fontSize={10}
              fill="#64748b"
              textAnchor="middle"
              fontWeight={600}
            >
              Feature x1
            </text>
            <text
              x={12}
              y={PAD.top + PLOT_H / 2}
              fontSize={10}
              fill="#64748b"
              textAnchor="middle"
              fontWeight={600}
              transform={`rotate(-90, 12, ${PAD.top + PLOT_H / 2})`}
            >
              Feature x2
            </text>

            {/* Color legend bar */}
            <g transform={`translate(${SVG_W - PAD.right - 20}, ${PAD.top + 5})`}>
              <defs>
                <linearGradient id="heatGrad" x1="0" y1="1" x2="0" y2="0">
                  <stop offset="0%" stopColor="rgb(239,68,68)" />
                  <stop offset="50%" stopColor="rgb(149,99,207)" />
                  <stop offset="100%" stopColor="rgb(59,130,246)" />
                </linearGradient>
              </defs>
              <rect x={0} y={0} width={12} height={80} fill="url(#heatGrad)" rx={2} opacity={0.7} />
              <text x={6} y={-4} fontSize={7} fill="#64748b" textAnchor="middle">P=1</text>
              <text x={6} y={92} fontSize={7} fill="#64748b" textAnchor="middle">P=0</text>
            </g>
          </svg>
        </div>
      </div>

      {/* Weight sliders */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="bg-white border border-slate-200 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-700">Weight w1</label>
            <span className="text-sm font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
              {w1.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.05}
            value={w1}
            onChange={(e) => setW1(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-700">Weight w2</label>
            <span className="text-sm font-mono font-bold text-sky-600 bg-sky-50 px-2 py-0.5 rounded">
              {w2.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.05}
            value={w2}
            onChange={(e) => setW2(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-sky-500"
          />
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-700">Bias (b)</label>
            <span className="text-sm font-mono font-bold text-amber-600 bg-amber-50 px-2 py-0.5 rounded">
              {bias.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.05}
            value={bias}
            onChange={(e) => setBias(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
        </div>
      </div>

      {/* Equation */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-center">
        <p className="text-sm font-mono font-semibold text-slate-800">
          Boundary: {w1.toFixed(2)}*x1 + {w2.toFixed(2)}*x2 + ({bias.toFixed(2)}) = 0
        </p>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════════

export default function LogisticRegressionActivity() {
  const [activeTab, setActiveTab] = useState<TabId>("sigmoid");

  const currentBanner = INFO_BANNERS[activeTab];

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 overflow-x-auto pb-1 border-b border-slate-200">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-t-lg whitespace-nowrap transition-colors ${
              activeTab === tab.id
                ? "bg-indigo-50 text-indigo-700 border border-b-0 border-indigo-200"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Info banner */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">
            {currentBanner.title}
          </h3>
          <p className="text-xs text-indigo-700 mt-1">{currentBanner.body}</p>
        </div>
      </div>

      {/* Tab content */}
      {activeTab === "sigmoid" && <SigmoidExplorerTab />}
      {activeTab === "threshold" && <ThresholdMetricsTab />}
      {activeTab === "boundary" && <DecisionBoundaryTab />}
    </div>
  );
}
