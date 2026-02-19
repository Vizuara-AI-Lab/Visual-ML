/**
 * Linear Regression Activity — comprehensive tabbed exploration
 *
 * 3 tabs:
 *   1. Fit the Line        – drag handles to minimise MSE
 *   2. Effect of Outliers   – see how outliers warp the best-fit line
 *   3. Gradient Descent Fit – watch the computer learn step by step
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Info,
  TrendingUp,
  RotateCcw,
  Sparkles,
  Shuffle,
  Plus,
  Trash2,
  Play,
  Pause,
  SkipForward,
  AlertTriangle,
  LineChart,
  CheckCircle2,
} from "lucide-react";

// ════════════════════════════════════════════════════════════════════════════
// SHARED UTILITIES
// ════════════════════════════════════════════════════════════════════════════

interface DataPoint {
  x: number;
  y: number;
}

/** Mulberry-32 seeded PRNG */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateLinearPoints(seed: number, n = 20): DataPoint[] {
  const rng = mulberry32(seed);
  const trueSlope = 0.4 + rng() * 1.4;
  const trueIntercept = 0.5 + rng() * 1.5;
  const noiseScale = 0.3 + rng() * 0.5;
  const pts: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = 0.5 + rng() * 9;
    const noise = (rng() - 0.5) * 2 * noiseScale * 2;
    pts.push({ x, y: trueSlope * x + trueIntercept + noise });
  }
  return pts;
}

function computeOLS(points: DataPoint[]): { m: number; b: number } {
  const n = points.length;
  if (n === 0) return { m: 0, b: 0 };
  let sumX = 0,
    sumY = 0;
  for (const p of points) {
    sumX += p.x;
    sumY += p.y;
  }
  const meanX = sumX / n;
  const meanY = sumY / n;
  let num = 0,
    den = 0;
  for (const p of points) {
    const dx = p.x - meanX;
    num += dx * (p.y - meanY);
    den += dx * dx;
  }
  const m = den !== 0 ? num / den : 0;
  return { m, b: meanY - m * meanX };
}

function computeMSE(points: DataPoint[], m: number, b: number): number {
  if (points.length === 0) return 0;
  let s = 0;
  for (const p of points) {
    const d = m * p.x + b - p.y;
    s += d * d;
  }
  return s / points.length;
}

function axisRange(points: DataPoint[], marginFrac = 0.15) {
  if (points.length === 0) return { xMin: 0, xMax: 10, yMin: 0, yMax: 10 };
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const dxMin = Math.min(...xs),
    dxMax = Math.max(...xs);
  const dyMin = Math.min(...ys),
    dyMax = Math.max(...ys);
  const xM = (dxMax - dxMin) * marginFrac;
  const yM = (dyMax - dyMin) * 0.25;
  return {
    xMin: Math.min(0, dxMin - xM),
    xMax: dxMax + xM,
    yMin: Math.min(0, dyMin - yM),
    yMax: dyMax + yM,
  };
}

/** Residual colour based on magnitude */
function residualColor(absError: number, maxError: number): string {
  const ratio = maxError > 0 ? absError / maxError : 0;
  if (ratio < 0.33) return "#22c55e";
  if (ratio < 0.66) return "#f59e0b";
  return "#ef4444";
}

// ── SVG coordinate helpers ──────────────────────────────────────────────
function toSX(x: number, xMin: number, xMax: number, pad: typeof PAD_DEFAULT, plotW: number) {
  return pad.left + ((x - xMin) / (xMax - xMin)) * plotW;
}
function toSY(y: number, yMin: number, yMax: number, pad: typeof PAD_DEFAULT, plotH: number) {
  return pad.top + plotH * (1 - (y - yMin) / (yMax - yMin));
}
function fromSX(sx: number, xMin: number, xMax: number, pad: typeof PAD_DEFAULT, plotW: number) {
  return xMin + ((sx - pad.left) / plotW) * (xMax - xMin);
}
function fromSY(sy: number, yMin: number, yMax: number, pad: typeof PAD_DEFAULT, plotH: number) {
  return yMax - ((sy - pad.top) / plotH) * (yMax - yMin);
}

const PAD_DEFAULT = { top: 30, right: 30, bottom: 45, left: 55 };

function makeTicks(lo: number, hi: number, count = 5): number[] {
  const step = (hi - lo) / count;
  return Array.from({ length: count + 1 }, (_, i) => lo + step * i);
}

// ── Small reusable SVG chart grid ───────────────────────────────────────
function ChartGrid({
  w,
  h,
  pad,
  xMin,
  xMax,
  yMin,
  yMax,
  xLabel,
  yLabel,
  clipId,
}: {
  w: number;
  h: number;
  pad: typeof PAD_DEFAULT;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  xLabel?: string;
  yLabel?: string;
  clipId?: string;
}) {
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const xTicks = makeTicks(xMin, xMax, 5);
  const yTicks = makeTicks(yMin, yMax, 5);
  return (
    <>
      <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="white" stroke="#e2e8f0" strokeWidth={1} rx={2} />
      {clipId && (
        <defs>
          <clipPath id={clipId}>
            <rect x={pad.left} y={pad.top} width={plotW} height={plotH} />
          </clipPath>
        </defs>
      )}
      {yTicks.map((t, i) => {
        const sy = toSY(t, yMin, yMax, pad, plotH);
        return (
          <g key={`yg-${i}`}>
            <line x1={pad.left} y1={sy} x2={pad.left + plotW} y2={sy} stroke="#f1f5f9" />
            <text x={pad.left - 6} y={sy + 3.5} fontSize={8} fill="#94a3b8" textAnchor="end">
              {Math.abs(t) < 0.001 ? "0" : t.toFixed(1)}
            </text>
          </g>
        );
      })}
      {xTicks.map((t, i) => {
        const sx = toSX(t, xMin, xMax, pad, plotW);
        return (
          <g key={`xg-${i}`}>
            <line x1={sx} y1={pad.top} x2={sx} y2={pad.top + plotH} stroke="#f1f5f9" />
            <text x={sx} y={h - pad.bottom + 14} fontSize={8} fill="#94a3b8" textAnchor="middle">
              {Math.abs(t) < 0.001 ? "0" : t.toFixed(1)}
            </text>
          </g>
        );
      })}
      {xLabel && (
        <text x={pad.left + plotW / 2} y={h - 4} fontSize={10} fill="#64748b" textAnchor="middle" fontWeight={600}>
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={12}
          y={pad.top + plotH / 2}
          fontSize={10}
          fill="#64748b"
          textAnchor="middle"
          fontWeight={600}
          transform={`rotate(-90, 12, ${pad.top + plotH / 2})`}
        >
          {yLabel}
        </text>
      )}
    </>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB DEFINITIONS
// ════════════════════════════════════════════════════════════════════════════

type TabKey = "fit" | "outliers" | "gradient";

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabDef[] = [
  { key: "fit", label: "Fit the Line", icon: <TrendingUp className="w-3.5 h-3.5" /> },
  { key: "outliers", label: "Effect of Outliers", icon: <AlertTriangle className="w-3.5 h-3.5" /> },
  { key: "gradient", label: "Gradient Descent", icon: <LineChart className="w-3.5 h-3.5" /> },
];

// ════════════════════════════════════════════════════════════════════════════
// TAB 1 — FIT THE LINE
// ════════════════════════════════════════════════════════════════════════════

const SVG_W = 520;
const SVG_H = 400;
const PAD = PAD_DEFAULT;
const PLOT_W = SVG_W - PAD.left - PAD.right;
const PLOT_H = SVG_H - PAD.top - PAD.bottom;

function FitTheLineTab() {
  const [seed, setSeed] = useState(42);
  const [slope, setSlope] = useState(0.5);
  const [intercept, setIntercept] = useState(2.0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [dragging, setDragging] = useState<"handle1" | "handle2" | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const animRef = useRef<number | null>(null);

  const points = useMemo(() => generateLinearPoints(seed), [seed]);
  const { xMin, xMax, yMin, yMax } = useMemo(() => axisRange(points), [points]);
  const mse = useMemo(() => computeMSE(points, slope, intercept), [points, slope, intercept]);
  const ols = useMemo(() => computeOLS(points), [points]);
  const olsMSE = useMemo(() => computeMSE(points, ols.m, ols.b), [points, ols]);

  const handle1X = xMin + (xMax - xMin) * 0.25;
  const handle2X = xMin + (xMax - xMin) * 0.75;

  const maxAbsResidual = useMemo(() => {
    let mx = 0;
    for (const p of points) mx = Math.max(mx, Math.abs(slope * p.x + intercept - p.y));
    return mx;
  }, [points, slope, intercept]);

  const getSVGPoint = useCallback((e: React.MouseEvent | MouseEvent) => {
    if (!svgRef.current) return { sx: 0, sy: 0 };
    const r = svgRef.current.getBoundingClientRect();
    return { sx: ((e.clientX - r.left) / r.width) * SVG_W, sy: ((e.clientY - r.top) / r.height) * SVG_H };
  }, []);

  const handleMouseDown = useCallback(
    (handle: "handle1" | "handle2") => (e: React.MouseEvent) => {
      e.preventDefault();
      setDragging(handle);
    },
    []
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging) return;
      const { sy } = getSVGPoint(e);
      const dataY = fromSY(sy, yMin, yMax, PAD, PLOT_H);
      if (dragging === "handle1") {
        const curH2Y = slope * handle2X + intercept;
        const newSlope = (curH2Y - dataY) / (handle2X - handle1X);
        setSlope(newSlope);
        setIntercept(dataY - newSlope * handle1X);
      } else {
        const curH1Y = slope * handle1X + intercept;
        const newSlope = (dataY - curH1Y) / (handle2X - handle1X);
        setSlope(newSlope);
        setIntercept(curH1Y - newSlope * handle1X);
      }
    },
    [dragging, getSVGPoint, slope, intercept, handle1X, handle2X, yMin, yMax]
  );

  const handleMouseUp = useCallback(() => setDragging(null), []);

  const animateToBestFit = useCallback(() => {
    if (isAnimating) return;
    setIsAnimating(true);
    const startM = slope,
      startB = intercept;
    const targetM = ols.m,
      targetB = ols.b;
    const dur = 800;
    const t0 = performance.now();
    const step = (now: number) => {
      const t = Math.min((now - t0) / dur, 1);
      const e = 1 - (1 - t) ** 3;
      setSlope(startM + (targetM - startM) * e);
      setIntercept(startB + (targetB - startB) * e);
      if (t < 1) animRef.current = requestAnimationFrame(step);
      else {
        setIsAnimating(false);
        animRef.current = null;
      }
    };
    animRef.current = requestAnimationFrame(step);
  }, [isAnimating, slope, intercept, ols]);

  const genNew = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    setIsAnimating(false);
    setSeed((s) => s + 1);
    setSlope(0.5);
    setIntercept(2.0);
  }, []);

  const resetLine = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    setIsAnimating(false);
    setSlope(0.5);
    setIntercept(2.0);
  }, []);

  const h1sx = toSX(handle1X, xMin, xMax, PAD, PLOT_W);
  const h1sy = toSY(slope * handle1X + intercept, yMin, yMax, PAD, PLOT_H);
  const h2sx = toSX(handle2X, xMin, xMax, PAD, PLOT_W);
  const h2sy = toSY(slope * handle2X + intercept, yMin, yMax, PAD, PLOT_H);

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">What is Linear Regression?</h3>
          <p className="text-xs text-indigo-700 mt-1">
            Linear regression finds the line that minimizes the <strong>sum of squared errors</strong> between predicted
            and actual values. Drag the two handles to adjust slope and intercept. Try to get MSE as low as possible,
            then click <strong>Show Best Fit</strong> to see the OLS solution.
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="w-full max-w-[560px] mx-auto select-none"
              style={{ aspectRatio: `${SVG_W}/${SVG_H}` }}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <ChartGrid w={SVG_W} h={SVG_H} pad={PAD} xMin={xMin} xMax={xMax} yMin={yMin} yMax={yMax} xLabel="x" yLabel="y" clipId="fitClip" />

              {/* Residuals */}
              <g clipPath="url(#fitClip)">
                {points.map((p, i) => {
                  const pred = slope * p.x + intercept;
                  const absE = Math.abs(pred - p.y);
                  return (
                    <line
                      key={`r-${i}`}
                      x1={toSX(p.x, xMin, xMax, PAD, PLOT_W)}
                      y1={toSY(p.y, yMin, yMax, PAD, PLOT_H)}
                      x2={toSX(p.x, xMin, xMax, PAD, PLOT_W)}
                      y2={toSY(pred, yMin, yMax, PAD, PLOT_H)}
                      stroke={residualColor(absE, maxAbsResidual)}
                      strokeWidth={1.5}
                      strokeDasharray="4,3"
                      opacity={0.7}
                    />
                  );
                })}
              </g>

              {/* Regression line */}
              <line
                x1={toSX(xMin, xMin, xMax, PAD, PLOT_W)}
                y1={toSY(slope * xMin + intercept, yMin, yMax, PAD, PLOT_H)}
                x2={toSX(xMax, xMin, xMax, PAD, PLOT_W)}
                y2={toSY(slope * xMax + intercept, yMin, yMax, PAD, PLOT_H)}
                stroke="#6366f1"
                strokeWidth={2.5}
                clipPath="url(#fitClip)"
              />

              {/* Points */}
              {points.map((p, i) => (
                <circle
                  key={`p-${i}`}
                  cx={toSX(p.x, xMin, xMax, PAD, PLOT_W)}
                  cy={toSY(p.y, yMin, yMax, PAD, PLOT_H)}
                  r={5}
                  fill="#3b82f6"
                  stroke="#fff"
                  strokeWidth={1.5}
                  opacity={0.9}
                />
              ))}

              {/* Handle 1 */}
              <g style={{ cursor: "grab" }} onMouseDown={handleMouseDown("handle1")}>
                <circle cx={h1sx} cy={h1sy} r={16} fill="transparent" />
                <circle cx={h1sx} cy={h1sy} r={8} fill={dragging === "handle1" ? "#4f46e5" : "#6366f1"} stroke="#fff" strokeWidth={2.5} clipPath="url(#fitClip)" />
                <circle cx={h1sx} cy={h1sy} r={3} fill="white" clipPath="url(#fitClip)" />
              </g>

              {/* Handle 2 */}
              <g style={{ cursor: "grab" }} onMouseDown={handleMouseDown("handle2")}>
                <circle cx={h2sx} cy={h2sy} r={16} fill="transparent" />
                <circle cx={h2sx} cy={h2sy} r={8} fill={dragging === "handle2" ? "#4f46e5" : "#6366f1"} stroke="#fff" strokeWidth={2.5} clipPath="url(#fitClip)" />
                <circle cx={h2sx} cy={h2sy} r={3} fill="white" clipPath="url(#fitClip)" />
              </g>

              {/* Equation overlay */}
              <g transform={`translate(${PAD.left + PLOT_W - 8}, ${PAD.top + 10})`}>
                <rect x={-170} y={-4} width={174} height={24} rx={4} fill="white" fillOpacity={0.92} stroke="#e2e8f0" strokeWidth={0.5} />
                <text x={-83} y={13} fontSize={12} fill="#6366f1" textAnchor="middle" fontWeight={600} fontFamily="monospace">
                  y = {slope >= 0 ? "" : "-"}
                  {Math.abs(slope).toFixed(2)}x {intercept >= 0 ? "+" : "-"} {Math.abs(intercept).toFixed(2)}
                </text>
              </g>

              {/* Legend */}
              <g transform={`translate(${PAD.left + 8}, ${PAD.top + 10})`}>
                <rect x={0} y={-4} width={130} height={62} rx={4} fill="white" fillOpacity={0.92} stroke="#e2e8f0" strokeWidth={0.5} />
                <circle cx={14} cy={10} r={4} fill="#3b82f6" />
                <text x={24} y={13} fontSize={9} fill="#64748b">Data points</text>
                <line x1={6} y1={28} x2={22} y2={28} stroke="#6366f1" strokeWidth={2.5} />
                <text x={28} y={31} fontSize={9} fill="#64748b">Your line</text>
                <line x1={6} y1={46} x2={22} y2={46} stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4,3" />
                <text x={28} y={49} fontSize={9} fill="#64748b">Residuals</text>
              </g>
            </svg>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-72 space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg p-4 text-center">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Mean Squared Error</p>
            <p className="text-3xl font-bold text-slate-800 mt-1 font-mono">{mse < 0.01 ? mse.toExponential(2) : mse.toFixed(3)}</p>
            <p className="text-[10px] text-slate-400 mt-1">
              Best (OLS): <span className="font-semibold text-green-600">{olsMSE < 0.01 ? olsMSE.toExponential(2) : olsMSE.toFixed(3)}</span>
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-1">Current Equation</p>
            <p className="text-sm font-mono text-slate-700">
              y = <span className="font-bold text-indigo-600">{slope.toFixed(3)}</span>x {intercept >= 0 ? "+" : "-"}{" "}
              <span className="font-bold text-indigo-600">{Math.abs(intercept).toFixed(3)}</span>
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white border border-slate-200 rounded-lg p-3 text-center">
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Slope (m)</p>
              <p className="text-xl font-bold text-slate-800 mt-1 font-mono">{slope.toFixed(3)}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3 text-center">
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Intercept (b)</p>
              <p className="text-xl font-bold text-slate-800 mt-1 font-mono">{intercept.toFixed(3)}</p>
            </div>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-3.5 h-3.5 text-green-600" />
              <p className="text-[11px] text-green-800 uppercase tracking-wide font-medium">OLS Best Fit</p>
            </div>
            <p className="text-xs font-mono text-green-700">
              m = {ols.m.toFixed(3)}, b = {ols.b.toFixed(3)}
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <button
              onClick={animateToBestFit}
              disabled={isAnimating}
              className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold bg-indigo-500 text-white hover:bg-indigo-600 disabled:opacity-50 transition-all"
            >
              <Sparkles className="w-4 h-4" />
              Show Best Fit
            </button>
            <div className="flex gap-2">
              <button
                onClick={genNew}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all"
              >
                <Shuffle className="w-4 h-4" />
                New Data
              </button>
              <button onClick={resetLine} className="px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all" title="Reset line">
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Residual Colors</p>
            <div className="flex items-center gap-4 text-xs text-slate-600">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-green-500 inline-block" /> Small
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-amber-500 inline-block" /> Medium
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> Large
              </span>
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Try these experiments:</p>
            <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
              <li>Drag the handles to minimize the MSE</li>
              <li>Watch residual colors change</li>
              <li>Compare your result to the OLS best fit</li>
              <li>Generate new data and try different patterns</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 2 — EFFECT OF OUTLIERS
// ════════════════════════════════════════════════════════════════════════════

function OutliersTab() {
  const [seed, setSeed] = useState(77);
  const [outliers, setOutliers] = useState<DataPoint[]>([]);
  const [showWithoutOutliers, setShowWithoutOutliers] = useState(true);
  const [showWithOutliers, setShowWithOutliers] = useState(true);
  const svgRef = useRef<SVGSVGElement>(null);
  const [clickMode, setClickMode] = useState(false);

  const basePoints = useMemo(() => generateLinearPoints(seed), [seed]);
  const allPoints = useMemo(() => [...basePoints, ...outliers], [basePoints, outliers]);

  const olsBase = useMemo(() => computeOLS(basePoints), [basePoints]);
  const olsAll = useMemo(() => computeOLS(allPoints), [allPoints]);
  const mseBase = useMemo(() => computeMSE(basePoints, olsBase.m, olsBase.b), [basePoints, olsBase]);
  const mseAll = useMemo(() => computeMSE(allPoints, olsAll.m, olsAll.b), [allPoints, olsAll]);

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const all = [...basePoints, ...outliers];
    const r = axisRange(all, 0.2);
    // ensure y range is generous to allow outlier placement
    return { ...r, yMin: Math.min(r.yMin, -2), yMax: Math.max(r.yMax, r.yMax + 3) };
  }, [basePoints, outliers]);

  const handleSvgClick = useCallback(
    (e: React.MouseEvent) => {
      if (!clickMode || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const sx = ((e.clientX - rect.left) / rect.width) * SVG_W;
      const sy = ((e.clientY - rect.top) / rect.height) * SVG_H;
      const x = fromSX(sx, xMin, xMax, PAD, PLOT_W);
      const y = fromSY(sy, yMin, yMax, PAD, PLOT_H);
      if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
        setOutliers((prev) => [...prev, { x, y }]);
      }
    },
    [clickMode, xMin, xMax, yMin, yMax]
  );

  const addRandomOutlier = useCallback(() => {
    const rng = mulberry32(Date.now());
    const x = xMin + rng() * (xMax - xMin);
    // place it far from the regression line
    const pred = olsBase.m * x + olsBase.b;
    const offset = (rng() > 0.5 ? 1 : -1) * (3 + rng() * 4);
    setOutliers((prev) => [...prev, { x, y: pred + offset }]);
  }, [xMin, xMax, olsBase]);

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 flex gap-3">
        <AlertTriangle className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-orange-900">Why Do Outliers Matter?</h3>
          <p className="text-xs text-orange-700 mt-1">
            OLS minimizes <strong>squared</strong> errors, so a single point far from the trend has an outsized effect on
            the line. Add outliers below and watch the best-fit line get pulled towards them. Toggle the lines to compare
            the fit with and without outliers. This is why data cleaning and robust regression techniques exist.
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="w-full max-w-[560px] mx-auto select-none"
              style={{ aspectRatio: `${SVG_W}/${SVG_H}`, cursor: clickMode ? "crosshair" : "default" }}
              onClick={handleSvgClick}
            >
              <ChartGrid w={SVG_W} h={SVG_H} pad={PAD} xMin={xMin} xMax={xMax} yMin={yMin} yMax={yMax} xLabel="x" yLabel="y" clipId="outlierClip" />

              {/* Line WITHOUT outliers */}
              {showWithoutOutliers && (
                <line
                  x1={toSX(xMin, xMin, xMax, PAD, PLOT_W)}
                  y1={toSY(olsBase.m * xMin + olsBase.b, yMin, yMax, PAD, PLOT_H)}
                  x2={toSX(xMax, xMin, xMax, PAD, PLOT_W)}
                  y2={toSY(olsBase.m * xMax + olsBase.b, yMin, yMax, PAD, PLOT_H)}
                  stroke="#22c55e"
                  strokeWidth={2.5}
                  clipPath="url(#outlierClip)"
                />
              )}

              {/* Line WITH outliers */}
              {showWithOutliers && outliers.length > 0 && (
                <line
                  x1={toSX(xMin, xMin, xMax, PAD, PLOT_W)}
                  y1={toSY(olsAll.m * xMin + olsAll.b, yMin, yMax, PAD, PLOT_H)}
                  x2={toSX(xMax, xMin, xMax, PAD, PLOT_W)}
                  y2={toSY(olsAll.m * xMax + olsAll.b, yMin, yMax, PAD, PLOT_H)}
                  stroke="#ef4444"
                  strokeWidth={2.5}
                  strokeDasharray="8,4"
                  clipPath="url(#outlierClip)"
                />
              )}

              {/* Base data points */}
              {basePoints.map((p, i) => (
                <circle
                  key={`bp-${i}`}
                  cx={toSX(p.x, xMin, xMax, PAD, PLOT_W)}
                  cy={toSY(p.y, yMin, yMax, PAD, PLOT_H)}
                  r={5}
                  fill="#3b82f6"
                  stroke="#fff"
                  strokeWidth={1.5}
                  opacity={0.9}
                />
              ))}

              {/* Outlier points */}
              {outliers.map((p, i) => (
                <g key={`op-${i}`}>
                  <circle
                    cx={toSX(p.x, xMin, xMax, PAD, PLOT_W)}
                    cy={toSY(p.y, yMin, yMax, PAD, PLOT_H)}
                    r={7}
                    fill="#ef4444"
                    stroke="#fff"
                    strokeWidth={2}
                    opacity={0.95}
                  />
                  <text
                    x={toSX(p.x, xMin, xMax, PAD, PLOT_W)}
                    y={toSY(p.y, yMin, yMax, PAD, PLOT_H) + 1}
                    fontSize={8}
                    fill="white"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontWeight={700}
                  >
                    !
                  </text>
                </g>
              ))}

              {/* Legend */}
              <g transform={`translate(${PAD.left + 8}, ${PAD.top + 10})`}>
                <rect x={0} y={-4} width={170} height={52} rx={4} fill="white" fillOpacity={0.92} stroke="#e2e8f0" strokeWidth={0.5} />
                <line x1={6} y1={10} x2={22} y2={10} stroke="#22c55e" strokeWidth={2.5} />
                <text x={28} y={13} fontSize={9} fill="#64748b">Fit without outliers</text>
                <line x1={6} y1={28} x2={22} y2={28} stroke="#ef4444" strokeWidth={2.5} strokeDasharray="6,3" />
                <text x={28} y={31} fontSize={9} fill="#64748b">Fit with outliers</text>
                <circle cx={14} cy={42} r={4} fill="#ef4444" />
                <text x={28} y={45} fontSize={9} fill="#64748b">Outlier points</text>
              </g>
            </svg>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-72 space-y-4">
          {/* MSE comparison */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-center">
              <p className="text-[10px] text-green-700 uppercase tracking-wide font-medium">MSE (no outliers)</p>
              <p className="text-lg font-bold text-green-800 mt-1 font-mono">{mseBase.toFixed(3)}</p>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-center">
              <p className="text-[10px] text-red-700 uppercase tracking-wide font-medium">MSE (with outliers)</p>
              <p className="text-lg font-bold text-red-800 mt-1 font-mono">{outliers.length > 0 ? mseAll.toFixed(3) : "—"}</p>
            </div>
          </div>

          {/* Equation comparison */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Equations</p>
            <p className="text-xs font-mono text-green-700">
              Without: y = {olsBase.m.toFixed(3)}x {olsBase.b >= 0 ? "+" : "-"} {Math.abs(olsBase.b).toFixed(3)}
            </p>
            {outliers.length > 0 && (
              <p className="text-xs font-mono text-red-700">
                With: y = {olsAll.m.toFixed(3)}x {olsAll.b >= 0 ? "+" : "-"} {Math.abs(olsAll.b).toFixed(3)}
              </p>
            )}
          </div>

          {/* Slope shift */}
          {outliers.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <p className="text-[11px] text-yellow-800 uppercase tracking-wide font-medium mb-1">Slope Shift</p>
              <p className="text-sm font-mono text-yellow-900 font-bold">
                {((olsAll.m - olsBase.m) / (Math.abs(olsBase.m) || 1) * 100).toFixed(1)}% change
              </p>
              <p className="text-[10px] text-yellow-700 mt-1">
                {Math.abs(olsAll.m - olsBase.m).toFixed(3)} absolute shift from {outliers.length} outlier{outliers.length > 1 ? "s" : ""}
              </p>
            </div>
          )}

          {/* Toggle lines */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Show Lines</p>
            <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input type="checkbox" checked={showWithoutOutliers} onChange={(e) => setShowWithoutOutliers(e.target.checked)} className="accent-green-600" />
              Without outliers (green)
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input type="checkbox" checked={showWithOutliers} onChange={(e) => setShowWithOutliers(e.target.checked)} className="accent-red-600" />
              With outliers (red dashed)
            </label>
          </div>

          {/* Buttons */}
          <div className="flex flex-col gap-2">
            <button
              onClick={addRandomOutlier}
              className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold bg-red-500 text-white hover:bg-red-600 transition-all"
            >
              <Plus className="w-4 h-4" />
              Add Random Outlier
            </button>
            <button
              onClick={() => setClickMode((c) => !c)}
              className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold border transition-all ${
                clickMode ? "bg-orange-100 border-orange-400 text-orange-800" : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
              }`}
            >
              {clickMode ? "Click Mode ON — click chart" : "Click to Place Outlier"}
            </button>
            <div className="flex gap-2">
              <button
                onClick={() => setOutliers([])}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all"
              >
                <Trash2 className="w-4 h-4" />
                Clear Outliers ({outliers.length})
              </button>
              <button
                onClick={() => setSeed((s) => s + 1)}
                className="px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all"
                title="New base data"
              >
                <Shuffle className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Things to try:</p>
            <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
              <li>Add one outlier far above the data</li>
              <li>Add several outliers on one side</li>
              <li>Toggle lines to compare the shift</li>
              <li>Notice how MSE jumps with outliers</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 3 — GRADIENT DESCENT FIT
// ════════════════════════════════════════════════════════════════════════════

function GradientDescentTab() {
  const [seed, setSeed] = useState(55);
  const [lr, setLr] = useState(0.01);
  const [m, setM] = useState(0);
  const [b, setB] = useState(0);
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<{ m: number; b: number; mse: number }[]>([{ m: 0, b: 0, mse: 0 }]);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const points = useMemo(() => generateLinearPoints(seed), [seed]);
  const ols = useMemo(() => computeOLS(points), [points]);
  const { xMin, xMax, yMin, yMax } = useMemo(() => axisRange(points), [points]);

  // Initialise history MSE once points are available
  useEffect(() => {
    const initMSE = computeMSE(points, 0, 0);
    setHistory([{ m: 0, b: 0, mse: initMSE }]);
  }, [points]);

  const doStep = useCallback(() => {
    setM((prevM) => {
      setB((prevB) => {
        const n = points.length;
        let dM = 0,
          dB = 0;
        for (const p of points) {
          const pred = prevM * p.x + prevB;
          const err = pred - p.y;
          dM += (2 / n) * err * p.x;
          dB += (2 / n) * err;
        }
        const newM = prevM - lr * dM;
        const newB = prevB - lr * dB;
        const newMSE = computeMSE(points, newM, newB);
        setStep((s) => s + 1);
        setHistory((h) => [...h, { m: newM, b: newB, mse: newMSE }]);
        setM(newM);
        return newB;
      });
      return prevM; // will be overwritten by inner setM
    });
  }, [points, lr]);

  // Proper step that avoids stale closures: compute directly from latest
  const doStepDirect = useCallback(() => {
    setHistory((prev) => {
      const last = prev[prev.length - 1];
      const n = points.length;
      let dM = 0,
        dB = 0;
      for (const p of points) {
        const pred = last.m * p.x + last.b;
        const err = pred - p.y;
        dM += (2 / n) * err * p.x;
        dB += (2 / n) * err;
      }
      const newM = last.m - lr * dM;
      const newB = last.b - lr * dB;
      const newMSE = computeMSE(points, newM, newB);
      setM(newM);
      setB(newB);
      setStep((s) => s + 1);
      return [...prev, { m: newM, b: newB, mse: newMSE }];
    });
  }, [points, lr]);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(doStepDirect, 60);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, doStepDirect]);

  const reset = useCallback(() => {
    setRunning(false);
    setM(0);
    setB(0);
    setStep(0);
    const initMSE = computeMSE(points, 0, 0);
    setHistory([{ m: 0, b: 0, mse: initMSE }]);
  }, [points]);

  const currentMSE = history.length > 0 ? history[history.length - 1].mse : computeMSE(points, m, b);
  const olsMSE = computeMSE(points, ols.m, ols.b);

  // Loss curve chart dimensions
  const LOSS_W = 260;
  const LOSS_H = 160;
  const LOSS_PAD = { top: 20, right: 15, bottom: 30, left: 45 };
  const LOSS_PW = LOSS_W - LOSS_PAD.left - LOSS_PAD.right;
  const LOSS_PH = LOSS_H - LOSS_PAD.top - LOSS_PAD.bottom;

  const lossRange = useMemo(() => {
    if (history.length === 0) return { minMSE: 0, maxMSE: 10 };
    const mses = history.map((h) => h.mse);
    return { minMSE: 0, maxMSE: Math.max(...mses) * 1.1 };
  }, [history]);

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 flex gap-3">
        <LineChart className="w-5 h-5 text-teal-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-teal-900">How Does the Computer Learn the Line?</h3>
          <p className="text-xs text-teal-700 mt-1">
            <strong>Gradient descent</strong> is an iterative algorithm: start with a guess (m=0, b=0), compute the
            gradient of the loss function, and take a small step downhill. The <strong>learning rate</strong> controls
            step size. Too small = slow convergence. Too large = overshooting. Watch the MSE decrease on the loss curve!
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        {/* Main scatter plot */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <svg
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="w-full max-w-[560px] mx-auto select-none"
              style={{ aspectRatio: `${SVG_W}/${SVG_H}` }}
            >
              <ChartGrid w={SVG_W} h={SVG_H} pad={PAD} xMin={xMin} xMax={xMax} yMin={yMin} yMax={yMax} xLabel="x" yLabel="y" clipId="gdClip" />

              {/* OLS line (faint reference) */}
              <line
                x1={toSX(xMin, xMin, xMax, PAD, PLOT_W)}
                y1={toSY(ols.m * xMin + ols.b, yMin, yMax, PAD, PLOT_H)}
                x2={toSX(xMax, xMin, xMax, PAD, PLOT_W)}
                y2={toSY(ols.m * xMax + ols.b, yMin, yMax, PAD, PLOT_H)}
                stroke="#22c55e"
                strokeWidth={1.5}
                strokeDasharray="6,4"
                opacity={0.5}
                clipPath="url(#gdClip)"
              />

              {/* GD line */}
              <line
                x1={toSX(xMin, xMin, xMax, PAD, PLOT_W)}
                y1={toSY(m * xMin + b, yMin, yMax, PAD, PLOT_H)}
                x2={toSX(xMax, xMin, xMax, PAD, PLOT_W)}
                y2={toSY(m * xMax + b, yMin, yMax, PAD, PLOT_H)}
                stroke="#6366f1"
                strokeWidth={2.5}
                clipPath="url(#gdClip)"
              />

              {/* Points */}
              {points.map((p, i) => (
                <circle
                  key={i}
                  cx={toSX(p.x, xMin, xMax, PAD, PLOT_W)}
                  cy={toSY(p.y, yMin, yMax, PAD, PLOT_H)}
                  r={5}
                  fill="#3b82f6"
                  stroke="#fff"
                  strokeWidth={1.5}
                  opacity={0.9}
                />
              ))}

              {/* Legend */}
              <g transform={`translate(${PAD.left + 8}, ${PAD.top + 10})`}>
                <rect x={0} y={-4} width={155} height={42} rx={4} fill="white" fillOpacity={0.92} stroke="#e2e8f0" strokeWidth={0.5} />
                <line x1={6} y1={10} x2={22} y2={10} stroke="#6366f1" strokeWidth={2.5} />
                <text x={28} y={13} fontSize={9} fill="#64748b">Gradient descent line</text>
                <line x1={6} y1={28} x2={22} y2={28} stroke="#22c55e" strokeWidth={1.5} strokeDasharray="6,4" />
                <text x={28} y={31} fontSize={9} fill="#64748b">OLS target (reference)</text>
              </g>
            </svg>
          </div>
        </div>

        {/* Controls & loss curve */}
        <div className="w-full lg:w-80 space-y-4">
          {/* Step & MSE */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white border border-slate-200 rounded-lg p-3 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium">Iteration</p>
              <p className="text-2xl font-bold text-slate-800 mt-1 font-mono">{step}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium">Current MSE</p>
              <p className="text-2xl font-bold text-slate-800 mt-1 font-mono">{currentMSE < 0.01 ? currentMSE.toExponential(2) : currentMSE.toFixed(3)}</p>
            </div>
          </div>

          {/* Current m, b */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-1">Current Parameters</p>
            <div className="grid grid-cols-2 gap-2 text-center">
              <div>
                <p className="text-[10px] text-slate-400">m (slope)</p>
                <p className="text-sm font-bold font-mono text-indigo-600">{m.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-slate-400">b (intercept)</p>
                <p className="text-sm font-bold font-mono text-indigo-600">{b.toFixed(4)}</p>
              </div>
            </div>
            <div className="mt-2 pt-2 border-t border-slate-100 grid grid-cols-2 gap-2 text-center">
              <div>
                <p className="text-[10px] text-slate-400">OLS m</p>
                <p className="text-xs font-mono text-green-600">{ols.m.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-slate-400">OLS b</p>
                <p className="text-xs font-mono text-green-600">{ols.b.toFixed(4)}</p>
              </div>
            </div>
          </div>

          {/* Learning rate slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Learning Rate</p>
              <span className="text-xs font-mono text-indigo-600 font-bold">{lr}</span>
            </div>
            <input
              type="range"
              min={0.001}
              max={0.5}
              step={0.001}
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="w-full accent-indigo-500"
            />
            <div className="flex justify-between text-[9px] text-slate-400 mt-1">
              <span>0.001 (slow)</span>
              <span>0.5 (fast)</span>
            </div>
          </div>

          {/* Buttons */}
          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <button
                onClick={doStepDirect}
                disabled={running}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold bg-indigo-500 text-white hover:bg-indigo-600 disabled:opacity-50 transition-all"
              >
                <SkipForward className="w-4 h-4" />
                Step
              </button>
              <button
                onClick={() => setRunning((r) => !r)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                  running ? "bg-amber-500 text-white hover:bg-amber-600" : "bg-teal-500 text-white hover:bg-teal-600"
                }`}
              >
                {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {running ? "Pause" : "Auto-Run"}
              </button>
            </div>
            <div className="flex gap-2">
              <button
                onClick={reset}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
              <button
                onClick={() => {
                  setRunning(false);
                  setSeed((s) => s + 1);
                  setM(0);
                  setB(0);
                  setStep(0);
                }}
                className="px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all"
                title="New Data"
              >
                <Shuffle className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Loss curve */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">MSE Loss Curve</p>
            <svg viewBox={`0 0 ${LOSS_W} ${LOSS_H}`} className="w-full" style={{ aspectRatio: `${LOSS_W}/${LOSS_H}` }}>
              <rect x={LOSS_PAD.left} y={LOSS_PAD.top} width={LOSS_PW} height={LOSS_PH} fill="white" stroke="#e2e8f0" rx={2} />

              {/* Y-axis labels */}
              {[0, 0.25, 0.5, 0.75, 1].map((frac, i) => {
                const val = lossRange.minMSE + frac * (lossRange.maxMSE - lossRange.minMSE);
                const sy = LOSS_PAD.top + LOSS_PH * (1 - frac);
                return (
                  <g key={i}>
                    <line x1={LOSS_PAD.left} y1={sy} x2={LOSS_PAD.left + LOSS_PW} y2={sy} stroke="#f1f5f9" />
                    <text x={LOSS_PAD.left - 4} y={sy + 3} fontSize={7} fill="#94a3b8" textAnchor="end">
                      {val < 0.01 && val !== 0 ? val.toExponential(1) : val.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* OLS MSE reference line */}
              {lossRange.maxMSE > 0 && (
                <>
                  <line
                    x1={LOSS_PAD.left}
                    y1={LOSS_PAD.top + LOSS_PH * (1 - (olsMSE - lossRange.minMSE) / (lossRange.maxMSE - lossRange.minMSE))}
                    x2={LOSS_PAD.left + LOSS_PW}
                    y2={LOSS_PAD.top + LOSS_PH * (1 - (olsMSE - lossRange.minMSE) / (lossRange.maxMSE - lossRange.minMSE))}
                    stroke="#22c55e"
                    strokeWidth={1}
                    strokeDasharray="4,3"
                    opacity={0.6}
                  />
                  <text
                    x={LOSS_PAD.left + LOSS_PW + 2}
                    y={LOSS_PAD.top + LOSS_PH * (1 - (olsMSE - lossRange.minMSE) / (lossRange.maxMSE - lossRange.minMSE)) + 3}
                    fontSize={7}
                    fill="#22c55e"
                  >
                    OLS
                  </text>
                </>
              )}

              {/* Loss curve path */}
              {history.length > 1 && (
                <polyline
                  points={history
                    .map((h, i) => {
                      const maxSteps = history.length - 1;
                      const sx = LOSS_PAD.left + (maxSteps > 0 ? (i / maxSteps) * LOSS_PW : 0);
                      const frac = lossRange.maxMSE > 0 ? (h.mse - lossRange.minMSE) / (lossRange.maxMSE - lossRange.minMSE) : 0;
                      const sy = LOSS_PAD.top + LOSS_PH * (1 - Math.max(0, Math.min(1, frac)));
                      return `${sx},${sy}`;
                    })
                    .join(" ")}
                  fill="none"
                  stroke="#6366f1"
                  strokeWidth={1.5}
                />
              )}

              <text x={LOSS_PAD.left + LOSS_PW / 2} y={LOSS_H - 4} fontSize={8} fill="#64748b" textAnchor="middle">
                Iteration
              </text>
            </svg>
          </div>

          {/* Convergence indicator */}
          {step > 0 && Math.abs(currentMSE - olsMSE) < 0.01 && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-600" />
              <p className="text-xs text-green-800 font-medium">Converged! MSE is within 0.01 of the OLS optimum.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════════════════

export default function LinearRegressionActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("fit");

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 overflow-x-auto pb-1 border-b border-slate-200">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-t-lg text-xs font-semibold whitespace-nowrap transition-all border-b-2 ${
              activeTab === tab.key
                ? "bg-indigo-50 text-indigo-700 border-indigo-500"
                : "text-slate-500 border-transparent hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "fit" && <FitTheLineTab />}
        {activeTab === "outliers" && <OutliersTab />}
        {activeTab === "gradient" && <GradientDescentTab />}
      </div>
    </div>
  );
}
