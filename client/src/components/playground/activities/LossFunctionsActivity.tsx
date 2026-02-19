/**
 * Loss Functions Activity — comprehensive 3-tab interactive SVG activity
 * Tab 1: MSE Explorer (drag prediction line, see squared residuals)
 * Tab 2: MAE vs MSE Comparison (side-by-side, outlier slider)
 * Tab 3: Loss Landscape (2D contour/heatmap, gradient descent animation)
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  BarChart3,
  GitCompareArrows,
  Mountain,
  Info,
  RotateCcw,
  Target,
  Play,
  Pause,
  Eye,
  EyeOff,
} from "lucide-react";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Seeded PRNG — mulberry32
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededGaussian(rng: () => number): number {
  const u1 = rng() || 1e-10;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Math helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function huberLoss(err: number, delta: number): number {
  const a = Math.abs(err);
  return a <= delta ? 0.5 * err * err : delta * (a - 0.5 * delta);
}

// OLS (Ordinary Least Squares) for simple linear regression
function olsFit(pts: { x: number; y: number }[]): { slope: number; intercept: number } {
  const n = pts.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of pts) {
    sx += p.x;
    sy += p.y;
    sxx += p.x * p.x;
    sxy += p.x * p.y;
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return { slope: 0, intercept: sy / n };
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  return { slope, intercept };
}

// Color interpolation for heatmaps
function heatColor(t: number): string {
  // t in [0,1] -> dark blue -> cyan -> green -> yellow -> red
  const clamped = clamp(t, 0, 1);
  const stops = [
    { t: 0.0, r: 13, g: 8, b: 135 },
    { t: 0.25, r: 84, g: 2, b: 163 },
    { t: 0.5, r: 190, g: 50, b: 70 },
    { t: 0.75, r: 243, g: 144, b: 29 },
    { t: 1.0, r: 252, g: 255, b: 164 },
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (clamped >= stops[i].t && clamped <= stops[i + 1].t) {
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }
  }
  const f = hi.t === lo.t ? 0 : (clamped - lo.t) / (hi.t - lo.t);
  const r = Math.round(lo.r + (hi.r - lo.r) * f);
  const g = Math.round(lo.g + (hi.g - lo.g) * f);
  const b = Math.round(lo.b + (hi.b - lo.b) * f);
  return `rgb(${r},${g},${b})`;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Data generation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const TRUE_SLOPE = 0.6;
const TRUE_INTERCEPT = 1.2;

function generateDataPoints(seed: number, n: number = 15): { x: number; y: number }[] {
  const rng = mulberry32(seed);
  const pts: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i++) {
    const x = 0.5 + (rng() * 8.5);
    const noise = seededGaussian(rng) * 0.6;
    const y = TRUE_SLOPE * x + TRUE_INTERCEPT + noise;
    pts.push({ x, y });
  }
  return pts.sort((a, b) => a.x - b.x);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
type TabKey = "mse" | "comparison" | "landscape";

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabDef[] = [
  { key: "mse", label: "MSE Explorer", icon: <BarChart3 className="w-4 h-4" /> },
  { key: "comparison", label: "MAE vs MSE", icon: <GitCompareArrows className="w-4 h-4" /> },
  { key: "landscape", label: "Loss Landscape", icon: <Mountain className="w-4 h-4" /> },
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SVG constants
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const W = 520;
const H = 360;
const PAD = { top: 25, right: 20, bottom: 40, left: 50 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

// Data domain for scatter plots
const X_MIN = 0;
const X_MAX = 10;
const Y_MIN = -1;
const Y_MAX = 9;

function toSX(x: number, xMin = X_MIN, xMax = X_MAX, padL = PAD.left, pw = PW): number {
  return padL + ((x - xMin) / (xMax - xMin)) * pw;
}
function toSY(y: number, yMin = Y_MIN, yMax = Y_MAX, padT = PAD.top, ph = PH): number {
  return padT + ph * (1 - (y - yMin) / (yMax - yMin));
}
function fromSX(sx: number, xMin = X_MIN, xMax = X_MAX, padL = PAD.left, pw = PW): number {
  return xMin + ((sx - padL) / pw) * (xMax - xMin);
}
function fromSY(sy: number, yMin = Y_MIN, yMax = Y_MAX, padT = PAD.top, ph = PH): number {
  return yMin + (1 - (sy - padT) / ph) * (yMax - yMin);
}

// SVG axis grid helper
function AxisGrid({
  xMin, xMax, yMin, yMax, xStep, yStep, w, h, pad, pw, ph,
}: {
  xMin: number; xMax: number; yMin: number; yMax: number;
  xStep: number; yStep: number;
  w: number; h: number; pad: typeof PAD; pw: number; ph: number;
}) {
  const xTicks: number[] = [];
  const yTicks: number[] = [];
  for (let v = xMin; v <= xMax + 0.001; v += xStep) xTicks.push(Math.round(v * 100) / 100);
  for (let v = yMin; v <= yMax + 0.001; v += yStep) yTicks.push(Math.round(v * 100) / 100);

  return (
    <>
      <rect x={pad.left} y={pad.top} width={pw} height={ph} fill="white" stroke="#e2e8f0" strokeWidth={1} />
      {yTicks.map((tick) => {
        const sy = toSY(tick, yMin, yMax, pad.top, ph);
        return (
          <g key={`y-${tick}`}>
            <line x1={pad.left} y1={sy} x2={pad.left + pw} y2={sy} stroke="#f1f5f9" strokeWidth={1} />
            <text x={pad.left - 8} y={sy + 3.5} fontSize={9} fill="#94a3b8" textAnchor="end">{tick}</text>
          </g>
        );
      })}
      {xTicks.map((tick) => {
        const sx = toSX(tick, xMin, xMax, pad.left, pw);
        return (
          <g key={`x-${tick}`}>
            <line x1={sx} y1={pad.top} x2={sx} y2={pad.top + ph} stroke="#f1f5f9" strokeWidth={1} />
            <text x={sx} y={h - pad.bottom + 16} fontSize={9} fill="#94a3b8" textAnchor="middle">{tick}</text>
          </g>
        );
      })}
    </>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 1 — MSE Explorer
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function Tab1MSEExplorer() {
  const dataPoints = useMemo(() => generateDataPoints(1001, 15), []);
  const olsSolution = useMemo(() => olsFit(dataPoints), [dataPoints]);

  const [slope, setSlope] = useState(0.3);
  const [intercept, setIntercept] = useState(2.5);
  const [dragging, setDragging] = useState<"slope" | "intercept" | null>(null);
  const [showBest, setShowBest] = useState(false);
  const [showSquares, setShowSquares] = useState(true);
  const svgRef = useRef<SVGSVGElement>(null);

  const predict = useCallback((x: number) => slope * x + intercept, [slope, intercept]);

  const residuals = useMemo(() =>
    dataPoints.map((p) => {
      const yHat = predict(p.x);
      const err = p.y - yHat;
      return { x: p.x, y: p.y, yHat, err, sqErr: err * err };
    }),
    [dataPoints, predict]
  );

  const mseValue = useMemo(() => {
    const sum = residuals.reduce((s, r) => s + r.sqErr, 0);
    return sum / residuals.length;
  }, [residuals]);

  const handleBestFit = useCallback(() => {
    setSlope(olsSolution.slope);
    setIntercept(olsSolution.intercept);
    setShowBest(true);
  }, [olsSolution]);

  const handleReset = useCallback(() => {
    setSlope(0.3);
    setIntercept(2.5);
    setShowBest(false);
  }, []);

  const getSvgCoords = useCallback((clientX: number, clientY: number) => {
    if (!svgRef.current) return { sx: 0, sy: 0 };
    const rect = svgRef.current.getBoundingClientRect();
    const sx = ((clientX - rect.left) / rect.width) * W;
    const sy = ((clientY - rect.top) / rect.height) * H;
    return { sx, sy };
  }, []);

  const handlePointerDown = useCallback((which: "slope" | "intercept") => {
    setDragging(which);
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging) return;
    const { sx, sy } = getSvgCoords(e.clientX, e.clientY);
    if (dragging === "intercept") {
      const newIntercept = fromSY(sy);
      setIntercept(clamp(newIntercept, -2, 8));
    } else if (dragging === "slope") {
      const dataX = fromSX(sx);
      const dataY = fromSY(sy);
      const newSlope = clamp((dataY - intercept) / Math.max(dataX, 0.1), -2, 3);
      setSlope(newSlope);
    }
  }, [dragging, intercept, getSvgCoords]);

  const handlePointerUp = useCallback(() => {
    setDragging(null);
  }, []);

  // Residual squares scaled to data space
  const maxSquarePx = 60;

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
        <p className="text-xs text-blue-700">
          <strong>Mean Squared Error</strong> measures the average of squared differences between predictions and actual values.
          Drag the <span className="text-indigo-600 font-semibold">prediction line</span> by its handles to see how MSE changes.
          Each squared residual is shown as a colored square -- larger squares mean larger errors.
        </p>
      </div>

      <div className="flex gap-4 flex-col lg:flex-row">
        {/* SVG Chart */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${W} ${H}`}
              className="w-full select-none"
              style={{ aspectRatio: `${W}/${H}`, touchAction: "none" }}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
            >
              <AxisGrid xMin={X_MIN} xMax={X_MAX} yMin={Y_MIN} yMax={Y_MAX} xStep={2} yStep={2} w={W} h={H} pad={PAD} pw={PW} ph={PH} />

              {/* Squared residual rectangles */}
              {showSquares && residuals.map((r, i) => {
                const sx = toSX(r.x);
                const syActual = toSY(r.y);
                const syPred = toSY(r.yHat);
                const sidePx = Math.min(Math.abs(syActual - syPred), maxSquarePx);
                if (sidePx < 2) return null;
                const topY = Math.min(syActual, syPred);
                const leftX = r.err > 0 ? sx : sx - sidePx;
                const alpha = clamp(0.12 + (r.sqErr / 10) * 0.4, 0.1, 0.5);
                return (
                  <rect
                    key={`sq-${i}`}
                    x={leftX}
                    y={topY}
                    width={sidePx}
                    height={sidePx}
                    fill="#ef4444"
                    opacity={alpha}
                    stroke="#ef4444"
                    strokeWidth={0.5}
                    strokeOpacity={0.4}
                    rx={2}
                  />
                );
              })}

              {/* Prediction line */}
              <line
                x1={toSX(X_MIN)}
                y1={toSY(predict(X_MIN))}
                x2={toSX(X_MAX)}
                y2={toSY(predict(X_MAX))}
                stroke="#6366f1"
                strokeWidth={2.5}
                strokeLinecap="round"
              />

              {/* Residual lines */}
              {residuals.map((r, i) => {
                if (Math.abs(r.err) < 0.05) return null;
                return (
                  <line
                    key={`res-${i}`}
                    x1={toSX(r.x)}
                    y1={toSY(r.y)}
                    x2={toSX(r.x)}
                    y2={toSY(r.yHat)}
                    stroke="#ef4444"
                    strokeWidth={1.5}
                    strokeDasharray="4,3"
                    opacity={0.7}
                  />
                );
              })}

              {/* Data points */}
              {dataPoints.map((p, i) => (
                <circle key={`dp-${i}`} cx={toSX(p.x)} cy={toSY(p.y)} r={5} fill="#10b981" stroke="#fff" strokeWidth={1.5} />
              ))}

              {/* Intercept handle (at x=0) */}
              <g
                onPointerDown={(e) => { e.preventDefault(); handlePointerDown("intercept"); }}
                style={{ cursor: "ns-resize" }}
              >
                <circle cx={toSX(0)} cy={toSY(intercept)} r={14} fill="transparent" />
                <circle cx={toSX(0)} cy={toSY(intercept)} r={8} fill="#6366f1" stroke="#fff" strokeWidth={2} />
                <text x={toSX(0) + 14} y={toSY(intercept) + 4} fontSize={9} fill="#6366f1" fontWeight={700}>b</text>
              </g>

              {/* Slope handle (at x=8) */}
              <g
                onPointerDown={(e) => { e.preventDefault(); handlePointerDown("slope"); }}
                style={{ cursor: "ns-resize" }}
              >
                <circle cx={toSX(8)} cy={toSY(predict(8))} r={14} fill="transparent" />
                <circle cx={toSX(8)} cy={toSY(predict(8))} r={8} fill="#8b5cf6" stroke="#fff" strokeWidth={2} />
                <text x={toSX(8) + 14} y={toSY(predict(8)) + 4} fontSize={9} fill="#8b5cf6" fontWeight={700}>m</text>
              </g>

              {/* Legend */}
              <g transform={`translate(${PAD.left + 8}, ${PAD.top + 8})`}>
                <rect x={0} y={0} width={140} height={52} rx={4} fill="white" fillOpacity={0.92} stroke="#e2e8f0" strokeWidth={0.5} />
                <circle cx={12} cy={14} r={4} fill="#10b981" />
                <text x={22} y={17} fontSize={9} fill="#64748b">Data points (actual)</text>
                <line x1={6} y1={30} x2={18} y2={30} stroke="#6366f1" strokeWidth={2} />
                <text x={22} y={33} fontSize={9} fill="#64748b">Prediction line</text>
                <rect x={6} y={40} width={12} height={8} fill="#ef4444" opacity={0.3} rx={1} />
                <text x={22} y={48} fontSize={9} fill="#64748b">Squared residuals</text>
              </g>
            </svg>
          </div>
        </div>

        {/* Controls panel */}
        <div className="w-full lg:w-72 space-y-3">
          {/* MSE readout */}
          <div className="bg-white border border-slate-200 rounded-lg p-4 text-center">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Mean Squared Error</p>
            <p className="text-3xl font-bold text-red-500 mt-1">{mseValue.toFixed(3)}</p>
            {showBest && <p className="text-[10px] text-emerald-600 mt-1">Optimal OLS Solution!</p>}
          </div>

          {/* Formula */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Formula</p>
            <div className="bg-slate-50 rounded p-2 font-mono text-xs text-slate-700 space-y-1">
              <p>MSE = (1/n) * Sum( (y_i - y_hat_i)^2 )</p>
              <p className="text-[10px] text-slate-500">= (1/{residuals.length}) * {residuals.reduce((s, r) => s + r.sqErr, 0).toFixed(3)}</p>
              <p className="text-[10px] text-slate-500">= <span className="text-red-500 font-bold">{mseValue.toFixed(3)}</span></p>
            </div>
          </div>

          {/* Line equation */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Prediction Line</p>
            <p className="font-mono text-sm text-indigo-600">y = {slope.toFixed(3)}x + {intercept.toFixed(3)}</p>
            <div className="mt-2 space-y-2">
              <div>
                <label className="text-[10px] text-slate-500">Slope (m): {slope.toFixed(2)}</label>
                <input type="range" min={-2} max={3} step={0.01} value={slope}
                  onChange={(e) => setSlope(parseFloat(e.target.value))}
                  className="w-full h-1.5 rounded-full appearance-none bg-indigo-100 accent-indigo-500" />
              </div>
              <div>
                <label className="text-[10px] text-slate-500">Intercept (b): {intercept.toFixed(2)}</label>
                <input type="range" min={-2} max={8} step={0.01} value={intercept}
                  onChange={(e) => setIntercept(parseFloat(e.target.value))}
                  className="w-full h-1.5 rounded-full appearance-none bg-indigo-100 accent-indigo-500" />
              </div>
            </div>
          </div>

          {/* Toggle squares */}
          <button
            onClick={() => setShowSquares(!showSquares)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200 transition-all border border-slate-200"
          >
            {showSquares ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            {showSquares ? "Hide" : "Show"} Squared Residuals
          </button>

          {/* Action buttons */}
          <div className="flex gap-2">
            <button onClick={handleReset}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200 transition-all border border-slate-200">
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
            <button onClick={handleBestFit}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-xs font-semibold bg-emerald-500 text-white hover:bg-emerald-600 transition-all">
              <Target className="w-3.5 h-3.5" /> Show Best Fit
            </button>
          </div>

          {/* Residual table */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 max-h-48 overflow-y-auto">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Residuals</p>
            <div className="space-y-1">
              {residuals.map((r, i) => (
                <div key={i} className="flex items-center gap-1 text-[10px] font-mono">
                  <span className="text-slate-400 w-8">pt{i + 1}</span>
                  <span className="text-slate-600 w-14">err={r.err.toFixed(2)}</span>
                  <div className="flex-1 bg-slate-100 rounded-full h-1.5 overflow-hidden">
                    <div className="h-full bg-red-400 rounded-full transition-all" style={{ width: `${clamp(r.sqErr / 5 * 100, 0, 100)}%` }} />
                  </div>
                  <span className="text-red-500 w-10 text-right">{r.sqErr.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 2 — MAE vs MSE Comparison
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const W2 = 280;
const H2 = 300;
const PAD2 = { top: 25, right: 15, bottom: 40, left: 45 };
const PW2 = W2 - PAD2.left - PAD2.right;
const PH2 = H2 - PAD2.top - PAD2.bottom;

function Tab2MAEMSEComparison() {
  const baseData = useMemo(() => generateDataPoints(2002, 12), []);
  const [outlierEnabled, setOutlierEnabled] = useState(true);
  const [outlierMagnitude, setOutlierMagnitude] = useState(4.0);
  const [slope, setSlope] = useState(0.55);
  const [intercept, setIntercept] = useState(1.5);

  const dataWithOutlier = useMemo(() => {
    const pts = [...baseData];
    if (outlierEnabled) {
      pts.push({ x: 5.0, y: TRUE_SLOPE * 5.0 + TRUE_INTERCEPT + outlierMagnitude });
    }
    return pts;
  }, [baseData, outlierEnabled, outlierMagnitude]);

  const predict = useCallback((x: number) => slope * x + intercept, [slope, intercept]);

  const olsBest = useMemo(() => olsFit(dataWithOutlier), [dataWithOutlier]);

  const maeResiduals = useMemo(() =>
    dataWithOutlier.map((p) => ({ x: p.x, y: p.y, yHat: predict(p.x), absErr: Math.abs(p.y - predict(p.x)) })),
    [dataWithOutlier, predict]
  );

  const mseResiduals = useMemo(() =>
    dataWithOutlier.map((p) => ({ x: p.x, y: p.y, yHat: predict(p.x), sqErr: (p.y - predict(p.x)) ** 2 })),
    [dataWithOutlier, predict]
  );

  const maeValue = useMemo(() => maeResiduals.reduce((s, r) => s + r.absErr, 0) / maeResiduals.length, [maeResiduals]);
  const mseValue = useMemo(() => mseResiduals.reduce((s, r) => s + r.sqErr, 0) / mseResiduals.length, [mseResiduals]);

  const handleBestFit = () => { setSlope(olsBest.slope); setIntercept(olsBest.intercept); };

  const toSX2 = (x: number) => PAD2.left + ((x - X_MIN) / (X_MAX - X_MIN)) * PW2;
  const toSY2 = (y: number) => PAD2.top + PH2 * (1 - (y - Y_MIN) / (Y_MAX - Y_MIN));

  const maxBarHeight = 50;

  function renderChart(title: string, type: "MAE" | "MSE", color: string) {
    const residuals = type === "MAE" ? maeResiduals : mseResiduals;
    return (
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-center mb-1" style={{ color }}>{title}</p>
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
          <svg viewBox={`0 0 ${W2} ${H2}`} className="w-full" style={{ aspectRatio: `${W2}/${H2}` }}>
            <AxisGrid xMin={X_MIN} xMax={X_MAX} yMin={Y_MIN} yMax={Y_MAX} xStep={2} yStep={2} w={W2} h={H2} pad={PAD2} pw={PW2} ph={PH2} />

            {/* Prediction line */}
            <line x1={toSX2(X_MIN)} y1={toSY2(predict(X_MIN))} x2={toSX2(X_MAX)} y2={toSY2(predict(X_MAX))}
              stroke={color} strokeWidth={2} strokeLinecap="round" />

            {/* Residual visualization */}
            {residuals.map((r, i) => {
              const sx = toSX2(r.x);
              const syActual = toSY2(r.y);
              const syPred = toSY2(r.yHat);
              const height = Math.abs(syActual - syPred);
              if (height < 1) return null;
              const isOutlier = outlierEnabled && i === residuals.length - 1;

              if (type === "MAE") {
                // vertical bar for absolute error
                return (
                  <rect key={`bar-${i}`} x={sx - 3} y={Math.min(syActual, syPred)} width={6} height={height}
                    fill={color} opacity={isOutlier ? 0.6 : 0.3} stroke={color} strokeWidth={0.5} rx={1} />
                );
              } else {
                // square for squared error
                const sidePx = Math.min(height, maxBarHeight);
                const topY = Math.min(syActual, syPred);
                return (
                  <rect key={`sq-${i}`} x={sx - sidePx / 2} y={topY} width={sidePx} height={sidePx}
                    fill={color} opacity={isOutlier ? 0.5 : 0.2} stroke={color} strokeWidth={0.5} rx={2} />
                );
              }
            })}

            {/* Residual lines */}
            {residuals.map((r, i) => {
              const err = Math.abs(r.y - r.yHat);
              if (err < 0.05) return null;
              return (
                <line key={`rl-${i}`} x1={toSX2(r.x)} y1={toSY2(r.y)} x2={toSX2(r.x)} y2={toSY2(r.yHat)}
                  stroke={color} strokeWidth={1} strokeDasharray="3,2" opacity={0.6} />
              );
            })}

            {/* Data points */}
            {dataWithOutlier.map((p, i) => {
              const isOutlier = outlierEnabled && i === dataWithOutlier.length - 1;
              return (
                <circle key={`dp-${i}`} cx={toSX2(p.x)} cy={toSY2(p.y)} r={isOutlier ? 6 : 4}
                  fill={isOutlier ? "#f59e0b" : "#10b981"} stroke="#fff" strokeWidth={1.5} />
              );
            })}

            {/* Chart title */}
            <text x={W2 / 2} y={14} fontSize={11} fill={color} textAnchor="middle" fontWeight={700}>
              {type} = {type === "MAE" ? maeValue.toFixed(3) : mseValue.toFixed(3)}
            </text>
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
        <p className="text-xs text-amber-700">
          <strong>MAE vs MSE:</strong> MAE treats all errors linearly while MSE squares them -- making MSE much more sensitive to outliers.
          Toggle the outlier and adjust its magnitude to see how differently the two losses respond.
        </p>
      </div>

      {/* Side-by-side charts */}
      <div className="flex gap-3">
        {renderChart("MAE (Absolute Residuals)", "MAE", "#3b82f6")}
        {renderChart("MSE (Squared Residuals)", "MSE", "#ef4444")}
      </div>

      {/* Loss comparison bar */}
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-3">Loss Comparison</p>
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-blue-500 w-12">MAE</span>
            <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
              <div className="h-full bg-blue-400 rounded-full transition-all duration-300"
                style={{ width: `${clamp((maeValue / Math.max(maeValue, mseValue, 0.01)) * 100, 1, 100)}%` }} />
            </div>
            <span className="text-xs font-bold text-blue-600 w-14 text-right">{maeValue.toFixed(3)}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-red-500 w-12">MSE</span>
            <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
              <div className="h-full bg-red-400 rounded-full transition-all duration-300"
                style={{ width: `${clamp((mseValue / Math.max(maeValue, mseValue, 0.01)) * 100, 1, 100)}%` }} />
            </div>
            <span className="text-xs font-bold text-red-600 w-14 text-right">{mseValue.toFixed(3)}</span>
          </div>
        </div>
        {mseValue > maeValue * 2 && (
          <p className="text-[10px] text-red-600 mt-2 font-medium">MSE is {(mseValue / Math.max(maeValue, 0.001)).toFixed(1)}x larger than MAE -- outliers dominate!</p>
        )}
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-3">
          <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Outlier Control</p>
          <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
            <input type="checkbox" checked={outlierEnabled} onChange={(e) => setOutlierEnabled(e.target.checked)}
              className="accent-amber-500" />
            Enable outlier point
          </label>
          {outlierEnabled && (
            <div>
              <label className="text-[10px] text-slate-500">Outlier magnitude: +{outlierMagnitude.toFixed(1)}</label>
              <input type="range" min={0.5} max={8} step={0.1} value={outlierMagnitude}
                onChange={(e) => setOutlierMagnitude(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none bg-amber-100 accent-amber-500" />
            </div>
          )}
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-3">
          <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Line Parameters</p>
          <div>
            <label className="text-[10px] text-slate-500">Slope: {slope.toFixed(2)}</label>
            <input type="range" min={-1} max={2} step={0.01} value={slope}
              onChange={(e) => setSlope(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none bg-slate-100 accent-indigo-500" />
          </div>
          <div>
            <label className="text-[10px] text-slate-500">Intercept: {intercept.toFixed(2)}</label>
            <input type="range" min={-2} max={6} step={0.01} value={intercept}
              onChange={(e) => setIntercept(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none bg-slate-100 accent-indigo-500" />
          </div>
          <button onClick={handleBestFit}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold bg-emerald-500 text-white hover:bg-emerald-600 transition-all">
            <Target className="w-3.5 h-3.5" /> OLS Best Fit
          </button>
        </div>
      </div>

      {/* Key insight */}
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
        <p className="text-xs text-violet-800 font-semibold mb-1">Key Insight</p>
        <p className="text-xs text-violet-700">
          When an outlier is present, MSE amplifies its contribution because squaring a large error makes it huge (e.g., error=4 gives MSE contribution=16 vs MAE=4).
          This makes MSE-fitted models sensitive to outliers. MAE is more robust but less smooth for optimization.
        </p>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 3 — Loss Landscape
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const WL = 400;
const HL = 400;
const PADL = { top: 30, right: 20, bottom: 45, left: 55 };
const PWL = WL - PADL.left - PADL.right;
const PHL = HL - PADL.top - PADL.bottom;

// Landscape parameter ranges
const SLOPE_MIN = -1;
const SLOPE_MAX = 2.5;
const INT_MIN = -2;
const INT_MAX = 5;

type LossLandscapeType = "MSE" | "MAE" | "Huber";

function Tab5LossLandscape() {
  const landscapeData = useMemo(() => generateDataPoints(5005, 10), []);
  const olsBest = useMemo(() => olsFit(landscapeData), [landscapeData]);

  const [lossType, setLossType] = useState<LossLandscapeType>("MSE");
  const [userSlope, setUserSlope] = useState(0.0);
  const [userIntercept, setUserIntercept] = useState(0.0);
  const [showGradient, setShowGradient] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);
  const [gdPath, setGdPath] = useState<{ slope: number; intercept: number }[]>([]);
  const animRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const GRID_RES = 50;
  const huberDelta = 1.5;

  // Compute loss at a given (slope, intercept)
  const computeLandscapeLoss = useCallback((s: number, b: number, type: LossLandscapeType): number => {
    let total = 0;
    for (const p of landscapeData) {
      const err = p.y - (s * p.x + b);
      switch (type) {
        case "MSE": total += err * err; break;
        case "MAE": total += Math.abs(err); break;
        case "Huber": total += huberLoss(err, huberDelta); break;
      }
    }
    return total / landscapeData.length;
  }, [landscapeData]);

  // Grid of loss values
  const { grid, minLoss, maxLoss } = useMemo(() => {
    const g: number[][] = [];
    let mn = Infinity, mx = -Infinity;
    for (let j = 0; j < GRID_RES; j++) {
      const row: number[] = [];
      const b = INT_MIN + (j / (GRID_RES - 1)) * (INT_MAX - INT_MIN);
      for (let i = 0; i < GRID_RES; i++) {
        const s = SLOPE_MIN + (i / (GRID_RES - 1)) * (SLOPE_MAX - SLOPE_MIN);
        const loss = computeLandscapeLoss(s, b, lossType);
        row.push(loss);
        mn = Math.min(mn, loss);
        mx = Math.max(mx, loss);
      }
      g.push(row);
    }
    return { grid: g, minLoss: mn, maxLoss: mx };
  }, [computeLandscapeLoss, lossType]);

  // Contour levels for contour lines
  const contourLevels = useMemo(() => {
    const levels: number[] = [];
    const range = maxLoss - minLoss;
    for (let i = 1; i <= 10; i++) {
      levels.push(minLoss + (range * i * i) / 100); // Quadratic spacing for nicer contours
    }
    return levels;
  }, [minLoss, maxLoss]);

  const toSXL = (s: number) => PADL.left + ((s - SLOPE_MIN) / (SLOPE_MAX - SLOPE_MIN)) * PWL;
  const toSYL = (b: number) => PADL.top + PHL * (1 - (b - INT_MIN) / (INT_MAX - INT_MIN));
  const fromSXL = (sx: number) => SLOPE_MIN + ((sx - PADL.left) / PWL) * (SLOPE_MAX - SLOPE_MIN);
  const fromSYL = (sy: number) => INT_MIN + (1 - (sy - PADL.top) / PHL) * (INT_MAX - INT_MIN);

  const userLoss = useMemo(() => computeLandscapeLoss(userSlope, userIntercept, lossType), [computeLandscapeLoss, userSlope, userIntercept, lossType]);

  // Gradient computation (numerical)
  const gradient = useMemo(() => {
    const eps = 0.01;
    const dLds = (computeLandscapeLoss(userSlope + eps, userIntercept, lossType) - computeLandscapeLoss(userSlope - eps, userIntercept, lossType)) / (2 * eps);
    const dLdb = (computeLandscapeLoss(userSlope, userIntercept + eps, lossType) - computeLandscapeLoss(userSlope, userIntercept - eps, lossType)) / (2 * eps);
    return { ds: dLds, db: dLdb };
  }, [computeLandscapeLoss, userSlope, userIntercept, lossType]);

  // Handle click on landscape
  const handleLandscapeClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || isAnimating) return;
    const rect = svgRef.current.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * WL;
    const sy = ((e.clientY - rect.top) / rect.height) * HL;
    const s = fromSXL(sx);
    const b = fromSYL(sy);
    if (s >= SLOPE_MIN && s <= SLOPE_MAX && b >= INT_MIN && b <= INT_MAX) {
      setUserSlope(clamp(s, SLOPE_MIN, SLOPE_MAX));
      setUserIntercept(clamp(b, INT_MIN, INT_MAX));
      setGdPath([]);
    }
  }, [isAnimating]);

  // Gradient descent animation
  const startGD = useCallback(() => {
    setIsAnimating(true);
    const lr = 0.05;
    const path: { slope: number; intercept: number }[] = [{ slope: userSlope, intercept: userIntercept }];
    let s = userSlope;
    let b = userIntercept;
    const eps = 0.01;

    let step = 0;
    const maxSteps = 100;

    const doStep = () => {
      if (step >= maxSteps) {
        setIsAnimating(false);
        return;
      }
      const dLds = (computeLandscapeLoss(s + eps, b, lossType) - computeLandscapeLoss(s - eps, b, lossType)) / (2 * eps);
      const dLdb = (computeLandscapeLoss(s, b + eps, lossType) - computeLandscapeLoss(s, b - eps, lossType)) / (2 * eps);

      s = clamp(s - lr * dLds, SLOPE_MIN, SLOPE_MAX);
      b = clamp(b - lr * dLdb, INT_MIN, INT_MAX);
      path.push({ slope: s, intercept: b });
      setGdPath([...path]);
      setUserSlope(s);
      setUserIntercept(b);

      const gradMag = Math.sqrt(dLds * dLds + dLdb * dLdb);
      if (gradMag < 0.001) {
        setIsAnimating(false);
        return;
      }

      step++;
      animRef.current = requestAnimationFrame(() => {
        setTimeout(doStep, 40);
      });
    };

    doStep();
  }, [userSlope, userIntercept, computeLandscapeLoss, lossType]);

  const stopGD = useCallback(() => {
    setIsAnimating(false);
    if (animRef.current) cancelAnimationFrame(animRef.current);
  }, []);

  useEffect(() => {
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  const resetLandscape = () => {
    stopGD();
    setUserSlope(0.0);
    setUserIntercept(0.0);
    setGdPath([]);
  };

  // Cell size for heatmap
  const cellW = PWL / GRID_RES;
  const cellH = PHL / GRID_RES;

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-teal-500 shrink-0 mt-0.5" />
        <p className="text-xs text-teal-700">
          <strong>Loss Landscape</strong> shows how the loss changes as a function of slope and intercept.
          Click anywhere on the heatmap to place your current parameters, see the corresponding line, and run gradient descent to find the minimum.
        </p>
      </div>

      <div className="flex gap-4 flex-col lg:flex-row">
        {/* Heatmap */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${WL} ${HL}`}
              className="w-full select-none cursor-crosshair"
              style={{ aspectRatio: `${WL}/${HL}`, touchAction: "none" }}
              onClick={handleLandscapeClick}
            >
              {/* Heatmap cells */}
              {grid.map((row, j) =>
                row.map((loss, i) => {
                  const t = maxLoss > minLoss ? (loss - minLoss) / (maxLoss - minLoss) : 0;
                  return (
                    <rect
                      key={`cell-${j}-${i}`}
                      x={PADL.left + i * cellW}
                      y={PADL.top + (GRID_RES - 1 - j) * cellH}
                      width={cellW + 0.5}
                      height={cellH + 0.5}
                      fill={heatColor(t)}
                    />
                  );
                })
              )}

              {/* Contour lines (simple marching squares approximation) */}
              {contourLevels.map((level, li) => {
                const segments: string[] = [];
                for (let j = 0; j < GRID_RES - 1; j++) {
                  for (let i = 0; i < GRID_RES - 1; i++) {
                    const v00 = grid[j][i];
                    const v10 = grid[j][i + 1];
                    const v01 = grid[j + 1][i];
                    const v11 = grid[j + 1][i + 1];
                    const crossings: { x: number; y: number }[] = [];

                    // Check each edge
                    if ((v00 - level) * (v10 - level) < 0) {
                      const frac = (level - v00) / (v10 - v00);
                      crossings.push({ x: PADL.left + (i + frac) * cellW, y: PADL.top + (GRID_RES - 1 - j) * cellH });
                    }
                    if ((v10 - level) * (v11 - level) < 0) {
                      const frac = (level - v10) / (v11 - v10);
                      crossings.push({ x: PADL.left + (i + 1) * cellW, y: PADL.top + (GRID_RES - 1 - j - frac) * cellH });
                    }
                    if ((v01 - level) * (v11 - level) < 0) {
                      const frac = (level - v01) / (v11 - v01);
                      crossings.push({ x: PADL.left + (i + frac) * cellW, y: PADL.top + (GRID_RES - 2 - j) * cellH });
                    }
                    if ((v00 - level) * (v01 - level) < 0) {
                      const frac = (level - v00) / (v01 - v00);
                      crossings.push({ x: PADL.left + i * cellW, y: PADL.top + (GRID_RES - 1 - j - frac) * cellH });
                    }
                    if (crossings.length >= 2) {
                      segments.push(`M ${crossings[0].x.toFixed(1)} ${crossings[0].y.toFixed(1)} L ${crossings[1].x.toFixed(1)} ${crossings[1].y.toFixed(1)}`);
                    }
                  }
                }
                if (segments.length === 0) return null;
                return (
                  <path key={`contour-${li}`} d={segments.join(" ")} fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={0.5} />
                );
              })}

              {/* Gradient descent path */}
              {gdPath.length > 1 && (
                <path
                  d={gdPath.map((p, i) => `${i === 0 ? "M" : "L"} ${toSXL(p.slope).toFixed(1)} ${toSYL(p.intercept).toFixed(1)}`).join(" ")}
                  fill="none" stroke="#fff" strokeWidth={2} opacity={0.8} strokeLinejoin="round"
                />
              )}
              {gdPath.map((p, i) => (
                <circle key={`gd-${i}`} cx={toSXL(p.slope)} cy={toSYL(p.intercept)} r={i === gdPath.length - 1 ? 4 : 1.5}
                  fill="#fff" opacity={i === gdPath.length - 1 ? 1 : 0.6} />
              ))}

              {/* Optimal point (star) */}
              <g transform={`translate(${toSXL(olsBest.slope)}, ${toSYL(olsBest.intercept)})`}>
                <polygon points="0,-8 2,-3 7,-3 3,1 5,6 0,3 -5,6 -3,1 -7,-3 -2,-3" fill="#fbbf24" stroke="#fff" strokeWidth={1} />
              </g>

              {/* User point */}
              <circle cx={toSXL(userSlope)} cy={toSYL(userIntercept)} r={7} fill="#fff" stroke="#000" strokeWidth={2} />

              {/* Gradient arrow */}
              {showGradient && (
                (() => {
                  const mag = Math.sqrt(gradient.ds * gradient.ds + gradient.db * gradient.db);
                  if (mag < 0.01) return null;
                  const scale = Math.min(30, 30 * mag / 5);
                  const nx = -gradient.ds / mag;
                  const ny = gradient.db / mag; // Flip because SVG y is inverted
                  const startX = toSXL(userSlope);
                  const startY = toSYL(userIntercept);
                  const endX = startX + nx * scale;
                  const endY = startY + ny * scale;
                  return (
                    <g>
                      <line x1={startX} y1={startY} x2={endX} y2={endY} stroke="#00ff88" strokeWidth={2.5} />
                      {/* Arrowhead */}
                      <circle cx={endX} cy={endY} r={3} fill="#00ff88" />
                    </g>
                  );
                })()
              )}

              {/* Axis labels */}
              {[SLOPE_MIN, 0, 0.5, 1, 1.5, 2, SLOPE_MAX].map((v) => (
                <text key={`xla-${v}`} x={toSXL(v)} y={HL - PADL.bottom + 18} fontSize={9} fill="#94a3b8" textAnchor="middle">{v}</text>
              ))}
              {[INT_MIN, 0, 1, 2, 3, 4, INT_MAX].map((v) => (
                <text key={`yla-${v}`} x={PADL.left - 8} y={toSYL(v) + 3} fontSize={9} fill="#94a3b8" textAnchor="end">{v}</text>
              ))}

              <text x={WL / 2} y={HL - 4} fontSize={10} fill="#64748b" textAnchor="middle" fontWeight={600}>Slope</text>
              <text x={12} y={PADL.top + PHL / 2} fontSize={10} fill="#64748b" textAnchor="middle" fontWeight={600}
                transform={`rotate(-90, 12, ${PADL.top + PHL / 2})`}>Intercept</text>

              {/* Title */}
              <text x={WL / 2} y={16} fontSize={11} fill="#fff" textAnchor="middle" fontWeight={700}>
                {lossType} Loss Landscape
              </text>
            </svg>
          </div>

          {/* Mini scatter plot showing the current line */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 mt-3">
            <p className="text-xs font-semibold text-slate-600 mb-1 text-center">
              Current Line: y = {userSlope.toFixed(2)}x + {userIntercept.toFixed(2)} (Loss: {userLoss.toFixed(3)})
            </p>
            <svg viewBox={`0 0 ${W} 200`} className="w-full" style={{ aspectRatio: `${W}/200` }}>
              <rect x={PAD.left} y={15} width={PW} height={160} fill="white" stroke="#e2e8f0" strokeWidth={1} />
              {/* Simple grid */}
              {[0, 2, 4, 6, 8, 10].map((v) => (
                <line key={`xg-${v}`} x1={toSX(v)} y1={15} x2={toSX(v)} y2={175} stroke="#f1f5f9" strokeWidth={1} />
              ))}
              {/* Prediction line */}
              <line
                x1={toSX(X_MIN)} y1={15 + 160 * (1 - (userSlope * X_MIN + userIntercept - Y_MIN) / (Y_MAX - Y_MIN))}
                x2={toSX(X_MAX)} y2={15 + 160 * (1 - (userSlope * X_MAX + userIntercept - Y_MIN) / (Y_MAX - Y_MIN))}
                stroke="#6366f1" strokeWidth={2}
              />
              {/* Data points */}
              {landscapeData.map((p, i) => {
                const sy = 15 + 160 * (1 - (p.y - Y_MIN) / (Y_MAX - Y_MIN));
                return <circle key={`ldp-${i}`} cx={toSX(p.x)} cy={sy} r={4} fill="#10b981" stroke="#fff" strokeWidth={1} />;
              })}
              {/* Residual lines */}
              {landscapeData.map((p, i) => {
                const syActual = 15 + 160 * (1 - (p.y - Y_MIN) / (Y_MAX - Y_MIN));
                const yHat = userSlope * p.x + userIntercept;
                const syPred = 15 + 160 * (1 - (yHat - Y_MIN) / (Y_MAX - Y_MIN));
                return (
                  <line key={`lrl-${i}`} x1={toSX(p.x)} y1={syActual} x2={toSX(p.x)} y2={syPred}
                    stroke="#ef4444" strokeWidth={1} strokeDasharray="3,2" opacity={0.5} />
                );
              })}
            </svg>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-64 space-y-3">
          {/* Loss type selector */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Loss Function</p>
            <div className="flex gap-1.5">
              {(["MSE", "MAE", "Huber"] as LossLandscapeType[]).map((t) => (
                <button key={t} onClick={() => { setLossType(t); setGdPath([]); }}
                  className={`flex-1 px-2 py-1.5 rounded-md text-xs font-semibold transition-all ${lossType === t
                    ? "bg-teal-500 text-white shadow-sm"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}>
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Current position */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Current Position</p>
            <div className="space-y-2">
              <div>
                <label className="text-[10px] text-slate-500">Slope: {userSlope.toFixed(3)}</label>
                <input type="range" min={SLOPE_MIN} max={SLOPE_MAX} step={0.01} value={userSlope}
                  onChange={(e) => { setUserSlope(parseFloat(e.target.value)); setGdPath([]); }}
                  className="w-full h-1.5 rounded-full appearance-none bg-teal-100 accent-teal-500" />
              </div>
              <div>
                <label className="text-[10px] text-slate-500">Intercept: {userIntercept.toFixed(3)}</label>
                <input type="range" min={INT_MIN} max={INT_MAX} step={0.01} value={userIntercept}
                  onChange={(e) => { setUserIntercept(parseFloat(e.target.value)); setGdPath([]); }}
                  className="w-full h-1.5 rounded-full appearance-none bg-teal-100 accent-teal-500" />
              </div>
            </div>
          </div>

          {/* Loss readout */}
          <div className="bg-white border border-slate-200 rounded-lg p-4 text-center">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">{lossType} at Current Point</p>
            <p className="text-2xl font-bold text-teal-600 mt-1">{userLoss.toFixed(4)}</p>
            <p className="text-[10px] text-slate-400 mt-1">Optimal: {computeLandscapeLoss(olsBest.slope, olsBest.intercept, lossType).toFixed(4)}</p>
          </div>

          {/* Gradient info */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Gradient</p>
              <label className="flex items-center gap-1 text-[10px] text-slate-500 cursor-pointer">
                <input type="checkbox" checked={showGradient} onChange={() => setShowGradient(!showGradient)} className="accent-teal-500" />
                Show arrow
              </label>
            </div>
            <div className="font-mono text-[10px] text-slate-600 space-y-0.5">
              <p>dL/d(slope) = {gradient.ds.toFixed(4)}</p>
              <p>dL/d(intercept) = {gradient.db.toFixed(4)}</p>
              <p>|grad| = {Math.sqrt(gradient.ds ** 2 + gradient.db ** 2).toFixed(4)}</p>
            </div>
          </div>

          {/* GD buttons */}
          <div className="flex gap-2">
            {!isAnimating ? (
              <button onClick={startGD}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-xs font-semibold bg-teal-500 text-white hover:bg-teal-600 transition-all">
                <Play className="w-3.5 h-3.5" /> Run Gradient Descent
              </button>
            ) : (
              <button onClick={stopGD}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-xs font-semibold bg-red-500 text-white hover:bg-red-600 transition-all">
                <Pause className="w-3.5 h-3.5" /> Stop
              </button>
            )}
          </div>

          <button onClick={resetLandscape}
            className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200 transition-all border border-slate-200">
            <RotateCcw className="w-3.5 h-3.5" /> Reset
          </button>

          {/* Legend */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Legend</p>
            <div className="space-y-1.5 text-xs text-slate-600">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-white border-2 border-black inline-block" /> Your position (click to move)
              </div>
              <div className="flex items-center gap-2">
                <span className="text-yellow-400 text-sm">&#9733;</span> Optimal point (OLS minimum)
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-green-400 inline-block" /> Gradient direction (steepest descent)
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-white inline-block border border-slate-300" /> Gradient descent path
              </div>
            </div>
          </div>

          {gdPath.length > 1 && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
              <p className="text-[10px] text-emerald-700 font-medium">
                Gradient descent completed {gdPath.length - 1} steps.
                Final loss: {userLoss.toFixed(4)}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main Component
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export default function LossFunctionsActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("mse");

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-100 rounded-xl p-1 overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
              activeTab === tab.key
                ? "bg-white text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "mse" && <Tab1MSEExplorer />}
      {activeTab === "comparison" && <Tab2MAEMSEComparison />}
      {activeTab === "landscape" && <Tab5LossLandscape />}
    </div>
  );
}
