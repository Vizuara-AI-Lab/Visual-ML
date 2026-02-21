/**
 * Overfitting Activity — comprehensive 3-tab interactive explorer
 *
 * Tab 1: Train vs Validation Loss — polynomial fitting with degree slider
 * Tab 2: Early Stopping — animated training with patience parameter
 * Tab 3: Dropout Effect — neural network visualization with dropout masks
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  TrendingUp,
  Clock,
  Zap,
  Play,
  Pause,
  RotateCcw,
  RefreshCw,
  Info,
  AlertTriangle,
} from "lucide-react";

// ══════════════════════════════════════════════════════════════════════════════
// SEEDED PRNG (mulberry32)
// ══════════════════════════════════════════════════════════════════════════════
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededGaussian(rng: () => number): number {
  const u1 = rng() + 1e-10;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ══════════════════════════════════════════════════════════════════════════════
// TYPES
// ══════════════════════════════════════════════════════════════════════════════
type TabId = "trainval" | "earlystop" | "dropout";

interface TabDef {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabDef[] = [
  { id: "trainval", label: "Train vs Val Loss", icon: <TrendingUp className="w-4 h-4" /> },
  { id: "earlystop", label: "Early Stopping", icon: <Clock className="w-4 h-4" /> },
  { id: "dropout", label: "Dropout Effect", icon: <Zap className="w-4 h-4" /> },
];

// ══════════════════════════════════════════════════════════════════════════════
// MATRIX MATH HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function transpose(m: number[][]): number[][] {
  const rows = m.length, cols = m[0].length;
  const r: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) r[j][i] = m[i][j];
  return r;
}

function matMul(a: number[][], b: number[][]): number[][] {
  const rows = a.length, cols = b[0].length, inner = b.length;
  const r: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      let s = 0;
      for (let k = 0; k < inner; k++) s += a[i][k] * b[k][j];
      r[i][j] = s;
    }
  return r;
}

function invertMatrix(m: number[][]): number[][] | null {
  const n = m.length;
  const aug = m.map((row, i) => {
    const ext = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) ext[j] = row[j];
    ext[n + i] = 1;
    return ext;
  });
  for (let col = 0; col < n; col++) {
    let maxRow = col, maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(aug[row][col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxVal < 1e-12) return null;
    if (maxRow !== col) [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

// ══════════════════════════════════════════════════════════════════════════════
// POLYNOMIAL HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function vandermonde(xs: number[], degree: number): number[][] {
  return xs.map((x) => {
    const row = new Array(degree + 1);
    row[0] = 1;
    for (let j = 1; j <= degree; j++) row[j] = row[j - 1] * x;
    return row;
  });
}

function fitPolynomial(xs: number[], ys: number[], degree: number, lambda = 0): number[] | null {
  const X = vandermonde(xs, degree);
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  for (let i = 0; i < XtX.length; i++) XtX[i][i] += lambda + 1e-10;
  const XtXinv = invertMatrix(XtX);
  if (!XtXinv) return null;
  const yCol = ys.map((y) => [y]);
  const XtY = matMul(Xt, yCol);
  const coeffs = matMul(XtXinv, XtY);
  return coeffs.map((row) => row[0]);
}

function evalPoly(coeffs: number[], x: number): number {
  let r = 0, p = 1;
  for (let i = 0; i < coeffs.length; i++) { r += coeffs[i] * p; p *= x; }
  return r;
}

function computeMSE(xs: number[], ys: number[], coeffs: number[]): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) { const d = evalPoly(coeffs, xs[i]) - ys[i]; s += d * d; }
  return s / xs.length;
}

// True function: sin(2*pi*x) * x + 0.5
function trueFunction(x: number): number {
  return Math.sin(2 * Math.PI * x) * x + 0.5;
}

// ══════════════════════════════════════════════════════════════════════════════
// DATA GENERATION
// ══════════════════════════════════════════════════════════════════════════════
function generateData(nTrain: number, noise: number, seed = 42) {
  const rng = mulberry32(seed);
  const nVal = Math.max(20, Math.floor(nTrain * 0.5));
  const trainX: number[] = [], trainY: number[] = [];
  const valX: number[] = [], valY: number[] = [];
  for (let i = 0; i < nTrain; i++) {
    const x = rng();
    trainX.push(x);
    trainY.push(trueFunction(x) + seededGaussian(rng) * noise);
  }
  for (let i = 0; i < nVal; i++) {
    const x = rng();
    valX.push(x);
    valY.push(trueFunction(x) + seededGaussian(rng) * noise);
  }
  return { trainX, trainY, valX, valY };
}

// ══════════════════════════════════════════════════════════════════════════════
// SVG CHART HELPERS
// ══════════════════════════════════════════════════════════════════════════════
const CW = 520, CH = 300;
const PAD = { top: 25, right: 20, bottom: 40, left: 50 };
const PW = CW - PAD.left - PAD.right;
const PH = CH - PAD.top - PAD.bottom;

function sx(val: number, minV: number, maxV: number): number {
  return PAD.left + ((val - minV) / (maxV - minV + 1e-10)) * PW;
}
function sy(val: number, minV: number, maxV: number): number {
  return PAD.top + PH * (1 - (val - minV) / (maxV - minV + 1e-10));
}

function buildPath(pts: { x: number; y: number }[], xMin: number, xMax: number, yMin: number, yMax: number): string {
  return pts.map((p, i) =>
    `${i === 0 ? "M" : "L"} ${sx(p.x, xMin, xMax).toFixed(1)} ${sy(p.y, yMin, yMax).toFixed(1)}`
  ).join(" ");
}

/** Grid lines and axes for a standard chart */
function ChartGrid({ xMin, xMax, yMin, yMax, xLabel, yLabel, xTicks, yTicks }: {
  xMin: number; xMax: number; yMin: number; yMax: number;
  xLabel: string; yLabel: string; xTicks: number[]; yTicks: number[];
}) {
  return (
    <g>
      <rect x={PAD.left} y={PAD.top} width={PW} height={PH} fill="white" stroke="#e2e8f0" />
      {yTicks.map((t, i) => {
        const yy = sy(t, yMin, yMax);
        return (
          <g key={`yt${i}`}>
            <line x1={PAD.left} y1={yy} x2={PAD.left + PW} y2={yy} stroke="#f1f5f9" />
            <text x={PAD.left - 6} y={yy + 3} fontSize={9} fill="#94a3b8" textAnchor="end">
              {t < 0.01 && t > 0 ? t.toExponential(0) : t % 1 === 0 ? t : t.toFixed(2)}
            </text>
          </g>
        );
      })}
      {xTicks.map((t, i) => (
        <text key={`xt${i}`} x={sx(t, xMin, xMax)} y={CH - PAD.bottom + 18} fontSize={9} fill="#94a3b8" textAnchor="middle">
          {t % 1 === 0 ? t : t.toFixed(2)}
        </text>
      ))}
      <text x={PAD.left + PW / 2} y={CH - 4} fontSize={10} fill="#64748b" textAnchor="middle" fontWeight={600}>{xLabel}</text>
      <text x={12} y={PAD.top + PH / 2} fontSize={10} fill="#64748b" textAnchor="middle" fontWeight={600}
        transform={`rotate(-90, 12, ${PAD.top + PH / 2})`}>{yLabel}</text>
    </g>
  );
}

function ChartLegend({ items, x: lx, y: ly }: { items: { color: string; label: string; dashed?: boolean }[]; x: number; y: number }) {
  const h = items.length * 16 + 8;
  return (
    <g transform={`translate(${lx}, ${ly})`}>
      <rect x={0} y={0} width={140} height={h} rx={4} fill="white" fillOpacity={0.92} stroke="#e2e8f0" strokeWidth={0.5} />
      {items.map((item, i) => (
        <g key={i}>
          <line x1={8} y1={12 + i * 16} x2={26} y2={12 + i * 16} stroke={item.color} strokeWidth={2}
            strokeDasharray={item.dashed ? "5,3" : undefined} />
          <text x={32} y={15 + i * 16} fontSize={9} fill="#64748b">{item.label}</text>
        </g>
      ))}
    </g>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 1: TRAIN VS VALIDATION LOSS
// ══════════════════════════════════════════════════════════════════════════════
function TrainValTab() {
  const [degree, setDegree] = useState(5);
  const [noise, setNoise] = useState(0.35);
  const [nPoints, setNPoints] = useState(30);

  const data = useMemo(() => generateData(nPoints, noise, 42), [nPoints, noise]);

  // Compute train/val loss for each degree 1..15
  const degreeLosses = useMemo(() => {
    const results: { deg: number; trainLoss: number; valLoss: number }[] = [];
    for (let d = 1; d <= 15; d++) {
      const c = fitPolynomial(data.trainX, data.trainY, d, 1e-8);
      if (!c) { results.push({ deg: d, trainLoss: 1, valLoss: 1 }); continue; }
      results.push({ deg: d, trainLoss: computeMSE(data.trainX, data.trainY, c), valLoss: computeMSE(data.valX, data.valY, c) });
    }
    return results;
  }, [data]);

  // Current fit
  const currentCoeffs = useMemo(() => fitPolynomial(data.trainX, data.trainY, degree, 1e-8), [data, degree]);

  // Fit curve points
  const fitCurve = useMemo(() => {
    if (!currentCoeffs) return [];
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= 200; i++) { const x = i / 200; pts.push({ x, y: evalPoly(currentCoeffs, x) }); }
    return pts;
  }, [currentCoeffs]);

  // True curve
  const trueCurve = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= 200; i++) { const x = i / 200; pts.push({ x, y: trueFunction(x) }); }
    return pts;
  }, []);

  // Y range for scatter
  const yRange = useMemo(() => {
    const allY = [...data.trainY, ...data.valY];
    const mn = Math.min(...allY, -0.5), mx = Math.max(...allY, 2);
    const margin = (mx - mn) * 0.1;
    return { min: mn - margin, max: mx + margin };
  }, [data]);

  // Loss chart ranges
  const lossMax = useMemo(() => {
    let mx = 0;
    for (const r of degreeLosses) mx = Math.max(mx, r.trainLoss, r.valLoss);
    return Math.min(mx * 1.15, 2);
  }, [degreeLosses]);

  // Find optimal degree (min val loss)
  const optimalDeg = useMemo(() => {
    let best = 1, bestVal = Infinity;
    for (const r of degreeLosses) if (r.valLoss < bestVal) { bestVal = r.valLoss; best = r.deg; }
    return best;
  }, [degreeLosses]);

  const currentLoss = degreeLosses[degree - 1];

  // Classification of current degree
  const fitLabel = degree <= Math.max(1, optimalDeg - 2) ? "Underfitting" : degree <= optimalDeg + 1 ? "Just Right" : "Overfitting";
  const fitColor = fitLabel === "Underfitting" ? "text-blue-600" : fitLabel === "Just Right" ? "text-green-600" : "text-red-600";
  const fitBg = fitLabel === "Underfitting" ? "bg-blue-50 border-blue-200" : fitLabel === "Just Right" ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200";

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
        <p className="text-xs text-orange-700">
          Adjust the polynomial degree to see how model complexity affects training and validation loss.
          Low degree = underfitting (both losses high). High degree = overfitting (train loss drops but val loss rises).
          The <strong>sweet spot</strong> minimizes validation loss.
        </p>
      </div>

      <div className="flex flex-col xl:flex-row gap-4">
        {/* Left: Charts */}
        <div className="flex-1 min-w-0 space-y-4">
          {/* Scatter plot with fit */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <p className="text-xs font-semibold text-slate-600 mb-2">Model Fit (Degree {degree}) - {fitLabel}</p>
            <svg viewBox={`0 0 ${CW} ${CH}`} className="w-full" style={{ maxWidth: 560 }}>
              <defs><clipPath id="t1fitClip"><rect x={PAD.left} y={PAD.top} width={PW} height={PH} /></clipPath></defs>
              <ChartGrid xMin={0} xMax={1} yMin={yRange.min} yMax={yRange.max} xLabel="x" yLabel="y"
                xTicks={[0, 0.25, 0.5, 0.75, 1]} yTicks={Array.from({ length: 5 }, (_, i) => yRange.min + (yRange.max - yRange.min) * i / 4)} />

              {/* True function */}
              <path d={buildPath(trueCurve, 0, 1, yRange.min, yRange.max)} fill="none" stroke="#22c55e" strokeWidth={1.5}
                strokeDasharray="5,3" opacity={0.7} clipPath="url(#t1fitClip)" />

              {/* Fit curve */}
              {fitCurve.length > 0 && (
                <path d={buildPath(fitCurve.map(p => ({ x: p.x, y: Math.max(yRange.min, Math.min(yRange.max, p.y)) })), 0, 1, yRange.min, yRange.max)}
                  fill="none" stroke="#ef4444" strokeWidth={2} clipPath="url(#t1fitClip)" />
              )}

              {/* Val points */}
              {data.valX.map((x, i) => {
                const py = sy(data.valY[i], yRange.min, yRange.max);
                if (py < PAD.top || py > PAD.top + PH) return null;
                return <circle key={`v${i}`} cx={sx(x, 0, 1)} cy={py} r={2.5} fill="#f97316" opacity={0.6} />;
              })}
              {/* Train points */}
              {data.trainX.map((x, i) => {
                const py = sy(data.trainY[i], yRange.min, yRange.max);
                if (py < PAD.top || py > PAD.top + PH) return null;
                return <circle key={`t${i}`} cx={sx(x, 0, 1)} cy={py} r={3} fill="#3b82f6" opacity={0.8} />;
              })}

              <ChartLegend x={PAD.left + 4} y={PAD.top + 4} items={[
                { color: "#3b82f6", label: "Training data" },
                { color: "#f97316", label: "Validation data" },
                { color: "#22c55e", label: "True function", dashed: true },
                { color: "#ef4444", label: "Model fit" },
              ]} />
            </svg>
          </div>

          {/* Loss vs Degree */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <p className="text-xs font-semibold text-slate-600 mb-2">Loss vs Polynomial Degree</p>
            <svg viewBox={`0 0 ${CW} ${CH}`} className="w-full" style={{ maxWidth: 560 }}>
              <defs><clipPath id="t1lossClip"><rect x={PAD.left} y={PAD.top} width={PW} height={PH} /></clipPath></defs>
              <ChartGrid xMin={1} xMax={15} yMin={0} yMax={lossMax} xLabel="Polynomial Degree" yLabel="Loss (MSE)"
                xTicks={[1, 3, 5, 7, 9, 11, 13, 15]}
                yTicks={Array.from({ length: 5 }, (_, i) => lossMax * i / 4)} />

              {/* Shaded regions */}
              {optimalDeg > 2 && (
                <rect x={sx(1, 1, 15)} y={PAD.top} width={sx(optimalDeg - 1, 1, 15) - sx(1, 1, 15)} height={PH}
                  fill="#3b82f6" opacity={0.05} />
              )}
              <rect x={sx(Math.max(1, optimalDeg - 1), 1, 15)} y={PAD.top}
                width={sx(optimalDeg + 1, 1, 15) - sx(Math.max(1, optimalDeg - 1), 1, 15)} height={PH}
                fill="#22c55e" opacity={0.08} />
              {optimalDeg + 2 <= 15 && (
                <rect x={sx(optimalDeg + 2, 1, 15)} y={PAD.top} width={sx(15, 1, 15) - sx(optimalDeg + 2, 1, 15)} height={PH}
                  fill="#ef4444" opacity={0.05} />
              )}

              {/* Region labels at top */}
              {optimalDeg > 3 && (
                <text x={sx((1 + optimalDeg - 1) / 2, 1, 15)} y={PAD.top + 14} fontSize={9} fill="#3b82f6"
                  textAnchor="middle" fontWeight={600}>Underfitting</text>
              )}
              <text x={sx(optimalDeg, 1, 15)} y={PAD.top + 14} fontSize={9} fill="#16a34a"
                textAnchor="middle" fontWeight={600}>Sweet Spot</text>
              {optimalDeg + 3 <= 15 && (
                <text x={sx((optimalDeg + 2 + 15) / 2, 1, 15)} y={PAD.top + 14} fontSize={9} fill="#ef4444"
                  textAnchor="middle" fontWeight={600}>Overfitting</text>
              )}

              {/* Train loss line */}
              <path d={degreeLosses.map((r, i) =>
                `${i === 0 ? "M" : "L"} ${sx(r.deg, 1, 15).toFixed(1)} ${sy(Math.min(r.trainLoss, lossMax), 0, lossMax).toFixed(1)}`
              ).join(" ")} fill="none" stroke="#3b82f6" strokeWidth={2.5} clipPath="url(#t1lossClip)" />

              {/* Val loss line */}
              <path d={degreeLosses.map((r, i) =>
                `${i === 0 ? "M" : "L"} ${sx(r.deg, 1, 15).toFixed(1)} ${sy(Math.min(r.valLoss, lossMax), 0, lossMax).toFixed(1)}`
              ).join(" ")} fill="none" stroke="#f97316" strokeWidth={2.5} clipPath="url(#t1lossClip)" />

              {/* Dots for each degree */}
              {degreeLosses.map((r) => (
                <g key={r.deg}>
                  <circle cx={sx(r.deg, 1, 15)} cy={sy(Math.min(r.trainLoss, lossMax), 0, lossMax)} r={r.deg === degree ? 5 : 3}
                    fill="#3b82f6" stroke={r.deg === degree ? "#1d4ed8" : "#fff"} strokeWidth={r.deg === degree ? 2 : 1} />
                  <circle cx={sx(r.deg, 1, 15)} cy={sy(Math.min(r.valLoss, lossMax), 0, lossMax)} r={r.deg === degree ? 5 : 3}
                    fill="#f97316" stroke={r.deg === degree ? "#c2410c" : "#fff"} strokeWidth={r.deg === degree ? 2 : 1} />
                </g>
              ))}

              {/* Current degree vertical line */}
              <line x1={sx(degree, 1, 15)} y1={PAD.top} x2={sx(degree, 1, 15)} y2={PAD.top + PH}
                stroke="#6366f1" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.6} />

              {/* Optimal degree marker */}
              <line x1={sx(optimalDeg, 1, 15)} y1={PAD.top + PH - 4} x2={sx(optimalDeg, 1, 15)} y2={PAD.top + PH + 2}
                stroke="#16a34a" strokeWidth={2} />
              <text x={sx(optimalDeg, 1, 15)} y={PAD.top + PH + 12} fontSize={8} fill="#16a34a" textAnchor="middle" fontWeight={600}>
                optimal
              </text>

              <ChartLegend x={PAD.left + PW - 145} y={PAD.top + 22} items={[
                { color: "#3b82f6", label: "Training Loss" },
                { color: "#f97316", label: "Validation Loss" },
              ]} />
            </svg>
          </div>
        </div>

        {/* Right: Controls */}
        <div className="w-full xl:w-64 space-y-3">
          {/* Status badge */}
          <div className={`${fitBg} border rounded-lg p-3 text-center`}>
            <p className="text-[10px] uppercase tracking-wide text-slate-500 font-medium">Status</p>
            <p className={`text-base font-bold ${fitColor} mt-1`}>{fitLabel}</p>
          </div>

          {/* Loss metrics */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-blue-500 uppercase font-medium">Train Loss</p>
              <p className="text-sm font-bold text-blue-600 mt-1">{currentLoss ? currentLoss.trainLoss.toFixed(4) : "---"}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-orange-500 uppercase font-medium">Val Loss</p>
              <p className="text-sm font-bold text-orange-600 mt-1">{currentLoss ? currentLoss.valLoss.toFixed(4) : "---"}</p>
            </div>
          </div>

          {/* Degree slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700">Polynomial Degree</label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">{degree}</span>
            </div>
            <input type="range" min={1} max={15} step={1} value={degree}
              onChange={(e) => setDegree(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>1 (linear)</span><span>15 (complex)</span>
            </div>
          </div>

          {/* Noise slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700">Noise Level</label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">{noise.toFixed(2)}</span>
            </div>
            <input type="range" min={0.05} max={1} step={0.05} value={noise}
              onChange={(e) => setNoise(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>0.05 (clean)</span><span>1.0 (noisy)</span>
            </div>
          </div>

          {/* Points slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700">Training Points</label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">{nPoints}</span>
            </div>
            <input type="range" min={10} max={100} step={5} value={nPoints}
              onChange={(e) => setNPoints(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>10 (few)</span><span>100 (many)</span>
            </div>
          </div>

          {/* Optimal degree info */}
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
            <p className="text-[10px] text-emerald-500 uppercase font-medium">Optimal Degree</p>
            <p className="text-lg font-bold text-emerald-700 mt-1">{optimalDeg}</p>
            <p className="text-[10px] text-emerald-600 mt-0.5">Lowest validation loss: {degreeLosses[optimalDeg - 1]?.valLoss.toFixed(4)}</p>
          </div>

          {/* Tips */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Experiments:</p>
            <ul className="text-[10px] text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Degree 1: straight line underfits curved data</li>
              <li>Degree 3-5: captures the shape well</li>
              <li>Degree 12+: wiggles through every point</li>
              <li>More noise makes overfitting worse</li>
              <li>More data points reduce overfitting</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 2: EARLY STOPPING
// ══════════════════════════════════════════════════════════════════════════════
const ES_EPOCHS = 200;

function EarlyStoppingTab() {
  const [patience, setPatience] = useState(15);
  const [degree, setDegree] = useState(9);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const animRef = useRef<number | null>(null);
  const epochRef = useRef(0);
  const lastTimeRef = useRef(0);
  epochRef.current = currentEpoch;

  const data = useMemo(() => generateData(30, 0.4, 42), []);

  // Pre-compute all epochs
  const epochResults = useMemo(() => {
    const results: { epoch: number; trainLoss: number; valLoss: number; coeffs: number[] }[] = [];
    const lambdaStart = 100, lambdaEnd = 1e-12;
    for (let ep = 0; ep <= ES_EPOCHS; ep++) {
      const t = ep / ES_EPOCHS;
      const lambda = lambdaStart * Math.pow(lambdaEnd / lambdaStart, t);
      const c = fitPolynomial(data.trainX, data.trainY, degree, lambda);
      if (!c) {
        const mean = data.trainY.reduce((a, b) => a + b, 0) / data.trainY.length;
        results.push({ epoch: ep, trainLoss: computeMSE(data.trainX, data.trainY, [mean]),
          valLoss: computeMSE(data.valX, data.valY, [mean]), coeffs: [mean] });
        continue;
      }
      results.push({ epoch: ep, trainLoss: computeMSE(data.trainX, data.trainY, c),
        valLoss: computeMSE(data.valX, data.valY, c), coeffs: c });
    }
    return results;
  }, [data, degree]);

  // Find best epoch (min val loss)
  const bestEpoch = useMemo(() => {
    let best = 0, bestVal = Infinity;
    for (const r of epochResults) if (r.valLoss < bestVal) { bestVal = r.valLoss; best = r.epoch; }
    return best;
  }, [epochResults]);

  // Early stop epoch based on patience
  const earlyStopEpoch = useMemo(() => {
    let bestVal = Infinity, bestEp = 0, wait = 0;
    for (const r of epochResults) {
      if (r.valLoss < bestVal - 1e-6) { bestVal = r.valLoss; bestEp = r.epoch; wait = 0; }
      else { wait++; }
      if (wait >= patience) return r.epoch;
    }
    return ES_EPOCHS;
  }, [epochResults, patience]);

  // Saved model epoch (the best before early stop)
  const savedModelEpoch = useMemo(() => {
    let bestVal = Infinity, bestEp = 0;
    for (const r of epochResults) {
      if (r.epoch > earlyStopEpoch) break;
      if (r.valLoss < bestVal) { bestVal = r.valLoss; bestEp = r.epoch; }
    }
    return bestEp;
  }, [epochResults, earlyStopEpoch]);

  const lossMax = useMemo(() => {
    let mx = 0;
    for (const r of epochResults) mx = Math.max(mx, r.trainLoss, r.valLoss);
    return Math.ceil(mx * 1.2 * 10) / 10;
  }, [epochResults]);

  const currentResult = epochResults[Math.min(currentEpoch, ES_EPOCHS)];
  const epochsSaved = ES_EPOCHS - earlyStopEpoch;
  const bestValLoss = epochResults[savedModelEpoch]?.valLoss ?? 0;
  const finalValLoss = epochResults[ES_EPOCHS]?.valLoss ?? 0;

  // Animation
  const tick = useCallback(() => {
    const now = performance.now();
    if (now - lastTimeRef.current > 25) {
      lastTimeRef.current = now;
      const next = epochRef.current + 1;
      if (next > ES_EPOCHS) { setIsRunning(false); return; }
      setCurrentEpoch(next);
    }
    animRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => {
    if (!isRunning) { if (animRef.current) { cancelAnimationFrame(animRef.current); animRef.current = null; } return; }
    lastTimeRef.current = performance.now();
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isRunning, tick]);

  const handleTrain = useCallback(() => {
    if (currentEpoch >= ES_EPOCHS) { setCurrentEpoch(0); epochRef.current = 0; }
    setIsRunning(true);
  }, [currentEpoch]);

  const handleReset = useCallback(() => { setIsRunning(false); setCurrentEpoch(0); epochRef.current = 0; }, []);

  // Build paths up to currentEpoch
  const trainPath = useMemo(() => {
    if (currentEpoch === 0) return "";
    return epochResults.slice(0, currentEpoch + 1).map((r, i) =>
      `${i === 0 ? "M" : "L"} ${sx(r.epoch, 0, ES_EPOCHS).toFixed(1)} ${sy(Math.min(r.trainLoss, lossMax), 0, lossMax).toFixed(1)}`
    ).join(" ");
  }, [currentEpoch, epochResults, lossMax]);

  const valPath = useMemo(() => {
    if (currentEpoch === 0) return "";
    return epochResults.slice(0, currentEpoch + 1).map((r, i) =>
      `${i === 0 ? "M" : "L"} ${sx(r.epoch, 0, ES_EPOCHS).toFixed(1)} ${sy(Math.min(r.valLoss, lossMax), 0, lossMax).toFixed(1)}`
    ).join(" ");
  }, [currentEpoch, epochResults, lossMax]);

  const showEarlyStop = currentEpoch >= earlyStopEpoch;

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
        <p className="text-xs text-orange-700">
          <strong>Early stopping</strong> monitors validation loss during training. If it does not improve for
          <strong> {patience} epochs</strong> (patience), training stops. The model saved at the best epoch prevents overfitting.
        </p>
      </div>

      <div className="flex flex-col xl:flex-row gap-4">
        {/* Chart */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <p className="text-xs font-semibold text-slate-600 mb-2">Training Progress (Epoch {currentEpoch}/{ES_EPOCHS})</p>
            <svg viewBox={`0 0 ${CW} ${CH + 20}`} className="w-full" style={{ maxWidth: 580 }}>
              <defs><clipPath id="esClip"><rect x={PAD.left} y={PAD.top} width={PW} height={PH} /></clipPath></defs>
              <ChartGrid xMin={0} xMax={ES_EPOCHS} yMin={0} yMax={lossMax} xLabel="Epoch" yLabel="Loss (MSE)"
                xTicks={[0, 50, 100, 150, 200]}
                yTicks={Array.from({ length: 5 }, (_, i) => lossMax * i / 4)} />

              {/* Train loss */}
              {trainPath && <path d={trainPath} fill="none" stroke="#3b82f6" strokeWidth={2.5} clipPath="url(#esClip)" />}
              {/* Val loss */}
              {valPath && <path d={valPath} fill="none" stroke="#f97316" strokeWidth={2.5} clipPath="url(#esClip)" />}

              {/* Early stop vertical line */}
              {showEarlyStop && (
                <g>
                  <line x1={sx(earlyStopEpoch, 0, ES_EPOCHS)} y1={PAD.top}
                    x2={sx(earlyStopEpoch, 0, ES_EPOCHS)} y2={PAD.top + PH}
                    stroke="#ef4444" strokeWidth={2} strokeDasharray="6,4" />
                  <text x={sx(earlyStopEpoch, 0, ES_EPOCHS) + 4} y={PAD.top + 14}
                    fontSize={8} fill="#ef4444" fontWeight={700}>EARLY STOP</text>
                  <text x={sx(earlyStopEpoch, 0, ES_EPOCHS) + 4} y={PAD.top + 24}
                    fontSize={8} fill="#ef4444">epoch {earlyStopEpoch}</text>
                </g>
              )}

              {/* Saved model checkpoint marker */}
              {showEarlyStop && (
                <g>
                  <circle cx={sx(savedModelEpoch, 0, ES_EPOCHS)}
                    cy={sy(Math.min(epochResults[savedModelEpoch].valLoss, lossMax), 0, lossMax)}
                    r={6} fill="#22c55e" stroke="#fff" strokeWidth={2} />
                  <text x={sx(savedModelEpoch, 0, ES_EPOCHS) - 2}
                    y={sy(Math.min(epochResults[savedModelEpoch].valLoss, lossMax), 0, lossMax) - 10}
                    fontSize={8} fill="#16a34a" fontWeight={700} textAnchor="middle">
                    Saved Model
                  </text>
                </g>
              )}

              {/* Current epoch dots */}
              {currentEpoch > 0 && currentResult && (
                <>
                  <circle cx={sx(currentEpoch, 0, ES_EPOCHS)} cy={sy(Math.min(currentResult.trainLoss, lossMax), 0, lossMax)}
                    r={4} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
                  <circle cx={sx(currentEpoch, 0, ES_EPOCHS)} cy={sy(Math.min(currentResult.valLoss, lossMax), 0, lossMax)}
                    r={4} fill="#f97316" stroke="#fff" strokeWidth={1.5} />
                </>
              )}

              {/* "Press Train" message */}
              {currentEpoch === 0 && (
                <text x={PAD.left + PW / 2} y={PAD.top + PH / 2} fontSize={14} fill="#94a3b8" textAnchor="middle" fontWeight={500}>
                  Press &quot;Train&quot; to start
                </text>
              )}

              <ChartLegend x={PAD.left + PW - 145} y={PAD.top + 6} items={[
                { color: "#3b82f6", label: "Training Loss" },
                { color: "#f97316", label: "Validation Loss" },
                { color: "#ef4444", label: "Early Stop Line", dashed: true },
                { color: "#22c55e", label: "Saved Checkpoint" },
              ]} />
            </svg>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full xl:w-64 space-y-3">
          {/* Buttons */}
          <div className="flex gap-2">
            <button onClick={handleTrain} disabled={isRunning}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-40 transition-all">
              <Play className="w-4 h-4" /> Train
            </button>
            <button onClick={() => setIsRunning(!isRunning)} disabled={currentEpoch === 0 && !isRunning}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold bg-slate-700 text-white hover:bg-slate-600 disabled:opacity-40 transition-all">
              {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Resume</>}
            </button>
            <button onClick={handleReset}
              className="px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Epoch and Status */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">Epoch</p>
              <p className="text-xl font-bold text-slate-800 mt-1">{currentEpoch}</p>
            </div>
            <div className={`rounded-lg p-2.5 text-center border ${
              currentEpoch >= earlyStopEpoch && currentEpoch > 0 ? "bg-red-50 border-red-200" : "bg-green-50 border-green-200"
            }`}>
              <p className="text-[10px] text-slate-500 uppercase font-medium">Status</p>
              <p className={`text-xs font-bold mt-1.5 ${currentEpoch >= earlyStopEpoch && currentEpoch > 0 ? "text-red-600" : "text-green-600"}`}>
                {currentEpoch === 0 ? "Ready" : currentEpoch >= earlyStopEpoch ? "Stopped" : "Training..."}
              </p>
            </div>
          </div>

          {/* Patience slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700">Patience</label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">{patience} epochs</span>
            </div>
            <input type="range" min={5} max={50} step={1} value={patience}
              onChange={(e) => setPatience(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-rose-500" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>5 (aggressive)</span><span>50 (patient)</span>
            </div>
          </div>

          {/* Degree slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700">Model Degree</label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">{degree}</span>
            </div>
            <input type="range" min={3} max={15} step={1} value={degree}
              onChange={(e) => { setDegree(parseInt(e.target.value)); handleReset(); }}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>3</span><span>15</span>
            </div>
          </div>

          {/* Metrics */}
          {currentEpoch > 0 && (
            <div className="space-y-2">
              <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
                <p className="text-[10px] text-emerald-500 uppercase font-medium">Best Checkpoint</p>
                <p className="text-sm font-bold text-emerald-700 mt-1">Epoch {savedModelEpoch}</p>
                <p className="text-[10px] text-emerald-600">Val Loss: {bestValLoss.toFixed(4)}</p>
              </div>
              {showEarlyStop && (
                <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
                  <p className="text-[10px] text-violet-500 uppercase font-medium">Comparison</p>
                  <div className="mt-1 space-y-1">
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-600">Early Stop val loss:</span>
                      <span className="font-bold text-emerald-600">{bestValLoss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-600">Final val loss (ep 200):</span>
                      <span className="font-bold text-red-600">{finalValLoss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-600">Epochs saved:</span>
                      <span className="font-bold text-violet-600">{epochsSaved}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Warning */}
          {currentEpoch > earlyStopEpoch && currentEpoch > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex gap-2">
              <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
              <p className="text-[10px] text-red-700">
                Training continued past the early stop point! The model is now overfitting.
                In practice, training would have stopped at epoch {earlyStopEpoch}.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 3: DROPOUT EFFECT
// ══════════════════════════════════════════════════════════════════════════════
const LAYER_SIZES = [6, 8, 8, 4];
const LAYER_LABELS = ["Input", "Hidden 1", "Hidden 2", "Output"];
const NN_W = 520, NN_H = 300;
const NN_PAD_X = 60, NN_PAD_Y = 30;

function DropoutTab() {
  const [dropoutRate, setDropoutRate] = useState(0.3);
  const [maskSeed, setMaskSeed] = useState(1);
  const [showTestMode, setShowTestMode] = useState(false);

  // Generate dropout mask
  const dropoutMask = useMemo(() => {
    const rng = mulberry32(maskSeed * 1000 + 7);
    const mask: boolean[][] = [];
    for (let l = 0; l < LAYER_SIZES.length; l++) {
      const layerMask: boolean[] = [];
      for (let n = 0; n < LAYER_SIZES[l]; n++) {
        // Don't dropout input or output layers
        if (l === 0 || l === LAYER_SIZES.length - 1) { layerMask.push(true); }
        else { layerMask.push(rng() > dropoutRate); }
      }
      mask.push(layerMask);
    }
    return mask;
  }, [dropoutRate, maskSeed]);

  // Compute neuron positions
  const neuronPositions = useMemo(() => {
    const positions: { x: number; y: number }[][] = [];
    const layerSpacing = (NN_W - 2 * NN_PAD_X) / (LAYER_SIZES.length - 1);
    for (let l = 0; l < LAYER_SIZES.length; l++) {
      const layerPos: { x: number; y: number }[] = [];
      const nNeurons = LAYER_SIZES[l];
      const neuronSpacing = (NN_H - 2 * NN_PAD_Y) / (nNeurons - 1 || 1);
      for (let n = 0; n < nNeurons; n++) {
        layerPos.push({
          x: NN_PAD_X + l * layerSpacing,
          y: NN_PAD_Y + n * neuronSpacing + (nNeurons === 1 ? (NN_H - 2 * NN_PAD_Y) / 2 : 0),
        });
      }
      positions.push(layerPos);
    }
    return positions;
  }, []);

  // Count active neurons
  const activeCount = useMemo(() => {
    let total = 0, active = 0;
    for (let l = 1; l < LAYER_SIZES.length - 1; l++) {
      for (let n = 0; n < LAYER_SIZES[l]; n++) {
        total++;
        if (dropoutMask[l][n]) active++;
      }
    }
    return { total, active, dropped: total - active };
  }, [dropoutMask]);

  // Simulated loss curves
  const lossCurves = useMemo(() => {
    const rng = mulberry32(99);
    const noDrop: { trainLoss: number; valLoss: number }[] = [];
    const withDrop: { trainLoss: number; valLoss: number }[] = [];
    for (let ep = 0; ep <= 100; ep++) {
      const t = ep / 100;
      // Without dropout: overfits
      const noDropTrain = 0.8 * Math.exp(-4 * t) + 0.02 + rng() * 0.01;
      const noDropVal = 0.8 * Math.exp(-2 * t) + 0.15 * t + 0.1 + rng() * 0.015;
      noDrop.push({ trainLoss: noDropTrain, valLoss: noDropVal });

      // With dropout: generalizes
      const scale = 1 - dropoutRate * 0.6;
      const withDropTrain = 0.8 * Math.exp(-3 * t * scale) + 0.05 + rng() * 0.01;
      const withDropVal = 0.8 * Math.exp(-2.5 * t * scale) + 0.06 + rng() * 0.012;
      withDrop.push({ trainLoss: withDropTrain, valLoss: withDropVal });
    }
    return { noDrop, withDrop };
  }, [dropoutRate]);

  const lossMax = 1.0;

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
        <p className="text-xs text-orange-700">
          <strong>Dropout</strong> randomly deactivates neurons during training, forcing the network to not rely on any single neuron.
          At test time, all neurons are active but outputs are scaled by (1-p) to compensate.
          This acts as an ensemble of thinned networks and reduces overfitting.
        </p>
      </div>

      <div className="flex flex-col xl:flex-row gap-4">
        <div className="flex-1 min-w-0 space-y-4">
          {/* Neural network visualization */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-semibold text-slate-600">
                Network Visualization ({showTestMode ? "Test Mode - All Active" : "Training Mode - Dropout Applied"})
              </p>
              <button onClick={() => setShowTestMode(!showTestMode)}
                className={`text-[10px] font-semibold px-3 py-1 rounded-full transition-all ${
                  showTestMode ? "bg-green-100 text-green-700" : "bg-blue-100 text-blue-700"
                }`}>
                {showTestMode ? "Test Mode" : "Train Mode"}
              </button>
            </div>
            <svg viewBox={`0 0 ${NN_W} ${NN_H}`} className="w-full" style={{ maxWidth: 560 }}>
              {/* Connections */}
              {neuronPositions.map((layer, l) => {
                if (l >= LAYER_SIZES.length - 1) return null;
                const nextLayer = neuronPositions[l + 1];
                return layer.map((from, ni) => {
                  const fromActive = showTestMode || dropoutMask[l][ni];
                  return nextLayer.map((to, nj) => {
                    const toActive = showTestMode || dropoutMask[l + 1][nj];
                    const active = fromActive && toActive;
                    return (
                      <line key={`c${l}-${ni}-${nj}`}
                        x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                        stroke={active ? "#6366f1" : "#e2e8f0"}
                        strokeWidth={active ? 1 : 0.5}
                        opacity={active ? (showTestMode ? 1 - dropoutRate * 0.3 : 0.6) : 0.15} />
                    );
                  });
                });
              })}

              {/* Neurons */}
              {neuronPositions.map((layer, l) =>
                layer.map((pos, ni) => {
                  const active = showTestMode || dropoutMask[l][ni];
                  const isHidden = l > 0 && l < LAYER_SIZES.length - 1;
                  return (
                    <g key={`n${l}-${ni}`}>
                      <circle cx={pos.x} cy={pos.y} r={12}
                        fill={!active ? "#f1f5f9" : l === 0 ? "#3b82f6" : l === LAYER_SIZES.length - 1 ? "#22c55e" : "#6366f1"}
                        stroke={!active ? "#cbd5e1" : "#fff"} strokeWidth={2}
                        opacity={active ? 1 : 0.4} />
                      {/* X on dropped neurons */}
                      {!active && isHidden && !showTestMode && (
                        <g>
                          <line x1={pos.x - 5} y1={pos.y - 5} x2={pos.x + 5} y2={pos.y + 5} stroke="#ef4444" strokeWidth={2} />
                          <line x1={pos.x + 5} y1={pos.y - 5} x2={pos.x - 5} y2={pos.y + 5} stroke="#ef4444" strokeWidth={2} />
                        </g>
                      )}
                      {/* Scale factor in test mode */}
                      {showTestMode && isHidden && (
                        <text x={pos.x} y={pos.y + 3} fontSize={6} fill="white" textAnchor="middle" fontWeight={700}>
                          x{(1 - dropoutRate).toFixed(1)}
                        </text>
                      )}
                    </g>
                  );
                })
              )}

              {/* Layer labels */}
              {LAYER_LABELS.map((label, l) => (
                <text key={`ll${l}`} x={neuronPositions[l][0].x} y={NN_H - 5}
                  fontSize={9} fill="#64748b" textAnchor="middle" fontWeight={600}>{label}</text>
              ))}
            </svg>
          </div>

          {/* Loss comparison */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <p className="text-xs font-semibold text-slate-600 mb-2">Loss Curves: With vs Without Dropout</p>
            <svg viewBox={`0 0 ${CW} ${CH}`} className="w-full" style={{ maxWidth: 560 }}>
              <defs><clipPath id="dropClip"><rect x={PAD.left} y={PAD.top} width={PW} height={PH} /></clipPath></defs>
              <ChartGrid xMin={0} xMax={100} yMin={0} yMax={lossMax} xLabel="Epoch" yLabel="Loss"
                xTicks={[0, 25, 50, 75, 100]} yTicks={[0, 0.25, 0.5, 0.75, 1.0]} />

              {/* No dropout train */}
              <path d={lossCurves.noDrop.map((r, i) =>
                `${i === 0 ? "M" : "L"} ${sx(i, 0, 100).toFixed(1)} ${sy(Math.min(r.trainLoss, lossMax), 0, lossMax).toFixed(1)}`
              ).join(" ")} fill="none" stroke="#ef4444" strokeWidth={2} clipPath="url(#dropClip)" />
              {/* No dropout val */}
              <path d={lossCurves.noDrop.map((r, i) =>
                `${i === 0 ? "M" : "L"} ${sx(i, 0, 100).toFixed(1)} ${sy(Math.min(r.valLoss, lossMax), 0, lossMax).toFixed(1)}`
              ).join(" ")} fill="none" stroke="#ef4444" strokeWidth={2} strokeDasharray="5,3" clipPath="url(#dropClip)" />

              {/* With dropout train */}
              <path d={lossCurves.withDrop.map((r, i) =>
                `${i === 0 ? "M" : "L"} ${sx(i, 0, 100).toFixed(1)} ${sy(Math.min(r.trainLoss, lossMax), 0, lossMax).toFixed(1)}`
              ).join(" ")} fill="none" stroke="#22c55e" strokeWidth={2} clipPath="url(#dropClip)" />
              {/* With dropout val */}
              <path d={lossCurves.withDrop.map((r, i) =>
                `${i === 0 ? "M" : "L"} ${sx(i, 0, 100).toFixed(1)} ${sy(Math.min(r.valLoss, lossMax), 0, lossMax).toFixed(1)}`
              ).join(" ")} fill="none" stroke="#22c55e" strokeWidth={2} strokeDasharray="5,3" clipPath="url(#dropClip)" />

              <ChartLegend x={PAD.left + PW - 148} y={PAD.top + 6} items={[
                { color: "#ef4444", label: "No Dropout (train)" },
                { color: "#ef4444", label: "No Dropout (val)", dashed: true },
                { color: "#22c55e", label: "With Dropout (train)" },
                { color: "#22c55e", label: "With Dropout (val)", dashed: true },
              ]} />
            </svg>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full xl:w-64 space-y-3">
          {/* Dropout rate slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700">Dropout Rate</label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">{(dropoutRate * 100).toFixed(0)}%</span>
            </div>
            <input type="range" min={0} max={0.8} step={0.05} value={dropoutRate}
              onChange={(e) => setDropoutRate(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-500" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>0% (none)</span><span>80% (extreme)</span>
            </div>
          </div>

          {/* Resample button */}
          <button onClick={() => setMaskSeed(maskSeed + 1)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold bg-violet-500 text-white hover:bg-violet-600 transition-all">
            <RefreshCw className="w-4 h-4" /> Resample Dropout Mask
          </button>

          {/* Neuron stats */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">Total</p>
              <p className="text-lg font-bold text-slate-700">{activeCount.total}</p>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-2 text-center">
              <p className="text-[10px] text-green-500 uppercase font-medium">Active</p>
              <p className="text-lg font-bold text-green-600">{activeCount.active}</p>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-2 text-center">
              <p className="text-[10px] text-red-500 uppercase font-medium">Dropped</p>
              <p className="text-lg font-bold text-red-600">{activeCount.dropped}</p>
            </div>
          </div>

          {/* Mode toggle */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-xs font-semibold text-slate-700 mb-2">Network Mode</p>
            <div className="flex gap-2">
              <button onClick={() => setShowTestMode(false)}
                className={`flex-1 py-1.5 rounded text-[10px] font-semibold transition-all ${
                  !showTestMode ? "bg-blue-500 text-white" : "bg-slate-100 text-slate-600"
                }`}>Training</button>
              <button onClick={() => setShowTestMode(true)}
                className={`flex-1 py-1.5 rounded text-[10px] font-semibold transition-all ${
                  showTestMode ? "bg-green-500 text-white" : "bg-slate-100 text-slate-600"
                }`}>Test</button>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">
              {showTestMode
                ? `Test mode: all neurons active, outputs scaled by ${(1 - dropoutRate).toFixed(2)}`
                : `Train mode: each hidden neuron dropped with p=${dropoutRate.toFixed(2)}`}
            </p>
          </div>

          {/* Scaling explanation */}
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
            <p className="text-xs font-semibold text-indigo-800 mb-1">Scaling at Test Time</p>
            <p className="text-[10px] text-indigo-700">
              During training, each neuron output is divided by <strong>(1 - p) = {(1 - dropoutRate).toFixed(2)}</strong> (inverted dropout).
              This way, at test time the network can use all neurons without any scaling adjustment.
            </p>
            <div className="mt-2 bg-white/60 rounded p-2 text-[10px] font-mono text-indigo-800 text-center">
              train: h = mask * f(x) / (1 - {dropoutRate.toFixed(2)})<br />
              test: h = f(x)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════════════
export default function OverfittingActivity() {
  const [activeTab, setActiveTab] = useState<TabId>("trainval");

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-100 rounded-xl p-1 overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
              activeTab === tab.id
                ? "bg-white text-violet-700 shadow-sm"
                : "text-slate-500 hover:text-slate-700 hover:bg-white/50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "trainval" && <TrainValTab />}
        {activeTab === "earlystop" && <EarlyStoppingTab />}
        {activeTab === "dropout" && <DropoutTab />}
      </div>
    </div>
  );
}
