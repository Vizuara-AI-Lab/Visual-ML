/**
 * LogisticRegressionAnimation - Interactive SVG visualization of Logistic Regression
 *
 * Educational component showing how the sigmoid function, weights, bias, and
 * decision threshold work together to classify data points into two classes.
 *
 * Features:
 * - Sigmoid curve that morphs as weight/bias sliders change
 * - Data points colored by class (blue = 0, red = 1)
 * - Adjustable decision threshold with shaded decision regions
 * - Live classification accuracy display
 */

import { useState, useMemo } from "react";
import { Info, SlidersHorizontal } from "lucide-react";

// ---------------------------------------------------------------------------
// Seeded PRNG (mulberry32) for reproducible data generation
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Gaussian-ish random using Box-Muller (uses two uniform samples)
// ---------------------------------------------------------------------------

function gaussianRandom(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

interface DataPoint {
  x: number;
  label: 0 | 1;
}

function generateData(seed: number = 42, count: number = 30): DataPoint[] {
  const rng = mulberry32(seed);
  const points: DataPoint[] = [];
  const half = Math.floor(count / 2);

  // Class 0: centered around x = -2
  for (let i = 0; i < half; i++) {
    points.push({ x: -2 + gaussianRandom(rng) * 1.0, label: 0 });
  }

  // Class 1: centered around x = 2
  for (let i = 0; i < count - half; i++) {
    points.push({ x: 2 + gaussianRandom(rng) * 1.0, label: 1 });
  }

  return points;
}

// ---------------------------------------------------------------------------
// Sigmoid function
// ---------------------------------------------------------------------------

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

// ---------------------------------------------------------------------------
// SVG coordinate helpers
// ---------------------------------------------------------------------------

// Data domain
const X_MIN = -6;
const X_MAX = 6;
const Y_MIN = -0.1;
const Y_MAX = 1.1;

// SVG viewBox: 0 0 500 400
// Plotting area with margins
const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };
const PLOT_W = 500 - MARGIN.left - MARGIN.right;
const PLOT_H = 400 - MARGIN.top - MARGIN.bottom;

function toSvgX(dataX: number): number {
  return MARGIN.left + ((dataX - X_MIN) / (X_MAX - X_MIN)) * PLOT_W;
}

function toSvgY(dataY: number): number {
  return MARGIN.top + (1 - (dataY - Y_MIN) / (Y_MAX - Y_MIN)) * PLOT_H;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const LogisticRegressionAnimation = () => {
  const [weight, setWeight] = useState(1.5);
  const [bias, setBias] = useState(0);
  const [threshold, setThreshold] = useState(0.5);

  // Generate fixed data points (seeded, so stable across re-renders)
  const dataPoints = useMemo(() => generateData(42, 30), []);

  // Build sigmoid curve path (100 samples)
  const sigmoidPath = useMemo(() => {
    const steps = 100;
    const parts: string[] = [];
    for (let i = 0; i <= steps; i++) {
      const x = X_MIN + (i / steps) * (X_MAX - X_MIN);
      const y = sigmoid(weight * x + bias);
      const sx = toSvgX(x);
      const sy = toSvgY(y);
      parts.push(`${i === 0 ? "M" : "L"}${sx.toFixed(2)},${sy.toFixed(2)}`);
    }
    return parts.join(" ");
  }, [weight, bias]);

  // Compute the x value where sigmoid(w*x + b) = threshold
  // threshold = 1/(1+e^(-(w*x+b)))  =>  w*x+b = -ln(1/threshold - 1)  =>  x = (-ln(1/threshold - 1) - b) / w
  const thresholdX = useMemo(() => {
    if (weight === 0) return null;
    const logit = -Math.log(1 / threshold - 1);
    const x = (logit - bias) / weight;
    if (x < X_MIN || x > X_MAX) return null;
    return x;
  }, [weight, bias, threshold]);

  // Classification accuracy
  const accuracy = useMemo(() => {
    let correct = 0;
    for (const pt of dataPoints) {
      const prob = sigmoid(weight * pt.x + bias);
      const predicted = prob >= threshold ? 1 : 0;
      if (predicted === pt.label) correct++;
    }
    return correct / dataPoints.length;
  }, [weight, bias, threshold, dataPoints]);

  // Grid lines for the y-axis (0, 0.25, 0.5, 0.75, 1.0)
  const yGridValues = [0, 0.25, 0.5, 0.75, 1.0];

  // Grid lines for the x-axis
  const xGridValues = [-6, -4, -2, 0, 2, 4, 6];

  // Decision region polygons
  const decisionRegions = useMemo(() => {
    if (thresholdX === null) {
      // If threshold line is off-screen, shade entire region based on weight sign
      const allBlue = weight > 0;
      return {
        leftColor: allBlue ? "rgba(59,130,246,0.08)" : "rgba(239,68,68,0.08)",
        rightColor: allBlue ? "rgba(239,68,68,0.08)" : "rgba(59,130,246,0.08)",
        thresholdSvgX: null,
      };
    }
    const svgX = toSvgX(thresholdX);
    // When weight > 0: left of threshold = class 0 (blue), right = class 1 (red)
    // When weight < 0: reversed
    const leftIsClassZero = weight >= 0;
    return {
      leftColor: leftIsClassZero
        ? "rgba(59,130,246,0.08)"
        : "rgba(239,68,68,0.08)",
      rightColor: leftIsClassZero
        ? "rgba(239,68,68,0.08)"
        : "rgba(59,130,246,0.08)",
      thresholdSvgX: svgX,
    };
  }, [thresholdX, weight]);

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 bg-emerald-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
            <Info className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-emerald-900">
              Logistic Regression
            </h3>
            <p className="text-xs text-emerald-700 mt-1 leading-relaxed">
              Logistic regression uses the <strong>sigmoid function</strong> to
              map any input to a probability between 0 and 1. By adjusting the{" "}
              <strong>weight</strong> and <strong>bias</strong>, the model learns
              where to place the decision boundary that separates the two
              classes. The <strong>threshold</strong> determines the cutoff
              probability for classification.
            </p>
          </div>
        </div>
      </div>

      {/* Main layout: SVG + Controls */}
      <div className="flex gap-6 flex-col lg:flex-row">
        {/* SVG visualization */}
        <div className="flex-1 bg-white border border-gray-200 rounded-xl p-4">
          <svg
            viewBox="0 0 500 400"
            className="w-full h-auto"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Decision region shading */}
            {decisionRegions.thresholdSvgX !== null ? (
              <>
                {/* Left region */}
                <rect
                  x={MARGIN.left}
                  y={MARGIN.top}
                  width={decisionRegions.thresholdSvgX - MARGIN.left}
                  height={PLOT_H}
                  fill={decisionRegions.leftColor}
                />
                {/* Right region */}
                <rect
                  x={decisionRegions.thresholdSvgX}
                  y={MARGIN.top}
                  width={
                    MARGIN.left + PLOT_W - decisionRegions.thresholdSvgX
                  }
                  height={PLOT_H}
                  fill={decisionRegions.rightColor}
                />
              </>
            ) : (
              <rect
                x={MARGIN.left}
                y={MARGIN.top}
                width={PLOT_W}
                height={PLOT_H}
                fill={decisionRegions.leftColor}
              />
            )}

            {/* Grid lines - horizontal */}
            {yGridValues.map((yVal) => {
              const sy = toSvgY(yVal);
              return (
                <g key={`hgrid-${yVal}`}>
                  <line
                    x1={MARGIN.left}
                    y1={sy}
                    x2={MARGIN.left + PLOT_W}
                    y2={sy}
                    stroke="#e5e7eb"
                    strokeWidth={0.5}
                  />
                  <text
                    x={MARGIN.left - 8}
                    y={sy + 3}
                    textAnchor="end"
                    fontSize={10}
                    fill="#6b7280"
                  >
                    {yVal.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* Grid lines - vertical */}
            {xGridValues.map((xVal) => {
              const sx = toSvgX(xVal);
              return (
                <g key={`vgrid-${xVal}`}>
                  <line
                    x1={sx}
                    y1={MARGIN.top}
                    x2={sx}
                    y2={MARGIN.top + PLOT_H}
                    stroke="#e5e7eb"
                    strokeWidth={0.5}
                  />
                  <text
                    x={sx}
                    y={MARGIN.top + PLOT_H + 16}
                    textAnchor="middle"
                    fontSize={10}
                    fill="#6b7280"
                  >
                    {xVal}
                  </text>
                </g>
              );
            })}

            {/* Axes */}
            {/* X-axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + PLOT_H}
              x2={MARGIN.left + PLOT_W}
              y2={MARGIN.top + PLOT_H}
              stroke="#374151"
              strokeWidth={1.5}
            />
            {/* Y-axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={MARGIN.top + PLOT_H}
              stroke="#374151"
              strokeWidth={1.5}
            />

            {/* Axis labels */}
            <text
              x={MARGIN.left + PLOT_W / 2}
              y={MARGIN.top + PLOT_H + 34}
              textAnchor="middle"
              fontSize={12}
              fill="#374151"
              fontWeight={600}
            >
              x (input)
            </text>
            <text
              x={14}
              y={MARGIN.top + PLOT_H / 2}
              textAnchor="middle"
              fontSize={12}
              fill="#374151"
              fontWeight={600}
              transform={`rotate(-90, 14, ${MARGIN.top + PLOT_H / 2})`}
            >
              P(y=1)
            </text>

            {/* Threshold horizontal line (y = threshold) */}
            <line
              x1={MARGIN.left}
              y1={toSvgY(threshold)}
              x2={MARGIN.left + PLOT_W}
              y2={toSvgY(threshold)}
              stroke="#a855f7"
              strokeWidth={1}
              strokeDasharray="4,3"
              opacity={0.6}
            />
            <text
              x={MARGIN.left + PLOT_W + 2}
              y={toSvgY(threshold) + 3}
              fontSize={9}
              fill="#a855f7"
              fontWeight={500}
            >
              t={threshold.toFixed(2)}
            </text>

            {/* Threshold vertical line (where sigmoid crosses threshold) */}
            {thresholdX !== null && (
              <>
                <line
                  x1={toSvgX(thresholdX)}
                  y1={MARGIN.top}
                  x2={toSvgX(thresholdX)}
                  y2={MARGIN.top + PLOT_H}
                  stroke="#a855f7"
                  strokeWidth={1.5}
                  strokeDasharray="6,4"
                />
                <text
                  x={toSvgX(thresholdX)}
                  y={MARGIN.top - 6}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#a855f7"
                  fontWeight={600}
                >
                  boundary
                </text>
              </>
            )}

            {/* Sigmoid curve */}
            <path
              d={sigmoidPath}
              fill="none"
              stroke="#059669"
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Data points */}
            {dataPoints.map((pt, idx) => {
              const prob = sigmoid(weight * pt.x + bias);
              const predicted = prob >= threshold ? 1 : 0;
              const isCorrect = predicted === pt.label;
              const cx = toSvgX(pt.x);
              // Place data points at y = sigmoid(w*x + b) so they "ride" the curve
              const cy = toSvgY(prob);
              const fillColor = pt.label === 0 ? "#3b82f6" : "#ef4444";
              return (
                <g key={idx}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={5}
                    fill={fillColor}
                    stroke={isCorrect ? fillColor : "#f59e0b"}
                    strokeWidth={isCorrect ? 1.5 : 2.5}
                    opacity={0.85}
                  />
                </g>
              );
            })}

            {/* Equation label */}
            <text
              x={MARGIN.left + PLOT_W - 4}
              y={MARGIN.top + 16}
              textAnchor="end"
              fontSize={11}
              fill="#374151"
              fontFamily="monospace"
            >
              {"\u03C3(z) = 1 / (1 + e"}
              <tspan baselineShift="super" fontSize={8}>
                {"-z"}
              </tspan>
              {")"}
            </text>
            <text
              x={MARGIN.left + PLOT_W - 4}
              y={MARGIN.top + 30}
              textAnchor="end"
              fontSize={10}
              fill="#6b7280"
              fontFamily="monospace"
            >
              z = {weight.toFixed(1)}x + ({bias.toFixed(1)})
            </text>
          </svg>

          {/* Legend below SVG */}
          <div className="flex items-center justify-center gap-6 mt-2 text-xs text-gray-600">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full bg-blue-500" />
              Class 0
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full bg-red-500" />
              Class 1
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-emerald-600" style={{ height: 3 }} />
              Sigmoid curve
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 border-t-2 border-dashed border-purple-500"
                style={{ height: 0 }}
              />
              Threshold
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full border-2 border-amber-400 bg-transparent" />
              Misclassified
            </span>
          </div>
        </div>

        {/* Controls panel */}
        <div className="w-full lg:w-64 space-y-4 flex-shrink-0">
          {/* Controls header */}
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
            <SlidersHorizontal className="w-4 h-4 text-emerald-600" />
            Controls
          </div>

          {/* Weight slider */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">
                Weight (w)
              </label>
              <span className="text-sm font-mono text-gray-900 bg-gray-100 px-2 py-0.5 rounded">
                {weight.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={-5}
              max={5}
              step={0.1}
              value={weight}
              onChange={(e) => setWeight(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>-5</span>
              <span>0</span>
              <span>5</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Controls the steepness of the sigmoid curve. Larger |w| = sharper
              boundary.
            </p>
          </div>

          {/* Bias slider */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">
                Bias (b)
              </label>
              <span className="text-sm font-mono text-gray-900 bg-gray-100 px-2 py-0.5 rounded">
                {bias.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={-5}
              max={5}
              step={0.1}
              value={bias}
              onChange={(e) => setBias(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>-5</span>
              <span>0</span>
              <span>5</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Shifts the sigmoid curve left or right. Positive b shifts the
              boundary left.
            </p>
          </div>

          {/* Threshold slider */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">
                Threshold
              </label>
              <span className="text-sm font-mono text-gray-900 bg-gray-100 px-2 py-0.5 rounded">
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0.1</span>
              <span>0.5</span>
              <span>0.9</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              If P(y=1) &ge; threshold, predict class 1; otherwise class 0.
            </p>
          </div>

          {/* Accuracy card */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-1">
              Classification Accuracy
            </div>
            <div className="text-3xl font-bold text-emerald-700 font-mono">
              {(accuracy * 100).toFixed(1)}%
            </div>
            <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden mt-2">
              <div
                className="h-full rounded-full bg-emerald-500 transition-all"
                style={{ width: `${accuracy * 100}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {Math.round(accuracy * dataPoints.length)} of {dataPoints.length}{" "}
              points classified correctly
            </p>
          </div>

          {/* Equation card */}
          <div className="bg-gray-900 rounded-xl p-4">
            <div className="text-xs text-gray-400 uppercase tracking-wide font-medium mb-2">
              Sigmoid Equation
            </div>
            <div className="text-sm text-green-400 font-mono leading-relaxed">
              <div>&sigma;(z) = 1 / (1 + e<sup>-z</sup>)</div>
              <div className="mt-1 text-gray-300">
                z = wx + b
              </div>
              <div className="mt-1 text-emerald-400">
                z = {weight.toFixed(1)}x + ({bias.toFixed(1)})
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LogisticRegressionAnimation;
