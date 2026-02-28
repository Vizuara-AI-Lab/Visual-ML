/**
 * LinearRegressionAnimation — SVG animation showing how Linear Regression
 * finds the best-fit line by minimizing residuals via gradient descent.
 * Uses actual model coefficients and feature data when available.
 */

import { useState, useRef, useCallback, useMemo } from "react";
import { Play, Pause, RotateCcw, SkipForward, Info } from "lucide-react";

// Seeded PRNG (mulberry32)
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface DataPoint {
  x: number;
  y: number;
}

interface AnimationProps {
  result?: any;
}

/**
 * Extract the best single-feature info from the result:
 * - targetSlope: coefficient of the most important feature
 * - targetIntercept: model intercept
 * - featureName: name of that feature
 * - targetName: name of target column
 */
function extractModelInfo(result: any) {
  const coefficients: number[] = Array.isArray(result?.coefficients)
    ? result.coefficients
    : [];
  const intercept: number =
    typeof result?.intercept === "number" ? result.intercept : 3;
  const featureNames: string[] = result?.metadata?.feature_names || [];
  const targetColumn: string =
    result?.metadata?.target_column || result?.target_column || "y";

  if (coefficients.length === 0) {
    return { targetSlope: 2, targetIntercept: 3, featureName: "x", targetName: "y", hasRealData: false };
  }

  // Pick the feature with the highest absolute coefficient
  let bestIdx = 0;
  let bestAbs = 0;
  for (let i = 0; i < coefficients.length; i++) {
    const abs = Math.abs(coefficients[i]);
    if (abs > bestAbs) {
      bestAbs = abs;
      bestIdx = i;
    }
  }

  return {
    targetSlope: coefficients[bestIdx],
    targetIntercept: intercept,
    featureName: featureNames[bestIdx] || `Feature ${bestIdx}`,
    targetName: targetColumn,
    hasRealData: true,
  };
}

/**
 * Generate synthetic data from y = slope * x + intercept + noise
 * using reasonable x-range based on the target values.
 */
function generateData(targetSlope: number, targetIntercept: number): DataPoint[] {
  const rng = mulberry32(42);
  const points: DataPoint[] = [];

  // Scale x-range so the line stays visually sensible
  const absSlope = Math.abs(targetSlope) || 1;
  const xSpan = Math.max(2, Math.min(20, 20 / absSlope));
  const xStart = 1;

  const noiseScale = Math.max(0.5, Math.abs(targetSlope) * xSpan * 0.08);

  for (let i = 0; i < 20; i++) {
    const x = rng() * xSpan + xStart;
    const noise = (rng() - 0.5) * noiseScale * 2;
    const y = targetSlope * x + targetIntercept + noise;
    points.push({ x, y });
  }
  return points.sort((a, b) => a.x - b.x);
}

// MSE loss
function mse(data: DataPoint[], m: number, b: number) {
  return data.reduce((sum, p) => sum + (p.y - (m * p.x + b)) ** 2, 0) / data.length;
}

const TOTAL_STEPS = 40;

export default function LinearRegressionAnimation({ result }: AnimationProps) {
  const { targetSlope, targetIntercept, featureName, targetName, hasRealData } =
    useMemo(() => extractModelInfo(result), [result]);

  const data = useMemo(
    () => generateData(targetSlope, targetIntercept),
    [targetSlope, targetIntercept]
  );

  // Compute appropriate axis ranges from data
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = data.map((p) => p.x);
    const ys = data.map((p) => p.y);
    const xLo = Math.min(...xs);
    const xHi = Math.max(...xs);
    const yLo = Math.min(...ys);
    const yHi = Math.max(...ys);
    const xPad = (xHi - xLo) * 0.15;
    const yPad = (yHi - yLo) * 0.2;
    return {
      xMin: Math.floor(xLo - xPad),
      xMax: Math.ceil(xHi + xPad),
      yMin: Math.floor(yLo - yPad),
      yMax: Math.ceil(yHi + yPad),
    };
  }, [data]);

  // Plot dimensions
  const PAD = { top: 20, right: 20, bottom: 30, left: 48 };
  const W = 500;
  const H = 400;
  const PW = W - PAD.left - PAD.right;
  const PH = H - PAD.top - PAD.bottom;

  const toSvgX = useCallback(
    (x: number) => PAD.left + ((x - xMin) / (xMax - xMin || 1)) * PW,
    [xMin, xMax]
  );
  const toSvgY = useCallback(
    (y: number) => PAD.top + PH - ((y - yMin) / (yMax - yMin || 1)) * PH,
    [yMin, yMax]
  );

  // Learning rate scaled to data magnitude
  const learningRate = useMemo(() => {
    const absSlope = Math.abs(targetSlope) || 1;
    const absIntercept = Math.abs(targetIntercept) || 1;
    const scale = Math.max(absSlope, absIntercept, 1);
    return Math.min(0.01, 0.004 / (scale * 0.1 + 1));
  }, [targetSlope, targetIntercept]);

  // Initial values: start far from target
  const initialSlope = 0;
  const initialIntercept = useMemo(() => {
    return (yMin + yMax) / 2;
  }, [yMin, yMax]);

  const [step, setStep] = useState(0);
  const [slope, setSlope] = useState(initialSlope);
  const [intercept, setIntercept] = useState(initialIntercept);
  const [isPlaying, setIsPlaying] = useState(false);
  const animRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<"left" | "right" | null>(null);

  const slopeRef = useRef(slope);
  const interceptRef = useRef(intercept);
  const stepRef = useRef(step);
  slopeRef.current = slope;
  interceptRef.current = intercept;
  stepRef.current = step;

  const doStep = useCallback(() => {
    const m = slopeRef.current;
    const b = interceptRef.current;
    const n = data.length;

    let dm = 0;
    let db = 0;
    for (const p of data) {
      const err = p.y - (m * p.x + b);
      dm += (-2 * p.x * err) / n;
      db += (-2 * err) / n;
    }

    const newM = m - learningRate * dm;
    const newB = b - learningRate * db;

    setSlope(newM);
    setIntercept(newB);
    setStep((s) => s + 1);
  }, [data, learningRate]);

  const startAnimation = useCallback(() => {
    setIsPlaying(true);
    lastTimeRef.current = 0;

    const animate = (time: number) => {
      if (stepRef.current >= TOTAL_STEPS) {
        setIsPlaying(false);
        return;
      }
      if (lastTimeRef.current === 0) lastTimeRef.current = time;
      if (time - lastTimeRef.current >= 100) {
        lastTimeRef.current = time;
        doStep();
      }
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
  }, [doStep]);

  const stopAnimation = useCallback(() => {
    setIsPlaying(false);
    if (animRef.current) cancelAnimationFrame(animRef.current);
  }, []);

  const handlePlayPause = () => {
    if (isPlaying) {
      stopAnimation();
    } else if (step < TOTAL_STEPS) {
      startAnimation();
    }
  };

  const handleStep = () => {
    if (step < TOTAL_STEPS) doStep();
  };

  const handleReset = () => {
    stopAnimation();
    setSlope(initialSlope);
    setIntercept(initialIntercept);
    setStep(0);
  };

  // --- Drag-to-adjust logic ---
  const fromSvgY = useCallback(
    (svgY: number) => yMax - ((svgY - PAD.top) / PH) * (yMax - yMin),
    [yMin, yMax]
  );

  const getSvgPoint = useCallback((e: React.MouseEvent) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const rect = svg.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * W,
      y: ((e.clientY - rect.top) / rect.height) * H,
    };
  }, []);

  const handleDragMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging) return;
      const pt = getSvgPoint(e);
      const newDataY = fromSvgY(pt.y);

      const curLeftY = slopeRef.current * xMin + interceptRef.current;
      const curRightY = slopeRef.current * xMax + interceptRef.current;

      const y1 = dragging === "left" ? newDataY : curLeftY;
      const y2 = dragging === "right" ? newDataY : curRightY;

      const newSlope = (y2 - y1) / (xMax - xMin || 1);
      const newIntercept = y1 - newSlope * xMin;

      setSlope(newSlope);
      setIntercept(newIntercept);
    },
    [dragging, xMin, xMax, fromSvgY, getSvgPoint]
  );

  const handleDragEnd = useCallback(() => setDragging(null), []);

  const currentMSE = mse(data, slope, intercept);

  // Line endpoints for SVG
  const lineY1 = slope * xMin + intercept;
  const lineY2 = slope * xMax + intercept;

  // Axis tick values
  const xTicks = useMemo(() => {
    const range = xMax - xMin;
    const rawStep = range / 5;
    const step = Math.ceil(rawStep) || 1;
    const ticks: number[] = [];
    for (let v = Math.ceil(xMin / step) * step; v <= xMax; v += step) {
      ticks.push(v);
    }
    return ticks;
  }, [xMin, xMax]);

  const yTicks = useMemo(() => {
    const range = yMax - yMin;
    const rawStep = range / 5;
    const step = Math.ceil(rawStep) || 1;
    const ticks: number[] = [];
    for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
      ticks.push(v);
    }
    return ticks;
  }, [yMin, yMax]);

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="flex items-start gap-3 p-4 bg-blue-50 rounded-xl border border-blue-100">
        <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
        <div className="text-[13px] text-blue-700 leading-relaxed">
          <strong>Linear Regression</strong> finds the best-fit line through data
          by minimizing the sum of squared residuals (the vertical distances from
          each point to the line). <strong>Drag the handles</strong> on the line
          to try finding the best fit yourself, or press Play to watch gradient
          descent do it automatically.
          {hasRealData && (
            <span className="block mt-1 text-blue-600">
              Using your model's strongest feature <strong>{featureName}</strong> to
              visualize how the line converges to the learned relationship.
            </span>
          )}
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        {/* SVG visualization */}
        <div className="flex-1 bg-white rounded-xl border border-gray-200 p-3">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            className={`w-full ${dragging ? "cursor-grabbing" : ""}`}
            onMouseMove={handleDragMove}
            onMouseUp={handleDragEnd}
            onMouseLeave={handleDragEnd}
          >
            {/* Grid lines */}
            {yTicks.map((y) => (
              <line
                key={`gy-${y}`}
                x1={PAD.left}
                y1={toSvgY(y)}
                x2={W - PAD.right}
                y2={toSvgY(y)}
                stroke="#f1f5f9"
                strokeWidth={1}
              />
            ))}
            {xTicks.map((x) => (
              <line
                key={`gx-${x}`}
                x1={toSvgX(x)}
                y1={PAD.top}
                x2={toSvgX(x)}
                y2={H - PAD.bottom}
                stroke="#f1f5f9"
                strokeWidth={1}
              />
            ))}

            {/* Axes */}
            <line
              x1={PAD.left}
              y1={H - PAD.bottom}
              x2={W - PAD.right}
              y2={H - PAD.bottom}
              stroke="#94a3b8"
              strokeWidth={1.5}
            />
            <line
              x1={PAD.left}
              y1={PAD.top}
              x2={PAD.left}
              y2={H - PAD.bottom}
              stroke="#94a3b8"
              strokeWidth={1.5}
            />

            {/* Axis tick labels */}
            {xTicks.map((x) => (
              <text
                key={`xl-${x}`}
                x={toSvgX(x)}
                y={H - PAD.bottom + 16}
                textAnchor="middle"
                fontSize={10}
                fill="#64748b"
              >
                {x}
              </text>
            ))}
            {yTicks.map((y) => (
              <text
                key={`yl-${y}`}
                x={PAD.left - 8}
                y={toSvgY(y) + 4}
                textAnchor="end"
                fontSize={10}
                fill="#64748b"
              >
                {y}
              </text>
            ))}

            {/* Axis titles */}
            <text
              x={W / 2}
              y={H - 2}
              textAnchor="middle"
              fontSize={11}
              fill="#64748b"
            >
              {featureName}
            </text>
            <text
              x={12}
              y={H / 2}
              textAnchor="middle"
              fontSize={11}
              fill="#64748b"
              transform={`rotate(-90, 12, ${H / 2})`}
            >
              {targetName}
            </text>

            {/* Residual lines */}
            {data.map((p, i) => {
              const predicted = slope * p.x + intercept;
              return (
                <line
                  key={`r-${i}`}
                  x1={toSvgX(p.x)}
                  y1={toSvgY(p.y)}
                  x2={toSvgX(p.x)}
                  y2={toSvgY(Math.max(yMin, Math.min(yMax, predicted)))}
                  stroke="#f87171"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  opacity={0.5}
                />
              );
            })}

            {/* Regression line */}
            <line
              x1={toSvgX(xMin)}
              y1={toSvgY(Math.max(yMin, Math.min(yMax, lineY1)))}
              x2={toSvgX(xMax)}
              y2={toSvgY(Math.max(yMin, Math.min(yMax, lineY2)))}
              stroke="#ef4444"
              strokeWidth={2.5}
              strokeLinecap="round"
            />

            {/* Data points */}
            {data.map((p, i) => (
              <circle
                key={`p-${i}`}
                cx={toSvgX(p.x)}
                cy={toSvgY(p.y)}
                r={5}
                fill="#3b82f6"
                stroke="white"
                strokeWidth={1.5}
              />
            ))}

            {/* Drag handles on line endpoints */}
            {[
              { pos: "left" as const, dataX: xMin, dataY: lineY1 },
              { pos: "right" as const, dataX: xMax, dataY: lineY2 },
            ].map((h) => (
              <circle
                key={h.pos}
                cx={toSvgX(h.dataX)}
                cy={toSvgY(Math.max(yMin, Math.min(yMax, h.dataY)))}
                r={dragging === h.pos ? 9 : 7}
                fill={dragging === h.pos ? "#ef4444" : "white"}
                stroke="#ef4444"
                strokeWidth={2.5}
                className={dragging ? "cursor-grabbing" : "cursor-grab"}
                onMouseDown={(e) => {
                  e.preventDefault();
                  stopAnimation();
                  setDragging(h.pos);
                }}
              />
            ))}

            {/* Equation display */}
            <text
              x={W - PAD.right - 10}
              y={PAD.top + 20}
              textAnchor="end"
              fontSize={13}
              fill="#1e293b"
              fontWeight={600}
            >
              {targetName} = {slope.toFixed(2)} * {featureName} + {intercept.toFixed(2)}
            </text>
          </svg>
        </div>

        {/* Controls panel */}
        <div className="lg:w-64 space-y-4">
          {/* Playback controls */}
          <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Controls
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={handlePlayPause}
                disabled={step >= TOTAL_STEPS && !isPlaying}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg
                           bg-blue-500 hover:bg-blue-600 text-white text-sm font-semibold
                           transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4" /> Pause
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> Play
                  </>
                )}
              </button>
              <button
                onClick={handleStep}
                disabled={step >= TOTAL_STEPS || isPlaying}
                className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors
                           disabled:opacity-50 disabled:cursor-not-allowed"
                title="Single step"
              >
                <SkipForward className="w-4 h-4" />
              </button>
              <button
                onClick={handleReset}
                className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors"
                title="Reset"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            {/* Progress */}
            <div>
              <div className="flex items-center justify-between text-[11px] text-gray-500 mb-1">
                <span>
                  Step {step}/{TOTAL_STEPS}
                </span>
                <span>{Math.round((step / TOTAL_STEPS) * 100)}%</span>
              </div>
              <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-150"
                  style={{ width: `${(step / TOTAL_STEPS) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Metrics
            </h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">MSE Loss</span>
                <span className="text-sm font-bold text-red-600">
                  {currentMSE.toFixed(2)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Slope (m)</span>
                <span className="text-sm font-mono font-semibold text-gray-800">
                  {slope.toFixed(3)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Intercept (b)</span>
                <span className="text-sm font-mono font-semibold text-gray-800">
                  {intercept.toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Target values */}
          <div className="bg-gray-50 rounded-xl border border-gray-200 p-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              {hasRealData ? "Your Model's Values" : "True Values"}
            </h3>
            <div className="text-sm text-gray-600 space-y-1">
              <div>
                Slope:{" "}
                <span className="font-mono font-semibold text-emerald-600">
                  {targetSlope.toFixed(4)}
                </span>
              </div>
              <div>
                Intercept:{" "}
                <span className="font-mono font-semibold text-emerald-600">
                  {targetIntercept.toFixed(4)}
                </span>
              </div>
            </div>
            <p className="text-[11px] text-gray-400 mt-2">
              {hasRealData
                ? `Coefficient of "${featureName}" from your trained model`
                : "Generated from y = 2x + 3 + noise"}
            </p>
          </div>

          {/* Tips */}
          <div className="bg-amber-50 rounded-xl border border-amber-100 p-4">
            <h4 className="text-xs font-semibold text-amber-700 mb-1.5">
              Things to notice
            </h4>
            <ul className="text-[11px] text-amber-700 space-y-1 list-disc pl-3">
              <li>Residual lines (dashed) shrink as the line fits better</li>
              <li>MSE loss decreases with each step</li>
              <li>
                Slope approaches {targetSlope.toFixed(2)} and intercept
                approaches {targetIntercept.toFixed(2)}
              </li>
              <li>Early steps make larger adjustments</li>
              <li>Drag the line handles and try to beat gradient descent's MSE!</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
