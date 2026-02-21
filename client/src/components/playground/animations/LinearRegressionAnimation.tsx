/**
 * LinearRegressionAnimation — SVG animation showing how Linear Regression
 * finds the best-fit line by minimizing residuals via gradient descent.
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

// Generate data from y = 2x + 3 + noise
function generateData(): DataPoint[] {
  const rng = mulberry32(42);
  const points: DataPoint[] = [];
  for (let i = 0; i < 20; i++) {
    const x = rng() * 8 + 1; // x in [1, 9]
    const noise = (rng() - 0.5) * 4;
    const y = 2 * x + 3 + noise;
    points.push({ x, y });
  }
  return points.sort((a, b) => a.x - b.x);
}

// Plot dimensions
const PAD = { top: 20, right: 20, bottom: 30, left: 40 };
const W = 500;
const H = 400;
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

const X_MIN = 0;
const X_MAX = 10;
const Y_MIN = 0;
const Y_MAX = 25;

function toSvgX(x: number) {
  return PAD.left + ((x - X_MIN) / (X_MAX - X_MIN)) * PW;
}
function toSvgY(y: number) {
  return PAD.top + PH - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * PH;
}

// MSE loss
function mse(data: DataPoint[], m: number, b: number) {
  return data.reduce((sum, p) => sum + (p.y - (m * p.x + b)) ** 2, 0) / data.length;
}

const TOTAL_STEPS = 40;
const LEARNING_RATE = 0.004;

export default function LinearRegressionAnimation() {
  const data = useMemo(generateData, []);

  const [step, setStep] = useState(0);
  const [slope, setSlope] = useState(0);
  const [intercept, setIntercept] = useState(12);
  const [isPlaying, setIsPlaying] = useState(false);
  const animRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

  // Current state refs for animation loop
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

    // Gradient of MSE w.r.t. m and b
    let dm = 0;
    let db = 0;
    for (const p of data) {
      const err = p.y - (m * p.x + b);
      dm += -2 * p.x * err / n;
      db += -2 * err / n;
    }

    const newM = m - LEARNING_RATE * dm;
    const newB = b - LEARNING_RATE * db;

    setSlope(newM);
    setIntercept(newB);
    setStep((s) => s + 1);
  }, [data]);

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
    setSlope(0);
    setIntercept(12);
    setStep(0);
  };

  const currentMSE = mse(data, slope, intercept);

  // Line endpoints for SVG
  const lineX1 = X_MIN;
  const lineX2 = X_MAX;
  const lineY1 = slope * lineX1 + intercept;
  const lineY2 = slope * lineX2 + intercept;

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="flex items-start gap-3 p-4 bg-blue-50 rounded-xl border border-blue-100">
        <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
        <div className="text-[13px] text-blue-700 leading-relaxed">
          <strong>Linear Regression</strong> finds the best-fit line through data
          by minimizing the sum of squared residuals (the vertical distances from
          each point to the line). Watch gradient descent iteratively adjust the
          slope and intercept to reduce the loss.
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        {/* SVG visualization */}
        <div className="flex-1 bg-white rounded-xl border border-gray-200 p-3">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
            {/* Grid lines */}
            {[0, 5, 10, 15, 20, 25].map((y) => (
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
            {[0, 2, 4, 6, 8, 10].map((x) => (
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

            {/* Axis labels */}
            {[0, 2, 4, 6, 8, 10].map((x) => (
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
            {[0, 5, 10, 15, 20, 25].map((y) => (
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

            {/* Residual lines */}
            {data.map((p, i) => {
              const predicted = slope * p.x + intercept;
              return (
                <line
                  key={`r-${i}`}
                  x1={toSvgX(p.x)}
                  y1={toSvgY(p.y)}
                  x2={toSvgX(p.x)}
                  y2={toSvgY(Math.max(Y_MIN, Math.min(Y_MAX, predicted)))}
                  stroke="#f87171"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  opacity={0.5}
                />
              );
            })}

            {/* Regression line */}
            <line
              x1={toSvgX(lineX1)}
              y1={toSvgY(Math.max(Y_MIN, Math.min(Y_MAX, lineY1)))}
              x2={toSvgX(lineX2)}
              y2={toSvgY(Math.max(Y_MIN, Math.min(Y_MAX, lineY2)))}
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

            {/* Equation display */}
            <text x={W - PAD.right - 10} y={PAD.top + 20} textAnchor="end" fontSize={13} fill="#1e293b" fontWeight={600}>
              y = {slope.toFixed(2)}x + {intercept.toFixed(2)}
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
                <span>Step {step}/{TOTAL_STEPS}</span>
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
              True Values
            </h3>
            <div className="text-sm text-gray-600 space-y-1">
              <div>
                Slope: <span className="font-mono font-semibold text-emerald-600">2.000</span>
              </div>
              <div>
                Intercept: <span className="font-mono font-semibold text-emerald-600">3.000</span>
              </div>
            </div>
            <p className="text-[11px] text-gray-400 mt-2">
              Generated from y = 2x + 3 + noise
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
              <li>Slope approaches 2 and intercept approaches 3</li>
              <li>Early steps make larger adjustments</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
