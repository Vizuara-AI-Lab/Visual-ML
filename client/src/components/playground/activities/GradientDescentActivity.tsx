/**
 * Gradient Descent Deep-Dive — 5-tab interactive SVG activity
 *
 * Tab 1: Step-by-Step 1D   — manual / auto stepping on a bumpy 1D loss curve
 * Tab 2: Learning Rate      — 3 simultaneous paths with different learning rates
 * Tab 3: Momentum & Variants— contour plot comparing Vanilla, Momentum, RMSProp, Adam
 * Tab 4: Stochastic vs Batch— SGD, Mini-Batch, Full-Batch on a regression dataset
 * Tab 5: Saddle Points      — navigating saddle points and local minima
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  TrendingDown,
  Zap,
  Info,
  Layers,
  BarChart3,
  Target,
  MousePointerClick,
} from "lucide-react";

// ═══════════════════════════════════════════════════════════════════════
// Seeded PRNG — mulberry32
// ═══════════════════════════════════════════════════════════════════════
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Common helpers
// ═══════════════════════════════════════════════════════════════════════
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

// ═══════════════════════════════════════════════════════════════════════
// Tab definition
// ═══════════════════════════════════════════════════════════════════════
interface TabDef {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabDef[] = [
  { id: "step1d", label: "Step-by-Step 1D", icon: <SkipForward className="w-4 h-4" /> },
  { id: "lreffect", label: "Learning Rate", icon: <TrendingDown className="w-4 h-4" /> },
  { id: "momentum", label: "Momentum & Variants", icon: <Zap className="w-4 h-4" /> },
  { id: "stochastic", label: "Stochastic vs Batch", icon: <BarChart3 className="w-4 h-4" /> },
  { id: "saddle", label: "Saddle Points", icon: <Target className="w-4 h-4" /> },
];

// ═══════════════════════════════════════════════════════════════════════
// Tab 1 — Step-by-Step 1D
// ═══════════════════════════════════════════════════════════════════════
const W1 = 560;
const H1 = 320;
const PAD1 = { top: 20, right: 20, bottom: 40, left: 50 };
const PW1 = W1 - PAD1.left - PAD1.right;
const PH1 = H1 - PAD1.top - PAD1.bottom;

function loss1D(x: number): number {
  return (x - 3) * (x - 3) + 2 * Math.sin(3 * x) + 0.5 * Math.cos(7 * x) + 5;
}
function gradLoss1D(x: number): number {
  return 2 * (x - 3) + 6 * Math.cos(3 * x) - 3.5 * Math.sin(7 * x);
}

function Tab1StepByStep() {
  const rng = useMemo(() => mulberry32(42), []);
  const [lr, setLr] = useState(0.05);
  const [pos, setPos] = useState(-2.0);
  const [trail, setTrail] = useState<{ x: number; loss: number }[]>([
    { x: -2.0, loss: loss1D(-2.0) },
  ]);
  const [stepCount, setStepCount] = useState(0);
  const [gradVal, setGradVal] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const posRef = useRef(pos);
  posRef.current = pos;
  const animRef = useRef<number | null>(null);
  const lrRef = useRef(lr);
  lrRef.current = lr;

  const xMin = -4;
  const xMax = 8;
  const samples = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= 300; i++) {
      const x = xMin + (i / 300) * (xMax - xMin);
      pts.push({ x, y: loss1D(x) });
    }
    return pts;
  }, []);

  const yMin = useMemo(() => Math.min(...samples.map((s) => s.y)) - 1, [samples]);
  const yMax = useMemo(() => Math.max(...samples.map((s) => s.y)) + 2, [samples]);

  const toSvgX = useCallback((x: number) => PAD1.left + ((x - xMin) / (xMax - xMin)) * PW1, []);
  const toSvgY = useCallback(
    (y: number) => PAD1.top + (1 - (y - yMin) / (yMax - yMin)) * PH1,
    [yMin, yMax]
  );

  const curvePath = useMemo(() => {
    return samples
      .map((s, i) => `${i === 0 ? "M" : "L"} ${toSvgX(s.x).toFixed(2)} ${toSvgY(s.y).toFixed(2)}`)
      .join(" ");
  }, [samples, toSvgX, toSvgY]);

  const takeStep = useCallback(() => {
    const x = posRef.current;
    const g = gradLoss1D(x);
    const newX = clamp(x - lrRef.current * g, xMin + 0.1, xMax - 0.1);
    setGradVal(g);
    setPos(newX);
    posRef.current = newX;
    setTrail((prev) => [...prev.slice(-150), { x: newX, loss: loss1D(newX) }]);
    setStepCount((s) => s + 1);
  }, []);

  useEffect(() => {
    if (!isRunning) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    let last = 0;
    const tick = (t: number) => {
      if (t - last > 120) {
        takeStep();
        last = t;
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isRunning, takeStep]);

  const reset = () => {
    setIsRunning(false);
    const r = rng();
    const startX = xMin + 0.5 + r * (xMax - xMin - 1);
    setPos(startX);
    posRef.current = startX;
    setTrail([{ x: startX, loss: loss1D(startX) }]);
    setStepCount(0);
    setGradVal(0);
  };

  const currentLoss = loss1D(pos);

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
        <p className="text-xs text-blue-800">
          Gradient descent moves downhill by computing the slope (gradient) and taking a step
          proportional to the <strong>learning rate</strong>. Press <strong>Take Step</strong> to
          advance one step, or <strong>Auto-Run</strong> to animate. The curve has bumps — watch
          how the ball navigates them!
        </p>
      </div>

      <div className="flex gap-5 flex-col lg:flex-row">
        {/* SVG chart */}
        <div className="flex-1 min-w-0 bg-slate-50 border border-slate-200 rounded-xl p-3">
          <svg viewBox={`0 0 ${W1} ${H1}`} className="w-full" style={{ maxHeight: 360 }}>
            {/* Grid lines */}
            {[0, 5, 10, 15, 20, 25, 30].map((v) => {
              if (v < yMin || v > yMax) return null;
              const sy = toSvgY(v);
              return (
                <g key={`g-${v}`}>
                  <line
                    x1={PAD1.left}
                    y1={sy}
                    x2={W1 - PAD1.right}
                    y2={sy}
                    stroke="#e2e8f0"
                    strokeWidth={0.8}
                  />
                  <text x={PAD1.left - 8} y={sy + 4} fontSize={10} fill="#94a3b8" textAnchor="end">
                    {v}
                  </text>
                </g>
              );
            })}

            {/* X-axis ticks */}
            {Array.from({ length: 13 }, (_, i) => xMin + i).map((v) => {
              const sx = toSvgX(v);
              return (
                <text key={`xt-${v}`} x={sx} y={H1 - 8} fontSize={10} fill="#94a3b8" textAnchor="middle">
                  {v}
                </text>
              );
            })}

            {/* Axis labels */}
            <text x={W1 / 2} y={H1 - 0} fontSize={11} fill="#64748b" textAnchor="middle" fontWeight={600}>
              x
            </text>
            <text
              x={12}
              y={H1 / 2}
              fontSize={11}
              fill="#64748b"
              textAnchor="middle"
              fontWeight={600}
              transform={`rotate(-90, 12, ${H1 / 2})`}
            >
              Loss f(x)
            </text>

            {/* Loss curve */}
            <path d={curvePath} fill="none" stroke="#6366f1" strokeWidth={2.5} />

            {/* Trail line */}
            {trail.length > 1 && (
              <polyline
                points={trail.map((p) => `${toSvgX(p.x).toFixed(1)},${toSvgY(p.loss).toFixed(1)}`).join(" ")}
                fill="none"
                stroke="#f97316"
                strokeWidth={1.2}
                opacity={0.5}
                strokeLinejoin="round"
              />
            )}

            {/* Trail dots */}
            {trail.map((p, i) => {
              const opacity = 0.15 + (i / trail.length) * 0.85;
              return (
                <circle
                  key={i}
                  cx={toSvgX(p.x)}
                  cy={toSvgY(p.loss)}
                  r={2.5}
                  fill="#f97316"
                  opacity={opacity}
                />
              );
            })}

            {/* Current position ball */}
            <circle
              cx={toSvgX(pos)}
              cy={toSvgY(currentLoss)}
              r={7}
              fill="#ef4444"
              stroke="#fff"
              strokeWidth={2.5}
            >
              <animate attributeName="r" values="6;8;6" dur="1.2s" repeatCount="indefinite" />
            </circle>

            {/* Gradient arrow */}
            {stepCount > 0 && (
              <>
                <defs>
                  <marker id="arrow1d" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#22c55e" />
                  </marker>
                </defs>
                <line
                  x1={toSvgX(pos)}
                  y1={toSvgY(currentLoss) - 14}
                  x2={toSvgX(pos) - clamp(gradVal * 3, -60, 60)}
                  y2={toSvgY(currentLoss) - 14}
                  stroke="#22c55e"
                  strokeWidth={2}
                  markerEnd="url(#arrow1d)"
                  opacity={0.8}
                />
                <text
                  x={toSvgX(pos)}
                  y={toSvgY(currentLoss) - 22}
                  fontSize={9}
                  fill="#22c55e"
                  textAnchor="middle"
                  fontWeight={600}
                >
                  grad = {gradVal.toFixed(2)}
                </text>
              </>
            )}

            {/* Minimum region indicator */}
            <circle cx={toSvgX(3)} cy={toSvgY(loss1D(3))} r={4} fill="#10b981" opacity={0.5} />
            <text x={toSvgX(3) + 8} y={toSvgY(loss1D(3)) - 4} fontSize={9} fill="#10b981" fontWeight={600}>
              global min
            </text>
          </svg>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-64 space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium">Steps</p>
              <p className="text-xl font-bold text-slate-800">{stepCount}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium">Loss</p>
              <p className="text-xl font-bold text-slate-800">
                {currentLoss < 0.01 ? currentLoss.toExponential(1) : currentLoss.toFixed(3)}
              </p>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-2.5">
            <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium mb-1">Position</p>
            <p className="text-sm font-mono text-slate-700">x = {pos.toFixed(4)}</p>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-2.5">
            <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium mb-1">Gradient</p>
            <p className="text-sm font-mono text-slate-700">df/dx = {gradVal.toFixed(4)}</p>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-2.5">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-slate-700">Learning Rate</label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded">
                {lr.toFixed(3)}
              </span>
            </div>
            <input
              type="range"
              min={0.001}
              max={1.0}
              step={0.001}
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-500"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
              <span>0.001</span>
              <span>1.0</span>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-sm font-semibold transition-all ${
                isRunning
                  ? "bg-slate-800 text-white hover:bg-slate-700"
                  : "bg-red-500 text-white hover:bg-red-600"
              }`}
            >
              {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isRunning ? "Pause" : "Auto-Run"}
            </button>
            <button
              onClick={() => !isRunning && takeStep()}
              disabled={isRunning}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 transition-all"
              title="Take one step"
            >
              <SkipForward className="w-4 h-4" />
            </button>
            <button
              onClick={reset}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all"
              title="Reset to random start"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-2.5">
            <p className="text-[10px] text-amber-800 font-semibold uppercase mb-1">Experiments</p>
            <ul className="text-xs text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Set learning rate to 0.8 — does it diverge?</li>
              <li>Try 0.01 — observe slow convergence</li>
              <li>Click Reset for a new random start</li>
              <li>Watch the gradient arrow change direction</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Tab 2 — Learning Rate Effect (3 simultaneous runners)
// ═══════════════════════════════════════════════════════════════════════
const W2 = 560;
const H2_TOP = 260;
const H2_BOT = 200;
const PAD2 = { top: 20, right: 20, bottom: 35, left: 55 };
const PW2 = W2 - PAD2.left - PAD2.right;
const PH2_TOP = H2_TOP - PAD2.top - PAD2.bottom;
const PH2_BOT = H2_BOT - PAD2.top - PAD2.bottom;

interface LRRunner {
  x: number;
  lossHistory: number[];
  converged: boolean;
  diverged: boolean;
}

function loss2LR(x: number): number {
  return (x - 3) * (x - 3) + 1.5 * Math.sin(2.5 * x) + 5;
}
function gradLoss2LR(x: number): number {
  return 2 * (x - 3) + 3.75 * Math.cos(2.5 * x);
}

const LR_COLORS = ["#3b82f6", "#22c55e", "#ef4444"];
const LR_LABELS = ["Small (slow)", "Medium (good)", "Large (unstable)"];

function Tab2LearningRate() {
  const [lrSmall, setLrSmall] = useState(0.005);
  const [lrMed, setLrMed] = useState(0.05);
  const [lrLarge, setLrLarge] = useState(0.4);
  const lrs = useMemo(() => [lrSmall, lrMed, lrLarge], [lrSmall, lrMed, lrLarge]);

  const startX = -2.0;

  const initRunners = useCallback(
    (): LRRunner[] =>
      [0, 1, 2].map(() => ({
        x: startX,
        lossHistory: [loss2LR(startX)],
        converged: false,
        diverged: false,
      })),
    []
  );

  const [runners, setRunners] = useState<LRRunner[]>(initRunners);
  const [stepCount, setStepCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const animRef = useRef<number | null>(null);
  const runnersRef = useRef(runners);
  runnersRef.current = runners;
  const lrsRef = useRef(lrs);
  lrsRef.current = lrs;
  const stepCountRef = useRef(stepCount);
  stepCountRef.current = stepCount;

  const xMin = -4;
  const xMax = 8;
  const curveSamples = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= 300; i++) {
      const x = xMin + (i / 300) * (xMax - xMin);
      pts.push({ x, y: loss2LR(x) });
    }
    return pts;
  }, []);

  const yMin = useMemo(() => Math.min(...curveSamples.map((s) => s.y)) - 1, [curveSamples]);
  const yMax = useMemo(() => Math.max(...curveSamples.map((s) => s.y)) + 3, [curveSamples]);

  const toX = useCallback((x: number) => PAD2.left + ((x - xMin) / (xMax - xMin)) * PW2, []);
  const toY = useCallback(
    (y: number) => PAD2.top + (1 - (y - yMin) / (yMax - yMin)) * PH2_TOP,
    [yMin, yMax]
  );

  const curvePath = useMemo(
    () =>
      curveSamples
        .map((s, i) => `${i === 0 ? "M" : "L"} ${toX(s.x).toFixed(1)} ${toY(s.y).toFixed(1)}`)
        .join(" "),
    [curveSamples, toX, toY]
  );

  const doStep = useCallback(() => {
    if (stepCountRef.current >= 200) {
      setIsRunning(false);
      return;
    }
    setRunners((prev) =>
      prev.map((r, idx) => {
        if (r.converged || r.diverged) return r;
        const g = gradLoss2LR(r.x);
        let newX = r.x - lrsRef.current[idx] * g;
        let diverged = false;
        if (Math.abs(newX) > 50 || !isFinite(newX)) {
          diverged = true;
          newX = r.x;
        }
        const newLoss = loss2LR(newX);
        const converged = Math.abs(g) < 0.01;
        return {
          x: newX,
          lossHistory: [...r.lossHistory, newLoss],
          converged,
          diverged,
        };
      })
    );
    setStepCount((s) => s + 1);
  }, []);

  useEffect(() => {
    if (!isRunning) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    let last = 0;
    const tick = (t: number) => {
      if (t - last > 80) {
        doStep();
        last = t;
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isRunning, doStep]);

  const reset = () => {
    setIsRunning(false);
    setRunners(initRunners());
    setStepCount(0);
  };

  // Loss history chart bounds
  const maxHistoryLoss = useMemo(() => {
    let mx = 10;
    runners.forEach((r) => {
      r.lossHistory.forEach((l) => {
        if (isFinite(l) && l < 200) mx = Math.max(mx, l);
      });
    });
    return mx + 2;
  }, [runners]);

  const histToX = useCallback((step: number) => PAD2.left + (step / 200) * PW2, []);
  const histToY = useCallback(
    (l: number) => PAD2.top + (1 - clamp(l, 0, maxHistoryLoss) / maxHistoryLoss) * PH2_BOT,
    [maxHistoryLoss]
  );

  return (
    <div className="space-y-4">
      <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-green-600 shrink-0 mt-0.5" />
        <p className="text-xs text-green-800">
          Three gradient descent runners start at the same point but use different learning rates.
          <strong> Blue</strong> = small (slow but steady), <strong>Green</strong> = medium (just
          right), <strong>Red</strong> = large (fast but may oscillate or diverge). Watch the
          loss-vs-iteration chart below to compare convergence.
        </p>
      </div>

      {/* LR sliders */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { val: lrSmall, set: setLrSmall, min: 0.001, max: 0.05, color: "blue" },
          { val: lrMed, set: setLrMed, min: 0.01, max: 0.2, color: "green" },
          { val: lrLarge, set: setLrLarge, min: 0.1, max: 1.0, color: "red" },
        ].map((cfg, i) => (
          <div key={i} className="bg-white border border-slate-200 rounded-lg p-2.5">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-semibold uppercase" style={{ color: LR_COLORS[i] }}>
                {LR_LABELS[i]}
              </span>
              <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded">
                {cfg.val.toFixed(3)}
              </span>
            </div>
            <input
              type="range"
              min={cfg.min}
              max={cfg.max}
              step={0.001}
              value={cfg.val}
              onChange={(e) => cfg.set(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              style={{ accentColor: LR_COLORS[i] }}
            />
          </div>
        ))}
      </div>

      {/* Buttons */}
      <div className="flex gap-2">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
            isRunning ? "bg-slate-800 text-white" : "bg-green-600 text-white hover:bg-green-700"
          }`}
        >
          {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isRunning ? "Pause" : "Run All"}
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 text-sm font-medium"
        >
          <RotateCcw className="w-4 h-4 inline mr-1" />
          Reset
        </button>
        <span className="self-center text-xs text-slate-500 ml-auto">
          Step: {stepCount} / 200
        </span>
      </div>

      {/* Top chart — loss curve with runners */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
        <p className="text-xs font-semibold text-slate-600 mb-1">Loss Landscape</p>
        <svg viewBox={`0 0 ${W2} ${H2_TOP}`} className="w-full" style={{ maxHeight: 300 }}>
          {/* Y grid */}
          {[0, 5, 10, 15, 20, 25, 30].map((v) => {
            if (v < yMin || v > yMax) return null;
            const sy = toY(v);
            return (
              <g key={v}>
                <line x1={PAD2.left} y1={sy} x2={W2 - PAD2.right} y2={sy} stroke="#e2e8f0" strokeWidth={0.7} />
                <text x={PAD2.left - 6} y={sy + 3} fontSize={9} fill="#94a3b8" textAnchor="end">
                  {v}
                </text>
              </g>
            );
          })}

          {/* Curve */}
          <path d={curvePath} fill="none" stroke="#6366f1" strokeWidth={2} opacity={0.6} />

          {/* Runner trails + balls */}
          {runners.map((r, idx) => {
            const history = r.lossHistory;
            if (history.length < 2) return null;
            // Build path on curve
            const pts: string[] = [];
            // Reconstruct x history from losses — we need to store x history too
            // Actually we only store current x. Let's re-derive: we store full history in lossHistory
            // but x changes. We need x trail. Let's just show current position + loss chart.
            return null;
          })}

          {/* Runner current positions */}
          {runners.map((r, idx) => {
            const sx = toX(clamp(r.x, xMin, xMax));
            const sy = toY(clamp(loss2LR(r.x), yMin, yMax));
            return (
              <g key={idx}>
                <circle cx={sx} cy={sy} r={7} fill={LR_COLORS[idx]} stroke="#fff" strokeWidth={2} opacity={0.9}>
                  <animate attributeName="r" values="6;8;6" dur="1.5s" repeatCount="indefinite" />
                </circle>
                {r.converged && (
                  <text x={sx + 10} y={sy - 4} fontSize={9} fill="#10b981" fontWeight={700}>
                    Converged!
                  </text>
                )}
                {r.diverged && (
                  <text x={sx + 10} y={sy - 4} fontSize={9} fill="#ef4444" fontWeight={700}>
                    Diverged!
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Bottom chart — loss vs iteration */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
        <p className="text-xs font-semibold text-slate-600 mb-1">Loss vs. Iteration</p>
        <svg viewBox={`0 0 ${W2} ${H2_BOT}`} className="w-full" style={{ maxHeight: 230 }}>
          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const y = PAD2.top + (1 - frac) * PH2_BOT;
            const val = frac * maxHistoryLoss;
            return (
              <g key={frac}>
                <line x1={PAD2.left} y1={y} x2={W2 - PAD2.right} y2={y} stroke="#e2e8f0" strokeWidth={0.7} />
                <text x={PAD2.left - 6} y={y + 3} fontSize={9} fill="#94a3b8" textAnchor="end">
                  {val.toFixed(0)}
                </text>
              </g>
            );
          })}
          {[0, 50, 100, 150, 200].map((s) => (
            <text key={s} x={histToX(s)} y={H2_BOT - 5} fontSize={9} fill="#94a3b8" textAnchor="middle">
              {s}
            </text>
          ))}

          {/* Loss lines for each runner */}
          {runners.map((r, idx) => {
            if (r.lossHistory.length < 2) return null;
            const path = r.lossHistory
              .map(
                (l, i) =>
                  `${i === 0 ? "M" : "L"} ${histToX(i).toFixed(1)} ${histToY(l).toFixed(1)}`
              )
              .join(" ");
            return (
              <path key={idx} d={path} fill="none" stroke={LR_COLORS[idx]} strokeWidth={2} opacity={0.8} />
            );
          })}

          {/* Legend */}
          {LR_LABELS.map((label, i) => (
            <g key={i} transform={`translate(${PAD2.left + i * 160}, ${PAD2.top - 8})`}>
              <rect width={12} height={3} fill={LR_COLORS[i]} rx={1} />
              <text x={16} y={3} fontSize={9} fill="#64748b">
                {label} (loss: {runners[i].lossHistory.length > 0 ? runners[i].lossHistory[runners[i].lossHistory.length - 1].toFixed(2) : "—"})
              </text>
            </g>
          ))}
        </svg>
      </div>

      {/* Status cards */}
      <div className="grid grid-cols-3 gap-3">
        {runners.map((r, i) => (
          <div key={i} className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
            <div className="flex items-center justify-center gap-1.5 mb-1">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: LR_COLORS[i] }} />
              <span className="text-[10px] font-semibold text-slate-600 uppercase">{LR_LABELS[i]}</span>
            </div>
            <p className="text-lg font-bold text-slate-800">
              {r.lossHistory.length > 0 ? r.lossHistory[r.lossHistory.length - 1].toFixed(3) : "—"}
            </p>
            <p className="text-[10px] text-slate-500">
              {r.converged ? "Converged" : r.diverged ? "Diverged" : `${r.lossHistory.length - 1} steps`}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Tab 3 — Momentum & Variants (2D contour plot)
// ═══════════════════════════════════════════════════════════════════════
const W3 = 480;
const H3 = 480;
const RANGE3 = 5;

interface Pt2D {
  x: number;
  y: number;
}

// Elliptical bowl: f(x,y) = 0.5*x^2 + 5*y^2 (elongated)
function lossMom(x: number, y: number): number {
  return 0.5 * x * x + 5 * y * y;
}
function gradMomX(x: number): number {
  return x;
}
function gradMomY(y: number): number {
  return 10 * y;
}

const CONTOUR_LEVELS_3 = [0.5, 1, 2, 4, 8, 14, 22, 32, 45];

function contourPath3(level: number): string {
  // 0.5*x^2 + 5*y^2 = level => x^2/(2*level) + y^2/(level/5) = 1
  const a = Math.sqrt(2 * level);
  const b = Math.sqrt(level / 5);
  const cx = W3 / 2;
  const cy = H3 / 2;
  const scale = W3 / (2 * RANGE3);
  const rx = a * scale;
  const ry = b * scale;
  if (rx > W3 || ry > H3) return "";
  return `M ${cx - rx} ${cy} A ${rx} ${ry} 0 1 1 ${cx + rx} ${cy} A ${rx} ${ry} 0 1 1 ${cx - rx} ${cy} Z`;
}

function toSvg3(x: number, y: number): [number, number] {
  return [((x + RANGE3) / (2 * RANGE3)) * W3, ((y + RANGE3) / (2 * RANGE3)) * H3];
}

interface OptimizerState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  sx: number;
  sy: number;
  mx: number;
  my: number;
  trail: Pt2D[];
  lossHistory: number[];
}

const OPT_COLORS = ["#3b82f6", "#f59e0b", "#8b5cf6", "#ef4444"];
const OPT_NAMES = ["Vanilla GD", "Momentum", "RMSProp", "Adam"];

function initOptState(x: number, y: number): OptimizerState {
  return {
    x,
    y,
    vx: 0,
    vy: 0,
    sx: 0,
    sy: 0,
    mx: 0,
    my: 0,
    trail: [{ x, y }],
    lossHistory: [lossMom(x, y)],
  };
}

function stepOptimizer(
  state: OptimizerState,
  optIdx: number,
  lr: number,
  t: number
): OptimizerState {
  const gx = gradMomX(state.x);
  const gy = gradMomY(state.y);
  let newX: number, newY: number;
  let vx = state.vx,
    vy = state.vy,
    sx = state.sx,
    sy = state.sy,
    mx = state.mx,
    my = state.my;

  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const rho = 0.9;

  switch (optIdx) {
    case 0: // Vanilla GD
      newX = state.x - lr * gx;
      newY = state.y - lr * gy;
      break;
    case 1: // Momentum
      vx = beta1 * vx - lr * gx;
      vy = beta1 * vy - lr * gy;
      newX = state.x + vx;
      newY = state.y + vy;
      break;
    case 2: // RMSProp
      sx = rho * sx + (1 - rho) * gx * gx;
      sy = rho * sy + (1 - rho) * gy * gy;
      newX = state.x - (lr / (Math.sqrt(sx) + eps)) * gx;
      newY = state.y - (lr / (Math.sqrt(sy) + eps)) * gy;
      break;
    case 3: {
      // Adam
      mx = beta1 * mx + (1 - beta1) * gx;
      my = beta1 * my + (1 - beta1) * gy;
      sx = beta2 * sx + (1 - beta2) * gx * gx;
      sy = beta2 * sy + (1 - beta2) * gy * gy;
      const mxHat = mx / (1 - Math.pow(beta1, t + 1));
      const myHat = my / (1 - Math.pow(beta1, t + 1));
      const sxHat = sx / (1 - Math.pow(beta2, t + 1));
      const syHat = sy / (1 - Math.pow(beta2, t + 1));
      newX = state.x - (lr / (Math.sqrt(sxHat) + eps)) * mxHat;
      newY = state.y - (lr / (Math.sqrt(syHat) + eps)) * myHat;
      break;
    }
    default:
      newX = state.x;
      newY = state.y;
  }

  newX = clamp(newX, -RANGE3, RANGE3);
  newY = clamp(newY, -RANGE3, RANGE3);
  const newLoss = lossMom(newX, newY);

  return {
    x: newX,
    y: newY,
    vx,
    vy,
    sx,
    sy,
    mx,
    my,
    trail: [...state.trail.slice(-300), { x: newX, y: newY }],
    lossHistory: [...state.lossHistory, newLoss],
  };
}

function Tab3Momentum() {
  const startPt: Pt2D = { x: -4.0, y: 2.5 };
  const [lr, setLr] = useState(0.08);
  const [speed, setSpeed] = useState(80);
  const [opts, setOpts] = useState<OptimizerState[]>(() =>
    [0, 1, 2, 3].map(() => initOptState(startPt.x, startPt.y))
  );
  const [stepCount, setStepCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [activeOpts, setActiveOpts] = useState([true, true, true, true]);
  const animRef = useRef<number | null>(null);
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const lrRef = useRef(lr);
  lrRef.current = lr;
  const stepRef = useRef(stepCount);
  stepRef.current = stepCount;
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const activeRef = useRef(activeOpts);
  activeRef.current = activeOpts;

  const doStep = useCallback(() => {
    setOpts((prev) =>
      prev.map((s, idx) => {
        if (!activeRef.current[idx]) return s;
        return stepOptimizer(s, idx, lrRef.current, stepRef.current);
      })
    );
    setStepCount((s) => s + 1);
  }, []);

  useEffect(() => {
    if (!isRunning) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    let last = 0;
    const tick = (t: number) => {
      if (t - last > speedRef.current) {
        if (stepRef.current < 300) {
          doStep();
        } else {
          setIsRunning(false);
        }
        last = t;
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isRunning, doStep]);

  const reset = () => {
    setIsRunning(false);
    setOpts([0, 1, 2, 3].map(() => initOptState(startPt.x, startPt.y)));
    setStepCount(0);
  };

  const toggleOpt = (idx: number) => {
    setActiveOpts((prev) => {
      const next = [...prev];
      next[idx] = !next[idx];
      return next;
    });
  };

  // Loss chart
  const maxLoss = useMemo(() => {
    let mx = 5;
    opts.forEach((o) => {
      o.lossHistory.forEach((l) => {
        if (isFinite(l) && l < 500) mx = Math.max(mx, l);
      });
    });
    return mx;
  }, [opts]);

  const H3_LOSS = 180;
  const lossToX = useCallback((s: number) => PAD2.left + (s / 300) * PW2, []);
  const lossToY = useCallback(
    (l: number) => PAD2.top + (1 - clamp(l, 0, maxLoss) / maxLoss) * (H3_LOSS - PAD2.top - PAD2.bottom),
    [maxLoss]
  );

  // Formula tooltips
  const formulas = [
    "x(t+1) = x(t) - lr * grad",
    "v(t+1) = beta*v(t) - lr*grad; x(t+1) = x(t) + v(t+1)",
    "s(t+1) = rho*s(t) + (1-rho)*grad^2; x -= lr*grad/sqrt(s+eps)",
    "m,v adapted; x -= lr*m_hat/sqrt(v_hat+eps)",
  ];

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-purple-600 shrink-0 mt-0.5" />
        <p className="text-xs text-purple-800">
          Compare four popular optimization algorithms on an elongated elliptical bowl.{" "}
          <strong>Vanilla GD</strong> oscillates in the steep direction.{" "}
          <strong>Momentum</strong> accelerates through flat regions.{" "}
          <strong>RMSProp</strong> adapts per-parameter.{" "}
          <strong>Adam</strong> combines momentum with adaptive rates.
        </p>
      </div>

      <div className="flex gap-5 flex-col xl:flex-row">
        {/* Contour plot */}
        <div className="flex-1 min-w-0 bg-slate-50 border border-slate-200 rounded-xl p-3">
          <p className="text-xs font-semibold text-slate-600 mb-1">
            Contour Plot: f(x,y) = 0.5x² + 5y²
          </p>
          <svg viewBox={`0 0 ${W3} ${W3}`} className="w-full" style={{ maxHeight: 440 }}>
            {/* Contours */}
            {CONTOUR_LEVELS_3.map((level, i) => {
              const p = contourPath3(level);
              if (!p) return null;
              return (
                <path
                  key={level}
                  d={p}
                  fill={`hsla(220, 50%, ${95 - i * 4}%, 0.4)`}
                  stroke={`hsl(220, 50%, ${75 - i * 5}%)`}
                  strokeWidth={1}
                />
              );
            })}

            {/* Axes */}
            <line x1={W3 / 2} y1={0} x2={W3 / 2} y2={W3} stroke="#cbd5e1" strokeWidth={0.5} strokeDasharray="4,4" />
            <line x1={0} y1={W3 / 2} x2={W3} y2={W3 / 2} stroke="#cbd5e1" strokeWidth={0.5} strokeDasharray="4,4" />

            {/* Minimum */}
            <circle cx={W3 / 2} cy={W3 / 2} r={4} fill="#10b981" opacity={0.7} />
            <text x={W3 / 2 + 8} y={W3 / 2 - 6} fontSize={10} fill="#10b981" fontWeight={600}>
              min (0,0)
            </text>

            {/* Optimizer trails */}
            {opts.map((o, idx) => {
              if (!activeOpts[idx]) return null;
              if (o.trail.length < 2) return null;
              const points = o.trail
                .map((p) => {
                  const [sx, sy] = toSvg3(p.x, p.y);
                  return `${sx.toFixed(1)},${sy.toFixed(1)}`;
                })
                .join(" ");
              return (
                <polyline
                  key={idx}
                  points={points}
                  fill="none"
                  stroke={OPT_COLORS[idx]}
                  strokeWidth={2}
                  opacity={0.7}
                  strokeLinejoin="round"
                />
              );
            })}

            {/* Optimizer current positions */}
            {opts.map((o, idx) => {
              if (!activeOpts[idx]) return null;
              const [sx, sy] = toSvg3(o.x, o.y);
              return (
                <g key={`pos-${idx}`}>
                  <circle cx={sx} cy={sy} r={6} fill={OPT_COLORS[idx]} stroke="#fff" strokeWidth={2}>
                    <animate attributeName="r" values="5;7;5" dur="1.5s" repeatCount="indefinite" />
                  </circle>
                </g>
              );
            })}

            {/* Start marker */}
            {(() => {
              const [sx, sy] = toSvg3(startPt.x, startPt.y);
              return (
                <g>
                  <circle cx={sx} cy={sy} r={5} fill="none" stroke="#64748b" strokeWidth={2} strokeDasharray="3,2" />
                  <text x={sx + 8} y={sy - 4} fontSize={9} fill="#64748b" fontWeight={600}>
                    start
                  </text>
                </g>
              );
            })()}
          </svg>
        </div>

        {/* Controls */}
        <div className="w-full xl:w-72 space-y-3">
          {/* Optimizer toggles */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-600">Optimizers</p>
            {OPT_NAMES.map((name, i) => (
              <button
                key={i}
                onClick={() => toggleOpt(i)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-all ${
                  activeOpts[i]
                    ? "bg-white border-slate-300 text-slate-800"
                    : "bg-slate-100 border-slate-200 text-slate-400"
                }`}
              >
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: activeOpts[i] ? OPT_COLORS[i] : "#cbd5e1" }}
                />
                <span className="flex-1 text-left">{name}</span>
                <span className="text-[9px] text-slate-400 font-mono text-right" style={{ maxWidth: 130 }}>
                  {formulas[i]}
                </span>
              </button>
            ))}
          </div>

          {/* LR slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-2.5">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-slate-700">Learning Rate</label>
              <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded">{lr.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min={0.001}
              max={0.3}
              step={0.001}
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
          </div>

          {/* Speed slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-2.5">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-slate-700">Animation Speed</label>
              <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded">{speed}ms</span>
            </div>
            <input
              type="range"
              min={20}
              max={200}
              step={5}
              value={speed}
              onChange={(e) => {
                setSpeed(parseInt(e.target.value));
                speedRef.current = parseInt(e.target.value);
              }}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
              <span>Fast</span>
              <span>Slow</span>
            </div>
          </div>

          {/* Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-sm font-semibold transition-all ${
                isRunning ? "bg-slate-800 text-white" : "bg-purple-600 text-white hover:bg-purple-700"
              }`}
            >
              {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isRunning ? "Pause" : "Start"}
            </button>
            <button
              onClick={reset}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <p className="text-[10px] text-slate-400 text-center">Step: {stepCount} / 300</p>

          {/* Metrics cards */}
          <div className="grid grid-cols-2 gap-2">
            {opts.map((o, i) => {
              if (!activeOpts[i]) return null;
              const currLoss = o.lossHistory[o.lossHistory.length - 1];
              return (
                <div key={i} className="bg-white border border-slate-200 rounded-lg p-2 text-center">
                  <div className="flex items-center justify-center gap-1 mb-0.5">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: OPT_COLORS[i] }} />
                    <span className="text-[9px] font-semibold text-slate-600">{OPT_NAMES[i]}</span>
                  </div>
                  <p className="text-sm font-bold text-slate-800">
                    {currLoss < 0.001 ? currLoss.toExponential(1) : currLoss.toFixed(3)}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Loss vs iteration chart */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
        <p className="text-xs font-semibold text-slate-600 mb-1">Loss vs. Iteration</p>
        <svg viewBox={`0 0 ${W2} ${H3_LOSS}`} className="w-full" style={{ maxHeight: 200 }}>
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const y = PAD2.top + (1 - frac) * (H3_LOSS - PAD2.top - PAD2.bottom);
            const val = frac * maxLoss;
            return (
              <g key={frac}>
                <line x1={PAD2.left} y1={y} x2={W2 - PAD2.right} y2={y} stroke="#e2e8f0" strokeWidth={0.5} />
                <text x={PAD2.left - 6} y={y + 3} fontSize={8} fill="#94a3b8" textAnchor="end">
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}
          {opts.map((o, idx) => {
            if (!activeOpts[idx] || o.lossHistory.length < 2) return null;
            const path = o.lossHistory
              .map((l, i) => `${i === 0 ? "M" : "L"} ${lossToX(i).toFixed(1)} ${lossToY(l).toFixed(1)}`)
              .join(" ");
            return <path key={idx} d={path} fill="none" stroke={OPT_COLORS[idx]} strokeWidth={1.5} opacity={0.8} />;
          })}
        </svg>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Tab 4 — Stochastic vs Batch GD
// ═══════════════════════════════════════════════════════════════════════
const W4 = 520;
const H4 = 300;
const PAD4 = { top: 20, right: 20, bottom: 35, left: 50 };
const PW4 = W4 - PAD4.left - PAD4.right;
const PH4 = H4 - PAD4.top - PAD4.bottom;

interface DataPoint {
  x: number;
  y: number;
}

function Tab4Stochastic() {
  const rng = useMemo(() => mulberry32(123), []);

  // Generate dataset: y = 2x + 1 + noise
  const dataset: DataPoint[] = useMemo(() => {
    const pts: DataPoint[] = [];
    for (let i = 0; i < 50; i++) {
      const x = -3 + rng() * 6;
      const noise = (rng() - 0.5) * 2.5;
      pts.push({ x, y: 2 * x + 1 + noise });
    }
    return pts;
  }, [rng]);

  // State: w (weight) and b (bias) for y = w*x + b
  const [batchSize, setBatchSize] = useState(10);
  const [lr, setLr] = useState(0.02);
  const [isRunning, setIsRunning] = useState(false);
  const [stepCount, setStepCount] = useState(0);

  // 3 modes: Full batch, Mini-batch, SGD
  interface GDState {
    w: number;
    b: number;
    lossHistory: number[];
    currentBatch: number[]; // indices
  }

  const initState = useCallback(
    (): GDState => ({
      w: -1 + rng() * 2,
      b: rng() * 2 - 1,
      lossHistory: [],
      currentBatch: [],
    }),
    [rng]
  );

  const [states, setStates] = useState<GDState[]>(() => {
    const seed = mulberry32(999);
    const w0 = -0.5;
    const b0 = -1.5;
    return [
      { w: w0, b: b0, lossHistory: [], currentBatch: [] },
      { w: w0, b: b0, lossHistory: [], currentBatch: [] },
      { w: w0, b: b0, lossHistory: [], currentBatch: [] },
    ];
  });

  const animRef = useRef<number | null>(null);
  const statesRef = useRef(states);
  statesRef.current = states;
  const lrRef = useRef(lr);
  lrRef.current = lr;
  const batchRef = useRef(batchSize);
  batchRef.current = batchSize;
  const stepRef = useRef(stepCount);
  stepRef.current = stepCount;

  const rngStep = useMemo(() => mulberry32(777), []);

  const computeLoss = useCallback(
    (w: number, b: number, indices: number[]): number => {
      let sum = 0;
      for (const idx of indices) {
        const p = dataset[idx];
        const pred = w * p.x + b;
        sum += (pred - p.y) ** 2;
      }
      return sum / indices.length;
    },
    [dataset]
  );

  const computeGrad = useCallback(
    (w: number, b: number, indices: number[]): { dw: number; db: number } => {
      let dw = 0;
      let db = 0;
      for (const idx of indices) {
        const p = dataset[idx];
        const pred = w * p.x + b;
        const err = pred - p.y;
        dw += (2 * err * p.x) / indices.length;
        db += (2 * err) / indices.length;
      }
      return { dw, db };
    },
    [dataset]
  );

  const allIndices = useMemo(() => Array.from({ length: dataset.length }, (_, i) => i), [dataset]);

  const doStep = useCallback(() => {
    if (stepRef.current >= 300) {
      setIsRunning(false);
      return;
    }

    const bs = batchRef.current;

    setStates((prev) =>
      prev.map((s, modeIdx) => {
        let batchIndices: number[];
        if (modeIdx === 0) {
          // Full batch
          batchIndices = allIndices;
        } else if (modeIdx === 1) {
          // Mini-batch
          batchIndices = [];
          const shuffled = [...allIndices];
          for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(rngStep() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
          }
          batchIndices = shuffled.slice(0, bs);
        } else {
          // SGD (single point)
          batchIndices = [Math.floor(rngStep() * dataset.length)];
        }

        const { dw, db } = computeGrad(s.w, s.b, batchIndices);
        const newW = s.w - lrRef.current * dw;
        const newB = s.b - lrRef.current * db;
        const fullLoss = computeLoss(newW, newB, allIndices);

        return {
          w: newW,
          b: newB,
          lossHistory: [...s.lossHistory, fullLoss],
          currentBatch: batchIndices,
        };
      })
    );
    setStepCount((s) => s + 1);
  }, [allIndices, computeGrad, computeLoss, dataset.length, rngStep]);

  useEffect(() => {
    if (!isRunning) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    let last = 0;
    const tick = (t: number) => {
      if (t - last > 100) {
        doStep();
        last = t;
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isRunning, doStep]);

  const reset = () => {
    setIsRunning(false);
    const w0 = -0.5;
    const b0 = -1.5;
    setStates([
      { w: w0, b: b0, lossHistory: [], currentBatch: [] },
      { w: w0, b: b0, lossHistory: [], currentBatch: [] },
      { w: w0, b: b0, lossHistory: [], currentBatch: [] },
    ]);
    setStepCount(0);
  };

  const MODE_COLORS = ["#3b82f6", "#f59e0b", "#ef4444"];
  const MODE_NAMES = ["Full Batch", `Mini-Batch (${batchSize})`, "SGD (1 pt)"];

  // Data range
  const dataXMin = -4;
  const dataXMax = 4;
  const dataYMin = -10;
  const dataYMax = 12;

  const dataSvgX = useCallback((x: number) => PAD4.left + ((x - dataXMin) / (dataXMax - dataXMin)) * PW4, []);
  const dataSvgY = useCallback((y: number) => PAD4.top + (1 - (y - dataYMin) / (dataYMax - dataYMin)) * PH4, []);

  // Loss chart
  const maxLoss = useMemo(() => {
    let mx = 5;
    states.forEach((s) => {
      s.lossHistory.forEach((l) => {
        if (isFinite(l) && l < 500) mx = Math.max(mx, l);
      });
    });
    return mx;
  }, [states]);

  const H4_LOSS = 180;
  const lossX = useCallback((s: number) => PAD4.left + (s / 300) * PW4, []);
  const lossY = useCallback(
    (l: number) => PAD4.top + (1 - clamp(l, 0, maxLoss) / maxLoss) * (H4_LOSS - PAD4.top - PAD4.bottom),
    [maxLoss]
  );

  return (
    <div className="space-y-4">
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <p className="text-xs text-amber-800">
          Compare how different batch strategies affect gradient descent on a linear regression problem.{" "}
          <strong>Full Batch</strong> uses all 50 points (smooth but slow per step).{" "}
          <strong>Mini-Batch</strong> uses a subset (good balance).{" "}
          <strong>SGD</strong> uses a single random point (noisy but fast updates).
          Highlighted dots show which data points each mode uses per step.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 items-end">
        <div className="bg-white border border-slate-200 rounded-lg p-2.5 flex-1 min-w-45">
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs font-semibold text-slate-700">Mini-Batch Size</label>
            <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded">{batchSize}</span>
          </div>
          <input
            type="range"
            min={2}
            max={50}
            step={1}
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
            <span>2</span>
            <span>50 (= full)</span>
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-2.5 flex-1 min-w-[180px]">
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs font-semibold text-slate-700">Learning Rate</label>
            <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded">{lr.toFixed(3)}</span>
          </div>
          <input
            type="range"
            min={0.001}
            max={0.1}
            step={0.001}
            value={lr}
            onChange={(e) => setLr(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              isRunning ? "bg-slate-800 text-white" : "bg-amber-500 text-white hover:bg-amber-600"
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? "Pause" : "Run"}
          </button>
          <button
            onClick={reset}
            className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 text-sm"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <span className="self-center text-xs text-slate-500">Step: {stepCount}/300</span>
        </div>
      </div>

      {/* 3 scatter plots side by side */}
      <div className="grid grid-cols-3 gap-3">
        {states.map((s, modeIdx) => (
          <div key={modeIdx} className="bg-slate-50 border border-slate-200 rounded-xl p-2">
            <div className="flex items-center gap-1.5 mb-1">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MODE_COLORS[modeIdx] }} />
              <p className="text-[10px] font-semibold text-slate-600">{MODE_NAMES[modeIdx]}</p>
            </div>
            <svg viewBox={`0 0 ${W4} ${H4}`} className="w-full" style={{ maxHeight: 240 }}>
              {/* Grid */}
              <line x1={PAD4.left} y1={PAD4.top} x2={PAD4.left} y2={H4 - PAD4.bottom} stroke="#e2e8f0" strokeWidth={0.7} />
              <line x1={PAD4.left} y1={H4 - PAD4.bottom} x2={W4 - PAD4.right} y2={H4 - PAD4.bottom} stroke="#e2e8f0" strokeWidth={0.7} />

              {/* Data points */}
              {dataset.map((p, i) => {
                const inBatch = s.currentBatch.includes(i);
                return (
                  <circle
                    key={i}
                    cx={dataSvgX(p.x)}
                    cy={dataSvgY(p.y)}
                    r={inBatch ? 5 : 3}
                    fill={inBatch ? MODE_COLORS[modeIdx] : "#94a3b8"}
                    opacity={inBatch ? 1 : 0.3}
                    stroke={inBatch ? "#fff" : "none"}
                    strokeWidth={inBatch ? 1.5 : 0}
                  />
                );
              })}

              {/* Regression line */}
              {(() => {
                const x1 = dataXMin;
                const x2 = dataXMax;
                const y1c = clamp(s.w * x1 + s.b, dataYMin, dataYMax);
                const y2c = clamp(s.w * x2 + s.b, dataYMin, dataYMax);
                return (
                  <line
                    x1={dataSvgX(x1)}
                    y1={dataSvgY(y1c)}
                    x2={dataSvgX(x2)}
                    y2={dataSvgY(y2c)}
                    stroke={MODE_COLORS[modeIdx]}
                    strokeWidth={2.5}
                    opacity={0.8}
                  />
                );
              })()}

              {/* True line */}
              {(() => {
                const x1 = dataXMin;
                const x2 = dataXMax;
                const y1c = 2 * x1 + 1;
                const y2c = 2 * x2 + 1;
                return (
                  <line
                    x1={dataSvgX(x1)}
                    y1={dataSvgY(y1c)}
                    x2={dataSvgX(x2)}
                    y2={dataSvgY(y2c)}
                    stroke="#10b981"
                    strokeWidth={1.5}
                    strokeDasharray="4,3"
                    opacity={0.6}
                  />
                );
              })()}

              {/* Labels */}
              <text x={PAD4.left + 4} y={PAD4.top + 12} fontSize={9} fill="#64748b">
                w={s.w.toFixed(2)}, b={s.b.toFixed(2)}
              </text>
            </svg>
          </div>
        ))}
      </div>

      {/* Loss convergence chart */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
        <p className="text-xs font-semibold text-slate-600 mb-1">Convergence: Loss vs. Iteration</p>
        <svg viewBox={`0 0 ${W4} ${H4_LOSS}`} className="w-full" style={{ maxHeight: 200 }}>
          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const y = PAD4.top + (1 - frac) * (H4_LOSS - PAD4.top - PAD4.bottom);
            return (
              <g key={frac}>
                <line x1={PAD4.left} y1={y} x2={W4 - PAD4.right} y2={y} stroke="#e2e8f0" strokeWidth={0.5} />
                <text x={PAD4.left - 6} y={y + 3} fontSize={8} fill="#94a3b8" textAnchor="end">
                  {(frac * maxLoss).toFixed(1)}
                </text>
              </g>
            );
          })}
          {[0, 100, 200, 300].map((s) => (
            <text key={s} x={lossX(s)} y={H4_LOSS - 5} fontSize={8} fill="#94a3b8" textAnchor="middle">
              {s}
            </text>
          ))}

          {states.map((s, idx) => {
            if (s.lossHistory.length < 2) return null;
            const path = s.lossHistory
              .map((l, i) => `${i === 0 ? "M" : "L"} ${lossX(i).toFixed(1)} ${lossY(l).toFixed(1)}`)
              .join(" ");
            return <path key={idx} d={path} fill="none" stroke={MODE_COLORS[idx]} strokeWidth={1.5} opacity={0.8} />;
          })}

          {/* Legend */}
          {MODE_NAMES.map((name, i) => (
            <g key={i} transform={`translate(${PAD4.left + i * 160}, ${PAD4.top - 6})`}>
              <rect width={12} height={3} fill={MODE_COLORS[i]} rx={1} />
              <text x={16} y={3} fontSize={8} fill="#64748b">
                {name}
              </text>
            </g>
          ))}
        </svg>
      </div>

      {/* Status */}
      <div className="grid grid-cols-3 gap-3">
        {states.map((s, i) => (
          <div key={i} className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
            <div className="flex items-center justify-center gap-1.5 mb-1">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MODE_COLORS[i] }} />
              <span className="text-[10px] font-semibold text-slate-600">{MODE_NAMES[i]}</span>
            </div>
            <p className="text-lg font-bold text-slate-800">
              {s.lossHistory.length > 0 ? s.lossHistory[s.lossHistory.length - 1].toFixed(3) : "—"}
            </p>
            <p className="text-[10px] text-slate-500 font-mono">
              w={s.w.toFixed(3)}, b={s.b.toFixed(3)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Tab 5 — Saddle Points & Local Minima
// ═══════════════════════════════════════════════════════════════════════
const W5 = 480;
const H5 = 480;
const RANGE5 = 2.5;

// f(x,y) = x^4 - 2x^2 + y^2  (saddle at origin, minima at (±1, 0))
function lossSaddle(x: number, y: number): number {
  return x * x * x * x - 2 * x * x + y * y;
}
function gradSaddleX(x: number): number {
  return 4 * x * x * x - 4 * x;
}
function gradSaddleY(y: number): number {
  return 2 * y;
}

function toSvg5(x: number, y: number): [number, number] {
  return [((x + RANGE5) / (2 * RANGE5)) * W5, ((-y + RANGE5) / (2 * RANGE5)) * H5];
}
function fromSvg5(sx: number, sy: number): [number, number] {
  return [(sx / W5) * 2 * RANGE5 - RANGE5, -((sy / H5) * 2 * RANGE5 - RANGE5)];
}

// Generate contour data with marching squares approximation
function generateContourPaths5(): { level: number; path: string }[] {
  const levels = [-1.0, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 1.0, 2.0, 3.5, 5.0];
  const resolution = 80;
  const results: { level: number; path: string }[] = [];

  for (const level of levels) {
    // Sample the grid and collect iso-line segments
    const segments: [number, number, number, number][] = [];
    const step = (2 * RANGE5) / resolution;

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x0 = -RANGE5 + i * step;
        const y0 = -RANGE5 + j * step;
        const x1 = x0 + step;
        const y1 = y0 + step;

        const v00 = lossSaddle(x0, y0) - level;
        const v10 = lossSaddle(x1, y0) - level;
        const v11 = lossSaddle(x1, y1) - level;
        const v01 = lossSaddle(x0, y1) - level;

        // Marching squares - simplified
        const code =
          (v00 > 0 ? 8 : 0) + (v10 > 0 ? 4 : 0) + (v11 > 0 ? 2 : 0) + (v01 > 0 ? 1 : 0);

        if (code === 0 || code === 15) continue;

        const interpX = (va: number, vb: number, a: number, b: number) =>
          va === vb ? (a + b) / 2 : a + (0 - va) * (b - a) / (vb - va);

        // Edge midpoints: top(v00-v10), right(v10-v11), bottom(v01-v11), left(v00-v01)
        const top = [interpX(v00, v10, x0, x1), y0] as const;
        const right = [x1, interpX(v10, v11, y0, y1)] as const;
        const bottom = [interpX(v01, v11, x0, x1), y1] as const;
        const left = [x0, interpX(v00, v01, y0, y1)] as const;

        const addSeg = (a: readonly [number, number], b: readonly [number, number]) => {
          const [sx1, sy1] = toSvg5(a[0], a[1]);
          const [sx2, sy2] = toSvg5(b[0], b[1]);
          segments.push([sx1, sy1, sx2, sy2]);
        };

        switch (code) {
          case 1:
          case 14:
            addSeg(bottom, left);
            break;
          case 2:
          case 13:
            addSeg(right, bottom);
            break;
          case 3:
          case 12:
            addSeg(right, left);
            break;
          case 4:
          case 11:
            addSeg(top, right);
            break;
          case 5:
            addSeg(top, left);
            addSeg(right, bottom);
            break;
          case 6:
          case 9:
            addSeg(top, bottom);
            break;
          case 7:
          case 8:
            addSeg(top, left);
            break;
          case 10:
            addSeg(top, right);
            addSeg(bottom, left);
            break;
        }
      }
    }

    // Convert segments to a path
    if (segments.length > 0) {
      const pathParts = segments.map(
        ([x1, y1, x2, y2]) =>
          `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)}`
      );
      results.push({ level, path: pathParts.join(" ") });
    }
  }

  return results;
}

interface SaddleRunner {
  x: number;
  y: number;
  vx: number;
  vy: number;
  trail: Pt2D[];
  lossHistory: number[];
  label: string;
  color: string;
}

function Tab5SaddlePoints() {
  const contours = useMemo(() => generateContourPaths5(), []);

  const [startPt, setStartPt] = useState<Pt2D>({ x: 0.05, y: 1.5 });
  const [lr, setLr] = useState(0.01);
  const [isRunning, setIsRunning] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [showGradArrows, setShowGradArrows] = useState(true);

  const initRunners = useCallback(
    (pt: Pt2D): SaddleRunner[] => [
      {
        x: pt.x,
        y: pt.y,
        vx: 0,
        vy: 0,
        trail: [{ x: pt.x, y: pt.y }],
        lossHistory: [lossSaddle(pt.x, pt.y)],
        label: "Vanilla GD",
        color: "#3b82f6",
      },
      {
        x: pt.x,
        y: pt.y,
        vx: 0,
        vy: 0,
        trail: [{ x: pt.x, y: pt.y }],
        lossHistory: [lossSaddle(pt.x, pt.y)],
        label: "Momentum (0.9)",
        color: "#ef4444",
      },
    ],
    []
  );

  const [runners, setRunners] = useState<SaddleRunner[]>(() => initRunners(startPt));
  const animRef = useRef<number | null>(null);
  const runnersRef = useRef(runners);
  runnersRef.current = runners;
  const lrRef = useRef(lr);
  lrRef.current = lr;
  const stepRef = useRef(stepCount);
  stepRef.current = stepCount;

  const doStep = useCallback(() => {
    if (stepRef.current >= 500) {
      setIsRunning(false);
      return;
    }
    setRunners((prev) =>
      prev.map((r, idx) => {
        const gx = gradSaddleX(r.x);
        const gy = gradSaddleY(r.y);
        let newX: number, newY: number;
        let vx = r.vx,
          vy = r.vy;

        if (idx === 0) {
          // Vanilla GD
          newX = r.x - lrRef.current * gx;
          newY = r.y - lrRef.current * gy;
        } else {
          // Momentum
          const beta = 0.9;
          vx = beta * vx - lrRef.current * gx;
          vy = beta * vy - lrRef.current * gy;
          newX = r.x + vx;
          newY = r.y + vy;
        }

        newX = clamp(newX, -RANGE5, RANGE5);
        newY = clamp(newY, -RANGE5, RANGE5);

        return {
          ...r,
          x: newX,
          y: newY,
          vx,
          vy,
          trail: [...r.trail.slice(-400), { x: newX, y: newY }],
          lossHistory: [...r.lossHistory, lossSaddle(newX, newY)],
        };
      })
    );
    setStepCount((s) => s + 1);
  }, []);

  useEffect(() => {
    if (!isRunning) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    let last = 0;
    const tick = (t: number) => {
      if (t - last > 60) {
        doStep();
        last = t;
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isRunning, doStep]);

  const handleSvgClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const scaleX = W5 / rect.width;
      const scaleY = H5 / rect.height;
      const sx = (e.clientX - rect.left) * scaleX;
      const sy = (e.clientY - rect.top) * scaleY;
      const [wx, wy] = fromSvg5(sx, sy);
      const pt = { x: clamp(wx, -RANGE5 + 0.1, RANGE5 - 0.1), y: clamp(wy, -RANGE5 + 0.1, RANGE5 - 0.1) };
      setStartPt(pt);
      setIsRunning(false);
      setRunners(initRunners(pt));
      setStepCount(0);
    },
    [initRunners]
  );

  const reset = () => {
    setIsRunning(false);
    setRunners(initRunners(startPt));
    setStepCount(0);
  };

  // Preset starting points
  const presets = [
    { label: "Near saddle", pt: { x: 0.05, y: 1.5 } },
    { label: "On saddle", pt: { x: 0.0, y: 0.5 } },
    { label: "Left basin", pt: { x: -2.0, y: 1.5 } },
    { label: "Right approach", pt: { x: 1.8, y: 1.8 } },
    { label: "Top center", pt: { x: 0.0, y: 2.2 } },
  ];

  // 3D-ish surface visualization using gradient fills
  const surfaceTiles = useMemo(() => {
    const tiles: { x: number; y: number; w: number; h: number; color: string }[] = [];
    const res = 40;
    const step = (2 * RANGE5) / res;
    let minL = Infinity;
    let maxL = -Infinity;
    const values: number[][] = [];

    for (let i = 0; i < res; i++) {
      values[i] = [];
      for (let j = 0; j < res; j++) {
        const x = -RANGE5 + (i + 0.5) * step;
        const y = -RANGE5 + (j + 0.5) * step;
        const l = lossSaddle(x, y);
        values[i][j] = l;
        if (l < minL) minL = l;
        if (l > maxL) maxL = l;
      }
    }

    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const frac = (values[i][j] - minL) / (maxL - minL);
        const hue = lerp(240, 0, clamp(frac, 0, 1));
        const lightness = lerp(85, 35, clamp(frac, 0, 1));
        const [sx, sy] = toSvg5(-RANGE5 + i * step, -RANGE5 + j * step);
        const cellW = (step / (2 * RANGE5)) * W5;
        const cellH = (step / (2 * RANGE5)) * H5;
        tiles.push({
          x: sx,
          y: sy,
          w: cellW + 0.5,
          h: cellH + 0.5,
          color: `hsl(${hue.toFixed(0)}, 50%, ${lightness.toFixed(0)}%)`,
        });
      }
    }
    return tiles;
  }, []);

  // Loss chart
  const maxLoss = useMemo(() => {
    let mx = 2;
    runners.forEach((r) => {
      r.lossHistory.forEach((l) => {
        if (isFinite(l) && l < 100) mx = Math.max(mx, l);
      });
    });
    return mx + 0.5;
  }, [runners]);

  const H5_LOSS = 160;
  const lossToX = useCallback((s: number) => PAD2.left + (s / 500) * PW2, []);
  const lossToY = useCallback(
    (l: number) => PAD2.top + (1 - clamp(l, -1.5, maxLoss) / maxLoss) * (H5_LOSS - PAD2.top - PAD2.bottom),
    [maxLoss]
  );

  return (
    <div className="space-y-4">
      <div className="bg-rose-50 border border-rose-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
        <div className="text-xs text-rose-800">
          <p>
            This surface f(x,y) = x<sup>4</sup> - 2x<sup>2</sup> + y<sup>2</sup> has a{" "}
            <strong>saddle point at (0,0)</strong> and two <strong>minima at (±1, 0)</strong>.
            Vanilla GD can get stuck near the saddle, while momentum helps escape it.
          </p>
          <p className="mt-1 flex items-center gap-1">
            <MousePointerClick className="w-3.5 h-3.5" />
            <strong>Click on the contour plot</strong> to place a new starting point!
          </p>
        </div>
      </div>

      <div className="flex gap-5 flex-col xl:flex-row">
        {/* Contour/surface plot */}
        <div className="flex-1 min-w-0 bg-slate-50 border border-slate-200 rounded-xl p-3">
          <p className="text-xs font-semibold text-slate-600 mb-1">
            Surface: f(x,y) = x⁴ - 2x² + y² &nbsp; (click to set start)
          </p>
          <svg
            viewBox={`0 0 ${W5} ${H5}`}
            className="w-full cursor-crosshair"
            style={{ maxHeight: 460 }}
            onClick={handleSvgClick}
          >
            {/* Heatmap tiles */}
            {surfaceTiles.map((t, i) => (
              <rect key={i} x={t.x} y={t.y} width={t.w} height={t.h} fill={t.color} opacity={0.6} />
            ))}

            {/* Contour lines */}
            {contours.map((c, i) => (
              <path
                key={i}
                d={c.path}
                fill="none"
                stroke={c.level === 0 ? "#000" : "#334155"}
                strokeWidth={c.level === 0 ? 1.5 : 0.8}
                opacity={c.level === 0 ? 0.6 : 0.4}
              />
            ))}

            {/* Critical points labels */}
            {/* Saddle at (0,0) */}
            {(() => {
              const [sx, sy] = toSvg5(0, 0);
              return (
                <g>
                  <circle cx={sx} cy={sy} r={5} fill="none" stroke="#000" strokeWidth={1.5} strokeDasharray="3,2" />
                  <text x={sx + 8} y={sy - 6} fontSize={10} fill="#1e293b" fontWeight={700}>
                    Saddle (0,0)
                  </text>
                </g>
              );
            })()}
            {/* Min at (-1,0) */}
            {(() => {
              const [sx, sy] = toSvg5(-1, 0);
              return (
                <g>
                  <circle cx={sx} cy={sy} r={4} fill="#10b981" opacity={0.7} />
                  <text x={sx + 7} y={sy - 5} fontSize={9} fill="#10b981" fontWeight={600}>
                    min (-1,0)
                  </text>
                </g>
              );
            })()}
            {/* Min at (1,0) */}
            {(() => {
              const [sx, sy] = toSvg5(1, 0);
              return (
                <g>
                  <circle cx={sx} cy={sy} r={4} fill="#10b981" opacity={0.7} />
                  <text x={sx + 7} y={sy - 5} fontSize={9} fill="#10b981" fontWeight={600}>
                    min (1,0)
                  </text>
                </g>
              );
            })()}

            {/* Runner trails and positions */}
            {runners.map((r, idx) => {
              if (r.trail.length < 2) return null;
              const points = r.trail
                .map((p) => {
                  const [sx, sy] = toSvg5(p.x, p.y);
                  return `${sx.toFixed(1)},${sy.toFixed(1)}`;
                })
                .join(" ");
              const [cx, cy] = toSvg5(r.x, r.y);
              return (
                <g key={idx}>
                  <polyline
                    points={points}
                    fill="none"
                    stroke={r.color}
                    strokeWidth={2}
                    opacity={0.7}
                    strokeLinejoin="round"
                  />
                  <circle cx={cx} cy={cy} r={6} fill={r.color} stroke="#fff" strokeWidth={2}>
                    <animate attributeName="r" values="5;7;5" dur="1.5s" repeatCount="indefinite" />
                  </circle>

                  {/* Gradient arrow */}
                  {showGradArrows && stepCount > 0 && (
                    <>
                      <defs>
                        <marker
                          id={`arrowSaddle${idx}`}
                          markerWidth="7"
                          markerHeight="5"
                          refX="6"
                          refY="2.5"
                          orient="auto"
                        >
                          <polygon points="0 0, 7 2.5, 0 5" fill={r.color} />
                        </marker>
                      </defs>
                      {(() => {
                        const gx = gradSaddleX(r.x);
                        const gy = gradSaddleY(r.y);
                        const mag = Math.sqrt(gx * gx + gy * gy);
                        if (mag < 0.001) return null;
                        const scale = Math.min(40, mag * 15);
                        const nx = -gx / mag;
                        const ny = gy / mag; // flip y for SVG
                        const [sx, sy] = toSvg5(r.x, r.y);
                        return (
                          <line
                            x1={sx}
                            y1={sy}
                            x2={sx + nx * scale}
                            y2={sy + ny * scale}
                            stroke={r.color}
                            strokeWidth={2}
                            markerEnd={`url(#arrowSaddle${idx})`}
                            opacity={0.6}
                          />
                        );
                      })()}
                    </>
                  )}
                </g>
              );
            })}

            {/* Start marker */}
            {(() => {
              const [sx, sy] = toSvg5(startPt.x, startPt.y);
              return (
                <g>
                  <circle cx={sx} cy={sy} r={5} fill="none" stroke="#64748b" strokeWidth={2} strokeDasharray="3,2" />
                  <text x={sx + 8} y={sy - 5} fontSize={9} fill="#64748b" fontWeight={600}>
                    start
                  </text>
                </g>
              );
            })()}
          </svg>
        </div>

        {/* Controls */}
        <div className="w-full xl:w-72 space-y-3">
          {/* Preset starts */}
          <div className="bg-white border border-slate-200 rounded-lg p-2.5">
            <p className="text-[10px] font-semibold text-slate-600 uppercase mb-1.5">Starting Points</p>
            <div className="flex flex-wrap gap-1.5">
              {presets.map((p, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setStartPt(p.pt);
                    setIsRunning(false);
                    setRunners(initRunners(p.pt));
                    setStepCount(0);
                  }}
                  className="px-2 py-1 text-[10px] rounded border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium transition-all"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Runner legend */}
          <div className="space-y-1.5">
            {runners.map((r, i) => (
              <div key={i} className="bg-white border border-slate-200 rounded-lg p-2 flex items-center gap-2">
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: r.color }} />
                <div className="flex-1">
                  <p className="text-[10px] font-semibold text-slate-700">{r.label}</p>
                  <p className="text-[9px] text-slate-500 font-mono">
                    ({r.x.toFixed(3)}, {r.y.toFixed(3)}) loss={lossSaddle(r.x, r.y).toFixed(4)}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* LR slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-2.5">
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-slate-700">Learning Rate</label>
              <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded">{lr.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min={0.001}
              max={0.05}
              step={0.001}
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-rose-500"
            />
          </div>

          {/* Show arrows toggle */}
          <label className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg p-2.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showGradArrows}
              onChange={(e) => setShowGradArrows(e.target.checked)}
              className="accent-rose-500"
            />
            <span className="text-xs text-slate-700 font-medium">Show gradient arrows</span>
          </label>

          {/* Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-sm font-semibold transition-all ${
                isRunning ? "bg-slate-800 text-white" : "bg-rose-500 text-white hover:bg-rose-600"
              }`}
            >
              {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isRunning ? "Pause" : "Start"}
            </button>
            <button
              onClick={() => !isRunning && doStep()}
              disabled={isRunning}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40"
            >
              <SkipForward className="w-4 h-4" />
            </button>
            <button
              onClick={reset}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <p className="text-[10px] text-slate-400 text-center">Step: {stepCount} / 500</p>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-2.5">
            <p className="text-[10px] text-amber-800 font-semibold uppercase mb-1">What to try</p>
            <ul className="text-[10px] text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Start "On saddle" — vanilla GD barely moves, momentum escapes</li>
              <li>Click different positions near the saddle</li>
              <li>Observe how momentum overshoots but finds the minimum</li>
              <li>Watch gradient arrows change direction</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Loss vs iteration */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
        <p className="text-xs font-semibold text-slate-600 mb-1">Loss vs. Iteration</p>
        <svg viewBox={`0 0 ${W2} ${H5_LOSS}`} className="w-full" style={{ maxHeight: 180 }}>
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const y = PAD2.top + (1 - frac) * (H5_LOSS - PAD2.top - PAD2.bottom);
            const val = lerp(-1.5, maxLoss, frac);
            return (
              <g key={frac}>
                <line x1={PAD2.left} y1={y} x2={W2 - PAD2.right} y2={y} stroke="#e2e8f0" strokeWidth={0.5} />
                <text x={PAD2.left - 6} y={y + 3} fontSize={8} fill="#94a3b8" textAnchor="end">
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Zero line */}
          {(() => {
            const zeroY = lossToY(0);
            return (
              <line
                x1={PAD2.left}
                y1={zeroY}
                x2={W2 - PAD2.right}
                y2={zeroY}
                stroke="#94a3b8"
                strokeWidth={0.5}
                strokeDasharray="4,3"
              />
            );
          })()}

          {/* Minimum reference line at -1 */}
          {(() => {
            const minY = lossToY(-1);
            return (
              <g>
                <line
                  x1={PAD2.left}
                  y1={minY}
                  x2={W2 - PAD2.right}
                  y2={minY}
                  stroke="#10b981"
                  strokeWidth={0.5}
                  strokeDasharray="3,3"
                />
                <text x={W2 - PAD2.right + 4} y={minY + 3} fontSize={8} fill="#10b981">
                  min=-1
                </text>
              </g>
            );
          })()}

          {runners.map((r, idx) => {
            if (r.lossHistory.length < 2) return null;
            const path = r.lossHistory
              .map((l, i) => `${i === 0 ? "M" : "L"} ${lossToX(i).toFixed(1)} ${lossToY(l).toFixed(1)}`)
              .join(" ");
            return <path key={idx} d={path} fill="none" stroke={r.color} strokeWidth={1.5} opacity={0.8} />;
          })}

          {/* Legend */}
          {runners.map((r, i) => (
            <g key={i} transform={`translate(${PAD2.left + i * 200}, ${PAD2.top - 6})`}>
              <rect width={12} height={3} fill={r.color} rx={1} />
              <text x={16} y={3} fontSize={9} fill="#64748b">
                {r.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════
export default function GradientDescentActivity() {
  const [activeTab, setActiveTab] = useState("step1d");

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 overflow-x-auto pb-1 border-b border-slate-200">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-t-lg text-xs font-semibold whitespace-nowrap transition-all border-b-2 ${
              activeTab === tab.id
                ? "bg-white text-red-600 border-red-500"
                : "text-slate-500 hover:text-slate-700 border-transparent hover:bg-slate-50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "step1d" && <Tab1StepByStep />}
        {activeTab === "lreffect" && <Tab2LearningRate />}
        {activeTab === "momentum" && <Tab3Momentum />}
        {activeTab === "stochastic" && <Tab4Stochastic />}
        {activeTab === "saddle" && <Tab5SaddlePoints />}
      </div>
    </div>
  );
}
