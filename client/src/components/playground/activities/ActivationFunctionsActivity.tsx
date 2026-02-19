/**
 * Activation Functions Activity — comprehensive 5-tab interactive SVG activity
 * Tab 1: Compare All Functions (6 activation functions with draggable probe)
 * Tab 2: Derivatives & Gradients (vanishing gradient & saturation visualization)
 * Tab 3: Dead ReLU Problem (mini neural network, ReLU vs Leaky ReLU)
 * Tab 4: Softmax Visualization (logit sliders, temperature, step-by-step)
 * Tab 5: Effect on Network Output (decision boundaries for different activations)
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  BarChart3,
  TrendingDown,
  Skull,
  PieChart,
  Grid3X3,
  Info,
  Eye,
  EyeOff,
  Zap,
  Shuffle,
  Play,
  RotateCcw,
  Thermometer,
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
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
type TabKey = "compare" | "derivatives" | "deadrelu" | "softmax" | "boundary";

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

interface ActivationDef {
  key: string;
  label: string;
  color: string;
  fn: (x: number) => number;
  dfn: (x: number) => number;
  range: string;
  monotonic: boolean;
  saturates: boolean;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Activation function definitions (6 functions including Swish)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const ALPHA_LEAKY = 0.1;
const ALPHA_ELU = 1.0;

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

const ACTIVATIONS: ActivationDef[] = [
  {
    key: "relu",
    label: "ReLU",
    color: "#ef4444",
    fn: (x) => Math.max(0, x),
    dfn: (x) => (x > 0 ? 1 : 0),
    range: "[0, +inf)",
    monotonic: true,
    saturates: false,
  },
  {
    key: "sigmoid",
    label: "Sigmoid",
    color: "#3b82f6",
    fn: (x) => sigmoid(x),
    dfn: (x) => {
      const s = sigmoid(x);
      return s * (1 - s);
    },
    range: "(0, 1)",
    monotonic: true,
    saturates: true,
  },
  {
    key: "tanh",
    label: "Tanh",
    color: "#22c55e",
    fn: (x) => Math.tanh(x),
    dfn: (x) => 1 - Math.tanh(x) ** 2,
    range: "(-1, 1)",
    monotonic: true,
    saturates: true,
  },
  {
    key: "leaky_relu",
    label: "Leaky ReLU",
    color: "#8b5cf6",
    fn: (x) => (x > 0 ? x : ALPHA_LEAKY * x),
    dfn: (x) => (x > 0 ? 1 : ALPHA_LEAKY),
    range: "(-inf, +inf)",
    monotonic: true,
    saturates: false,
  },
  {
    key: "elu",
    label: "ELU",
    color: "#f97316",
    fn: (x) => (x > 0 ? x : ALPHA_ELU * (Math.exp(x) - 1)),
    dfn: (x) => (x > 0 ? 1 : ALPHA_ELU * Math.exp(x)),
    range: "(-1, +inf)",
    monotonic: true,
    saturates: false,
  },
  {
    key: "swish",
    label: "Swish",
    color: "#ec4899",
    fn: (x) => x * sigmoid(x),
    dfn: (x) => {
      const s = sigmoid(x);
      return s + x * s * (1 - s);
    },
    range: "(-0.28, +inf)",
    monotonic: false,
    saturates: false,
  },
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Shared SVG utilities
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function fmt(n: number, decimals = 4): string {
  if (Math.abs(n) < 0.00005) return "0.0000";
  return n.toFixed(decimals);
}

function buildPolyline(
  fn: (x: number) => number,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  toSvgX: (v: number) => number,
  toSvgY: (v: number) => number,
  samples = 400
): string {
  const pts: string[] = [];
  for (let i = 0; i <= samples; i++) {
    const x = xMin + (i / samples) * (xMax - xMin);
    const y = clamp(fn(x), yMin - 0.5, yMax + 0.5);
    pts.push(`${toSvgX(x).toFixed(1)},${toSvgY(y).toFixed(1)}`);
  }
  return pts.join(" ");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Reusable chart grid
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
interface GridProps {
  svgW: number;
  svgH: number;
  pad: { top: number; right: number; bottom: number; left: number };
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  xStep?: number;
  yStep?: number;
  xLabel?: string;
  yLabel?: string;
}

function ChartGrid({
  svgW,
  svgH,
  pad,
  xMin,
  xMax,
  yMin,
  yMax,
  xStep = 1,
  yStep = 0.5,
  xLabel = "x",
  yLabel = "f(x)",
}: GridProps) {
  const plotW = svgW - pad.left - pad.right;
  const plotH = svgH - pad.top - pad.bottom;
  const toX = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const toY = (v: number) => pad.top + ((yMax - v) / (yMax - yMin)) * plotH;

  const xTicks: number[] = [];
  for (let v = xMin; v <= xMax + xStep * 0.01; v += xStep) {
    xTicks.push(Math.round(v * 100) / 100);
  }
  const yTicks: number[] = [];
  for (let v = yMin; v <= yMax + yStep * 0.01; v += yStep) {
    yTicks.push(Math.round(v * 100) / 100);
  }

  return (
    <>
      <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="#fafafa" stroke="#e2e8f0" strokeWidth={1} />
      {yTicks.map((t) => (
        <g key={`hy-${t}`}>
          <line x1={pad.left} y1={toY(t)} x2={pad.left + plotW} y2={toY(t)} stroke={t === 0 ? "#94a3b8" : "#f1f5f9"} strokeWidth={t === 0 ? 1 : 0.5} />
          <text x={pad.left - 6} y={toY(t) + 3.5} fontSize={9} fill="#94a3b8" textAnchor="end">{t % 1 === 0 ? t : t.toFixed(1)}</text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g key={`vx-${t}`}>
          <line x1={toX(t)} y1={pad.top} x2={toX(t)} y2={pad.top + plotH} stroke={t === 0 ? "#94a3b8" : "#f1f5f9"} strokeWidth={t === 0 ? 1 : 0.5} />
          <text x={toX(t)} y={pad.top + plotH + 14} fontSize={9} fill="#94a3b8" textAnchor="middle">{t}</text>
        </g>
      ))}
      <text x={pad.left + plotW / 2} y={svgH - 2} fontSize={10} fill="#64748b" textAnchor="middle" fontWeight={600}>{xLabel}</text>
      <text x={10} y={pad.top + plotH / 2} fontSize={10} fill="#64748b" textAnchor="middle" fontWeight={600} transform={`rotate(-90, 10, ${pad.top + plotH / 2})`}>{yLabel}</text>
    </>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab definitions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const TABS: TabDef[] = [
  { key: "compare", label: "Compare All", icon: <BarChart3 className="w-3.5 h-3.5" /> },
  { key: "derivatives", label: "Derivatives & Gradients", icon: <TrendingDown className="w-3.5 h-3.5" /> },
  { key: "deadrelu", label: "Dead ReLU Problem", icon: <Skull className="w-3.5 h-3.5" /> },
  { key: "softmax", label: "Softmax", icon: <PieChart className="w-3.5 h-3.5" /> },
  { key: "boundary", label: "Decision Boundaries", icon: <Grid3X3 className="w-3.5 h-3.5" /> },
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main component
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export default function ActivationFunctionsActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("compare");

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-100 rounded-lg p-1 overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium whitespace-nowrap transition-all ${
              activeTab === tab.key
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "compare" && <CompareAllTab />}
      {activeTab === "derivatives" && <DerivativesTab />}
      {activeTab === "deadrelu" && <DeadReLUTab />}
      {activeTab === "softmax" && <SoftmaxTab />}
      {activeTab === "boundary" && <DecisionBoundaryTab />}
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 1 — COMPARE ALL FUNCTIONS                                           ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const CMP_SVG_W = 640;
const CMP_SVG_H = 220;
const CMP_PAD = { top: 16, right: 20, bottom: 24, left: 45 };
const CMP_PLOT_W = CMP_SVG_W - CMP_PAD.left - CMP_PAD.right;
const CMP_PLOT_H = CMP_SVG_H - CMP_PAD.top - CMP_PAD.bottom;
const CMP_X_MIN = -5;
const CMP_X_MAX = 5;
const CMP_Y_MIN = -2;
const CMP_Y_MAX = 5;

function cmpToSvgX(v: number): number {
  return CMP_PAD.left + ((v - CMP_X_MIN) / (CMP_X_MAX - CMP_X_MIN)) * CMP_PLOT_W;
}
function cmpToSvgY(v: number): number {
  return CMP_PAD.top + ((CMP_Y_MAX - v) / (CMP_Y_MAX - CMP_Y_MIN)) * CMP_PLOT_H;
}
function cmpFromSvgX(sx: number, rect: DOMRect, svgW: number): number {
  const scale = svgW / rect.width;
  const svgX = (sx - rect.left) * scale;
  return CMP_X_MIN + ((svgX - CMP_PAD.left) / CMP_PLOT_W) * (CMP_X_MAX - CMP_X_MIN);
}

function CompareAllTab() {
  const [enabled, setEnabled] = useState<Record<string, boolean>>({
    relu: true, sigmoid: true, tanh: true, leaky_relu: true, elu: true, swish: true,
  });
  const [probeX, setProbeX] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const toggle = useCallback((key: string) => {
    setEnabled((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const activeActs = useMemo(() => ACTIVATIONS.filter((a) => enabled[a.key]), [enabled]);

  const fnPaths = useMemo(
    () =>
      activeActs.map((a) => ({
        key: a.key,
        color: a.color,
        path: buildPolyline(a.fn, CMP_X_MIN, CMP_X_MAX, CMP_Y_MIN, CMP_Y_MAX, cmpToSvgX, cmpToSvgY),
      })),
    [activeActs]
  );

  const probeValues = useMemo(
    () =>
      activeActs.map((a) => ({
        key: a.key,
        label: a.label,
        color: a.color,
        fx: a.fn(probeX),
        dfx: a.dfn(probeX),
        range: a.range,
        monotonic: a.monotonic,
        saturates: a.saturates,
      })),
    [activeActs, probeX]
  );

  const handleSvgMouse = useCallback(
    (e: React.MouseEvent<SVGSVGElement>, forceSet = false) => {
      if (!isDragging && !forceSet) return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = cmpFromSvgX(e.clientX, rect, CMP_SVG_W);
      setProbeX(clamp(x, CMP_X_MIN, CMP_X_MAX));
    },
    [isDragging]
  );

  const probeSvgX = cmpToSvgX(probeX);

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">Compare 6 Activation Functions</h3>
          <p className="text-xs text-indigo-700 mt-1">
            Drag the vertical probe line to inspect <strong>f(x)</strong> and <strong>f'(x)</strong> for each function at
            any x value. Toggle functions on/off using the checkboxes. Notice how each function handles negative inputs
            differently and how their output ranges vary.
          </p>
        </div>
      </div>

      {/* Function toggles */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-slate-600 mr-1 flex items-center gap-1.5">
          <Zap className="w-3.5 h-3.5 text-amber-500" />
          Functions:
        </span>
        {ACTIVATIONS.map((a) => {
          const isOn = enabled[a.key];
          return (
            <button
              key={a.key}
              onClick={() => toggle(a.key)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
                isOn
                  ? "bg-white text-slate-800 border-slate-300 shadow-sm"
                  : "bg-slate-50 text-slate-400 border-slate-200 hover:bg-slate-100"
              }`}
            >
              {isOn ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: a.color, opacity: isOn ? 1 : 0.25 }}
              />
              {a.label}
            </button>
          );
        })}
      </div>

      {/* SVG Chart */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-700">Activation Functions f(x)</span>
          <span className="text-[10px] text-slate-400">Click or drag to move probe | x = {probeX.toFixed(2)}</span>
        </div>
        <div className="p-2">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${CMP_SVG_W} ${CMP_SVG_H}`}
            className="w-full select-none"
            style={{ aspectRatio: `${CMP_SVG_W} / ${CMP_SVG_H}`, cursor: isDragging ? "grabbing" : "crosshair" }}
            onMouseDown={(e) => { setIsDragging(true); handleSvgMouse(e, true); }}
            onMouseMove={(e) => handleSvgMouse(e)}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
          >
            <ChartGrid
              svgW={CMP_SVG_W} svgH={CMP_SVG_H} pad={CMP_PAD}
              xMin={CMP_X_MIN} xMax={CMP_X_MAX} yMin={CMP_Y_MIN} yMax={CMP_Y_MAX}
              yStep={1} xLabel="x (input)" yLabel="f(x)"
            />

            {/* Function curves */}
            {fnPaths.map((fp) => (
              <polyline
                key={fp.key}
                points={fp.path}
                fill="none"
                stroke={fp.color}
                strokeWidth={2.5}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={0.85}
              />
            ))}

            {/* Probe line */}
            <line
              x1={probeSvgX} y1={CMP_PAD.top} x2={probeSvgX} y2={CMP_PAD.top + CMP_PLOT_H}
              stroke="#475569" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}
            />

            {/* Probe dots */}
            {probeValues.map((pv) => {
              const cy = cmpToSvgY(clamp(pv.fx, CMP_Y_MIN, CMP_Y_MAX));
              return (
                <circle key={pv.key} cx={probeSvgX} cy={cy} r={5} fill={pv.color} stroke="#fff" strokeWidth={2} />
              );
            })}

            {/* Probe label */}
            <text x={probeSvgX} y={CMP_PAD.top - 5} fontSize={10} fill="#475569" textAnchor="middle" fontWeight={600}>
              x = {probeX.toFixed(2)}
            </text>

            {/* Legend */}
            {activeActs.map((a, i) => (
              <g key={a.key} transform={`translate(${CMP_PAD.left + 10}, ${CMP_PAD.top + 12 + i * 16})`}>
                <line x1={0} y1={0} x2={16} y2={0} stroke={a.color} strokeWidth={2.5} strokeLinecap="round" />
                <text x={20} y={4} fontSize={10} fill="#475569" fontWeight={500}>{a.label}</text>
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Probe slider */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-xs font-semibold text-slate-600">Probe Position:</span>
          <input
            type="range"
            min={CMP_X_MIN}
            max={CMP_X_MAX}
            step={0.01}
            value={probeX}
            onChange={(e) => setProbeX(parseFloat(e.target.value))}
            className="flex-1 h-2 accent-indigo-500"
          />
          <span className="text-xs font-mono text-slate-700 w-14 text-right">{probeX.toFixed(2)}</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {[-4, -2, -1, 0, 1, 2, 4].map((v) => (
            <button
              key={v}
              onClick={() => setProbeX(v)}
              className={`px-2.5 py-1 rounded-md text-xs font-mono transition-all ${
                Math.abs(probeX - v) < 0.05
                  ? "bg-indigo-100 text-indigo-700 font-semibold"
                  : "bg-slate-100 text-slate-500 hover:bg-slate-200"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Values table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-500" />
          <span className="text-xs font-semibold text-slate-700">Values at x = {probeX.toFixed(2)}</span>
        </div>
        {activeActs.length === 0 ? (
          <div className="p-6 text-center text-xs text-slate-400">Enable at least one function to see values.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50/50">
                  <th className="text-left px-4 py-2 font-semibold text-slate-500">Function</th>
                  <th className="text-right px-4 py-2 font-semibold text-slate-500">f(x)</th>
                  <th className="text-right px-4 py-2 font-semibold text-slate-500">f'(x)</th>
                  <th className="text-center px-4 py-2 font-semibold text-slate-500">Range</th>
                  <th className="text-center px-4 py-2 font-semibold text-slate-500">Monotonic</th>
                  <th className="text-center px-4 py-2 font-semibold text-slate-500">Saturates</th>
                </tr>
              </thead>
              <tbody>
                {probeValues.map((pv) => (
                  <tr key={pv.key} className="border-b border-slate-50 hover:bg-slate-50/50">
                    <td className="px-4 py-2.5 font-medium text-slate-700 flex items-center gap-2">
                      <span className="inline-block w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: pv.color }} />
                      {pv.label}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-slate-700">{fmt(pv.fx)}</td>
                    <td className={`px-4 py-2.5 text-right font-mono ${Math.abs(pv.dfx) < 0.01 ? "text-red-500 font-bold" : "text-slate-700"}`}>
                      {fmt(pv.dfx)}
                    </td>
                    <td className="px-4 py-2.5 text-center font-mono text-slate-500 text-[10px]">{pv.range}</td>
                    <td className="px-4 py-2.5 text-center">{pv.monotonic ? <span className="text-green-600">Yes</span> : <span className="text-amber-600">No</span>}</td>
                    <td className="px-4 py-2.5 text-center">{pv.saturates ? <span className="text-red-500">Yes</span> : <span className="text-green-600">No</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Saturation regions highlight */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Key observations at x = {probeX.toFixed(2)}:</p>
        <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
          {probeX < 0 && enabled.relu && (
            <li>ReLU outputs exactly 0 for negative inputs — gradient is dead (0). This causes the "dead neuron" problem.</li>
          )}
          {Math.abs(probeX) > 3 && enabled.sigmoid && (
            <li>Sigmoid is in its <strong>saturation zone</strong> — gradient is near 0, causing vanishing gradients during backpropagation.</li>
          )}
          {Math.abs(probeX) > 3 && enabled.tanh && (
            <li>Tanh is also saturating — similar vanishing gradient issue as sigmoid but with a wider active region.</li>
          )}
          {probeX < 0 && enabled.leaky_relu && (
            <li>Leaky ReLU maintains a small gradient ({ALPHA_LEAKY}) for negative inputs — neurons never fully die.</li>
          )}
          {Math.abs(probeX) < 0.5 && (
            <li>Near x=0, most functions have their steepest gradients. This is the "sweet spot" for learning.</li>
          )}
          {enabled.swish && probeX < -3 && (
            <li>Swish is non-monotonic: it dips slightly below 0 for negative inputs before approaching 0, making it smoother than ReLU.</li>
          )}
        </ul>
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 2 — DERIVATIVES & GRADIENTS                                         ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const DRV_SVG_W = 640;
const DRV_SVG_H = 240;
const DRV_PAD = { top: 20, right: 20, bottom: 30, left: 45 };
const DRV_PLOT_W = DRV_SVG_W - DRV_PAD.left - DRV_PAD.right;
const DRV_PLOT_H = DRV_SVG_H - DRV_PAD.top - DRV_PAD.bottom;
const DRV_X_MIN = -5;
const DRV_X_MAX = 5;

function drvToSvgX(v: number): number {
  return DRV_PAD.left + ((v - DRV_X_MIN) / (DRV_X_MAX - DRV_X_MIN)) * DRV_PLOT_W;
}

function DerivativesTab() {
  const [selectedKey, setSelectedKey] = useState("sigmoid");
  const [probeX, setProbeX] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const svgTopRef = useRef<SVGSVGElement>(null);
  const svgBotRef = useRef<SVGSVGElement>(null);

  const act = useMemo(() => ACTIVATIONS.find((a) => a.key === selectedKey)!, [selectedKey]);

  // y ranges differ per function for better visualization
  const fnYMin = selectedKey === "relu" || selectedKey === "leaky_relu" || selectedKey === "elu" || selectedKey === "swish" ? -2 : -1.5;
  const fnYMax = selectedKey === "relu" || selectedKey === "leaky_relu" || selectedKey === "elu" || selectedKey === "swish" ? 5 : 1.5;
  const dfnYMin = -0.5;
  const dfnYMax = 1.5;

  const fnToSvgY = useCallback(
    (v: number) => DRV_PAD.top + ((fnYMax - v) / (fnYMax - fnYMin)) * DRV_PLOT_H,
    [fnYMin, fnYMax]
  );
  const dfnToSvgY = useCallback(
    (v: number) => DRV_PAD.top + ((dfnYMax - v) / (dfnYMax - dfnYMin)) * DRV_PLOT_H,
    []
  );

  const fnPath = useMemo(
    () => buildPolyline(act.fn, DRV_X_MIN, DRV_X_MAX, fnYMin, fnYMax, drvToSvgX, fnToSvgY),
    [act, fnYMin, fnYMax, fnToSvgY]
  );

  const dfnPath = useMemo(
    () => buildPolyline(act.dfn, DRV_X_MIN, DRV_X_MAX, dfnYMin, dfnYMax, drvToSvgX, dfnToSvgY),
    [act, dfnToSvgY]
  );

  // Build gradient-colored segments for the function curve
  const gradientSegments = useMemo(() => {
    const segments: { x1: number; y1: number; x2: number; y2: number; intensity: number }[] = [];
    const samples = 200;
    for (let i = 0; i < samples; i++) {
      const x0 = DRV_X_MIN + (i / samples) * (DRV_X_MAX - DRV_X_MIN);
      const x1 = DRV_X_MIN + ((i + 1) / samples) * (DRV_X_MAX - DRV_X_MIN);
      const y0 = clamp(act.fn(x0), fnYMin, fnYMax);
      const y1 = clamp(act.fn(x1), fnYMin, fnYMax);
      const grad = Math.abs(act.dfn((x0 + x1) / 2));
      segments.push({
        x1: drvToSvgX(x0),
        y1: fnToSvgY(y0),
        x2: drvToSvgX(x1),
        y2: fnToSvgY(y1),
        intensity: Math.min(grad, 1),
      });
    }
    return segments;
  }, [act, fnYMin, fnYMax, fnToSvgY]);

  // Vanishing gradient regions
  const vanishingRegions = useMemo(() => {
    const regions: { xStart: number; xEnd: number }[] = [];
    const threshold = 0.05;
    let start: number | null = null;
    const samples = 400;
    for (let i = 0; i <= samples; i++) {
      const x = DRV_X_MIN + (i / samples) * (DRV_X_MAX - DRV_X_MIN);
      const dval = Math.abs(act.dfn(x));
      if (dval < threshold) {
        if (start === null) start = x;
      } else {
        if (start !== null) {
          regions.push({ xStart: start, xEnd: x });
          start = null;
        }
      }
    }
    if (start !== null) regions.push({ xStart: start, xEnd: DRV_X_MAX });
    return regions;
  }, [act]);

  const handleDrag = useCallback(
    (e: React.MouseEvent<SVGSVGElement>, svg: SVGSVGElement | null, forceSet = false) => {
      if (!isDragging && !forceSet) return;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const scale = DRV_SVG_W / rect.width;
      const sx = (e.clientX - rect.left) * scale;
      const x = DRV_X_MIN + ((sx - DRV_PAD.left) / DRV_PLOT_W) * (DRV_X_MAX - DRV_X_MIN);
      setProbeX(clamp(x, DRV_X_MIN, DRV_X_MAX));
    },
    [isDragging]
  );

  const probeSvgX = drvToSvgX(probeX);
  const fxVal = act.fn(probeX);
  const dfxVal = act.dfn(probeX);

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">Derivatives and the Vanishing Gradient Problem</h3>
          <p className="text-xs text-indigo-700 mt-1">
            During backpropagation, gradients are multiplied through each layer. If the activation derivative is close to 0,
            gradients <strong>vanish</strong> — earlier layers learn extremely slowly. The red-shaded regions below show where
            the gradient is dangerously small. The color intensity on the function curve represents gradient magnitude.
          </p>
        </div>
      </div>

      {/* Function selector */}
      <div className="flex flex-wrap gap-2">
        {ACTIVATIONS.map((a) => (
          <button
            key={a.key}
            onClick={() => setSelectedKey(a.key)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all border ${
              selectedKey === a.key
                ? "bg-white text-slate-800 border-slate-300 shadow-sm ring-2 ring-indigo-200"
                : "bg-slate-50 text-slate-500 border-slate-200 hover:bg-slate-100"
            }`}
          >
            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: a.color }} />
            {a.label}
          </button>
        ))}
      </div>

      {/* Function chart (with gradient coloring) */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
          <span className="text-xs font-semibold text-slate-700">{act.label} — f(x)</span>
          <span className="text-[10px] text-slate-400 ml-2">Color intensity = gradient magnitude</span>
        </div>
        <div className="p-2">
          <svg
            ref={svgTopRef}
            viewBox={`0 0 ${DRV_SVG_W} ${DRV_SVG_H}`}
            className="w-full select-none"
            style={{ aspectRatio: `${DRV_SVG_W} / ${DRV_SVG_H}`, cursor: isDragging ? "grabbing" : "crosshair" }}
            onMouseDown={(e) => { setIsDragging(true); handleDrag(e, svgTopRef.current, true); }}
            onMouseMove={(e) => handleDrag(e, svgTopRef.current)}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
          >
            <ChartGrid
              svgW={DRV_SVG_W} svgH={DRV_SVG_H} pad={DRV_PAD}
              xMin={DRV_X_MIN} xMax={DRV_X_MAX} yMin={fnYMin} yMax={fnYMax}
              yStep={fnYMax > 2 ? 1 : 0.5} xLabel="x" yLabel="f(x)"
            />

            {/* Vanishing gradient regions on function chart */}
            {vanishingRegions.map((r, i) => (
              <rect
                key={i}
                x={drvToSvgX(r.xStart)}
                y={DRV_PAD.top}
                width={drvToSvgX(r.xEnd) - drvToSvgX(r.xStart)}
                height={DRV_PLOT_H}
                fill="#fecaca"
                opacity={0.3}
              />
            ))}

            {/* Gradient-colored curve segments */}
            {gradientSegments.map((seg, i) => (
              <line
                key={i}
                x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2}
                stroke={act.color}
                strokeWidth={3}
                opacity={0.2 + seg.intensity * 0.8}
                strokeLinecap="round"
              />
            ))}

            {/* Probe line */}
            <line
              x1={probeSvgX} y1={DRV_PAD.top} x2={probeSvgX} y2={DRV_PAD.top + DRV_PLOT_H}
              stroke="#475569" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}
            />
            <circle cx={probeSvgX} cy={fnToSvgY(clamp(fxVal, fnYMin, fnYMax))} r={5} fill={act.color} stroke="#fff" strokeWidth={2} />
            <text x={probeSvgX} y={DRV_PAD.top - 5} fontSize={10} fill="#475569" textAnchor="middle" fontWeight={600}>
              x = {probeX.toFixed(2)}, f(x) = {fmt(fxVal, 3)}
            </text>
          </svg>
        </div>
      </div>

      {/* Derivative chart */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
          <span className="text-xs font-semibold text-slate-700">{act.label} — f'(x) (Derivative)</span>
          <span className="text-[10px] text-slate-400 ml-2">
            {vanishingRegions.length > 0 ? "Red zones = vanishing gradient" : "No vanishing gradient regions"}
          </span>
        </div>
        <div className="p-2">
          <svg
            ref={svgBotRef}
            viewBox={`0 0 ${DRV_SVG_W} ${DRV_SVG_H}`}
            className="w-full select-none"
            style={{ aspectRatio: `${DRV_SVG_W} / ${DRV_SVG_H}`, cursor: isDragging ? "grabbing" : "crosshair" }}
            onMouseDown={(e) => { setIsDragging(true); handleDrag(e, svgBotRef.current, true); }}
            onMouseMove={(e) => handleDrag(e, svgBotRef.current)}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
          >
            <ChartGrid
              svgW={DRV_SVG_W} svgH={DRV_SVG_H} pad={DRV_PAD}
              xMin={DRV_X_MIN} xMax={DRV_X_MAX} yMin={dfnYMin} yMax={dfnYMax}
              yStep={0.25} xLabel="x" yLabel="f'(x)"
            />

            {/* Vanishing gradient regions */}
            {vanishingRegions.map((r, i) => (
              <rect
                key={i}
                x={drvToSvgX(r.xStart)}
                y={DRV_PAD.top}
                width={drvToSvgX(r.xEnd) - drvToSvgX(r.xStart)}
                height={DRV_PLOT_H}
                fill="#fecaca"
                opacity={0.3}
              />
            ))}

            {/* Zero line emphasized */}
            <line
              x1={DRV_PAD.left} y1={dfnToSvgY(0)}
              x2={DRV_PAD.left + DRV_PLOT_W} y2={dfnToSvgY(0)}
              stroke="#ef4444" strokeWidth={1} strokeDasharray="2,2" opacity={0.5}
            />

            {/* Derivative curve */}
            <polyline
              points={dfnPath}
              fill="none"
              stroke={act.color}
              strokeWidth={2.5}
              strokeLinejoin="round"
              strokeLinecap="round"
              strokeDasharray="6,3"
              opacity={0.85}
            />

            {/* Probe */}
            <line
              x1={probeSvgX} y1={DRV_PAD.top} x2={probeSvgX} y2={DRV_PAD.top + DRV_PLOT_H}
              stroke="#475569" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}
            />
            <circle cx={probeSvgX} cy={dfnToSvgY(clamp(dfxVal, dfnYMin, dfnYMax))} r={5} fill={act.color} stroke="#fff" strokeWidth={2} />
            <text x={probeSvgX} y={DRV_PAD.top - 5} fontSize={10} fill="#475569" textAnchor="middle" fontWeight={600}>
              f'({probeX.toFixed(2)}) = {fmt(dfxVal, 4)}
            </text>

            {/* Dead zone / saturation annotations */}
            {selectedKey === "relu" && (
              <text x={drvToSvgX(-2.5)} y={DRV_PAD.top + 16} fontSize={10} fill="#dc2626" textAnchor="middle" fontWeight={700}>
                DEAD ZONE (gradient = 0)
              </text>
            )}
            {(selectedKey === "sigmoid" || selectedKey === "tanh") && (
              <>
                <text x={drvToSvgX(-4)} y={DRV_PAD.top + 16} fontSize={9} fill="#dc2626" textAnchor="middle" fontWeight={600}>
                  Saturation
                </text>
                <text x={drvToSvgX(4)} y={DRV_PAD.top + 16} fontSize={9} fill="#dc2626" textAnchor="middle" fontWeight={600}>
                  Saturation
                </text>
              </>
            )}
          </svg>
        </div>
      </div>

      {/* Probe slider */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-slate-600">Probe:</span>
          <input
            type="range" min={DRV_X_MIN} max={DRV_X_MAX} step={0.01} value={probeX}
            onChange={(e) => setProbeX(parseFloat(e.target.value))}
            className="flex-1 h-2 accent-indigo-500"
          />
          <span className="text-xs font-mono text-slate-700 w-14 text-right">{probeX.toFixed(2)}</span>
        </div>
        <div className="grid grid-cols-3 gap-3 mt-3 text-xs">
          <div className="bg-slate-50 rounded-lg p-2 text-center">
            <div className="text-slate-400 mb-0.5">f(x)</div>
            <div className="font-mono font-semibold text-slate-700">{fmt(fxVal, 4)}</div>
          </div>
          <div className="bg-slate-50 rounded-lg p-2 text-center">
            <div className="text-slate-400 mb-0.5">f'(x)</div>
            <div className={`font-mono font-semibold ${Math.abs(dfxVal) < 0.05 ? "text-red-500" : "text-slate-700"}`}>
              {fmt(dfxVal, 4)}
            </div>
          </div>
          <div className="bg-slate-50 rounded-lg p-2 text-center">
            <div className="text-slate-400 mb-0.5">Gradient Status</div>
            <div className={`font-semibold ${Math.abs(dfxVal) < 0.05 ? "text-red-500" : Math.abs(dfxVal) < 0.2 ? "text-amber-500" : "text-green-600"}`}>
              {Math.abs(dfxVal) < 0.05 ? "Vanishing!" : Math.abs(dfxVal) < 0.2 ? "Weak" : "Healthy"}
            </div>
          </div>
        </div>
      </div>

      {/* Educational explanation */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Why vanishing gradients matter:</p>
        <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
          <li>In a deep network, gradients are <strong>multiplied</strong> through each layer during backpropagation.</li>
          <li>If f'(x) is less than 0.25 at each of 10 layers, the gradient shrinks by 0.25^10 = 0.00000095 — effectively zero.</li>
          <li>This means early layers <strong>stop learning</strong>, making deep networks with sigmoid/tanh hard to train.</li>
          <li>ReLU avoids this for x &gt; 0 (f'(x) = 1), but creates "dead neurons" for x &lt; 0 where f'(x) = 0 permanently.</li>
          <li>Leaky ReLU and ELU solve this by ensuring a small but non-zero gradient everywhere.</li>
        </ul>
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 3 — DEAD RELU PROBLEM                                               ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const DEAD_SVG_W = 520;
const DEAD_SVG_H = 340;

interface NeuronInfo {
  z: number;  // pre-activation
  a: number;  // post-activation
  isDead: boolean;
}

interface NetworkResult {
  hidden: NeuronInfo[];
  output: NeuronInfo;
}

function computeMiniNetwork(
  x1: number,
  x2: number,
  wIn: number[][],     // 2x4 input->hidden weights
  bHidden: number[],   // 4 hidden biases
  wOut: number[],      // 4 hidden->output weights
  bOut: number,        // output bias
  activation: "relu" | "leaky_relu"
): NetworkResult {
  const alpha = 0.1;
  const hidden: NeuronInfo[] = [];
  for (let j = 0; j < 4; j++) {
    const z = x1 * wIn[0][j] + x2 * wIn[1][j] + bHidden[j];
    let a: number;
    if (activation === "relu") {
      a = Math.max(0, z);
    } else {
      a = z > 0 ? z : alpha * z;
    }
    hidden.push({ z, a, isDead: activation === "relu" && z <= 0 });
  }
  let zOut = bOut;
  for (let j = 0; j < 4; j++) {
    zOut += hidden[j].a * wOut[j];
  }
  const aOut = zOut; // linear output
  return { hidden, output: { z: zOut, a: aOut, isDead: false } };
}

function DeadReLUTab() {
  const [x1, setX1] = useState(0.5);
  const [x2, setX2] = useState(-0.3);
  const [seed, setSeed] = useState(42);

  // Generate weights from seed
  const { wIn, bHidden, wOut, bOut } = useMemo(() => {
    const rng = mulberry32(seed);
    const w: number[][] = [[], []];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 4; j++) {
        w[i].push((rng() - 0.5) * 4);
      }
    }
    const bH = Array.from({ length: 4 }, () => (rng() - 0.5) * 2);
    const wO = Array.from({ length: 4 }, () => (rng() - 0.5) * 2);
    const bO = (rng() - 0.5) * 1;
    return { wIn: w, bHidden: bH, wOut: wO, bOut: bO };
  }, [seed]);

  const reluResult = useMemo(() => computeMiniNetwork(x1, x2, wIn, bHidden, wOut, bOut, "relu"), [x1, x2, wIn, bHidden, wOut, bOut]);
  const leakyResult = useMemo(() => computeMiniNetwork(x1, x2, wIn, bHidden, wOut, bOut, "leaky_relu"), [x1, x2, wIn, bHidden, wOut, bOut]);

  const reluDeadCount = reluResult.hidden.filter((n) => n.isDead).length;
  const leakyDeadCount = leakyResult.hidden.filter((n) => n.isDead).length;

  const randomizeWeights = useCallback(() => {
    setSeed((prev) => prev + 7);
  }, []);

  // Neuron positions for SVG
  const inputNeurons = [
    { x: 80, y: 100, label: "x1" },
    { x: 80, y: 240, label: "x2" },
  ];
  const hiddenNeurons = [
    { x: 230, y: 55 },
    { x: 230, y: 135 },
    { x: 230, y: 215 },
    { x: 230, y: 295 },
  ];
  const outputNeuron = { x: 420, y: 170 };

  function renderNetwork(
    result: NetworkResult,
    activationType: string,
    label: string,
    deadCount: number
  ) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-700">{label}</span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
            deadCount > 0 ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
          }`}>
            {deadCount} / 4 dead neurons
          </span>
        </div>
        <div className="p-2">
          <svg viewBox={`0 0 ${DEAD_SVG_W} ${DEAD_SVG_H}`} className="w-full" style={{ aspectRatio: `${DEAD_SVG_W} / ${DEAD_SVG_H}` }}>
            {/* Connections: input -> hidden */}
            {inputNeurons.map((inp, i) =>
              hiddenNeurons.map((hid, j) => {
                const w = wIn[i][j];
                const opacity = 0.2 + Math.min(Math.abs(w) / 2, 1) * 0.6;
                return (
                  <line
                    key={`ih-${i}-${j}`}
                    x1={inp.x + 20} y1={inp.y} x2={hid.x - 22} y2={hid.y}
                    stroke={w >= 0 ? "#3b82f6" : "#ef4444"}
                    strokeWidth={1 + Math.abs(w) * 0.5}
                    opacity={result.hidden[j].isDead ? 0.15 : opacity}
                  />
                );
              })
            )}

            {/* Connections: hidden -> output */}
            {hiddenNeurons.map((hid, j) => {
              const w = wOut[j];
              const opacity = 0.2 + Math.min(Math.abs(w) / 2, 1) * 0.6;
              return (
                <line
                  key={`ho-${j}`}
                  x1={hid.x + 22} y1={hid.y} x2={outputNeuron.x - 22} y2={outputNeuron.y}
                  stroke={w >= 0 ? "#3b82f6" : "#ef4444"}
                  strokeWidth={1 + Math.abs(w) * 0.5}
                  opacity={result.hidden[j].isDead ? 0.15 : opacity}
                />
              );
            })}

            {/* Input neurons */}
            {inputNeurons.map((inp, i) => (
              <g key={`in-${i}`}>
                <circle cx={inp.x} cy={inp.y} r={20} fill="#e0e7ff" stroke="#6366f1" strokeWidth={2} />
                <text x={inp.x} y={inp.y - 5} textAnchor="middle" fontSize={10} fontWeight={700} fill="#4338ca">{inp.label}</text>
                <text x={inp.x} y={inp.y + 8} textAnchor="middle" fontSize={9} fill="#6366f1" fontFamily="monospace">
                  {(i === 0 ? x1 : x2).toFixed(2)}
                </text>
              </g>
            ))}

            {/* Hidden neurons */}
            {hiddenNeurons.map((hid, j) => {
              const neuron = result.hidden[j];
              const dead = neuron.isDead;
              return (
                <g key={`hid-${j}`}>
                  <circle
                    cx={hid.x} cy={hid.y} r={22}
                    fill={dead ? "#f1f5f9" : "#dcfce7"}
                    stroke={dead ? "#94a3b8" : "#22c55e"}
                    strokeWidth={2}
                    opacity={dead ? 0.5 : 1}
                  />
                  {dead && (
                    <>
                      <line x1={hid.x - 12} y1={hid.y - 12} x2={hid.x + 12} y2={hid.y + 12} stroke="#dc2626" strokeWidth={2} />
                      <line x1={hid.x + 12} y1={hid.y - 12} x2={hid.x - 12} y2={hid.y + 12} stroke="#dc2626" strokeWidth={2} />
                    </>
                  )}
                  <text x={hid.x} y={hid.y - 7} textAnchor="middle" fontSize={8} fill={dead ? "#94a3b8" : "#15803d"} fontWeight={600}>
                    z={neuron.z.toFixed(2)}
                  </text>
                  <text x={hid.x} y={hid.y + 5} textAnchor="middle" fontSize={8} fill={dead ? "#94a3b8" : "#15803d"} fontFamily="monospace">
                    a={neuron.a.toFixed(2)}
                  </text>
                  <text x={hid.x} y={hid.y + 16} textAnchor="middle" fontSize={7} fill={dead ? "#dc2626" : "#6b7280"} fontWeight={dead ? 700 : 400}>
                    {dead ? "DEAD" : "active"}
                  </text>
                </g>
              );
            })}

            {/* Output neuron */}
            <g>
              <circle cx={outputNeuron.x} cy={outputNeuron.y} r={22} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
              <text x={outputNeuron.x} y={outputNeuron.y - 4} textAnchor="middle" fontSize={9} fontWeight={700} fill="#92400e">output</text>
              <text x={outputNeuron.x} y={outputNeuron.y + 9} textAnchor="middle" fontSize={9} fill="#92400e" fontFamily="monospace">
                {result.output.a.toFixed(3)}
              </text>
            </g>

            {/* Layer labels */}
            <text x={80} y={25} textAnchor="middle" fontSize={10} fill="#64748b" fontWeight={600}>Input</text>
            <text x={230} y={25} textAnchor="middle" fontSize={10} fill="#64748b" fontWeight={600}>Hidden ({activationType})</text>
            <text x={420} y={25} textAnchor="middle" fontSize={10} fill="#64748b" fontWeight={600}>Output</text>
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Info */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">The Dead ReLU Problem</h3>
          <p className="text-xs text-indigo-700 mt-1">
            When a neuron's pre-activation value (z) is negative, ReLU outputs exactly 0. If this happens consistently,
            the neuron <strong>never activates</strong> and its gradient is always 0 — it's "dead" and will never learn again.
            Leaky ReLU solves this by allowing a small negative output (alpha * z) so the gradient is never exactly zero.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="text-xs font-semibold text-slate-600 block mb-1.5">Input x1 = {x1.toFixed(2)}</label>
            <input
              type="range" min={-2} max={2} step={0.01} value={x1}
              onChange={(e) => setX1(parseFloat(e.target.value))}
              className="w-full h-2 accent-indigo-500"
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-600 block mb-1.5">Input x2 = {x2.toFixed(2)}</label>
            <input
              type="range" min={-2} max={2} step={0.01} value={x2}
              onChange={(e) => setX2(parseFloat(e.target.value))}
              className="w-full h-2 accent-indigo-500"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={randomizeWeights}
              className="flex items-center gap-1.5 px-4 py-2 bg-indigo-600 text-white text-xs font-semibold rounded-lg hover:bg-indigo-700 transition-colors w-full justify-center"
            >
              <Shuffle className="w-3.5 h-3.5" />
              Randomize Weights
            </button>
          </div>
        </div>
      </div>

      {/* Side-by-side comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {renderNetwork(reluResult, "ReLU", "Network with ReLU", reluDeadCount)}
        {renderNetwork(leakyResult, "Leaky ReLU", "Network with Leaky ReLU (alpha=0.1)", leakyDeadCount)}
      </div>

      {/* Statistics comparison */}
      <div className="grid grid-cols-2 gap-4">
        <div className={`rounded-xl p-4 border-2 ${reluDeadCount > 0 ? "border-red-200 bg-red-50" : "border-green-200 bg-green-50"}`}>
          <div className="text-xs font-semibold text-slate-700 mb-2">ReLU Network</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{4 - reluDeadCount}</div>
              <div className="text-[10px] text-slate-500">Active</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-500">{reluDeadCount}</div>
              <div className="text-[10px] text-slate-500">Dead</div>
            </div>
          </div>
          <div className="mt-2 flex gap-1">
            {reluResult.hidden.map((n, i) => (
              <div key={i} className={`flex-1 h-2 rounded-full ${n.isDead ? "bg-red-400" : "bg-green-400"}`} />
            ))}
          </div>
        </div>
        <div className="border-2 border-green-200 bg-green-50 rounded-xl p-4">
          <div className="text-xs font-semibold text-slate-700 mb-2">Leaky ReLU Network</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">4</div>
              <div className="text-[10px] text-slate-500">Active</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">0</div>
              <div className="text-[10px] text-slate-500">Dead</div>
            </div>
          </div>
          <div className="mt-2 flex gap-1">
            {leakyResult.hidden.map((_, i) => (
              <div key={i} className="flex-1 h-2 rounded-full bg-green-400" />
            ))}
          </div>
        </div>
      </div>

      {/* Detailed comparison table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50">
          <span className="text-xs font-semibold text-slate-700">Neuron-by-Neuron Comparison</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50/50">
                <th className="text-left px-4 py-2 font-semibold text-slate-500">Neuron</th>
                <th className="text-right px-3 py-2 font-semibold text-slate-500">z (pre-act)</th>
                <th className="text-right px-3 py-2 font-semibold text-slate-500">ReLU a</th>
                <th className="text-center px-3 py-2 font-semibold text-slate-500">ReLU Status</th>
                <th className="text-right px-3 py-2 font-semibold text-slate-500">Leaky a</th>
                <th className="text-center px-3 py-2 font-semibold text-slate-500">Leaky Status</th>
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2, 3].map((j) => (
                <tr key={j} className="border-b border-slate-50 hover:bg-slate-50/50">
                  <td className="px-4 py-2 font-medium text-slate-700">H{j + 1}</td>
                  <td className="px-3 py-2 text-right font-mono text-slate-700">{reluResult.hidden[j].z.toFixed(3)}</td>
                  <td className="px-3 py-2 text-right font-mono text-slate-700">{reluResult.hidden[j].a.toFixed(3)}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                      reluResult.hidden[j].isDead ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                    }`}>
                      {reluResult.hidden[j].isDead ? "DEAD" : "active"}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-slate-700">{leakyResult.hidden[j].a.toFixed(3)}</td>
                  <td className="px-3 py-2 text-center">
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-green-100 text-green-700">
                      active
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Tips */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Things to explore:</p>
        <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
          <li>Click <strong>Randomize Weights</strong> repeatedly — some initializations kill 3-4 neurons at once!</li>
          <li>Drag x1 and x2 sliders to negative values — more ReLU neurons tend to die.</li>
          <li>Notice that Leaky ReLU <strong>never has dead neurons</strong> — the small alpha=0.1 slope keeps gradients flowing.</li>
          <li>In practice, about 10-40% of ReLU neurons can be dead in a trained network, wasting capacity.</li>
        </ul>
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 4 — SOFTMAX VISUALIZATION                                           ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const SM_BAR_SVG_W = 500;
const SM_BAR_SVG_H = 220;
const SM_BAR_PAD = { top: 25, right: 20, bottom: 40, left: 50 };

const CLASS_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f97316", "#8b5cf6"];
const CLASS_LABELS = ["Cat", "Dog", "Bird", "Fish", "Frog"];

function softmax(logits: number[], temperature: number): number[] {
  const scaled = logits.map((z) => z / Math.max(temperature, 0.01));
  const maxZ = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - maxZ));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExp);
}

function SoftmaxTab() {
  const [logits, setLogits] = useState([2.0, 1.0, 0.5, -0.5, -1.0]);
  const [temperature, setTemperature] = useState(1.0);
  const [showPie, setShowPie] = useState(false);

  const probs = useMemo(() => softmax(logits, temperature), [logits, temperature]);
  const exps = useMemo(() => {
    const scaled = logits.map((z) => z / Math.max(temperature, 0.01));
    const maxZ = Math.max(...scaled);
    return scaled.map((z) => Math.exp(z - maxZ));
  }, [logits, temperature]);
  const sumExp = useMemo(() => exps.reduce((a, b) => a + b, 0), [exps]);

  const setLogit = useCallback((idx: number, val: number) => {
    setLogits((prev) => {
      const next = [...prev];
      next[idx] = val;
      return next;
    });
  }, []);

  const maxProb = Math.max(...probs);
  const maxIdx = probs.indexOf(maxProb);

  // Bar chart dimensions
  const barPlotW = SM_BAR_SVG_W - SM_BAR_PAD.left - SM_BAR_PAD.right;
  const barPlotH = SM_BAR_SVG_H - SM_BAR_PAD.top - SM_BAR_PAD.bottom;
  const barW = barPlotW / logits.length * 0.6;
  const barGap = barPlotW / logits.length;

  // Pie chart computation
  const pieData = useMemo(() => {
    let cumAngle = -Math.PI / 2;
    return probs.map((p, i) => {
      const angle = p * 2 * Math.PI;
      const startAngle = cumAngle;
      cumAngle += angle;
      const endAngle = cumAngle;
      const midAngle = (startAngle + endAngle) / 2;
      const r = 80;
      const x1 = 100 + r * Math.cos(startAngle);
      const y1 = 100 + r * Math.sin(startAngle);
      const x2 = 100 + r * Math.cos(endAngle);
      const y2 = 100 + r * Math.sin(endAngle);
      const largeArc = angle > Math.PI ? 1 : 0;
      const labelR = r * 0.65;
      const lx = 100 + labelR * Math.cos(midAngle);
      const ly = 100 + labelR * Math.sin(midAngle);
      return {
        path: `M 100 100 L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`,
        labelX: lx,
        labelY: ly,
        prob: p,
        color: CLASS_COLORS[i],
        label: CLASS_LABELS[i],
      };
    });
  }, [probs]);

  return (
    <div className="space-y-4">
      {/* Info */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">Softmax: From Raw Scores to Probabilities</h3>
          <p className="text-xs text-indigo-700 mt-1">
            Softmax converts raw model outputs (logits) into a probability distribution that sums to 1.0.
            The <strong>temperature</strong> parameter controls how "sharp" or "uniform" the distribution is:
            low temperature makes the model more confident, high temperature spreads probability more evenly.
          </p>
        </div>
      </div>

      {/* Formula display */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
        <div className="text-xs font-semibold text-slate-600 mb-2">Softmax Formula:</div>
        <div className="text-sm font-mono text-slate-800 text-center py-2 bg-white rounded-lg border border-slate-200">
          P(class_i) = exp(z_i / T) / sum( exp(z_j / T) )
        </div>
        <div className="text-[10px] text-slate-500 mt-2 text-center">
          Where z_i is the logit for class i, and T is the temperature parameter
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Logit sliders */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs font-semibold text-slate-700 mb-3">Raw Logits (Model Outputs)</div>
          {logits.map((z, i) => (
            <div key={i} className="flex items-center gap-2 mb-2.5">
              <span className="inline-block w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: CLASS_COLORS[i] }} />
              <span className="text-xs font-medium text-slate-600 w-10">{CLASS_LABELS[i]}</span>
              <input
                type="range" min={-5} max={5} step={0.1} value={z}
                onChange={(e) => setLogit(i, parseFloat(e.target.value))}
                className="flex-1 h-2 accent-indigo-500"
              />
              <span className="text-xs font-mono text-slate-700 w-12 text-right">{z.toFixed(1)}</span>
            </div>
          ))}
          <div className="mt-4 pt-3 border-t border-slate-100">
            <div className="flex items-center gap-2 mb-2">
              <Thermometer className="w-3.5 h-3.5 text-amber-500" />
              <span className="text-xs font-semibold text-slate-600">Temperature: {temperature.toFixed(2)}</span>
            </div>
            <input
              type="range" min={0.1} max={5} step={0.05} value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-2 accent-amber-500"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>0.1 (very confident)</span>
              <span>1.0 (normal)</span>
              <span>5.0 (uniform)</span>
            </div>
          </div>
        </div>

        {/* Step-by-step computation */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs font-semibold text-slate-700 mb-3">Step-by-Step Computation</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="text-left px-2 py-1.5 font-semibold text-slate-500">Class</th>
                  <th className="text-right px-2 py-1.5 font-semibold text-slate-500">z/T</th>
                  <th className="text-right px-2 py-1.5 font-semibold text-slate-500">exp(z/T)</th>
                  <th className="text-right px-2 py-1.5 font-semibold text-slate-500">Probability</th>
                </tr>
              </thead>
              <tbody>
                {logits.map((z, i) => (
                  <tr key={i} className="border-b border-slate-50">
                    <td className="px-2 py-1.5 flex items-center gap-1.5">
                      <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLASS_COLORS[i] }} />
                      <span className="font-medium text-slate-700">{CLASS_LABELS[i]}</span>
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-slate-600">{(z / Math.max(temperature, 0.01)).toFixed(2)}</td>
                    <td className="px-2 py-1.5 text-right font-mono text-slate-600">{exps[i].toFixed(3)}</td>
                    <td className={`px-2 py-1.5 text-right font-mono font-semibold ${i === maxIdx ? "text-indigo-700" : "text-slate-700"}`}>
                      {(probs[i] * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
                <tr className="bg-slate-50">
                  <td className="px-2 py-1.5 font-semibold text-slate-600" colSpan={2}>Sum</td>
                  <td className="px-2 py-1.5 text-right font-mono font-semibold text-slate-700">{sumExp.toFixed(3)}</td>
                  <td className="px-2 py-1.5 text-right font-mono font-semibold text-green-600">100.0%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Toggle view */}
      <div className="flex justify-center">
        <button
          onClick={() => setShowPie(!showPie)}
          className="flex items-center gap-1.5 px-4 py-2 bg-slate-100 text-slate-700 text-xs font-semibold rounded-lg hover:bg-slate-200 transition-colors"
        >
          {showPie ? <BarChart3 className="w-3.5 h-3.5" /> : <PieChart className="w-3.5 h-3.5" />}
          Switch to {showPie ? "Bar Chart" : "Pie Chart"}
        </button>
      </div>

      {/* Chart visualization */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
          <span className="text-xs font-semibold text-slate-700">
            Probability Distribution — Predicted: <span style={{ color: CLASS_COLORS[maxIdx] }}>{CLASS_LABELS[maxIdx]}</span> ({(maxProb * 100).toFixed(1)}%)
          </span>
        </div>
        <div className="p-3 flex justify-center">
          {!showPie ? (
            <svg viewBox={`0 0 ${SM_BAR_SVG_W} ${SM_BAR_SVG_H}`} className="w-full max-w-lg" style={{ aspectRatio: `${SM_BAR_SVG_W} / ${SM_BAR_SVG_H}` }}>
              {/* Y-axis grid lines */}
              {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v) => {
                const y = SM_BAR_PAD.top + barPlotH * (1 - v);
                return (
                  <g key={v}>
                    <line x1={SM_BAR_PAD.left} y1={y} x2={SM_BAR_PAD.left + barPlotW} y2={y} stroke="#f1f5f9" strokeWidth={0.5} />
                    <text x={SM_BAR_PAD.left - 6} y={y + 3} fontSize={9} fill="#94a3b8" textAnchor="end">{(v * 100).toFixed(0)}%</text>
                  </g>
                );
              })}

              {/* Bars */}
              {probs.map((p, i) => {
                const bx = SM_BAR_PAD.left + i * barGap + (barGap - barW) / 2;
                const bh = Math.max(1, p * barPlotH);
                const by = SM_BAR_PAD.top + barPlotH - bh;
                return (
                  <g key={i}>
                    <rect x={bx} y={by} width={barW} height={bh} fill={CLASS_COLORS[i]} rx={3} opacity={0.85} />
                    <text x={bx + barW / 2} y={by - 6} textAnchor="middle" fontSize={10} fontWeight={600} fill={CLASS_COLORS[i]}>
                      {(p * 100).toFixed(1)}%
                    </text>
                    <text x={bx + barW / 2} y={SM_BAR_PAD.top + barPlotH + 16} textAnchor="middle" fontSize={10} fill="#475569" fontWeight={500}>
                      {CLASS_LABELS[i]}
                    </text>
                  </g>
                );
              })}

              {/* Axes */}
              <line x1={SM_BAR_PAD.left} y1={SM_BAR_PAD.top} x2={SM_BAR_PAD.left} y2={SM_BAR_PAD.top + barPlotH} stroke="#cbd5e1" strokeWidth={1} />
              <line x1={SM_BAR_PAD.left} y1={SM_BAR_PAD.top + barPlotH} x2={SM_BAR_PAD.left + barPlotW} y2={SM_BAR_PAD.top + barPlotH} stroke="#cbd5e1" strokeWidth={1} />
            </svg>
          ) : (
            <svg viewBox="0 0 200 200" className="w-64 h-64">
              {pieData.map((d, i) => (
                <g key={i}>
                  <path d={d.path} fill={d.color} stroke="#fff" strokeWidth={2} opacity={0.85} />
                  {d.prob > 0.05 && (
                    <text x={d.labelX} y={d.labelY} textAnchor="middle" dominantBaseline="middle" fontSize={9} fill="#fff" fontWeight={700}>
                      {(d.prob * 100).toFixed(0)}%
                    </text>
                  )}
                </g>
              ))}
              <circle cx={100} cy={100} r={20} fill="#fff" />
              <text x={100} y={100} textAnchor="middle" dominantBaseline="middle" fontSize={8} fill="#475569" fontWeight={600}>
                {CLASS_LABELS[maxIdx]}
              </text>
            </svg>
          )}
        </div>
      </div>

      {/* Temperature effect visualization */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
        <div className="text-xs font-semibold text-slate-700 mb-3">Temperature Effect on Probability Distribution</div>
        <div className="grid grid-cols-3 gap-3">
          {[0.1, 1.0, 5.0].map((t) => {
            const p = softmax(logits, t);
            const maxP = Math.max(...p);
            return (
              <div key={t} className={`rounded-lg border p-3 ${Math.abs(temperature - t) < 0.05 ? "border-indigo-300 bg-indigo-50" : "border-slate-200 bg-slate-50"}`}>
                <div className="text-[10px] font-semibold text-slate-600 mb-2 text-center">T = {t}</div>
                {p.map((prob, i) => (
                  <div key={i} className="flex items-center gap-1 mb-1">
                    <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: CLASS_COLORS[i] }} />
                    <div className="flex-1 bg-slate-200 rounded-full h-2 overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${prob * 100}%`, backgroundColor: CLASS_COLORS[i], opacity: 0.8 }} />
                    </div>
                    <span className="text-[10px] font-mono text-slate-600 w-10 text-right">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
                <div className="text-[10px] text-center mt-1.5 font-medium text-slate-500">
                  {t <= 0.2 ? "Almost one-hot" : t <= 1.5 ? "Standard" : "Nearly uniform"}
                </div>
                <div className="text-[10px] text-center text-slate-400">
                  max = {(maxP * 100).toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Tips */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Things to explore:</p>
        <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
          <li>Increase one logit by 1 — notice how the probability changes are <strong>non-linear</strong> (disproportionate).</li>
          <li>Set all logits to the same value — probabilities become uniform (1/5 = 20% each).</li>
          <li>Set temperature very low (0.1) — the highest logit dominates almost completely.</li>
          <li>Set temperature very high (5.0) — even large logit differences produce similar probabilities.</li>
          <li>This is why softmax is used in classification: it emphasizes the most confident prediction.</li>
        </ul>
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 5 — EFFECT ON NETWORK OUTPUT (Decision Boundaries)                  ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const DB_SVG_W = 280;
const DB_SVG_H = 280;
const DB_PAD = 30;
const DB_PLOT = DB_SVG_W - 2 * DB_PAD;
const DB_RESOLUTION = 40; // grid resolution for decision boundary
const DB_CELL = DB_PLOT / DB_RESOLUTION;
const DB_DATA_RANGE = 3; // data spans [-3, 3]

type BoundaryActivation = "relu" | "sigmoid" | "tanh" | "linear";

const BOUNDARY_ACTS: { key: BoundaryActivation; label: string; color: string }[] = [
  { key: "relu", label: "ReLU", color: "#ef4444" },
  { key: "sigmoid", label: "Sigmoid", color: "#3b82f6" },
  { key: "tanh", label: "Tanh", color: "#22c55e" },
  { key: "linear", label: "Linear", color: "#94a3b8" },
];

function applyBoundaryAct(x: number, act: BoundaryActivation): number {
  switch (act) {
    case "relu": return Math.max(0, x);
    case "sigmoid": return sigmoid(x);
    case "tanh": return Math.tanh(x);
    case "linear": return x;
  }
}

function applyBoundaryActDeriv(x: number, act: BoundaryActivation): number {
  switch (act) {
    case "relu": return x > 0 ? 1 : 0;
    case "sigmoid": { const s = sigmoid(x); return s * (1 - s); }
    case "tanh": { const t = Math.tanh(x); return 1 - t * t; }
    case "linear": return 1;
  }
}

// Tiny network: 2 -> 4 -> 1 with trainable weights
interface TinyNetwork {
  w1: number[][]; // 2x4
  b1: number[];   // 4
  w2: number[];   // 4
  b2: number;     // 1
}

function initTinyNetwork(rng: () => number): TinyNetwork {
  const w1: number[][] = [[], []];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 4; j++) {
      w1[i].push((rng() - 0.5) * 2);
    }
  }
  const b1 = Array.from({ length: 4 }, () => (rng() - 0.5) * 0.5);
  const w2 = Array.from({ length: 4 }, () => (rng() - 0.5) * 2);
  const b2 = (rng() - 0.5) * 0.5;
  return { w1, b1, w2, b2 };
}

function forwardTiny(net: TinyNetwork, x1: number, x2: number, act: BoundaryActivation): number {
  const hidden: number[] = [];
  for (let j = 0; j < 4; j++) {
    const z = net.w1[0][j] * x1 + net.w1[1][j] * x2 + net.b1[j];
    hidden.push(applyBoundaryAct(z, act));
  }
  let out = net.b2;
  for (let j = 0; j < 4; j++) {
    out += hidden[j] * net.w2[j];
  }
  return sigmoid(out); // output sigmoid for binary classification
}

function trainTinyStep(
  net: TinyNetwork,
  data: { x1: number; x2: number; label: number }[],
  act: BoundaryActivation,
  lr: number
): TinyNetwork {
  // Copy network
  const newW1 = net.w1.map((r) => [...r]);
  const newB1 = [...net.b1];
  const newW2 = [...net.w2];
  let newB2 = net.b2;

  // Mini-batch gradient descent
  const dW1 = [[0, 0, 0, 0], [0, 0, 0, 0]];
  const dB1 = [0, 0, 0, 0];
  const dW2 = [0, 0, 0, 0];
  let dB2 = 0;

  for (const d of data) {
    // Forward
    const z1: number[] = [];
    const a1: number[] = [];
    for (let j = 0; j < 4; j++) {
      const z = net.w1[0][j] * d.x1 + net.w1[1][j] * d.x2 + net.b1[j];
      z1.push(z);
      a1.push(applyBoundaryAct(z, act));
    }
    let z2 = net.b2;
    for (let j = 0; j < 4; j++) z2 += a1[j] * net.w2[j];
    const a2 = sigmoid(z2);

    // Backward
    const dLoss = a2 - d.label; // derivative of BCE loss
    const dZ2 = dLoss * a2 * (1 - a2);

    dB2 += dZ2;
    for (let j = 0; j < 4; j++) {
      dW2[j] += dZ2 * a1[j];
      const dA1 = dZ2 * net.w2[j];
      const dZ1 = dA1 * applyBoundaryActDeriv(z1[j], act);
      dB1[j] += dZ1;
      dW1[0][j] += dZ1 * d.x1;
      dW1[1][j] += dZ1 * d.x2;
    }
  }

  const n = data.length;
  for (let j = 0; j < 4; j++) {
    newW2[j] -= lr * dW2[j] / n;
    newB1[j] -= lr * dB1[j] / n;
    for (let i = 0; i < 2; i++) {
      newW1[i][j] -= lr * dW1[i][j] / n;
    }
  }
  newB2 -= lr * dB2 / n;

  return { w1: newW1, b1: newB1, w2: newW2, b2: newB2 };
}

function generateXORData(rng: () => number, count: number): { x1: number; x2: number; label: number }[] {
  const data: { x1: number; x2: number; label: number }[] = [];
  for (let i = 0; i < count; i++) {
    const cx = rng() > 0.5 ? 1.2 : -1.2;
    const cy = rng() > 0.5 ? 1.2 : -1.2;
    const label = (cx > 0) !== (cy > 0) ? 1 : 0; // XOR pattern
    const x1 = cx + seededGaussian(rng) * 0.5;
    const x2 = cy + seededGaussian(rng) * 0.5;
    data.push({ x1, x2, label });
  }
  return data;
}

function computeAccuracy(
  net: TinyNetwork,
  data: { x1: number; x2: number; label: number }[],
  act: BoundaryActivation
): number {
  let correct = 0;
  for (const d of data) {
    const pred = forwardTiny(net, d.x1, d.x2, act) > 0.5 ? 1 : 0;
    if (pred === d.label) correct++;
  }
  return correct / data.length;
}

function DecisionBoundaryTab() {
  const [dataSeed] = useState(123);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const trainingRef = useRef(false);
  const animRef = useRef<number | null>(null);

  // Generate data once
  const data = useMemo(() => {
    const rng = mulberry32(dataSeed);
    return generateXORData(rng, 120);
  }, [dataSeed]);

  // Separate networks per activation
  const [networks, setNetworks] = useState<Record<BoundaryActivation, TinyNetwork>>(() => {
    const rng = mulberry32(999);
    const base = initTinyNetwork(rng);
    // Same initial weights for fair comparison
    return {
      relu: JSON.parse(JSON.stringify(base)),
      sigmoid: JSON.parse(JSON.stringify(base)),
      tanh: JSON.parse(JSON.stringify(base)),
      linear: JSON.parse(JSON.stringify(base)),
    };
  });

  // Compute decision boundaries
  const boundaries = useMemo(() => {
    const result: Record<string, number[][]> = {};
    for (const act of BOUNDARY_ACTS) {
      const grid: number[][] = [];
      for (let gy = 0; gy < DB_RESOLUTION; gy++) {
        const row: number[] = [];
        for (let gx = 0; gx < DB_RESOLUTION; gx++) {
          const x1 = -DB_DATA_RANGE + (gx + 0.5) / DB_RESOLUTION * 2 * DB_DATA_RANGE;
          const x2 = -DB_DATA_RANGE + (gy + 0.5) / DB_RESOLUTION * 2 * DB_DATA_RANGE;
          row.push(forwardTiny(networks[act.key], x1, x2, act.key));
        }
        grid.push(row);
      }
      result[act.key] = grid;
    }
    return result;
  }, [networks]);

  // Accuracies
  const accuracies = useMemo(() => {
    const result: Record<string, number> = {};
    for (const act of BOUNDARY_ACTS) {
      result[act.key] = computeAccuracy(networks[act.key], data, act.key);
    }
    return result;
  }, [networks, data]);

  // Training loop
  const trainStep = useCallback(() => {
    if (!trainingRef.current) return;

    setNetworks((prev) => {
      const next = { ...prev };
      for (const act of BOUNDARY_ACTS) {
        next[act.key] = trainTinyStep(prev[act.key], data, act.key, 1.5);
      }
      return next;
    });
    setEpoch((e) => e + 1);

    animRef.current = requestAnimationFrame(trainStep);
  }, [data]);

  const startTraining = useCallback(() => {
    trainingRef.current = true;
    setIsTraining(true);
    animRef.current = requestAnimationFrame(trainStep);
  }, [trainStep]);

  const stopTraining = useCallback(() => {
    trainingRef.current = false;
    setIsTraining(false);
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
  }, []);

  const resetNetworks = useCallback(() => {
    stopTraining();
    const rng = mulberry32(999);
    const base = initTinyNetwork(rng);
    setNetworks({
      relu: JSON.parse(JSON.stringify(base)),
      sigmoid: JSON.parse(JSON.stringify(base)),
      tanh: JSON.parse(JSON.stringify(base)),
      linear: JSON.parse(JSON.stringify(base)),
    });
    setEpoch(0);
  }, [stopTraining]);

  useEffect(() => {
    return () => {
      trainingRef.current = false;
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  function dataToSvg(v: number): number {
    return DB_PAD + ((v + DB_DATA_RANGE) / (2 * DB_DATA_RANGE)) * DB_PLOT;
  }

  function renderBoundaryChart(actKey: BoundaryActivation, actLabel: string, actColor: string) {
    const grid = boundaries[actKey];
    const acc = accuracies[actKey];

    return (
      <div key={actKey} className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
          <span className="text-xs font-semibold" style={{ color: actColor }}>{actLabel}</span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
            acc > 0.85 ? "bg-green-100 text-green-700" : acc > 0.6 ? "bg-amber-100 text-amber-700" : "bg-red-100 text-red-700"
          }`}>
            {(acc * 100).toFixed(1)}% accuracy
          </span>
        </div>
        <div className="p-1">
          <svg viewBox={`0 0 ${DB_SVG_W} ${DB_SVG_H}`} className="w-full" style={{ aspectRatio: "1" }}>
            {/* Decision boundary heatmap */}
            {grid.map((row, gy) =>
              row.map((val, gx) => {
                const x = DB_PAD + gx * DB_CELL;
                const y = DB_PAD + gy * DB_CELL;
                // Blue for class 0, red for class 1
                const r = Math.round(val * 220 + 35);
                const b = Math.round((1 - val) * 220 + 35);
                const g = Math.round(80 + (1 - Math.abs(val - 0.5) * 2) * 50);
                return (
                  <rect
                    key={`${gx}-${gy}`}
                    x={x} y={y} width={DB_CELL + 0.5} height={DB_CELL + 0.5}
                    fill={`rgb(${r},${g},${b})`}
                    opacity={0.6}
                  />
                );
              })
            )}

            {/* Data points */}
            {data.map((d, i) => {
              const cx = dataToSvg(d.x1);
              const cy = dataToSvg(d.x2);
              if (cx < DB_PAD || cx > DB_PAD + DB_PLOT || cy < DB_PAD || cy > DB_PAD + DB_PLOT) return null;
              return (
                <circle
                  key={i}
                  cx={cx} cy={cy} r={2.5}
                  fill={d.label === 1 ? "#ef4444" : "#3b82f6"}
                  stroke="#fff"
                  strokeWidth={0.8}
                />
              );
            })}

            {/* Border */}
            <rect x={DB_PAD} y={DB_PAD} width={DB_PLOT} height={DB_PLOT} fill="none" stroke="#cbd5e1" strokeWidth={1} />

            {/* Axis labels */}
            <text x={DB_SVG_W / 2} y={DB_SVG_H - 5} textAnchor="middle" fontSize={9} fill="#64748b">x1</text>
            <text x={8} y={DB_SVG_H / 2} textAnchor="middle" fontSize={9} fill="#64748b" transform={`rotate(-90, 8, ${DB_SVG_H / 2})`}>x2</text>
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Info */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">How Activation Functions Shape Decision Boundaries</h3>
          <p className="text-xs text-indigo-700 mt-1">
            A neural network's decision boundary depends heavily on its activation function. <strong>Linear</strong> activation
            can only create straight boundaries (it cannot solve XOR!). Non-linear activations like ReLU, Sigmoid, and Tanh
            create curved boundaries that can separate complex patterns. Train all four networks simultaneously and compare.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={isTraining ? stopTraining : startTraining}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg transition-colors ${
              isTraining
                ? "bg-amber-500 text-white hover:bg-amber-600"
                : "bg-indigo-600 text-white hover:bg-indigo-700"
            }`}
          >
            {isTraining ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            {isTraining ? "Stop Training" : "Start Training"}
          </button>
          <button
            onClick={resetNetworks}
            className="flex items-center gap-1.5 px-4 py-2 bg-slate-100 text-slate-700 text-xs font-semibold rounded-lg hover:bg-slate-200 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
          <div className="text-xs text-slate-500">
            Epoch: <span className="font-mono font-semibold text-slate-700">{epoch}</span>
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <div className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-full bg-red-500" />
              <span className="text-[10px] text-slate-500">Class 1</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-[10px] text-slate-500">Class 0</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2x2 grid of decision boundaries */}
      <div className="grid grid-cols-2 gap-3">
        {BOUNDARY_ACTS.map((act) => renderBoundaryChart(act.key, act.label, act.color))}
      </div>

      {/* Accuracy comparison table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50">
          <span className="text-xs font-semibold text-slate-700">Accuracy Comparison (after {epoch} epochs)</span>
        </div>
        <div className="p-3">
          <div className="grid grid-cols-4 gap-3">
            {BOUNDARY_ACTS.map((act) => {
              const acc = accuracies[act.key];
              return (
                <div key={act.key} className="text-center">
                  <div className="text-xs font-semibold mb-1" style={{ color: act.color }}>{act.label}</div>
                  <div className="text-2xl font-bold text-slate-800">{(acc * 100).toFixed(1)}%</div>
                  <div className="mt-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ width: `${acc * 100}%`, backgroundColor: act.color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Observation prompts */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Things to observe while training:</p>
        <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
          <li><strong>Linear</strong> activation will plateau around 50% — it simply cannot solve the XOR problem with a straight line.</li>
          <li><strong>ReLU</strong> creates piecewise-linear boundaries with sharp angular edges.</li>
          <li><strong>Sigmoid</strong> and <strong>Tanh</strong> create smooth, curved boundaries but may train slower due to saturation.</li>
          <li>Watch the heatmap colors: <strong>sharp color transitions</strong> = confident model, <strong>gradual blending</strong> = uncertain model.</li>
          <li>The XOR pattern requires the network to learn that "same sign = class 0, different sign = class 1" — this is fundamentally non-linear.</li>
          <li>Try clicking Reset and then Start multiple times — observe how the final accuracy varies per activation type.</li>
        </ul>
      </div>

      {/* Detailed explanation */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <p className="text-xs text-slate-700 font-medium mb-1">Why linear activation fails:</p>
        <p className="text-xs text-slate-600">
          With linear activation, f(x) = x, so the composition of two linear layers is still linear:
          W2 * (W1 * x + b1) + b2 = (W2*W1) * x + (W2*b1 + b2) — just another linear function.
          No matter how many linear layers you stack, the result is a single linear transformation.
          Non-linear activations break this limitation, allowing the network to learn curved decision boundaries.
        </p>
      </div>
    </div>
  );
}
