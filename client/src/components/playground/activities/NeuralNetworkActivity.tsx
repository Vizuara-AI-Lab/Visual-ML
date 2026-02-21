/**
 * Neural Network Explorer — comprehensive interactive SVG activity
 * Five tabbed explorations: Build & Propagate, Weight Inspector,
 * Activation Comparison, XOR Problem, and Depth Experiment.
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Network,
  Info,
  Play,
  Shuffle,
  Plus,
  Minus,
  Eye,
  Layers,
  GitBranch,
  BarChart3,
  Pause,
  RotateCcw,
  Sliders,
} from "lucide-react";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
type ActivationFn = "relu" | "sigmoid" | "tanh";
type TabKey =
  | "build"
  | "weights"
  | "activations"
  | "xor"
  | "depth";

interface NeuronState {
  value: number;
  activated: number;
  glowing: boolean;
}

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Activation functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function applyActivation(x: number, fn: ActivationFn): number {
  switch (fn) {
    case "relu":
      return Math.max(0, x);
    case "sigmoid":
      return 1 / (1 + Math.exp(-clamp(x, -500, 500)));
    case "tanh":
      return Math.tanh(x);
  }
}

function activationDerivative(x: number, fn: ActivationFn): number {
  switch (fn) {
    case "relu":
      return x > 0 ? 1 : 0;
    case "sigmoid": {
      const s = applyActivation(x, "sigmoid");
      return s * (1 - s);
    }
    case "tanh": {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Weight initialization strategies
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function randomWeight(): number {
  return (Math.random() - 0.5) * 2;
}

function xavierWeight(fanIn: number, fanOut: number): number {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  return gaussianRandom() * std;
}

function heWeight(fanIn: number): number {
  const std = Math.sqrt(2 / fanIn);
  return gaussianRandom() * std;
}

function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function initWeights(layerSizes: number[]): number[][][] {
  const weights: number[][][] = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const lw: number[][] = [];
    for (let i = 0; i < layerSizes[l]; i++) {
      const nw: number[] = [];
      for (let j = 0; j < layerSizes[l + 1]; j++) {
        nw.push(randomWeight());
      }
      lw.push(nw);
    }
    weights.push(lw);
  }
  return weights;
}

function initWeightsXavier(layerSizes: number[]): number[][][] {
  const weights: number[][][] = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const lw: number[][] = [];
    for (let i = 0; i < layerSizes[l]; i++) {
      const nw: number[] = [];
      for (let j = 0; j < layerSizes[l + 1]; j++) {
        nw.push(xavierWeight(layerSizes[l], layerSizes[l + 1]));
      }
      lw.push(nw);
    }
    weights.push(lw);
  }
  return weights;
}

function initWeightsHe(layerSizes: number[]): number[][][] {
  const weights: number[][][] = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const lw: number[][] = [];
    for (let i = 0; i < layerSizes[l]; i++) {
      const nw: number[] = [];
      for (let j = 0; j < layerSizes[l + 1]; j++) {
        nw.push(heWeight(layerSizes[l]));
      }
      lw.push(nw);
    }
    weights.push(lw);
  }
  return weights;
}

function initBiases(layerSizes: number[]): number[][] {
  const biases: number[][] = [];
  for (let l = 0; l < layerSizes.length; l++) {
    const lb: number[] = [];
    for (let i = 0; i < layerSizes[l]; i++) {
      lb.push(l === 0 ? 0 : randomWeight() * 0.3);
    }
    biases.push(lb);
  }
  return biases;
}

function initBiasesZero(layerSizes: number[]): number[][] {
  return layerSizes.map((s) => Array(s).fill(0));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Forward pass (generic)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function forwardPass(
  inputValues: number[],
  layerSizes: number[],
  weights: number[][][],
  biases: number[][],
  activation: ActivationFn,
): NeuronState[][] {
  const states: NeuronState[][] = [];
  states.push(
    inputValues.map((v) => ({ value: v, activated: v, glowing: false })),
  );
  for (let l = 1; l < layerSizes.length; l++) {
    const layer: NeuronState[] = [];
    for (let j = 0; j < layerSizes[l]; j++) {
      let sum = biases[l]?.[j] ?? 0;
      for (let i = 0; i < layerSizes[l - 1]; i++) {
        sum += (states[l - 1][i].activated) * (weights[l - 1]?.[i]?.[j] ?? 0);
      }
      const act = applyActivation(sum, activation);
      layer.push({ value: sum, activated: act, glowing: false });
    }
    states.push(layer);
  }
  return states;
}

// Flat forward for raw numbers — returns activations per layer (including input)
function forwardRaw(
  input: number[],
  layerSizes: number[],
  weights: number[][][],
  biases: number[][],
  activation: ActivationFn,
): number[][] {
  const acts: number[][] = [[...input]];
  for (let l = 1; l < layerSizes.length; l++) {
    const layer: number[] = [];
    for (let j = 0; j < layerSizes[l]; j++) {
      let sum = biases[l]?.[j] ?? 0;
      for (let i = 0; i < layerSizes[l - 1]; i++) {
        sum += acts[l - 1][i] * (weights[l - 1]?.[i]?.[j] ?? 0);
      }
      layer.push(applyActivation(sum, activation));
    }
    acts.push(layer);
  }
  return acts;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SVG helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const NEURON_R = 18;

function weightColor(w: number): string {
  const c = clamp(w, -2, 2) / 2;
  if (c >= 0) {
    return `rgba(59, 130, 246, ${0.15 + Math.abs(c) * 0.85})`;
  }
  return `rgba(239, 68, 68, ${0.15 + Math.abs(c) * 0.85})`;
}

function weightStrokeWidth(w: number): number {
  return 0.5 + Math.min(Math.abs(w), 2) * 1.5;
}

function computePositions(
  layerSizes: number[],
  svgW: number,
  svgH: number,
  padX: number,
  padY: number,
): { x: number; y: number }[][] {
  const positions: { x: number; y: number }[][] = [];
  const numLayers = layerSizes.length;
  const xStep = numLayers > 1 ? (svgW - 2 * padX) / (numLayers - 1) : 0;
  for (let l = 0; l < numLayers; l++) {
    const lp: { x: number; y: number }[] = [];
    const count = layerSizes[l];
    const totalH = svgH - 2 * padY;
    const yStep = count > 1 ? totalH / (count - 1) : 0;
    const yStart = count > 1 ? padY : svgH / 2;
    for (let n = 0; n < count; n++) {
      lp.push({ x: padX + l * xStep, y: yStart + n * yStep });
    }
    positions.push(lp);
  }
  return positions;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Shared SVG network renderer
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function NetworkSVG({
  layerSizes,
  weights,
  neuronStates,
  positions,
  svgW,
  svgH,
  neuronR = NEURON_R,
  animDots,
  highlightConn,
  onClickConn,
  filterId,
  showLabels = true,
  inputValues,
}: {
  layerSizes: number[];
  weights: number[][][];
  neuronStates: NeuronState[][] | null;
  positions: { x: number; y: number }[][];
  svgW: number;
  svgH: number;
  neuronR?: number;
  animDots?: { x: number; y: number; opacity: number }[];
  highlightConn?: { l: number; i: number; j: number } | null;
  onClickConn?: (l: number, i: number, j: number) => void;
  filterId?: string;
  showLabels?: boolean;
  inputValues?: number[];
}) {
  const fid = filterId || "nn-glow";
  const layerLabels = useMemo(() => {
    const labels: string[] = ["Input"];
    for (let i = 1; i < layerSizes.length - 1; i++) labels.push(`H${i}`);
    labels.push("Output");
    return labels;
  }, [layerSizes]);

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className="w-full mx-auto"
      style={{ aspectRatio: `${svgW}/${svgH}` }}
    >
      <defs>
        <filter id={fid} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id={`${fid}-dot`}>
          <stop offset="0%" stopColor="#facc15" stopOpacity="1" />
          <stop offset="100%" stopColor="#facc15" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Layer labels */}
      {showLabels &&
        layerLabels.map((label, l) => {
          const x = positions[l]?.[0]?.x ?? 70;
          return (
            <text
              key={`lbl-${l}`}
              x={x}
              y={28}
              textAnchor="middle"
              fontSize={11}
              fontWeight={600}
              fill="#475569"
            >
              {label}
            </text>
          );
        })}

      {/* Connections */}
      {positions.map((layer, l) => {
        if (l >= positions.length - 1) return null;
        const next = positions[l + 1];
        return layer.map((from, i) =>
          next.map((to, j) => {
            const w = weights[l]?.[i]?.[j] ?? 0;
            const isHL =
              highlightConn &&
              highlightConn.l === l &&
              highlightConn.i === i &&
              highlightConn.j === j;
            return (
              <line
                key={`c-${l}-${i}-${j}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={isHL ? "#f59e0b" : weightColor(w)}
                strokeWidth={isHL ? 3 : weightStrokeWidth(w)}
                style={{ cursor: onClickConn ? "pointer" : "default" }}
                onClick={() => onClickConn?.(l, i, j)}
              />
            );
          }),
        );
      })}

      {/* Animated dots */}
      {animDots?.map((dot, idx) => (
        <circle
          key={`dot-${idx}`}
          cx={dot.x}
          cy={dot.y}
          r={4}
          fill={`url(#${fid}-dot)`}
          opacity={dot.opacity}
        />
      ))}

      {/* Neurons */}
      {positions.map((layer, l) =>
        layer.map((pos, n) => {
          const state = neuronStates?.[l]?.[n];
          const isGlowing = state?.glowing ?? false;
          const displayValue = state
            ? state.activated
            : l === 0
              ? (inputValues?.[n] ?? 0)
              : null;
          let fillColor = "#f1f5f9";
          let strokeColor = "#94a3b8";
          if (l === 0) {
            fillColor = "#eff6ff";
            strokeColor = "#3b82f6";
          } else if (l === layerSizes.length - 1) {
            fillColor = "#f0fdf4";
            strokeColor = "#22c55e";
          } else {
            fillColor = "#faf5ff";
            strokeColor = "#a855f7";
          }
          return (
            <g key={`n-${l}-${n}`}>
              {isGlowing && (
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={neuronR + 6}
                  fill="none"
                  stroke="#facc15"
                  strokeWidth={3}
                  opacity={0.7}
                  filter={`url(#${fid})`}
                >
                  <animate
                    attributeName="opacity"
                    values="0.7;0.3;0.7"
                    dur="0.8s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={neuronR}
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth={2}
              />
              {displayValue !== null && (
                <text
                  x={pos.x}
                  y={pos.y + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={neuronR > 14 ? 10 : 8}
                  fontWeight={600}
                  fontFamily="monospace"
                  fill="#334155"
                >
                  {displayValue >= 0
                    ? displayValue.toFixed(2)
                    : displayValue.toFixed(1)}
                </text>
              )}
            </g>
          );
        }),
      )}
    </svg>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab definitions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const TABS: TabDef[] = [
  { key: "build", label: "Build & Propagate", icon: <Network className="w-3.5 h-3.5" /> },
  { key: "weights", label: "Weight Inspector", icon: <Sliders className="w-3.5 h-3.5" /> },
  { key: "activations", label: "Activation Comparison", icon: <BarChart3 className="w-3.5 h-3.5" /> },
  { key: "xor", label: "XOR Problem", icon: <GitBranch className="w-3.5 h-3.5" /> },
  { key: "depth", label: "Depth Experiment", icon: <Layers className="w-3.5 h-3.5" /> },
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main component
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export default function NeuralNetworkActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("build");

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
                ? "bg-white text-violet-700 shadow-sm"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "build" && <BuildTab />}
      {activeTab === "weights" && <WeightInspectorTab />}
      {activeTab === "activations" && <ActivationComparisonTab />}
      {activeTab === "xor" && <XORTab />}
      {activeTab === "depth" && <DepthExperimentTab />}
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 1 — BUILD & PROPAGATE                                               ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const BUILD_SVG_W = 700;
const BUILD_SVG_H = 450;
const BUILD_PAD_X = 70;
const BUILD_PAD_Y = 50;

function BuildTab() {
  const [inputCount, setInputCount] = useState(3);
  const [hiddenLayerCount, setHiddenLayerCount] = useState(2);
  const [neuronsPerHidden, setNeuronsPerHidden] = useState(4);
  const [outputCount, setOutputCount] = useState(1);
  const [activationFn, setActivationFn] = useState<ActivationFn>("relu");
  const [inputValues, setInputValues] = useState<number[]>(() =>
    Array.from({ length: 3 }, () => +(Math.random() * 2 - 1).toFixed(2)),
  );

  const layerSizes = useMemo(() => {
    const s = [inputCount];
    for (let i = 0; i < hiddenLayerCount; i++) s.push(neuronsPerHidden);
    s.push(outputCount);
    return s;
  }, [inputCount, hiddenLayerCount, neuronsPerHidden, outputCount]);

  const [weights, setWeights] = useState<number[][][]>(() => initWeights(layerSizes));
  const [biases, setBiases] = useState<number[][]>(() => initBiases(layerSizes));
  const [neuronStates, setNeuronStates] = useState<NeuronState[][] | null>(null);
  const [animProgress, setAnimProgress] = useState<number | null>(null);
  const animRef = useRef<number | null>(null);

  const prevTopologyKey = useRef(layerSizes.join(","));
  useEffect(() => {
    const key = layerSizes.join(",");
    if (key !== prevTopologyKey.current) {
      prevTopologyKey.current = key;
      setWeights(initWeights(layerSizes));
      setBiases(initBiases(layerSizes));
      setNeuronStates(null);
      setAnimProgress(null);
    }
  }, [layerSizes]);

  useEffect(() => {
    setInputValues((prev) => {
      if (prev.length === inputCount) return prev;
      return Array.from({ length: inputCount }, (_, i) =>
        i < prev.length ? prev[i] : +(Math.random() * 2 - 1).toFixed(2),
      );
    });
  }, [inputCount]);

  const positions = useMemo(
    () => computePositions(layerSizes, BUILD_SVG_W, BUILD_SVG_H, BUILD_PAD_X, BUILD_PAD_Y),
    [layerSizes],
  );

  const computeForwardPass = useCallback((): NeuronState[][] => {
    return forwardPass(inputValues, layerSizes, weights, biases, activationFn);
  }, [inputValues, layerSizes, weights, biases, activationFn]);

  const propagate = useCallback(() => {
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
    const allStates = computeForwardPass();
    const totalLayers = layerSizes.length;
    const durationPerLayer = 600;
    const startTime = performance.now();
    const totalDuration = (totalLayers - 1) * durationPerLayer;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / totalDuration, 1) * (totalLayers - 1);
      const reachedLayer = Math.floor(progress);
      const partialStates: NeuronState[][] = allStates.map((layer, l) =>
        layer.map((neuron) => ({
          ...neuron,
          glowing: l === reachedLayer + 1 && progress % 1 > 0.5,
          activated: l <= reachedLayer ? neuron.activated : 0,
          value: l <= reachedLayer ? neuron.value : 0,
        })),
      );
      partialStates[0] = allStates[0].map((n) => ({
        ...n,
        glowing: reachedLayer === 0 && progress < 0.5,
      }));
      setNeuronStates(partialStates);
      setAnimProgress(progress);
      if (progress < totalLayers - 1) {
        animRef.current = requestAnimationFrame(tick);
      } else {
        const finalStates = allStates.map((layer, l) =>
          layer.map((neuron) => ({
            ...neuron,
            glowing: l === totalLayers - 1,
          })),
        );
        setNeuronStates(finalStates);
        setAnimProgress(null);
        animRef.current = null;
      }
    };
    animRef.current = requestAnimationFrame(tick);
  }, [computeForwardPass, layerSizes]);

  useEffect(() => {
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  const randomizeWeights = () => {
    setWeights(initWeights(layerSizes));
    setBiases(initBiases(layerSizes));
    setNeuronStates(null);
    setAnimProgress(null);
  };

  const isAnimating = animProgress !== null;

  const animDots = useMemo(() => {
    if (animProgress === null) return [];
    const currentTransition = Math.floor(animProgress);
    const frac = animProgress - currentTransition;
    if (currentTransition >= layerSizes.length - 1) return [];
    const fromLayer = positions[currentTransition];
    const toLayer = positions[currentTransition + 1];
    if (!fromLayer || !toLayer) return [];
    const dots: { x: number; y: number; opacity: number }[] = [];
    for (let i = 0; i < fromLayer.length; i++) {
      for (let j = 0; j < toLayer.length; j++) {
        dots.push({
          x: fromLayer[i].x + (toLayer[j].x - fromLayer[i].x) * frac,
          y: fromLayer[i].y + (toLayer[j].y - fromLayer[i].y) * frac,
          opacity: 0.4 + 0.6 * Math.sin(frac * Math.PI),
        });
      }
    }
    return dots;
  }, [animProgress, positions, layerSizes.length]);

  const outputValues = useMemo(() => {
    if (!neuronStates) return null;
    return neuronStates[neuronStates.length - 1].map((n) => n.activated);
  }, [neuronStates]);

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-violet-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-violet-900">
            How Neural Networks Work
          </h3>
          <p className="text-xs text-violet-700 mt-1">
            A neural network transforms inputs through layers of interconnected neurons.
            Each connection has a <strong>weight</strong>, and each neuron applies an{" "}
            <strong>activation function</strong> to the weighted sum of its inputs plus a bias.
            During <strong>forward propagation</strong>, data flows from the input layer through
            hidden layers to produce an output. Adjust the topology, weights, and inputs below.
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <NetworkSVG
              layerSizes={layerSizes}
              weights={weights}
              neuronStates={neuronStates}
              positions={positions}
              svgW={BUILD_SVG_W}
              svgH={BUILD_SVG_H}
              animDots={animDots}
              filterId="build-glow"
              inputValues={inputValues}
            />
            {/* Weight legend */}
            <div className="flex items-center justify-center gap-4 mt-3">
              <span className="text-[11px] text-slate-500 font-medium">Weight Legend:</span>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-sm bg-red-500 opacity-90" />
                <span className="text-[10px] text-slate-500">Negative</span>
              </div>
              <div
                className="h-2 rounded-full"
                style={{
                  width: 100,
                  background:
                    "linear-gradient(to right, rgba(239,68,68,0.9), rgba(239,68,68,0.15), rgba(203,213,225,0.3), rgba(59,130,246,0.15), rgba(59,130,246,0.9))",
                }}
              />
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-sm bg-blue-500 opacity-90" />
                <span className="text-[10px] text-slate-500">Positive</span>
              </div>
              <span className="text-[10px] text-slate-400 ml-2">Thickness = magnitude</span>
            </div>
          </div>
        </div>

        <div className="w-full lg:w-72 space-y-4">
          {/* Input sliders */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Input Values
            </p>
            <div className="space-y-2">
              {inputValues.map((val, i) => (
                <div key={`inp-${i}`} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-500 w-6 shrink-0">
                    x{i + 1}
                  </span>
                  <input
                    type="range"
                    min={-1}
                    max={1}
                    step={0.01}
                    value={val}
                    onChange={(e) => {
                      const next = [...inputValues];
                      next[i] = parseFloat(e.target.value);
                      setInputValues(next);
                    }}
                    className="flex-1 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded w-10 text-center">
                    {val.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Topology controls */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">
              Network Topology
            </p>
            <CounterRow
              label="Input Neurons"
              value={inputCount}
              min={2}
              max={4}
              onChange={setInputCount}
            />
            <CounterRow
              label="Hidden Layers"
              value={hiddenLayerCount}
              min={1}
              max={3}
              onChange={setHiddenLayerCount}
            />
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-slate-700 font-medium">Neurons / Hidden</span>
                <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                  {neuronsPerHidden}
                </span>
              </div>
              <input
                type="range"
                min={2}
                max={6}
                step={1}
                value={neuronsPerHidden}
                onChange={(e) => setNeuronsPerHidden(parseInt(e.target.value, 10))}
                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-500"
              />
              <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                <span>2</span>
                <span>6</span>
              </div>
            </div>
            <CounterRow
              label="Output Neurons"
              value={outputCount}
              min={1}
              max={2}
              onChange={setOutputCount}
            />
          </div>

          {/* Activation selector */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                <Network className="w-3.5 h-3.5 text-violet-500" />
                Activation Function
              </label>
            </div>
            <div className="grid grid-cols-3 gap-1.5">
              {(["relu", "sigmoid", "tanh"] as ActivationFn[]).map((fn) => (
                <button
                  key={fn}
                  onClick={() => setActivationFn(fn)}
                  className={`px-2 py-1.5 rounded text-xs font-medium transition-all ${
                    activationFn === fn
                      ? "bg-violet-500 text-white shadow-sm"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {fn === "relu" ? "ReLU" : fn === "sigmoid" ? "Sigmoid" : "Tanh"}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-400 mt-1.5">
              {activationFn === "relu" && "max(0, x) -- fast, sparse activations"}
              {activationFn === "sigmoid" && "1/(1+e^-x) -- smooth, output in (0,1)"}
              {activationFn === "tanh" && "tanh(x) -- smooth, output in (-1,1)"}
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={propagate}
              disabled={isAnimating}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                isAnimating
                  ? "bg-violet-300 text-white cursor-not-allowed"
                  : "bg-violet-500 text-white hover:bg-violet-600"
              }`}
            >
              <Play className="w-4 h-4" />
              {isAnimating ? "Propagating..." : "Propagate"}
            </button>
            <button
              onClick={randomizeWeights}
              disabled={isAnimating}
              className="px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 transition-all"
              title="Randomize Weights"
            >
              <Shuffle className="w-4 h-4" />
            </button>
          </div>

          {/* Output */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Output Values
            </p>
            {outputValues ? (
              <div className="space-y-1.5">
                {outputValues.map((v, i) => (
                  <div
                    key={`out-${i}`}
                    className="flex items-center justify-between bg-green-50 border border-green-200 rounded px-3 py-2"
                  >
                    <span className="text-xs font-mono text-green-700">y{i + 1}</span>
                    <span className="text-sm font-bold font-mono text-green-800">
                      {v.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-400 italic">
                Click &quot;Propagate&quot; to compute output
              </p>
            )}
          </div>

          {/* Tips */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Try these experiments:</p>
            <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
              <li>Change activation to Sigmoid and watch values compress to (0,1)</li>
              <li>Add more hidden layers and see how values transform</li>
              <li>Randomize weights and propagate to see different outputs</li>
              <li>Set all inputs to 1.0 and compare activation functions</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 2 — WEIGHT INSPECTOR                                                ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const WI_SVG_W = 600;
const WI_SVG_H = 380;

function WeightInspectorTab() {
  const layerSizes = [3, 4, 4, 2];
  const [weights, setWeights] = useState<number[][][]>(() => initWeights(layerSizes));
  const [biases] = useState<number[][]>(() => initBiasesZero(layerSizes));
  const [activationFn] = useState<ActivationFn>("relu");
  const inputValues = [0.5, -0.3, 0.8];
  const [selectedConn, setSelectedConn] = useState<{
    l: number;
    i: number;
    j: number;
  } | null>(null);

  const positions = useMemo(
    () => computePositions(layerSizes, WI_SVG_W, WI_SVG_H, 70, 50),
    [],
  );

  const neuronStates = useMemo(
    () => forwardPass(inputValues, layerSizes, weights, biases, activationFn),
    [weights, biases, activationFn],
  );

  const outputValues = useMemo(
    () => neuronStates[neuronStates.length - 1].map((n) => n.activated),
    [neuronStates],
  );

  const handleClickConn = useCallback((l: number, i: number, j: number) => {
    setSelectedConn({ l, i, j });
  }, []);

  const handleWeightChange = useCallback(
    (val: number) => {
      if (!selectedConn) return;
      setWeights((prev) => {
        const next = prev.map((l) => l.map((row) => [...row]));
        next[selectedConn.l][selectedConn.i][selectedConn.j] = val;
        return next;
      });
    },
    [selectedConn],
  );

  // Gather all weights for histogram and heatmap
  const allWeightValues = useMemo(() => {
    const vals: number[] = [];
    for (const layer of weights) {
      for (const row of layer) {
        for (const w of row) {
          vals.push(w);
        }
      }
    }
    return vals;
  }, [weights]);

  // Histogram bins
  const histogram = useMemo(() => {
    const bins = 20;
    const min = -2;
    const max = 2;
    const step = (max - min) / bins;
    const counts = Array(bins).fill(0);
    for (const w of allWeightValues) {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor((w - min) / step)));
      counts[idx]++;
    }
    const maxCount = Math.max(...counts, 1);
    return { counts, min, max, step, maxCount, bins };
  }, [allWeightValues]);

  const applyXavier = () => {
    setWeights(initWeightsXavier(layerSizes));
    setSelectedConn(null);
  };

  const applyHe = () => {
    setWeights(initWeightsHe(layerSizes));
    setSelectedConn(null);
  };

  const applyRandom = () => {
    setWeights(initWeights(layerSizes));
    setSelectedConn(null);
  };

  const selectedW = selectedConn
    ? weights[selectedConn.l]?.[selectedConn.i]?.[selectedConn.j] ?? 0
    : null;

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-blue-900">Weight Inspector</h3>
          <p className="text-xs text-blue-700 mt-1">
            Weights are the <strong>learnable parameters</strong> of a neural network. Click any
            connection to inspect and adjust its weight. Watch how the output changes in real
            time. Initialization strategy matters &mdash; Xavier works well for Sigmoid/Tanh,
            while He initialization is designed for ReLU networks.
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        {/* Network SVG */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <p className="text-[10px] text-slate-400 mb-2 text-center">
              Click any connection line to inspect its weight
            </p>
            <NetworkSVG
              layerSizes={layerSizes}
              weights={weights}
              neuronStates={neuronStates}
              positions={positions}
              svgW={WI_SVG_W}
              svgH={WI_SVG_H}
              highlightConn={selectedConn}
              onClickConn={handleClickConn}
              filterId="wi-glow"
              inputValues={inputValues}
            />
          </div>

          {/* Weight heatmap grid */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 mt-4">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Weight Heatmap (per layer)
            </p>
            <div className="flex gap-4 overflow-x-auto pb-2">
              {weights.map((layerW, l) => (
                <div key={`hm-${l}`} className="shrink-0">
                  <p className="text-[10px] text-slate-500 mb-1 text-center">
                    Layer {l} &rarr; {l + 1}
                  </p>
                  <div className="flex flex-col gap-px">
                    {layerW.map((row, i) => (
                      <div key={`hm-${l}-${i}`} className="flex gap-px">
                        {row.map((w, j) => {
                          const isSelected =
                            selectedConn?.l === l &&
                            selectedConn?.i === i &&
                            selectedConn?.j === j;
                          const norm = clamp(w / 2, -1, 1);
                          const r = norm < 0 ? 239 : Math.round(59 + (1 - norm) * 180);
                          const g = norm < 0 ? Math.round(68 + (1 + norm) * 180) : 130;
                          const b = norm < 0 ? Math.round(68 + (1 + norm) * 180) : 246;
                          return (
                            <div
                              key={`hm-${l}-${i}-${j}`}
                              className={`w-7 h-7 rounded-sm flex items-center justify-center cursor-pointer transition-all ${
                                isSelected ? "ring-2 ring-amber-400 ring-offset-1" : ""
                              }`}
                              style={{ backgroundColor: `rgb(${r},${g},${b})` }}
                              onClick={() => setSelectedConn({ l, i, j })}
                              title={`w[${l}][${i}][${j}] = ${w.toFixed(3)}`}
                            >
                              <span className="text-[7px] font-mono text-white font-bold drop-shadow-sm">
                                {w.toFixed(1)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-80 space-y-4">
          {/* Selected weight display */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Selected Weight
            </p>
            {selectedConn && selectedW !== null ? (
              <div className="space-y-3">
                <div className="text-center">
                  <span className="text-[10px] text-slate-400 block">
                    Layer {selectedConn.l}, Neuron {selectedConn.i} &rarr; Neuron{" "}
                    {selectedConn.j}
                  </span>
                  <span
                    className="text-3xl font-bold font-mono"
                    style={{ color: selectedW >= 0 ? "#3b82f6" : "#ef4444" }}
                  >
                    {selectedW.toFixed(4)}
                  </span>
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 block mb-1">
                    Adjust weight (-2 to +2):
                  </label>
                  <input
                    type="range"
                    min={-2}
                    max={2}
                    step={0.01}
                    value={selectedW}
                    onChange={(e) => handleWeightChange(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                    <span>-2.0</span>
                    <span>0</span>
                    <span>+2.0</span>
                  </div>
                </div>
                <div className="bg-green-50 border border-green-200 rounded p-2">
                  <p className="text-[10px] text-green-700 font-medium mb-1">
                    Live output:
                  </p>
                  <div className="flex gap-2">
                    {outputValues.map((v, idx) => (
                      <span key={idx} className="text-sm font-bold font-mono text-green-800">
                        y{idx + 1}={v.toFixed(4)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400 italic text-center py-4">
                Click a connection in the network to inspect it
              </p>
            )}
          </div>

          {/* Initialization strategies */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Initialization Strategy
            </p>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={applyRandom}
                className="px-2 py-2 rounded text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 transition-colors"
              >
                Random
              </button>
              <button
                onClick={applyXavier}
                className="px-2 py-2 rounded text-xs font-medium bg-blue-100 text-blue-700 hover:bg-blue-200 transition-colors"
              >
                Xavier
              </button>
              <button
                onClick={applyHe}
                className="px-2 py-2 rounded text-xs font-medium bg-violet-100 text-violet-700 hover:bg-violet-200 transition-colors"
              >
                He Init
              </button>
            </div>
            <p className="text-[10px] text-slate-400 mt-2">
              <strong>Xavier:</strong> std = sqrt(2 / (fan_in + fan_out)) &mdash; good for
              Sigmoid/Tanh.
              <br />
              <strong>He:</strong> std = sqrt(2 / fan_in) &mdash; designed for ReLU.
            </p>
          </div>

          {/* Histogram */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Weight Distribution
            </p>
            <svg viewBox="0 0 220 100" className="w-full" style={{ aspectRatio: "220/100" }}>
              <line x1="20" y1="85" x2="210" y2="85" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="20" y1="5" x2="20" y2="85" stroke="#cbd5e1" strokeWidth="1" />
              {histogram.counts.map((count: number, idx: number) => {
                const barW = 190 / histogram.bins - 1;
                const barH = (count / histogram.maxCount) * 75;
                const x = 20 + (idx / histogram.bins) * 190 + 0.5;
                const midVal = histogram.min + (idx + 0.5) * histogram.step;
                const color = midVal >= 0 ? "rgba(59,130,246,0.6)" : "rgba(239,68,68,0.6)";
                return (
                  <rect
                    key={idx}
                    x={x}
                    y={85 - barH}
                    width={barW}
                    height={barH}
                    fill={color}
                    rx={1}
                  />
                );
              })}
              <text x="20" y="97" fontSize="7" fill="#94a3b8" textAnchor="middle">
                -2
              </text>
              <text x="115" y="97" fontSize="7" fill="#94a3b8" textAnchor="middle">
                0
              </text>
              <text x="210" y="97" fontSize="7" fill="#94a3b8" textAnchor="middle">
                +2
              </text>
              <text
                x="10"
                y="50"
                fontSize="7"
                fill="#94a3b8"
                textAnchor="middle"
                transform="rotate(-90, 10, 50)"
              >
                Count
              </text>
            </svg>
          </div>

          {/* Input display */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium mb-1">
              Fixed Inputs
            </p>
            <div className="flex gap-2">
              {inputValues.map((v, i) => (
                <span
                  key={i}
                  className="text-xs font-mono bg-blue-50 text-blue-700 px-2 py-1 rounded border border-blue-200"
                >
                  x{i + 1}={v}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 3 — ACTIVATION COMPARISON                                            ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const AC_SVG_W = 250;
const AC_SVG_H = 200;
const AC_CURVE_W = 120;
const AC_CURVE_H = 80;

function ActivationComparisonTab() {
  const layerSizes = [3, 4, 3, 2];
  const [weights] = useState<number[][][]>(() => initWeights(layerSizes));
  const [biases] = useState<number[][]>(() => initBiasesZero(layerSizes));
  const [inputValues, setInputValues] = useState<number[]>([0.6, -0.4, 0.9]);

  const activations: ActivationFn[] = ["relu", "sigmoid", "tanh"];
  const activationLabels = { relu: "ReLU", sigmoid: "Sigmoid", tanh: "Tanh" };
  const activationColors = { relu: "#ef4444", sigmoid: "#3b82f6", tanh: "#22c55e" };

  const positions = useMemo(
    () => computePositions(layerSizes, AC_SVG_W, AC_SVG_H, 35, 30),
    [],
  );

  const allStates = useMemo(() => {
    return activations.map((fn) =>
      forwardPass(inputValues, layerSizes, weights, biases, fn),
    );
  }, [inputValues, weights, biases]);

  // Bar chart data: activations per layer for each activation function
  const barData = useMemo(() => {
    return allStates.map((states) =>
      states.map((layer) => layer.map((n) => n.activated)),
    );
  }, [allStates]);

  // Generate activation curve points
  const curvePoints = useCallback(
    (fn: ActivationFn): string => {
      const points: string[] = [];
      const padX = 10;
      const padY = 10;
      const plotW = AC_CURVE_W - 2 * padX;
      const plotH = AC_CURVE_H - 2 * padY;
      for (let px = 0; px <= plotW; px++) {
        const x = -4 + (px / plotW) * 8;
        const y = applyActivation(x, fn);
        const svgX = padX + px;
        const svgY = padY + plotH / 2 - (y / 2) * plotH;
        points.push(`${svgX},${clamp(svgY, padY, padY + plotH)}`);
      }
      return points.join(" ");
    },
    [],
  );

  return (
    <div className="space-y-4">
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-emerald-900">Activation Comparison</h3>
          <p className="text-xs text-emerald-700 mt-1">
            Activation functions shape how information flows through the network. Here,
            the <strong>same input</strong> and <strong>same weights</strong> produce
            different outputs depending on the activation function. ReLU creates sparse
            activations, Sigmoid squashes everything to (0,1), and Tanh outputs (-1,1).
          </p>
        </div>
      </div>

      {/* Input sliders */}
      <div className="bg-white border border-slate-200 rounded-lg p-3">
        <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
          Shared Input Values
        </p>
        <div className="flex gap-4 flex-wrap">
          {inputValues.map((val, i) => (
            <div key={`ac-inp-${i}`} className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-slate-500">x{i + 1}</span>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.01}
                value={val}
                onChange={(e) => {
                  const next = [...inputValues];
                  next[i] = parseFloat(e.target.value);
                  setInputValues(next);
                }}
                className="w-24 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded">
                {val.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Three networks side by side */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {activations.map((fn, fnIdx) => {
          const states = allStates[fnIdx];
          const layerActs = barData[fnIdx];
          const color = activationColors[fn];
          const outVals = states[states.length - 1].map((n) => n.activated);

          return (
            <div
              key={fn}
              className="bg-white border border-slate-200 rounded-xl p-3 space-y-3"
            >
              <div className="flex items-center justify-between">
                <h4
                  className="text-sm font-bold"
                  style={{ color }}
                >
                  {activationLabels[fn]}
                </h4>
                <div className="flex gap-1">
                  {outVals.map((v, i) => (
                    <span
                      key={i}
                      className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded"
                    >
                      {v.toFixed(3)}
                    </span>
                  ))}
                </div>
              </div>

              {/* Mini network SVG */}
              <div className="bg-slate-50 rounded-lg p-1">
                <NetworkSVG
                  layerSizes={layerSizes}
                  weights={weights}
                  neuronStates={states}
                  positions={positions}
                  svgW={AC_SVG_W}
                  svgH={AC_SVG_H}
                  neuronR={12}
                  filterId={`ac-${fn}`}
                  showLabels={false}
                  inputValues={inputValues}
                />
              </div>

              {/* Activation curve */}
              <div className="bg-slate-50 rounded-lg p-1">
                <svg
                  viewBox={`0 0 ${AC_CURVE_W} ${AC_CURVE_H}`}
                  className="w-full"
                  style={{ aspectRatio: `${AC_CURVE_W}/${AC_CURVE_H}` }}
                >
                  {/* Axes */}
                  <line x1="10" y1={AC_CURVE_H / 2} x2={AC_CURVE_W - 10} y2={AC_CURVE_H / 2} stroke="#e2e8f0" strokeWidth="0.5" />
                  <line x1={AC_CURVE_W / 2} y1="10" x2={AC_CURVE_W / 2} y2={AC_CURVE_H - 10} stroke="#e2e8f0" strokeWidth="0.5" />
                  {/* Curve */}
                  <polyline
                    points={curvePoints(fn)}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                  />
                  <text x={AC_CURVE_W / 2} y={AC_CURVE_H - 2} fontSize="7" fill="#94a3b8" textAnchor="middle">
                    {fn === "relu" ? "max(0,x)" : fn === "sigmoid" ? "1/(1+e^-x)" : "tanh(x)"}
                  </text>
                </svg>
              </div>

              {/* Bar charts per layer */}
              <div className="space-y-1">
                {layerActs.map((layerVals, lIdx) => {
                  if (lIdx === 0) return null;
                  const maxAbs = Math.max(...layerVals.map(Math.abs), 0.01);
                  return (
                    <div key={`bar-${fn}-${lIdx}`}>
                      <p className="text-[9px] text-slate-400">
                        {lIdx < layerSizes.length - 1
                          ? `Hidden ${lIdx}`
                          : "Output"}
                      </p>
                      <div className="flex gap-0.5 items-end h-8">
                        {layerVals.map((v, nIdx) => {
                          const h = (Math.abs(v) / maxAbs) * 28;
                          return (
                            <div
                              key={nIdx}
                              className="flex-1 flex flex-col items-center justify-end"
                            >
                              <div
                                className="w-full rounded-t-sm"
                                style={{
                                  height: Math.max(2, h),
                                  backgroundColor: v >= 0 ? color : "#94a3b8",
                                  opacity: 0.7,
                                }}
                              />
                              <span className="text-[7px] font-mono text-slate-400 mt-0.5">
                                {v.toFixed(1)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 4 — XOR PROBLEM                                                     ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const XOR_DATA: { input: number[]; target: number }[] = [
  { input: [0, 0], target: 0 },
  { input: [0, 1], target: 1 },
  { input: [1, 0], target: 1 },
  { input: [1, 1], target: 0 },
];

const XOR_SVG_W = 500;
const XOR_SVG_H = 300;
const XOR_HM_SIZE = 200;

function XORTab() {
  const layerSizes = [2, 4, 1];
  const [weights, setWeights] = useState<number[][][]>(() => initWeights(layerSizes));
  const [biases, setBiases] = useState<number[][]>(() => initBiases(layerSizes));
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const trainRef = useRef<number | null>(null);
  const [predictions, setPredictions] = useState<number[]>([0, 0, 0, 0]);
  const activationFn: ActivationFn = "sigmoid";

  const positions = useMemo(
    () => computePositions(layerSizes, XOR_SVG_W, XOR_SVG_H, 80, 50),
    [],
  );

  // Compute predictions
  const computePredictions = useCallback(
    (w: number[][][], b: number[][]) => {
      return XOR_DATA.map((d) => {
        const acts = forwardRaw(d.input, layerSizes, w, b, activationFn);
        return acts[acts.length - 1][0];
      });
    },
    [],
  );

  // Compute loss
  const computeLoss = useCallback(
    (w: number[][][], b: number[][]) => {
      let loss = 0;
      for (const d of XOR_DATA) {
        const acts = forwardRaw(d.input, layerSizes, w, b, activationFn);
        const pred = acts[acts.length - 1][0];
        const err = pred - d.target;
        loss += err * err;
      }
      return loss / XOR_DATA.length;
    },
    [],
  );

  // One step of gradient descent (manual backprop for 2-layer net with sigmoid)
  const trainStep = useCallback(
    (
      w: number[][][],
      b: number[][],
      lr: number,
    ): { w: number[][][]; b: number[][] } => {
      const nw = w.map((l) => l.map((r) => [...r]));
      const nb = b.map((l) => [...l]);

      // Gradient accumulators
      const dw0 = w[0].map((r) => r.map(() => 0));
      const dw1 = w[1].map((r) => r.map(() => 0));
      const db1 = Array(layerSizes[1]).fill(0);
      const db2 = Array(layerSizes[2]).fill(0);

      for (const data of XOR_DATA) {
        // Forward pass (with pre-activations stored)
        const z: number[][] = [data.input];
        const a: number[][] = [data.input];

        // Hidden layer
        const hZ: number[] = [];
        const hA: number[] = [];
        for (let j = 0; j < layerSizes[1]; j++) {
          let sum = b[1][j];
          for (let i = 0; i < layerSizes[0]; i++) {
            sum += a[0][i] * w[0][i][j];
          }
          hZ.push(sum);
          hA.push(applyActivation(sum, activationFn));
        }
        z.push(hZ);
        a.push(hA);

        // Output layer
        const oZ: number[] = [];
        const oA: number[] = [];
        for (let j = 0; j < layerSizes[2]; j++) {
          let sum = b[2][j];
          for (let i = 0; i < layerSizes[1]; i++) {
            sum += a[1][i] * w[1][i][j];
          }
          oZ.push(sum);
          oA.push(applyActivation(sum, activationFn));
        }
        z.push(oZ);
        a.push(oA);

        // Backprop
        const pred = oA[0];
        const dLoss = 2 * (pred - data.target) / XOR_DATA.length;
        const dOut = dLoss * activationDerivative(oZ[0], activationFn);

        db2[0] += dOut;
        for (let i = 0; i < layerSizes[1]; i++) {
          dw1[i][0] += dOut * hA[i];
        }

        for (let j = 0; j < layerSizes[1]; j++) {
          const dHidden = dOut * w[1][j][0] * activationDerivative(hZ[j], activationFn);
          db1[j] += dHidden;
          for (let i = 0; i < layerSizes[0]; i++) {
            dw0[i][j] += dHidden * data.input[i];
          }
        }
      }

      // Apply gradients
      for (let i = 0; i < layerSizes[0]; i++) {
        for (let j = 0; j < layerSizes[1]; j++) {
          nw[0][i][j] -= lr * dw0[i][j];
        }
      }
      for (let i = 0; i < layerSizes[1]; i++) {
        for (let j = 0; j < layerSizes[2]; j++) {
          nw[1][i][j] -= lr * dw1[i][j];
        }
      }
      for (let j = 0; j < layerSizes[1]; j++) {
        nb[1][j] -= lr * db1[j];
      }
      for (let j = 0; j < layerSizes[2]; j++) {
        nb[2][j] -= lr * db2[j];
      }

      return { w: nw, b: nb };
    },
    [],
  );

  // Training loop
  const startTraining = useCallback(() => {
    setIsTraining(true);
    let currentW = weights.map((l) => l.map((r) => [...r]));
    let currentB = biases.map((l) => [...l]);
    let currentEpoch = epoch;
    let losses = [...lossHistory];
    const lr = 2.0;
    const stepsPerFrame = 5;

    const tick = () => {
      for (let s = 0; s < stepsPerFrame; s++) {
        const result = trainStep(currentW, currentB, lr);
        currentW = result.w;
        currentB = result.b;
        currentEpoch++;
        const loss = computeLoss(currentW, currentB);
        losses.push(loss);
      }

      setWeights(currentW.map((l) => l.map((r) => [...r])));
      setBiases(currentB.map((l) => [...l]));
      setEpoch(currentEpoch);
      setLossHistory([...losses]);
      setPredictions(computePredictions(currentW, currentB));

      if (currentEpoch < 2000) {
        trainRef.current = requestAnimationFrame(tick);
      } else {
        setIsTraining(false);
        trainRef.current = null;
      }
    };

    trainRef.current = requestAnimationFrame(tick);
  }, [weights, biases, epoch, lossHistory, trainStep, computeLoss, computePredictions]);

  const stopTraining = useCallback(() => {
    if (trainRef.current) {
      cancelAnimationFrame(trainRef.current);
      trainRef.current = null;
    }
    setIsTraining(false);
  }, []);

  const resetXOR = useCallback(() => {
    stopTraining();
    setWeights(initWeights(layerSizes));
    setBiases(initBiases(layerSizes));
    setLossHistory([]);
    setEpoch(0);
    setPredictions([0, 0, 0, 0]);
  }, [stopTraining]);

  useEffect(() => {
    setPredictions(computePredictions(weights, biases));
  }, []);

  useEffect(() => {
    return () => {
      if (trainRef.current) cancelAnimationFrame(trainRef.current);
    };
  }, []);

  // Neuron states for current XOR input (0,1) for visualization
  const neuronStates = useMemo(() => {
    return forwardPass([0, 1], layerSizes, weights, biases, activationFn);
  }, [weights, biases]);

  // Decision boundary heatmap
  const heatmapData = useMemo(() => {
    const res = 25;
    const grid: number[][] = [];
    for (let r = 0; r < res; r++) {
      const row: number[] = [];
      for (let c = 0; c < res; c++) {
        const x = c / (res - 1);
        const y = 1 - r / (res - 1);
        const acts = forwardRaw([x, y], layerSizes, weights, biases, activationFn);
        row.push(acts[acts.length - 1][0]);
      }
      grid.push(row);
    }
    return grid;
  }, [weights, biases]);

  // Loss curve dimensions
  const lossCurveW = 300;
  const lossCurveH = 120;

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-orange-900">The XOR Problem</h3>
          <p className="text-xs text-orange-700 mt-1">
            XOR (exclusive or) is a classic problem: a single-layer perceptron <strong>cannot</strong> solve
            it because the data is not linearly separable. But a network with a <strong>hidden layer</strong> can!
            Watch as gradient descent trains the weights to correctly classify all four XOR inputs.
            The decision boundary transforms from a line into a curved region.
          </p>
        </div>
      </div>

      <div className="flex gap-6 flex-col lg:flex-row">
        {/* Network + heatmap */}
        <div className="flex-1 min-w-0 space-y-4">
          {/* Network SVG */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <p className="text-[10px] text-slate-400 mb-1 text-center">
              Network: 2 inputs &rarr; 4 hidden (sigmoid) &rarr; 1 output
            </p>
            <NetworkSVG
              layerSizes={layerSizes}
              weights={weights}
              neuronStates={neuronStates}
              positions={positions}
              svgW={XOR_SVG_W}
              svgH={XOR_SVG_H}
              filterId="xor-glow"
              inputValues={[0, 1]}
            />
          </div>

          {/* Decision boundary heatmap */}
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Decision Boundary
            </p>
            <div className="flex justify-center">
              <svg
                viewBox={`0 0 ${XOR_HM_SIZE + 40} ${XOR_HM_SIZE + 40}`}
                className="w-full max-w-70"
                style={{ aspectRatio: "1/1" }}
              >
                {/* Heatmap grid */}
                {heatmapData.map((row, r) =>
                  row.map((val, c) => {
                    const cellSize = XOR_HM_SIZE / heatmapData.length;
                    const intensity = clamp(val, 0, 1);
                    const red = Math.round(255 * (1 - intensity));
                    const green = Math.round(100 + 155 * intensity);
                    const blue = Math.round(255 * intensity);
                    return (
                      <rect
                        key={`hm-${r}-${c}`}
                        x={30 + c * cellSize}
                        y={10 + r * cellSize}
                        width={cellSize + 0.5}
                        height={cellSize + 0.5}
                        fill={`rgb(${red},${green},${blue})`}
                      />
                    );
                  }),
                )}
                {/* XOR data points */}
                {XOR_DATA.map((d, i) => {
                  const cx = 30 + d.input[0] * XOR_HM_SIZE;
                  const cy = 10 + (1 - d.input[1]) * XOR_HM_SIZE;
                  return (
                    <g key={`xor-pt-${i}`}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={8}
                        fill={d.target === 1 ? "#22c55e" : "#ef4444"}
                        stroke="white"
                        strokeWidth={2}
                      />
                      <text
                        x={cx}
                        y={cy + 1}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fontSize="8"
                        fill="white"
                        fontWeight="bold"
                      >
                        {d.target}
                      </text>
                    </g>
                  );
                })}
                {/* Axis labels */}
                <text x={30 + XOR_HM_SIZE / 2} y={XOR_HM_SIZE + 30} fontSize="9" fill="#64748b" textAnchor="middle">
                  x1
                </text>
                <text
                  x={15}
                  y={10 + XOR_HM_SIZE / 2}
                  fontSize="9"
                  fill="#64748b"
                  textAnchor="middle"
                  transform={`rotate(-90, 15, ${10 + XOR_HM_SIZE / 2})`}
                >
                  x2
                </text>
                <text x={30} y={XOR_HM_SIZE + 22} fontSize="7" fill="#94a3b8">0</text>
                <text x={30 + XOR_HM_SIZE - 5} y={XOR_HM_SIZE + 22} fontSize="7" fill="#94a3b8">1</text>
              </svg>
            </div>
            <div className="flex items-center justify-center gap-3 mt-2">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <span className="text-[10px] text-slate-500">Output = 0</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-[10px] text-slate-500">Output = 1</span>
              </div>
              <span className="text-[10px] text-slate-400">Background = network prediction</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-72 space-y-4">
          {/* Training controls */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">
              Training
            </p>
            <div className="flex gap-2">
              <button
                onClick={isTraining ? stopTraining : startTraining}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                  isTraining
                    ? "bg-red-500 text-white hover:bg-red-600"
                    : "bg-orange-500 text-white hover:bg-orange-600"
                }`}
              >
                {isTraining ? (
                  <>
                    <Pause className="w-4 h-4" />
                    Stop
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Train
                  </>
                )}
              </button>
              <button
                onClick={resetXOR}
                className="px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all"
                title="Reset"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-600">Epoch:</span>
              <span className="text-sm font-bold font-mono text-slate-800">{epoch}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-600">Loss:</span>
              <span className="text-sm font-bold font-mono text-slate-800">
                {lossHistory.length > 0
                  ? lossHistory[lossHistory.length - 1].toFixed(6)
                  : computeLoss(weights, biases).toFixed(6)}
              </span>
            </div>
          </div>

          {/* Predictions table */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Predictions
            </p>
            <div className="space-y-1.5">
              {XOR_DATA.map((d, i) => {
                const pred = predictions[i];
                const correct =
                  (d.target === 1 && pred > 0.5) || (d.target === 0 && pred < 0.5);
                return (
                  <div
                    key={`xor-pred-${i}`}
                    className={`flex items-center justify-between px-3 py-1.5 rounded text-xs font-mono ${
                      correct
                        ? "bg-green-50 border border-green-200"
                        : "bg-red-50 border border-red-200"
                    }`}
                  >
                    <span className="text-slate-600">
                      ({d.input[0]},{d.input[1]})&rarr;{d.target}
                    </span>
                    <span
                      className={`font-bold ${
                        correct ? "text-green-700" : "text-red-700"
                      }`}
                    >
                      {pred.toFixed(3)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Loss curve */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Loss Curve
            </p>
            {lossHistory.length > 1 ? (
              <svg
                viewBox={`0 0 ${lossCurveW} ${lossCurveH}`}
                className="w-full"
                style={{ aspectRatio: `${lossCurveW}/${lossCurveH}` }}
              >
                <rect x="30" y="5" width={lossCurveW - 40} height={lossCurveH - 25} fill="#f8fafc" rx="2" />
                {/* Axes */}
                <line x1="30" y1={lossCurveH - 20} x2={lossCurveW - 10} y2={lossCurveH - 20} stroke="#cbd5e1" strokeWidth="1" />
                <line x1="30" y1="5" x2="30" y2={lossCurveH - 20} stroke="#cbd5e1" strokeWidth="1" />
                {/* Loss curve */}
                <polyline
                  points={(() => {
                    const maxLoss = Math.max(...lossHistory, 0.01);
                    const step = Math.max(1, Math.floor(lossHistory.length / 200));
                    const pts: string[] = [];
                    for (let i = 0; i < lossHistory.length; i += step) {
                      const x = 30 + (i / (lossHistory.length - 1)) * (lossCurveW - 40);
                      const y = 5 + (1 - lossHistory[i] / maxLoss) * (lossCurveH - 25);
                      pts.push(`${x},${y}`);
                    }
                    return pts.join(" ");
                  })()}
                  fill="none"
                  stroke="#f97316"
                  strokeWidth="1.5"
                />
                <text x={lossCurveW / 2} y={lossCurveH - 2} fontSize="8" fill="#94a3b8" textAnchor="middle">
                  Epoch
                </text>
                <text x="5" y={lossCurveH / 2} fontSize="8" fill="#94a3b8" textAnchor="middle" transform={`rotate(-90, 10, ${lossCurveH / 2})`}>
                  Loss
                </text>
              </svg>
            ) : (
              <p className="text-xs text-slate-400 italic text-center py-4">
                Press &quot;Train&quot; to see the loss decrease
              </p>
            )}
          </div>

          {/* Tip */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Key insight:</p>
            <p className="text-xs text-amber-700">
              A single-layer network (no hidden layer) can only draw a straight line as a
              decision boundary. XOR needs a <strong>curved</strong> boundary, which requires
              at least one hidden layer. This is why depth matters!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ╔════════════════════════════════════════════════════════════════════════════╗
// ║  TAB 5 — DEPTH EXPERIMENT                                                ║
// ╚════════════════════════════════════════════════════════════════════════════╝
const DEPTH_SCATTER_SIZE = 160;

function DepthExperimentTab() {
  const [selectedDepth, setSelectedDepth] = useState(2);
  const depthOptions = [1, 2, 3, 4];
  const hiddenSize = 4;
  const [inputValues, setInputValues] = useState<number[]>([0.7, -0.5]);
  const activationFn: ActivationFn = "relu";

  // Build layer sizes for each depth
  const allLayerSizes = useMemo(() => {
    const result: Record<number, number[]> = {};
    for (const d of depthOptions) {
      const sizes = [2];
      for (let i = 0; i < d; i++) sizes.push(hiddenSize);
      sizes.push(2);
      result[d] = sizes;
    }
    return result;
  }, []);

  // Shared initial weights (seeded so results are consistent), one set per depth
  const [allWeights, setAllWeights] = useState<Record<number, number[][][]>>(() => {
    const result: Record<number, number[][][]> = {};
    for (const d of depthOptions) {
      result[d] = initWeights(allLayerSizes[d]);
    }
    return result;
  });

  const [allBiases, setAllBiases] = useState<Record<number, number[][]>>(() => {
    const result: Record<number, number[][]> = {};
    for (const d of depthOptions) {
      result[d] = initBiasesZero(allLayerSizes[d]);
    }
    return result;
  });

  // Compute activations for each depth
  const allActivations = useMemo(() => {
    const result: Record<number, number[][]> = {};
    for (const d of depthOptions) {
      result[d] = forwardRaw(
        inputValues,
        allLayerSizes[d],
        allWeights[d],
        allBiases[d],
        activationFn,
      );
    }
    return result;
  }, [inputValues, allWeights, allBiases, allLayerSizes]);

  // Generate a cloud of 2D points and transform them through each network
  const scatterData = useMemo(() => {
    const numPoints = 60;
    const rawPoints: number[][] = [];
    // Generate a circle of points
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      const r = 0.3 + 0.5 * ((i % 3) / 3);
      rawPoints.push([
        0.5 + r * Math.cos(angle),
        0.5 + r * Math.sin(angle),
      ]);
    }

    const result: Record<number, number[][][]> = {};
    for (const d of depthOptions) {
      const layerTransforms: number[][][] = [];
      const sizes = allLayerSizes[d];
      const w = allWeights[d];
      const b = allBiases[d];

      // For each point, compute activations at every layer
      const perLayer: number[][][] = Array.from(
        { length: sizes.length },
        () => [],
      );

      for (const pt of rawPoints) {
        const acts = forwardRaw(pt, sizes, w, b, activationFn);
        for (let l = 0; l < acts.length; l++) {
          perLayer[l].push(acts[l]);
        }
      }

      layerTransforms.push(...perLayer);
      result[d] = layerTransforms;
    }
    return result;
  }, [allWeights, allBiases, allLayerSizes]);

  const randomizeAll = () => {
    const nw: Record<number, number[][][]> = {};
    const nb: Record<number, number[][]> = {};
    for (const d of depthOptions) {
      nw[d] = initWeights(allLayerSizes[d]);
      nb[d] = initBiasesZero(allLayerSizes[d]);
    }
    setAllWeights(nw);
    setAllBiases(nb);
  };

  // Compute how "spread" the representations are (simple measure: std of all values)
  const representationalSpread = useMemo(() => {
    const result: Record<number, number[]> = {};
    for (const d of depthOptions) {
      const layers = scatterData[d];
      const spreads: number[] = [];
      for (let l = 0; l < layers.length; l++) {
        const allVals = layers[l].flatMap((p) => p.slice(0, 2));
        if (allVals.length === 0) {
          spreads.push(0);
          continue;
        }
        const mean = allVals.reduce((a, b) => a + b, 0) / allVals.length;
        const variance =
          allVals.reduce((a, b) => a + (b - mean) ** 2, 0) / allVals.length;
        spreads.push(Math.sqrt(variance));
      }
      result[d] = spreads;
    }
    return result;
  }, [scatterData]);

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-purple-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-purple-900">Depth Experiment</h3>
          <p className="text-xs text-purple-700 mt-1">
            Deeper networks can learn <strong>more complex functions</strong>. Each hidden layer
            applies a nonlinear transformation, progressively reshaping the input space. Watch how
            a cloud of 2D points gets transformed through layers of increasing depth. Deeper
            networks &quot;fold&quot; and &quot;twist&quot; the space in more intricate ways, giving them greater{" "}
            <strong>representational power</strong>.
          </p>
        </div>
      </div>

      {/* Depth selector */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="bg-white border border-slate-200 rounded-lg p-3 flex items-center gap-3">
          <span className="text-xs text-slate-600 font-medium">Hidden Layers:</span>
          <div className="flex gap-1.5">
            {depthOptions.map((d) => (
              <button
                key={d}
                onClick={() => setSelectedDepth(d)}
                className={`w-8 h-8 rounded-lg text-sm font-bold transition-all ${
                  selectedDepth === d
                    ? "bg-purple-500 text-white shadow-sm"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {inputValues.map((val, i) => (
            <div key={`de-inp-${i}`} className="flex items-center gap-1">
              <span className="text-[10px] font-mono text-slate-500">x{i + 1}</span>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.01}
                value={val}
                onChange={(e) => {
                  const next = [...inputValues];
                  next[i] = parseFloat(e.target.value);
                  setInputValues(next);
                }}
                className="w-20 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
              />
              <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1 py-0.5 rounded">
                {val.toFixed(2)}
              </span>
            </div>
          ))}
        </div>

        <button
          onClick={randomizeAll}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all text-xs font-medium"
        >
          <Shuffle className="w-3.5 h-3.5" />
          Randomize Weights
        </button>
      </div>

      {/* Comparison: all depths */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {depthOptions.map((d) => {
          const layers = scatterData[d];
          const sizes = allLayerSizes[d];
          const acts = allActivations[d];
          const isSelected = d === selectedDepth;
          const outVals = acts[acts.length - 1];

          return (
            <div
              key={`depth-${d}`}
              className={`rounded-xl border-2 p-3 transition-all cursor-pointer ${
                isSelected
                  ? "border-purple-400 bg-purple-50/50"
                  : "border-slate-200 bg-white hover:border-slate-300"
              }`}
              onClick={() => setSelectedDepth(d)}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-bold text-slate-800">
                  {d} Hidden Layer{d > 1 ? "s" : ""}
                </h4>
                <span className="text-[10px] text-slate-400">
                  {sizes.join(" -> ")}
                </span>
              </div>

              {/* Output */}
              <div className="flex gap-1 mb-2">
                {outVals.map((v, i) => (
                  <span
                    key={i}
                    className="text-[10px] font-mono bg-green-50 text-green-700 px-1.5 py-0.5 rounded border border-green-200"
                  >
                    y{i + 1}={v.toFixed(3)}
                  </span>
                ))}
              </div>

              {/* Representation spread bar */}
              <div className="mb-2">
                <p className="text-[9px] text-slate-400 mb-0.5">
                  Representation complexity
                </p>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-purple-400 rounded-full transition-all"
                    style={{
                      width: `${Math.min(
                        100,
                        ((representationalSpread[d]?.[sizes.length - 1] ?? 0) /
                          Math.max(
                            ...depthOptions.map(
                              (dd) =>
                                representationalSpread[dd]?.[
                                  allLayerSizes[dd].length - 1
                                ] ?? 1,
                            ),
                            0.01,
                          )) *
                          100,
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Detailed scatter plots for selected depth */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-slate-800">
            Data Transformation Through {selectedDepth} Hidden Layer
            {selectedDepth > 1 ? "s" : ""}{" "}
            <span className="text-slate-400 font-normal">
              ({allLayerSizes[selectedDepth].join(" -> ")})
            </span>
          </h4>
          <p className="text-[10px] text-slate-400">
            Each plot shows how points are represented at that layer (first 2 dimensions)
          </p>
        </div>

        <div className="flex gap-3 overflow-x-auto pb-2">
          {scatterData[selectedDepth]?.map((layerPoints, lIdx) => {
            const sizes = allLayerSizes[selectedDepth];
            const isInput = lIdx === 0;
            const isOutput = lIdx === sizes.length - 1;
            const label = isInput
              ? "Input"
              : isOutput
                ? "Output"
                : `Hidden ${lIdx}`;

            // Get first 2 dimensions of each point
            const pts = layerPoints.map((p) => [p[0] ?? 0, p[1] ?? 0]);

            // Find bounds
            let minX = Infinity,
              maxX = -Infinity,
              minY = Infinity,
              maxY = -Infinity;
            for (const [x, y] of pts) {
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (y < minY) minY = y;
              if (y > maxY) maxY = y;
            }
            const rangeX = maxX - minX || 1;
            const rangeY = maxY - minY || 1;
            const padFrac = 0.1;
            const effMinX = minX - rangeX * padFrac;
            const effMaxX = maxX + rangeX * padFrac;
            const effMinY = minY - rangeY * padFrac;
            const effMaxY = maxY + rangeY * padFrac;
            const eRangeX = effMaxX - effMinX;
            const eRangeY = effMaxY - effMinY;

            const plotPad = 15;
            const plotSize = DEPTH_SCATTER_SIZE - 2 * plotPad;

            return (
              <div key={`scatter-${lIdx}`} className="shrink-0">
                <p className="text-[10px] text-slate-500 mb-1 text-center font-medium">
                  {label}
                </p>
                <svg
                  viewBox={`0 0 ${DEPTH_SCATTER_SIZE} ${DEPTH_SCATTER_SIZE}`}
                  width={DEPTH_SCATTER_SIZE}
                  height={DEPTH_SCATTER_SIZE}
                  className="bg-slate-50 rounded-lg border border-slate-100"
                >
                  {/* Grid */}
                  <line
                    x1={plotPad}
                    y1={DEPTH_SCATTER_SIZE - plotPad}
                    x2={DEPTH_SCATTER_SIZE - plotPad}
                    y2={DEPTH_SCATTER_SIZE - plotPad}
                    stroke="#e2e8f0"
                    strokeWidth="0.5"
                  />
                  <line
                    x1={plotPad}
                    y1={plotPad}
                    x2={plotPad}
                    y2={DEPTH_SCATTER_SIZE - plotPad}
                    stroke="#e2e8f0"
                    strokeWidth="0.5"
                  />

                  {/* Points colored by index (to show how structure changes) */}
                  {pts.map(([x, y], pIdx) => {
                    const sx =
                      plotPad + ((x - effMinX) / eRangeX) * plotSize;
                    const sy =
                      DEPTH_SCATTER_SIZE -
                      plotPad -
                      ((y - effMinY) / eRangeY) * plotSize;
                    const hue = (pIdx / pts.length) * 300;
                    return (
                      <circle
                        key={pIdx}
                        cx={clamp(sx, plotPad, DEPTH_SCATTER_SIZE - plotPad)}
                        cy={clamp(sy, plotPad, DEPTH_SCATTER_SIZE - plotPad)}
                        r={3}
                        fill={`hsl(${hue}, 70%, 55%)`}
                        opacity={0.8}
                      />
                    );
                  })}
                </svg>
                <p className="text-[8px] text-slate-400 text-center mt-0.5">
                  dim {sizes[lIdx]} neurons
                </p>
              </div>
            );
          })}

          {/* Arrow between scatter plots */}
        </div>

        <div className="mt-3 bg-slate-50 rounded-lg p-3">
          <p className="text-xs text-slate-600">
            <strong>How to read this:</strong> Each dot is the same data point, shown as it
            passes through layers. The colors are consistent across plots so you can track
            individual points. Notice how deeper layers create more complex spatial
            arrangements &mdash; this is the network learning increasingly abstract
            representations.
          </p>
        </div>
      </div>

      {/* Activation values for the single input */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <h4 className="text-sm font-semibold text-slate-800 mb-3">
          Neuron Activations for Current Input
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {depthOptions.map((d) => {
            const acts = allActivations[d];
            const sizes = allLayerSizes[d];
            return (
              <div
                key={`act-${d}`}
                className={`rounded-lg border p-2 ${
                  d === selectedDepth
                    ? "border-purple-300 bg-purple-50/30"
                    : "border-slate-200"
                }`}
              >
                <p className="text-[10px] text-slate-500 font-medium mb-1">
                  Depth {d}
                </p>
                {acts.map((layerActs, lIdx) => {
                  if (lIdx === 0) return null;
                  const maxAbs = Math.max(...layerActs.map(Math.abs), 0.01);
                  const isOutput = lIdx === sizes.length - 1;
                  return (
                    <div key={`act-${d}-${lIdx}`} className="mb-1">
                      <p className="text-[8px] text-slate-400">
                        {isOutput ? "Out" : `H${lIdx}`}
                      </p>
                      <div className="flex gap-0.5 items-end h-6">
                        {layerActs.map((v, nIdx) => {
                          const h = (Math.abs(v) / maxAbs) * 20;
                          return (
                            <div
                              key={nIdx}
                              className="flex-1 flex flex-col items-center justify-end"
                            >
                              <div
                                className="w-full rounded-t-sm"
                                style={{
                                  height: Math.max(1, h),
                                  backgroundColor:
                                    v > 0 ? "#a855f7" : "#94a3b8",
                                  opacity: 0.7,
                                }}
                              />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      {/* Lesson tip */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <p className="text-xs text-amber-800 font-medium mb-1">Key takeaway:</p>
        <p className="text-xs text-amber-700">
          Each layer applies a <strong>nonlinear transformation</strong>. With 1 hidden
          layer, the network can approximate simple functions. With 2+ layers, it can fold
          and warp the input space in complex ways. However, more depth is not always better
          &mdash; very deep networks are harder to train (vanishing/exploding gradients)
          without techniques like batch normalization and skip connections.
        </p>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Shared UI helper
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function CounterRow({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-700 font-medium">{label}</span>
      <div className="flex items-center gap-1.5">
        <button
          onClick={() => onChange(Math.max(min, value - 1))}
          className="w-6 h-6 rounded border border-slate-200 flex items-center justify-center hover:bg-slate-50 transition-colors"
          disabled={value <= min}
        >
          <Minus className="w-3 h-3 text-slate-600" />
        </button>
        <span className="text-xs font-mono text-slate-700 w-5 text-center">
          {value}
        </span>
        <button
          onClick={() => onChange(Math.min(max, value + 1))}
          className="w-6 h-6 rounded border border-slate-200 flex items-center justify-center hover:bg-slate-50 transition-colors"
          disabled={value >= max}
        >
          <Plus className="w-3 h-3 text-slate-600" />
        </button>
      </div>
    </div>
  );
}
