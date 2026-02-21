/**
 * Backpropagation Visualizer — comprehensive 5-tab interactive SVG activity
 * Tab 1: Forward Pass Visualization (2→3→1 network with animated data flow)
 * Tab 2: Backward Pass Step-by-Step (gradient computation right-to-left)
 * Tab 3: Chain Rule Explorer (configurable chain of functions)
 * Tab 4: Gradient Flow Visualization (deep network, vanishing gradients)
 * Tab 5: Train & Watch (multi-epoch training on XOR with decision boundary)
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  ArrowRight,
  ArrowLeft,
  Play,
  RotateCcw,
  Info,
  Eye,
  Link,
  Layers,
  TrendingUp,
  Pause,
  RefreshCw,
  Zap,
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
type TabKey = "forward" | "backward" | "chain" | "gradient" | "train";

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

interface NeuronState {
  z: number;
  a: number;
  dA: number;
  dZ: number;
}

interface NetworkWeights {
  w: number[][][];
  b: number[][];
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab definitions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const TABS: TabDef[] = [
  { key: "forward", label: "Forward Pass", icon: <ArrowRight className="w-3.5 h-3.5" /> },
  { key: "backward", label: "Backward Pass", icon: <ArrowLeft className="w-3.5 h-3.5" /> },
  { key: "chain", label: "Chain Rule", icon: <Link className="w-3.5 h-3.5" /> },
  { key: "gradient", label: "Gradient Flow", icon: <Layers className="w-3.5 h-3.5" /> },
  { key: "train", label: "Train & Watch", icon: <TrendingUp className="w-3.5 h-3.5" /> },
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Math helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-clamp(x, -500, 500)));
}

function sigmoidDeriv(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

function relu(x: number): number {
  return Math.max(0, x);
}

function reluDeriv(x: number): number {
  return x > 0 ? 1 : 0;
}

function tanhAct(x: number): number {
  return Math.tanh(x);
}

function tanhDeriv(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

function linear(x: number): number {
  return x;
}

function linearDeriv(_x: number): number {
  return 1;
}

type ActivationFn = (x: number) => number;
type ActivationDerivFn = (x: number) => number;

function getActivation(name: string): [ActivationFn, ActivationDerivFn] {
  switch (name) {
    case "sigmoid": return [sigmoid, sigmoidDeriv];
    case "relu": return [relu, reluDeriv];
    case "tanh": return [tanhAct, tanhDeriv];
    case "linear": return [linear, linearDeriv];
    default: return [sigmoid, sigmoidDeriv];
  }
}

// Color helpers
function activationColor(a: number): string {
  const t = clamp(a, 0, 1);
  const r = Math.round(59 + t * 180);
  const g = Math.round(130 - t * 60);
  const b = Math.round(246 - t * 200);
  return `rgb(${r},${g},${b})`;
}

function gradMagnitudeColor(g: number, maxG: number): string {
  if (maxG < 1e-10) return "rgba(100,150,255,0.3)";
  const t = clamp(Math.abs(g) / maxG, 0, 1);
  const r = Math.round(60 + t * 195);
  const gb = Math.round(130 - t * 90);
  const bb = Math.round(246 - t * 210);
  return `rgb(${r},${gb},${bb})`;
}

function gradStrokeWidth(g: number, maxG: number): number {
  if (maxG < 1e-10) return 1;
  return 1 + clamp(Math.abs(g) / maxG, 0, 1) * 4;
}

function weightColor(w: number): string {
  const t = clamp(Math.abs(w), 0, 2) / 2;
  if (w >= 0) return `rgba(59,130,246,${0.2 + t * 0.8})`;
  return `rgba(239,68,68,${0.2 + t * 0.8})`;
}

function weightWidth(w: number): number {
  return 1 + clamp(Math.abs(w), 0, 2) * 1.5;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Network init helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function initNetworkWeights(layers: number[], rng: () => number): NetworkWeights {
  const w: number[][][] = [];
  const b: number[][] = [];
  for (let l = 0; l < layers.length - 1; l++) {
    const fanIn = layers[l];
    const fanOut = layers[l + 1];
    const layerW: number[][] = [];
    const layerB: number[] = [];
    const std = Math.sqrt(2 / (fanIn + fanOut));
    for (let j = 0; j < fanOut; j++) {
      const row: number[] = [];
      for (let i = 0; i < fanIn; i++) {
        row.push(seededGaussian(rng) * std);
      }
      layerW.push(row);
      layerB.push(seededGaussian(rng) * 0.1);
    }
    w.push(layerW);
    b.push(layerB);
  }
  return { w, b };
}

function forwardPass(
  inputs: number[],
  net: NetworkWeights,
  actFn: ActivationFn,
  outputActFn?: ActivationFn,
): NeuronState[][] {
  const layers: NeuronState[][] = [];
  layers.push(inputs.map(v => ({ z: v, a: v, dA: 0, dZ: 0 })));
  for (let l = 0; l < net.w.length; l++) {
    const prevA = layers[l].map(n => n.a);
    const layerNeurons: NeuronState[] = [];
    const isOutput = l === net.w.length - 1;
    const fn = isOutput && outputActFn ? outputActFn : actFn;
    for (let j = 0; j < net.w[l].length; j++) {
      let z = net.b[l][j];
      for (let i = 0; i < prevA.length; i++) {
        z += prevA[i] * net.w[l][j][i];
      }
      const a = fn(z);
      layerNeurons.push({ z, a, dA: 0, dZ: 0 });
    }
    layers.push(layerNeurons);
  }
  return layers;
}

function backwardPass(
  layers: NeuronState[][],
  net: NetworkWeights,
  target: number,
  actDerivFn: ActivationDerivFn,
  outputActDerivFn?: ActivationDerivFn,
): { layers: NeuronState[][]; dW: number[][][]; dB: number[][] } {
  const L = layers.length;
  const result: NeuronState[][] = layers.map(layer => layer.map(n => ({ ...n })));
  const dW: number[][][] = net.w.map(lw => lw.map(nw => nw.map(() => 0)));
  const dB: number[][] = net.b.map(lb => lb.map(() => 0));

  // Output layer: dL/da = 2*(a - target)
  const outIdx = L - 1;
  const outA = result[outIdx][0].a;
  result[outIdx][0].dA = 2 * (outA - target);
  const outDerivFn = outputActDerivFn || actDerivFn;
  result[outIdx][0].dZ = result[outIdx][0].dA * outDerivFn(result[outIdx][0].z);

  // Backward through layers
  for (let l = L - 2; l >= 0; l--) {
    const nextLayer = l + 1;
    // Compute weight and bias gradients for layer l -> l+1
    for (let j = 0; j < result[nextLayer].length; j++) {
      const dZ = result[nextLayer][j].dZ;
      for (let i = 0; i < result[l].length; i++) {
        dW[l][j][i] = dZ * result[l][i].a;
      }
      dB[l][j] = dZ;
    }
    // Propagate gradients to layer l (if not input)
    if (l > 0) {
      for (let i = 0; i < result[l].length; i++) {
        let da = 0;
        for (let j = 0; j < result[nextLayer].length; j++) {
          da += result[nextLayer][j].dZ * net.w[l][j][i];
        }
        result[l][i].dA = da;
        result[l][i].dZ = da * actDerivFn(result[l][i].z);
      }
    } else {
      for (let i = 0; i < result[l].length; i++) {
        let da = 0;
        for (let j = 0; j < result[nextLayer].length; j++) {
          da += result[nextLayer][j].dZ * net.w[l][j][i];
        }
        result[l][i].dA = da;
        result[l][i].dZ = da;
      }
    }
  }
  return { layers: result, dW, dB };
}

function neuronPositions(layerSizes: number[], svgW: number, svgH: number, padX: number, padY: number) {
  const positions: { x: number; y: number }[][] = [];
  const nLayers = layerSizes.length;
  const xStep = nLayers > 1 ? (svgW - 2 * padX) / (nLayers - 1) : 0;
  for (let l = 0; l < nLayers; l++) {
    const count = layerSizes[l];
    const totalH = svgH - 2 * padY;
    const yStep = count > 1 ? totalH / (count - 1) : 0;
    const yStart = count > 1 ? padY : svgH / 2;
    const layerPos: { x: number; y: number }[] = [];
    for (let n = 0; n < count; n++) {
      layerPos.push({ x: padX + l * xStep, y: yStart + n * yStep });
    }
    positions.push(layerPos);
  }
  return positions;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 1: Forward Pass Visualization
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const FWD_LAYERS = [2, 3, 1];
const FWD_LABELS = ["Input", "Hidden", "Output"];
const FWD_W = 620;
const FWD_H = 380;

function ForwardPassTab() {
  const [x1, setX1] = useState(0.8);
  const [x2, setX2] = useState(0.4);
  const [target, setTarget] = useState(0.7);
  const [net, setNet] = useState(() => initNetworkWeights(FWD_LAYERS, mulberry32(42)));
  const [animLayer, setAnimLayer] = useState(-1);
  const [propagated, setPropagated] = useState(false);
  const [selectedNeuron, setSelectedNeuron] = useState<{ l: number; n: number } | null>(null);
  const animRef = useRef<number | null>(null);

  useEffect(() => () => { if (animRef.current) cancelAnimationFrame(animRef.current); }, []);

  const positions = useMemo(() => neuronPositions(FWD_LAYERS, FWD_W, FWD_H, 90, 60), []);

  const layers = useMemo(() => forwardPass([x1, x2], net, sigmoid, sigmoid), [x1, x2, net]);

  const loss = (layers[2][0].a - target) ** 2;

  const propagate = useCallback(() => {
    if (animRef.current) return;
    setPropagated(false);
    setAnimLayer(0);
    const start = performance.now();
    const dur = 600;
    const tick = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / (dur * 2), 1);
      const layer = Math.min(Math.floor(progress * 3), 2);
      setAnimLayer(layer);
      if (progress < 1) {
        animRef.current = requestAnimationFrame(tick);
      } else {
        setAnimLayer(-1);
        setPropagated(true);
        animRef.current = null;
      }
    };
    animRef.current = requestAnimationFrame(tick);
  }, []);

  const resetNet = useCallback(() => {
    if (animRef.current) { cancelAnimationFrame(animRef.current); animRef.current = null; }
    setNet(initNetworkWeights(FWD_LAYERS, mulberry32(Math.floor(Math.random() * 100000))));
    setPropagated(false);
    setAnimLayer(-1);
    setSelectedNeuron(null);
  }, []);

  const sel = selectedNeuron;

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex gap-2">
        <Info className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
        <p className="text-xs text-blue-700">
          The <strong>forward pass</strong> sends input values through the network layer by layer.
          Each neuron computes a weighted sum z = Sum(w*x) + b, then applies the sigmoid activation a = 1/(1+e^(-z)).
          Adjust inputs and click <strong>Propagate</strong> to watch data flow. Click any hidden/output neuron to see its math.
        </p>
      </div>

      <div className="flex gap-4 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-3">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg viewBox={`0 0 ${FWD_W} ${FWD_H}`} className="w-full" style={{ aspectRatio: `${FWD_W}/${FWD_H}` }}>
              <defs>
                <filter id="fwd-glow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>

              {/* Layer labels */}
              {FWD_LABELS.map((label, l) => (
                <text key={l} x={positions[l][0].x} y={30} textAnchor="middle" fontSize={11} fontWeight={600} fill="#475569">{label}</text>
              ))}

              {/* Connections */}
              {net.w.map((layerW, l) =>
                layerW.map((neuronW, j) =>
                  neuronW.map((w, i) => {
                    const from = positions[l][i];
                    const to = positions[l + 1][j];
                    return (
                      <line key={`c-${l}-${i}-${j}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                        stroke={weightColor(w)} strokeWidth={weightWidth(w)} />
                    );
                  })
                )
              )}

              {/* Weight labels */}
              {net.w.map((layerW, l) =>
                layerW.map((neuronW, j) =>
                  neuronW.map((w, i) => {
                    const from = positions[l][i];
                    const to = positions[l + 1][j];
                    const mx = (from.x + to.x) / 2 + (l === 0 ? -8 : 12);
                    const my = (from.y + to.y) / 2 + (j - (FWD_LAYERS[l + 1] - 1) / 2) * 3;
                    return (
                      <text key={`wl-${l}-${i}-${j}`} x={mx} y={my} textAnchor="middle"
                        fontSize={7} fontFamily="monospace" fill="#64748b">{w.toFixed(2)}</text>
                    );
                  })
                )
              )}

              {/* Neurons */}
              {positions.map((layer, l) =>
                layer.map((pos, n) => {
                  const neuron = layers[l][n];
                  const isGlowing = animLayer === l;
                  const showVal = propagated || animLayer >= l;
                  const fillC = showVal && l > 0 ? activationColor(neuron.a) : (l === 0 ? "#eff6ff" : "#f1f5f9");
                  const strokeC = l === 0 ? "#3b82f6" : l === 2 ? "#22c55e" : "#a855f7";
                  const isSel = sel && sel.l === l && sel.n === n;
                  return (
                    <g key={`n-${l}-${n}`} onClick={() => { if (l > 0) setSelectedNeuron({ l, n }); }}
                      className={l > 0 ? "cursor-pointer" : ""}>
                      {isGlowing && (
                        <circle cx={pos.x} cy={pos.y} r={30} fill="none" stroke="#facc15" strokeWidth={3} opacity={0.6} filter="url(#fwd-glow)">
                          <animate attributeName="opacity" values="0.6;0.2;0.6" dur="0.5s" repeatCount="indefinite" />
                        </circle>
                      )}
                      {isSel && <circle cx={pos.x} cy={pos.y} r={27} fill="none" stroke="#f97316" strokeWidth={2} strokeDasharray="4 2" />}
                      <circle cx={pos.x} cy={pos.y} r={22} fill={fillC} stroke={strokeC} strokeWidth={2} />
                      {showVal && (
                        <>
                          <text x={pos.x} y={pos.y - 4} textAnchor="middle" dominantBaseline="central"
                            fontSize={7} fontFamily="monospace" fill="#94a3b8">
                            {l > 0 ? `z=${neuron.z.toFixed(2)}` : ""}
                          </text>
                          <text x={pos.x} y={pos.y + 6} textAnchor="middle" dominantBaseline="central"
                            fontSize={8} fontWeight={600} fontFamily="monospace" fill="#1e293b">
                            {l === 0 ? neuron.a.toFixed(2) : `a=${neuron.a.toFixed(3)}`}
                          </text>
                        </>
                      )}
                      {!showVal && l === 0 && (
                        <text x={pos.x} y={pos.y} textAnchor="middle" dominantBaseline="central"
                          fontSize={9} fontWeight={600} fontFamily="monospace" fill="#1e293b">{neuron.a.toFixed(2)}</text>
                      )}
                      {/* Bias label */}
                      {l > 0 && (
                        <text x={pos.x} y={pos.y - 28} textAnchor="middle" fontSize={7} fontFamily="monospace" fill="#94a3b8">
                          b={net.b[l - 1][n].toFixed(2)}
                        </text>
                      )}
                      {/* Node label */}
                      <text x={pos.x} y={pos.y + 34} textAnchor="middle" fontSize={9} fill="#94a3b8" fontWeight={500}>
                        {l === 0 ? `x${n + 1}` : l === 1 ? `h${n + 1}` : "out"}
                      </text>
                    </g>
                  );
                })
              )}

              {/* Loss display */}
              {propagated && (
                <g>
                  <text x={positions[2][0].x + 60} y={positions[2][0].y - 16} textAnchor="middle"
                    fontSize={10} fontWeight={700} fill="#dc2626" fontFamily="monospace">
                    Loss = {loss.toFixed(4)}
                  </text>
                  <text x={positions[2][0].x + 60} y={positions[2][0].y} textAnchor="middle"
                    fontSize={8} fill="#94a3b8" fontFamily="monospace">
                    (out - target)^2
                  </text>
                  <text x={positions[2][0].x + 60} y={positions[2][0].y + 14} textAnchor="middle"
                    fontSize={8} fill="#64748b" fontFamily="monospace">
                    target = {target.toFixed(2)}
                  </text>
                </g>
              )}
            </svg>
          </div>

          <div className="flex gap-2">
            <button onClick={propagate} disabled={animRef.current !== null}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-white hover:bg-amber-600 disabled:bg-amber-300 disabled:cursor-not-allowed transition-all">
              <Play className="w-4 h-4" /><ArrowRight className="w-3.5 h-3.5" /> Propagate
            </button>
            <button onClick={resetNet}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Right panel */}
        <div className="w-full lg:w-72 space-y-3">
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Input Values</p>
            {[{ label: "x1", val: x1, set: setX1 }, { label: "x2", val: x2, set: setX2 }].map(({ label, val, set }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-500 w-5">{label}</span>
                <input type="range" min={0} max={1} step={0.01} value={val}
                  onChange={e => { set(parseFloat(e.target.value)); setPropagated(false); }}
                  className="flex-1 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded w-9 text-center">{val.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-700">Target</span>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded">{target.toFixed(2)}</span>
            </div>
            <input type="range" min={0} max={1} step={0.01} value={target}
              onChange={e => setTarget(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-500" />
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">Output</p>
              <p className="text-base font-bold font-mono text-slate-800">{propagated ? layers[2][0].a.toFixed(4) : "---"}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">Loss</p>
              <p className="text-base font-bold font-mono text-red-600">{propagated ? loss.toFixed(4) : "---"}</p>
            </div>
          </div>

          {/* Math detail for selected neuron */}
          {sel && sel.l > 0 && (
            <div className="bg-white border border-slate-200 rounded-lg p-3">
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
                Math: {sel.l === 1 ? `h${sel.n + 1}` : "output"}
              </p>
              <div className="text-[10px] font-mono text-slate-700 bg-slate-50 rounded-md p-2 space-y-1.5">
                <div>
                  <span className="text-slate-400">Weighted sum:</span>
                  <div className="ml-2">z = {layers[sel.l - 1].map((prev, i) =>
                    `${prev.a.toFixed(2)}*${net.w[sel.l - 1][sel.n][i].toFixed(2)}`
                  ).join(" + ")} + {net.b[sel.l - 1][sel.n].toFixed(2)}</div>
                  <div className="ml-2 font-semibold">= {layers[sel.l][sel.n].z.toFixed(4)}</div>
                </div>
                <div>
                  <span className="text-slate-400">Activation:</span>
                  <div className="ml-2">a = sigmoid({layers[sel.l][sel.n].z.toFixed(3)}) = <span className="font-semibold">{layers[sel.l][sel.n].a.toFixed(4)}</span></div>
                </div>
              </div>
            </div>
          )}

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Tips</p>
            <ul className="text-[10px] text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Click Propagate to see data flow layer by layer</li>
              <li>Click hidden/output neurons to inspect their computation</li>
              <li>Adjust inputs and observe how outputs change</li>
              <li>Blue edges = positive weights, red = negative</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 2: Backward Pass Step-by-Step
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function BackwardPassTab() {
  const [x1, setX1] = useState(0.8);
  const [x2, setX2] = useState(0.4);
  const [target, setTarget] = useState(0.7);
  const [lr, setLr] = useState(0.5);
  const [net, setNet] = useState(() => initNetworkWeights(FWD_LAYERS, mulberry32(77)));
  const [backStep, setBackStep] = useState(-1); // -1=none, 0=output grads, 1=hidden grads, 2=input grads
  const [updated, setUpdated] = useState(false);

  const positions = useMemo(() => neuronPositions(FWD_LAYERS, FWD_W, FWD_H, 90, 60), []);

  const fwd = useMemo(() => forwardPass([x1, x2], net, sigmoid, sigmoid), [x1, x2, net]);
  const loss = (fwd[2][0].a - target) ** 2;
  const bwd = useMemo(() => backwardPass(fwd, net, target, sigmoidDeriv, sigmoidDeriv), [fwd, net, target]);

  const maxGrad = useMemo(() => {
    let mx = 1e-10;
    for (const lw of bwd.dW) for (const nw of lw) for (const g of nw) mx = Math.max(mx, Math.abs(g));
    return mx;
  }, [bwd]);

  const stepBack = useCallback(() => {
    setBackStep(prev => Math.min(prev + 1, 2));
    setUpdated(false);
  }, []);

  const applyUpdate = useCallback(() => {
    if (backStep < 1) return;
    setNet(prev => {
      const newW = prev.w.map((lw, l) =>
        lw.map((nw, j) => nw.map((w, i) => w - lr * bwd.dW[l][j][i]))
      );
      const newB = prev.b.map((lb, l) =>
        lb.map((b, j) => b - lr * bwd.dB[l][j])
      );
      return { w: newW, b: newB };
    });
    setBackStep(-1);
    setUpdated(true);
  }, [backStep, lr, bwd]);

  const resetAll = useCallback(() => {
    setNet(initNetworkWeights(FWD_LAYERS, mulberry32(77)));
    setBackStep(-1);
    setUpdated(false);
  }, []);

  const showOutputGrad = backStep >= 0;
  const showHiddenGrad = backStep >= 1;
  const showEdgeGrad = backStep >= 1;

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 flex gap-2">
        <Info className="w-4 h-4 text-orange-500 shrink-0 mt-0.5" />
        <p className="text-xs text-orange-700">
          The <strong>backward pass</strong> computes gradients from the output back to the inputs using the chain rule.
          Click <strong>Back Step</strong> to step through gradient computation one layer at a time (right to left).
          Then click <strong>Apply Update</strong> to adjust weights by -lr * gradient.
        </p>
      </div>

      <div className="flex gap-4 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-3">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg viewBox={`0 0 ${FWD_W} ${FWD_H}`} className="w-full" style={{ aspectRatio: `${FWD_W}/${FWD_H}` }}>
              <defs>
                <filter id="bwd-glow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>

              {FWD_LABELS.map((label, l) => (
                <text key={l} x={positions[l][0].x} y={30} textAnchor="middle" fontSize={11} fontWeight={600} fill="#475569">{label}</text>
              ))}

              {/* Connections with gradient coloring */}
              {net.w.map((layerW, l) =>
                layerW.map((neuronW, j) =>
                  neuronW.map((w, i) => {
                    const from = positions[l][i];
                    const to = positions[l + 1][j];
                    const useGrad = showEdgeGrad;
                    const sc = useGrad ? gradMagnitudeColor(bwd.dW[l][j][i], maxGrad) : weightColor(w);
                    const sw = useGrad ? gradStrokeWidth(bwd.dW[l][j][i], maxGrad) : weightWidth(w);
                    return (
                      <line key={`c-${l}-${i}-${j}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke={sc} strokeWidth={sw} />
                    );
                  })
                )
              )}

              {/* Gradient labels on edges */}
              {showEdgeGrad && net.w.map((layerW, l) =>
                layerW.map((neuronW, j) =>
                  neuronW.map((_w, i) => {
                    const from = positions[l][i];
                    const to = positions[l + 1][j];
                    const mx = (from.x + to.x) / 2 + (l === 0 ? -10 : 14);
                    const my = (from.y + to.y) / 2 + (j - (FWD_LAYERS[l + 1] - 1) / 2) * 4;
                    const grad = bwd.dW[l][j][i];
                    return (
                      <text key={`gl-${l}-${i}-${j}`} x={mx} y={my} textAnchor="middle"
                        fontSize={7} fontFamily="monospace" fontWeight={600} fill="#ea580c">
                        {grad >= 0 ? "+" : ""}{grad.toFixed(3)}
                      </text>
                    );
                  })
                )
              )}

              {/* Neurons */}
              {positions.map((layer, l) =>
                layer.map((pos, n) => {
                  const neuron = fwd[l][n];
                  const bwdNeuron = bwd.layers[l][n];
                  const showGrad = (l === 2 && showOutputGrad) || (l === 1 && showHiddenGrad) || (l === 0 && backStep >= 2);
                  const isGlowLayer = (l === 2 && backStep === 0) || (l === 1 && backStep === 1) || (l === 0 && backStep === 2);
                  const fillC = activationColor(neuron.a);
                  const strokeC = l === 0 ? "#3b82f6" : l === 2 ? "#22c55e" : "#a855f7";
                  return (
                    <g key={`n-${l}-${n}`}>
                      {isGlowLayer && (
                        <circle cx={pos.x} cy={pos.y} r={30} fill="none" stroke="#f97316" strokeWidth={3} opacity={0.6} filter="url(#bwd-glow)">
                          <animate attributeName="opacity" values="0.6;0.2;0.6" dur="0.5s" repeatCount="indefinite" />
                        </circle>
                      )}
                      <circle cx={pos.x} cy={pos.y} r={22} fill={fillC} stroke={strokeC} strokeWidth={2} />
                      <text x={pos.x} y={pos.y - 4} textAnchor="middle" dominantBaseline="central"
                        fontSize={8} fontWeight={600} fontFamily="monospace" fill="#1e293b">
                        {neuron.a.toFixed(3)}
                      </text>
                      {showGrad && l > 0 && (
                        <text x={pos.x} y={pos.y + 7} textAnchor="middle" dominantBaseline="central"
                          fontSize={6.5} fontWeight={600} fontFamily="monospace" fill="#ea580c">
                          dz={bwdNeuron.dZ.toFixed(3)}
                        </text>
                      )}
                      <text x={pos.x} y={pos.y + 34} textAnchor="middle" fontSize={9} fill="#94a3b8" fontWeight={500}>
                        {l === 0 ? `x${n + 1}` : l === 1 ? `h${n + 1}` : "out"}
                      </text>
                    </g>
                  );
                })
              )}

              {/* Loss */}
              <text x={positions[2][0].x + 60} y={positions[2][0].y - 10} textAnchor="middle"
                fontSize={10} fontWeight={700} fill="#dc2626" fontFamily="monospace">Loss = {loss.toFixed(4)}</text>
              <text x={positions[2][0].x + 60} y={positions[2][0].y + 4} textAnchor="middle"
                fontSize={8} fill="#64748b" fontFamily="monospace">target = {target.toFixed(2)}</text>
            </svg>
          </div>

          <div className="flex gap-2">
            <button onClick={stepBack} disabled={backStep >= 2}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-orange-500 text-white hover:bg-orange-600 disabled:bg-orange-300 disabled:cursor-not-allowed transition-all">
              <ArrowLeft className="w-3.5 h-3.5" /> Back Step ({backStep + 2}/3)
            </button>
            <button onClick={applyUpdate} disabled={backStep < 1}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-emerald-500 text-white hover:bg-emerald-600 disabled:bg-emerald-300 disabled:cursor-not-allowed transition-all">
              <RefreshCw className="w-4 h-4" /> Apply Update
            </button>
            <button onClick={resetAll}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="w-full lg:w-80 space-y-3">
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Inputs & Target</p>
            {[{ label: "x1", val: x1, set: setX1 }, { label: "x2", val: x2, set: setX2 }, { label: "target", val: target, set: setTarget }].map(({ label, val, set }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-500 w-10">{label}</span>
                <input type="range" min={0} max={1} step={0.01} value={val}
                  onChange={e => { set(parseFloat(e.target.value)); setBackStep(-1); setUpdated(false); }}
                  className="flex-1 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded w-9 text-center">{val.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-700">Learning Rate</span>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded">{lr.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={2} step={0.01} value={lr}
              onChange={e => setLr(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
          </div>

          {/* Gradient table */}
          {showHiddenGrad && (
            <div className="bg-white border border-slate-200 rounded-lg p-3">
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Partial Derivatives</p>
              <div className="text-[9px] font-mono space-y-1 max-h-48 overflow-y-auto">
                <div className="text-slate-400 mb-1">Chain rule: dL/dw = dL/da * da/dz * dz/dw</div>
                {/* Output neuron */}
                <div className="bg-green-50 rounded p-1.5 space-y-0.5">
                  <div className="text-green-700 font-semibold">Output neuron:</div>
                  <div>dL/da_out = 2*(a-t) = {bwd.layers[2][0].dA.toFixed(4)}</div>
                  <div>da/dz = sig'(z) = {sigmoidDeriv(fwd[2][0].z).toFixed(4)}</div>
                  <div className="font-semibold text-orange-700">dL/dz_out = {bwd.layers[2][0].dZ.toFixed(4)}</div>
                </div>
                {/* Hidden neurons */}
                {[0, 1, 2].map(j => (
                  <div key={j} className="bg-purple-50 rounded p-1.5 space-y-0.5">
                    <div className="text-purple-700 font-semibold">h{j + 1}:</div>
                    <div>dL/da_h{j + 1} = dz_out * w = {bwd.layers[1][j].dA.toFixed(4)}</div>
                    <div>da/dz = sig'(z) = {sigmoidDeriv(fwd[1][j].z).toFixed(4)}</div>
                    <div className="font-semibold text-orange-700">dL/dz_h{j + 1} = {bwd.layers[1][j].dZ.toFixed(4)}</div>
                  </div>
                ))}
                {/* Weight gradients */}
                <div className="bg-slate-50 rounded p-1.5 space-y-0.5 mt-1">
                  <div className="text-slate-600 font-semibold">Weight gradients (dL/dw):</div>
                  {bwd.dW.map((lw, l) =>
                    lw.map((nw, j) =>
                      nw.map((g, i) => (
                        <div key={`${l}-${j}-${i}`} className="text-orange-700">
                          w[{l === 0 ? `x${i + 1}` : `h${i + 1}`} -&gt; {l === 0 ? `h${j + 1}` : "out"}]: {g.toFixed(4)}
                          {updated && <span className="text-emerald-600 ml-1">(-{(lr * g).toFixed(4)})</span>}
                        </div>
                      ))
                    )
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 3: Chain Rule Explorer
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const CHAIN_SVG_W = 620;
const CHAIN_SVG_H = 200;

type ChainFnName = "linear" | "sigmoid" | "relu";

function chainFn(name: ChainFnName, x: number): number {
  switch (name) {
    case "linear": return x;
    case "sigmoid": return sigmoid(x);
    case "relu": return relu(x);
  }
}

function chainFnDeriv(name: ChainFnName, x: number): number {
  switch (name) {
    case "linear": return 1;
    case "sigmoid": return sigmoidDeriv(x);
    case "relu": return reluDeriv(x);
  }
}

function ChainRuleTab() {
  const [depth, setDepth] = useState(3);
  const [xVal, setXVal] = useState(0.5);
  const [targetVal, setTargetVal] = useState(1.0);
  const [fns, setFns] = useState<ChainFnName[]>(["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"]);

  const computeChain = useCallback((d: number, x: number, t: number, fnList: ChainFnName[]) => {
    const vals: number[] = [x];
    const derivs: number[] = [];
    let current = x;
    for (let i = 0; i < d; i++) {
      const fn = fnList[i] || "sigmoid";
      derivs.push(chainFnDeriv(fn, current));
      current = chainFn(fn, current);
      vals.push(current);
    }
    const finalOut = current;
    const lossVal = (finalOut - t) ** 2;
    const dLoss_dOut = 2 * (finalOut - t);

    // Build chain of derivatives for dL/dx
    const chainDerivs = [dLoss_dOut, ...derivs.slice().reverse()];
    let product = dLoss_dOut;
    const products: number[] = [dLoss_dOut];
    for (let i = derivs.length - 1; i >= 0; i--) {
      product *= derivs[i];
      products.push(product);
    }

    return { vals, derivs, lossVal, dLoss_dOut, products, finalOut };
  }, []);

  const chain = useMemo(
    () => computeChain(depth, xVal, targetVal, fns),
    [depth, xVal, targetVal, fns, computeChain]
  );

  const nodeCount = depth + 2; // input + depth hidden + loss
  const nodeX = (idx: number) => 50 + idx * ((CHAIN_SVG_W - 100) / (nodeCount - 1));

  const setFn = (idx: number, fn: ChainFnName) => {
    setFns(prev => { const next = [...prev]; next[idx] = fn; return next; });
  };

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-3 flex gap-2">
        <Info className="w-4 h-4 text-violet-500 shrink-0 mt-0.5" />
        <p className="text-xs text-violet-700">
          The <strong>chain rule</strong> is the mathematical backbone of backpropagation. For a chain of functions
          x -&gt; f1 -&gt; f2 -&gt; ... -&gt; L, the derivative dL/dx is the product of all local derivatives along the chain.
          Explore how gradient signals multiply and how they can vanish with sigmoid or explode with certain configurations.
          Increase depth to see the effect.
        </p>
      </div>

      <div className="flex gap-4 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-3">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg viewBox={`0 0 ${CHAIN_SVG_W} ${CHAIN_SVG_H}`} className="w-full" style={{ aspectRatio: `${CHAIN_SVG_W}/${CHAIN_SVG_H}` }}>
              {/* Connections */}
              {Array.from({ length: nodeCount - 1 }).map((_, i) => {
                const x1 = nodeX(i) + 18;
                const x2 = nodeX(i + 1) - 18;
                return (
                  <g key={`edge-${i}`}>
                    <line x1={x1} y1={90} x2={x2} y2={90} stroke="#94a3b8" strokeWidth={2} />
                    <polygon points={`${x2},86 ${x2 + 6},90 ${x2},94`} fill="#94a3b8" />
                    {i < depth && (
                      <text x={(x1 + x2) / 2} y={78} textAnchor="middle" fontSize={9} fontWeight={600} fill="#7c3aed">
                        {fns[i] || "sigmoid"}
                      </text>
                    )}
                    {i === depth && (
                      <text x={(x1 + x2) / 2} y={78} textAnchor="middle" fontSize={9} fontWeight={600} fill="#dc2626">
                        (y-t)^2
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {Array.from({ length: nodeCount }).map((_, i) => {
                const cx = nodeX(i);
                const isInput = i === 0;
                const isLoss = i === nodeCount - 1;
                const isOutput = i === depth;
                const fillC = isInput ? "#eff6ff" : isLoss ? "#fef2f2" : isOutput ? "#f0fdf4" : "#faf5ff";
                const strokeC = isInput ? "#3b82f6" : isLoss ? "#dc2626" : isOutput ? "#22c55e" : "#a855f7";
                const val = isLoss ? chain.lossVal : chain.vals[i];
                const label = isInput ? "x" : isLoss ? "L" : isOutput ? `y (f${depth})` : `f${i}`;
                return (
                  <g key={`node-${i}`}>
                    <circle cx={cx} cy={90} r={18} fill={fillC} stroke={strokeC} strokeWidth={2} />
                    <text x={cx} y={87} textAnchor="middle" dominantBaseline="central" fontSize={9} fontWeight={600} fontFamily="monospace" fill="#1e293b">
                      {val !== undefined ? val.toFixed(3) : ""}
                    </text>
                    <text x={cx} y={122} textAnchor="middle" fontSize={8} fill="#64748b" fontWeight={500}>{label}</text>
                  </g>
                );
              })}

              {/* Derivative chain below */}
              <text x={CHAIN_SVG_W / 2} y={150} textAnchor="middle" fontSize={10} fontWeight={600} fill="#475569">
                dL/dx = {chain.products.length > 1
                  ? chain.products.slice(0, -1).map((_, idx) => {
                      if (idx === 0) return `dL/dy(${chain.dLoss_dOut.toFixed(3)})`;
                      return `df${depth - idx + 1}'(${chain.derivs[depth - idx].toFixed(3)})`;
                    }).join(" * ")
                  : chain.dLoss_dOut.toFixed(3)}
              </text>
              <text x={CHAIN_SVG_W / 2} y={170} textAnchor="middle" fontSize={12} fontWeight={700} fill={Math.abs(chain.products[chain.products.length - 1]) < 0.01 ? "#dc2626" : "#059669"}>
                = {chain.products[chain.products.length - 1].toFixed(6)}
                {Math.abs(chain.products[chain.products.length - 1]) < 0.01 && " (vanishing!)"}
                {Math.abs(chain.products[chain.products.length - 1]) > 10 && " (exploding!)"}
              </text>
            </svg>
          </div>

          {/* Gradient multiplication visualization */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Gradient Magnitude at Each Stage</p>
            <div className="flex items-end gap-1 h-24">
              {chain.products.map((p, i) => {
                const maxP = Math.max(...chain.products.map(Math.abs), 0.001);
                const h = clamp(Math.abs(p) / maxP, 0.02, 1) * 80;
                const isVanish = Math.abs(p) < 0.01;
                return (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1">
                    <span className="text-[8px] font-mono text-slate-500">{p.toFixed(3)}</span>
                    <div style={{ height: h, background: isVanish ? "#fca5a5" : "#86efac" }}
                      className="w-full rounded-t-sm transition-all" />
                    <span className="text-[7px] text-slate-400">{i === 0 ? "dL/dy" : `*df${depth - i + 1}'`}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="w-full lg:w-64 space-y-3">
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-700">Chain Depth</span>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded">{depth}</span>
            </div>
            <input type="range" min={2} max={6} step={1} value={depth}
              onChange={e => setDepth(parseInt(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-500" />
            <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
              <span>2 (shallow)</span><span>6 (deep)</span>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Functions</p>
            {Array.from({ length: depth }).map((_, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-500 w-6">f{i + 1}</span>
                <select value={fns[i] || "sigmoid"} onChange={e => setFn(i, e.target.value as ChainFnName)}
                  className="flex-1 text-[10px] px-2 py-1 rounded border border-slate-200 bg-white text-slate-700">
                  <option value="sigmoid">sigmoid</option>
                  <option value="relu">relu</option>
                  <option value="linear">linear</option>
                </select>
              </div>
            ))}
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            {[
              { label: "x (input)", val: xVal, set: setXVal, min: -2, max: 2, color: "accent-blue-500" },
              { label: "target", val: targetVal, set: setTargetVal, min: -1, max: 2, color: "accent-green-500" },
            ].map(({ label, val, set, min, max, color }) => (
              <div key={label}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-mono text-slate-500">{label}</span>
                  <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1 py-0.5 rounded">{val.toFixed(2)}</span>
                </div>
                <input type="range" min={min} max={max} step={0.01} value={val}
                  onChange={e => set(parseFloat(e.target.value))}
                  className={`w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer ${color}`} />
              </div>
            ))}
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Experiment</p>
            <ul className="text-[10px] text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Set all functions to sigmoid, increase depth to 6 -- see gradients vanish</li>
              <li>Switch to relu -- gradients pass through without shrinking</li>
              <li>Set x near 0 with sigmoid -- derivative is ~0.25 at each layer</li>
              <li>Try linear functions -- gradient stays constant</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 4: Gradient Flow Visualization
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const DEEP_LAYERS = [2, 4, 4, 4, 1];
const GRAD_SVG_W = 620;
const GRAD_SVG_H = 340;

function GradientFlowTab() {
  const [actName, setActName] = useState<"sigmoid" | "relu">("sigmoid");
  const [net, setNet] = useState(() => initNetworkWeights(DEEP_LAYERS, mulberry32(123)));
  const [seed, setSeed] = useState(123);
  const [x1, setX1] = useState(0.6);
  const [x2, setX2] = useState(0.3);
  const [target, setTarget] = useState(0.8);
  const [trained, setTrained] = useState(false);

  const [actFn, actDerivFnVal] = useMemo(() => getActivation(actName), [actName]);

  const positions = useMemo(() => neuronPositions(DEEP_LAYERS, GRAD_SVG_W, GRAD_SVG_H, 70, 50), []);

  const fwd = useMemo(() => forwardPass([x1, x2], net, actFn, sigmoid), [x1, x2, net, actFn]);
  const bwd = useMemo(() => backwardPass(fwd, net, target, actDerivFnVal, sigmoidDeriv), [fwd, net, target, actDerivFnVal]);

  // Per-layer gradient stats
  const layerGradStats = useMemo(() => {
    return bwd.dW.map((lw) => {
      const allG = lw.flatMap(nw => nw);
      const mean = allG.reduce((s, g) => s + Math.abs(g), 0) / Math.max(allG.length, 1);
      const max = Math.max(...allG.map(Math.abs), 1e-10);
      return { mean, max, values: allG };
    });
  }, [bwd]);

  const globalMaxGrad = Math.max(...layerGradStats.map(s => s.max), 1e-10);

  const trainStep = useCallback(() => {
    const lr = 0.1;
    setNet(prev => {
      const fwdLocal = forwardPass([x1, x2], prev, actFn, sigmoid);
      const bwdLocal = backwardPass(fwdLocal, prev, target, actDerivFnVal, sigmoidDeriv);
      const newW = prev.w.map((lw, l) =>
        lw.map((nw, j) => nw.map((w, i) => w - lr * bwdLocal.dW[l][j][i]))
      );
      const newB = prev.b.map((lb, l) =>
        lb.map((b, j) => b - lr * bwdLocal.dB[l][j])
      );
      return { w: newW, b: newB };
    });
    setTrained(true);
  }, [x1, x2, target, actFn, actDerivFnVal]);

  const resetNet = useCallback(() => {
    const newSeed = seed + 1;
    setSeed(newSeed);
    setNet(initNetworkWeights(DEEP_LAYERS, mulberry32(newSeed)));
    setTrained(false);
  }, [seed]);

  return (
    <div className="space-y-4">
      <div className="bg-rose-50 border border-rose-200 rounded-lg p-3 flex gap-2">
        <Info className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
        <p className="text-xs text-rose-700">
          In deeper networks, gradients can <strong>vanish</strong> as they propagate backward through sigmoid activations
          (since sigmoid' is at most 0.25). Compare <strong>sigmoid vs ReLU</strong> to see the dramatic difference.
          The heatmap and histograms show gradient magnitude per layer.
        </p>
      </div>

      <div className="flex gap-4 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-3">
          {/* Network visualization */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg viewBox={`0 0 ${GRAD_SVG_W} ${GRAD_SVG_H}`} className="w-full" style={{ aspectRatio: `${GRAD_SVG_W}/${GRAD_SVG_H}` }}>
              {/* Layer labels */}
              {["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"].map((label, l) => (
                <text key={l} x={positions[l][0].x} y={25} textAnchor="middle" fontSize={10} fontWeight={600} fill="#475569">{label}</text>
              ))}

              {/* Connections colored by gradient */}
              {net.w.map((layerW, l) =>
                layerW.map((neuronW, j) =>
                  neuronW.map((_w, i) => {
                    const from = positions[l][i];
                    const to = positions[l + 1][j];
                    const grad = bwd.dW[l][j][i];
                    return (
                      <line key={`c-${l}-${i}-${j}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                        stroke={gradMagnitudeColor(grad, globalMaxGrad)}
                        strokeWidth={gradStrokeWidth(grad, globalMaxGrad)} />
                    );
                  })
                )
              )}

              {/* Neurons */}
              {positions.map((layer, l) =>
                layer.map((pos, n) => {
                  const neuron = fwd[l][n];
                  return (
                    <g key={`n-${l}-${n}`}>
                      <circle cx={pos.x} cy={pos.y} r={16} fill={activationColor(clamp(neuron.a, 0, 1))}
                        stroke={l === 0 ? "#3b82f6" : l === DEEP_LAYERS.length - 1 ? "#22c55e" : "#a855f7"} strokeWidth={1.5} />
                      <text x={pos.x} y={pos.y} textAnchor="middle" dominantBaseline="central"
                        fontSize={7} fontWeight={600} fontFamily="monospace" fill="#1e293b">{neuron.a.toFixed(2)}</text>
                    </g>
                  );
                })
              )}
            </svg>
          </div>

          {/* Gradient heatmap per layer */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Gradient Magnitude Heatmap (per layer)</p>
            <div className="flex gap-2">
              {layerGradStats.map((stats, l) => {
                const layerLabel = l === 0 ? "L0-1" : l === 1 ? "L1-2" : l === 2 ? "L2-3" : "L3-4";
                return (
                  <div key={l} className="flex-1">
                    <p className="text-[9px] text-center text-slate-400 mb-1">{layerLabel}</p>
                    <div className="grid grid-cols-2 gap-0.5">
                      {stats.values.map((g, gi) => {
                        const intensity = clamp(Math.abs(g) / globalMaxGrad, 0, 1);
                        const r = Math.round(255);
                        const gb = Math.round(255 - intensity * 200);
                        return (
                          <div key={gi} className="aspect-square rounded-sm flex items-center justify-center"
                            style={{ background: `rgb(${r},${gb},${gb})` }}>
                            <span className="text-[6px] font-mono text-slate-700">{Math.abs(g).toFixed(3)}</span>
                          </div>
                        );
                      })}
                    </div>
                    <p className="text-[8px] text-center text-slate-500 mt-1 font-mono">avg: {stats.mean.toFixed(4)}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Weight update magnitude bars */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Weight Update Magnitude (lr=0.1)</p>
            <div className="space-y-1.5">
              {layerGradStats.map((stats, l) => {
                const updateMag = stats.mean * 0.1;
                const maxUpdate = Math.max(...layerGradStats.map(s => s.mean * 0.1), 1e-10);
                const barW = clamp(updateMag / maxUpdate, 0.01, 1) * 100;
                const label = l === 0 ? "Layer 0->1" : l === 1 ? "Layer 1->2" : l === 2 ? "Layer 2->3" : "Layer 3->4";
                return (
                  <div key={l} className="flex items-center gap-2">
                    <span className="text-[9px] font-mono text-slate-500 w-16">{label}</span>
                    <div className="flex-1 bg-slate-100 rounded-full h-3 overflow-hidden">
                      <div className="h-full rounded-full transition-all" style={{
                        width: `${barW}%`,
                        background: barW < 10 ? "#fca5a5" : barW < 40 ? "#fde68a" : "#86efac"
                      }} />
                    </div>
                    <span className="text-[8px] font-mono text-slate-600 w-14 text-right">{updateMag.toFixed(5)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="w-full lg:w-64 space-y-3">
          {/* Activation selector */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Activation Function</p>
            <div className="flex gap-2">
              {(["sigmoid", "relu"] as const).map(name => (
                <button key={name} onClick={() => setActName(name)}
                  className={`flex-1 text-xs py-1.5 rounded-md font-semibold transition-all ${
                    actName === name ? "bg-violet-500 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}>
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Inputs & Target</p>
            {[
              { label: "x1", val: x1, set: setX1 },
              { label: "x2", val: x2, set: setX2 },
              { label: "target", val: target, set: setTarget },
            ].map(({ label, val, set }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-500 w-10">{label}</span>
                <input type="range" min={0} max={1} step={0.01} value={val}
                  onChange={e => set(parseFloat(e.target.value))}
                  className="flex-1 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1 py-0.5 rounded w-8 text-center">{val.toFixed(2)}</span>
              </div>
            ))}
          </div>

          <div className="flex gap-2">
            <button onClick={trainStep}
              className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-emerald-500 text-white hover:bg-emerald-600 transition-all">
              <Zap className="w-3.5 h-3.5" /> Train 1 Step
            </button>
            <button onClick={resetNet}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">Output</p>
              <p className="text-sm font-bold font-mono text-slate-800">{fwd[DEEP_LAYERS.length - 1][0].a.toFixed(4)}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">Loss</p>
              <p className="text-sm font-bold font-mono text-red-600">{((fwd[DEEP_LAYERS.length - 1][0].a - target) ** 2).toFixed(4)}</p>
            </div>
          </div>

          {/* Gradient histogram per layer */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Gradient Histogram</p>
            <div className="space-y-2">
              {layerGradStats.map((stats, l) => {
                const bins = [0, 0, 0, 0, 0]; // 5 bins: [0,0.001), [0.001,0.01), [0.01,0.1), [0.1,1), [1+)
                for (const g of stats.values) {
                  const ag = Math.abs(g);
                  if (ag < 0.001) bins[0]++;
                  else if (ag < 0.01) bins[1]++;
                  else if (ag < 0.1) bins[2]++;
                  else if (ag < 1) bins[3]++;
                  else bins[4]++;
                }
                const maxBin = Math.max(...bins, 1);
                return (
                  <div key={l}>
                    <p className="text-[8px] text-slate-400 mb-0.5">Layer {l}-{l + 1}</p>
                    <div className="flex items-end gap-0.5 h-8">
                      {bins.map((count, bi) => (
                        <div key={bi} className="flex-1 flex flex-col items-center">
                          <div className="w-full rounded-t-sm" style={{
                            height: `${(count / maxBin) * 28}px`,
                            background: bi < 2 ? "#fca5a5" : bi < 4 ? "#fde68a" : "#86efac"
                          }} />
                        </div>
                      ))}
                    </div>
                    <div className="flex gap-0.5">
                      {["<.001", "<.01", "<.1", "<1", "1+"].map((lb, bi) => (
                        <span key={bi} className="flex-1 text-center text-[6px] text-slate-400">{lb}</span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tab 5: Train & Watch
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const XOR_DATA: [number, number, number][] = [
  [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0],
];
const TRAIN_LAYERS = [2, 4, 4, 1];
const TRAIN_SVG_W = 400;
const TRAIN_SVG_H = 300;
const DB_SVG_W = 220;
const DB_SVG_H = 220;
const LOSS_SVG_W = 400;
const LOSS_SVG_H = 120;

function TrainWatchTab() {
  const [net, setNet] = useState(() => initNetworkWeights(TRAIN_LAYERS, mulberry32(999)));
  const [epoch, setEpoch] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [speed, setSpeed] = useState(50);
  const [lr, setLr] = useState(0.5);
  const trainRef = useRef<number | null>(null);
  const netRef = useRef(net);
  const epochRef = useRef(0);
  const lossRef = useRef<number[]>([]);

  useEffect(() => { netRef.current = net; }, [net]);
  useEffect(() => { epochRef.current = epoch; }, [epoch]);
  useEffect(() => { lossRef.current = lossHistory; }, [lossHistory]);

  useEffect(() => () => { if (trainRef.current) clearInterval(trainRef.current); }, []);

  const positions = useMemo(() => neuronPositions(TRAIN_LAYERS, TRAIN_SVG_W, TRAIN_SVG_H, 60, 40), []);

  const currentFwd = useMemo(() => forwardPass([0.5, 0.5], net, relu, sigmoid), [net]);

  const currentLoss = useMemo(() => {
    let total = 0;
    for (const [x1, x2, t] of XOR_DATA) {
      const out = forwardPass([x1, x2], net, relu, sigmoid);
      total += (out[TRAIN_LAYERS.length - 1][0].a - t) ** 2;
    }
    return total / XOR_DATA.length;
  }, [net]);

  const accuracy = useMemo(() => {
    let correct = 0;
    for (const [x1, x2, t] of XOR_DATA) {
      const out = forwardPass([x1, x2], net, relu, sigmoid);
      const pred = out[TRAIN_LAYERS.length - 1][0].a > 0.5 ? 1 : 0;
      if (pred === t) correct++;
    }
    return correct / XOR_DATA.length;
  }, [net]);

  // Decision boundary grid
  const dbGrid = useMemo(() => {
    const res = 25;
    const grid: number[][] = [];
    for (let r = 0; r < res; r++) {
      const row: number[] = [];
      for (let c = 0; c < res; c++) {
        const x1 = c / (res - 1);
        const x2 = r / (res - 1);
        const out = forwardPass([x1, x2], net, relu, sigmoid);
        row.push(out[TRAIN_LAYERS.length - 1][0].a);
      }
      grid.push(row);
    }
    return grid;
  }, [net]);

  const trainOneEpoch = useCallback((currentNet: NetworkWeights): NetworkWeights => {
    let w = currentNet.w.map(lw => lw.map(nw => [...nw]));
    let b = currentNet.b.map(lb => [...lb]);
    const lrVal = lr;

    for (const [x1, x2, t] of XOR_DATA) {
      const tempNet = { w, b };
      const fwdRes = forwardPass([x1, x2], tempNet, relu, sigmoid);
      const bwdRes = backwardPass(fwdRes, tempNet, t, reluDeriv, sigmoidDeriv);

      w = w.map((lw, l) =>
        lw.map((nw, j) => nw.map((wv, i) => wv - lrVal * bwdRes.dW[l][j][i]))
      );
      b = b.map((lb, l) =>
        lb.map((bv, j) => bv - lrVal * bwdRes.dB[l][j])
      );
    }
    return { w, b };
  }, [lr]);

  const startTraining = useCallback(() => {
    if (isTraining) return;
    setIsTraining(true);
    const interval = Math.max(10, 200 - speed * 2);
    trainRef.current = window.setInterval(() => {
      const newNet = trainOneEpoch(netRef.current);
      netRef.current = newNet;
      setNet(newNet);
      epochRef.current += 1;
      setEpoch(epochRef.current);

      let totalLoss = 0;
      for (const [x1, x2, t] of XOR_DATA) {
        const out = forwardPass([x1, x2], newNet, relu, sigmoid);
        totalLoss += (out[TRAIN_LAYERS.length - 1][0].a - t) ** 2;
      }
      const avgLoss = totalLoss / XOR_DATA.length;
      lossRef.current = [...lossRef.current, avgLoss];
      setLossHistory([...lossRef.current]);

      if (epochRef.current >= 2000 || avgLoss < 0.001) {
        if (trainRef.current) clearInterval(trainRef.current);
        trainRef.current = null;
        setIsTraining(false);
      }
    }, interval);
  }, [isTraining, speed, trainOneEpoch]);

  const stopTraining = useCallback(() => {
    if (trainRef.current) { clearInterval(trainRef.current); trainRef.current = null; }
    setIsTraining(false);
  }, []);

  const resetAll = useCallback(() => {
    stopTraining();
    const newNet = initNetworkWeights(TRAIN_LAYERS, mulberry32(Math.floor(Math.random() * 100000)));
    setNet(newNet);
    netRef.current = newNet;
    setEpoch(0);
    epochRef.current = 0;
    setLossHistory([]);
    lossRef.current = [];
  }, [stopTraining]);

  // Loss curve path
  const lossPath = useMemo(() => {
    if (lossHistory.length < 2) return "";
    const maxL = Math.max(...lossHistory, 0.1);
    const xScale = (LOSS_SVG_W - 40) / Math.max(lossHistory.length - 1, 1);
    return lossHistory.map((l, i) => {
      const x = 30 + i * xScale;
      const y = 10 + (1 - clamp(l / maxL, 0, 1)) * (LOSS_SVG_H - 30);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    }).join(" ");
  }, [lossHistory]);

  return (
    <div className="space-y-4">
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 flex gap-2">
        <Info className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
        <p className="text-xs text-emerald-700">
          Watch backpropagation in action on the <strong>XOR problem</strong>: a classic non-linearly-separable dataset.
          The network learns a decision boundary as it trains. Watch the loss curve decrease and the decision boundary evolve
          over epochs. Uses <strong>ReLU</strong> hidden layers and <strong>sigmoid</strong> output.
        </p>
      </div>

      <div className="flex gap-4 flex-col lg:flex-row">
        <div className="flex-1 min-w-0 space-y-3">
          <div className="flex gap-3 flex-col md:flex-row">
            {/* Network visualization */}
            <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-3">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium mb-1 text-center">Network State</p>
              <svg viewBox={`0 0 ${TRAIN_SVG_W} ${TRAIN_SVG_H}`} className="w-full" style={{ aspectRatio: `${TRAIN_SVG_W}/${TRAIN_SVG_H}` }}>
                {/* Connections */}
                {net.w.map((layerW, l) =>
                  layerW.map((neuronW, j) =>
                    neuronW.map((w, i) => {
                      const from = positions[l][i];
                      const to = positions[l + 1][j];
                      return (
                        <line key={`c-${l}-${i}-${j}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                          stroke={weightColor(w)} strokeWidth={weightWidth(w)} opacity={0.7} />
                      );
                    })
                  )
                )}
                {/* Neurons */}
                {positions.map((layer, l) =>
                  layer.map((pos, n) => {
                    const neuron = currentFwd[l][n];
                    return (
                      <g key={`n-${l}-${n}`}>
                        <circle cx={pos.x} cy={pos.y} r={14}
                          fill={activationColor(clamp(neuron.a, 0, 1))}
                          stroke={l === 0 ? "#3b82f6" : l === TRAIN_LAYERS.length - 1 ? "#22c55e" : "#a855f7"}
                          strokeWidth={1.5} />
                        <text x={pos.x} y={pos.y} textAnchor="middle" dominantBaseline="central"
                          fontSize={7} fontWeight={600} fontFamily="monospace" fill="#1e293b">{neuron.a.toFixed(2)}</text>
                      </g>
                    );
                  })
                )}
              </svg>
            </div>

            {/* Decision boundary */}
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium mb-1 text-center">Decision Boundary</p>
              <svg viewBox={`0 0 ${DB_SVG_W} ${DB_SVG_H}`} className="w-full" style={{ aspectRatio: "1" }}>
                {/* Grid cells */}
                {dbGrid.map((row, r) =>
                  row.map((val, c) => {
                    const cellW = DB_SVG_W / dbGrid.length;
                    const cellH = DB_SVG_H / dbGrid.length;
                    const t = clamp(val, 0, 1);
                    const red = Math.round(239 * (1 - t) + 59 * t);
                    const green = Math.round(68 * (1 - t) + 130 * t);
                    const blue = Math.round(68 * (1 - t) + 246 * t);
                    return (
                      <rect key={`${r}-${c}`} x={c * cellW} y={r * cellH} width={cellW + 0.5} height={cellH + 0.5}
                        fill={`rgb(${red},${green},${blue})`} />
                    );
                  })
                )}
                {/* Data points */}
                {XOR_DATA.map(([x1, x2, t], i) => (
                  <g key={i}>
                    <circle cx={x1 * DB_SVG_W} cy={x2 * DB_SVG_H} r={8}
                      fill={t === 1 ? "#3b82f6" : "#ef4444"} stroke="white" strokeWidth={2} />
                    <text x={x1 * DB_SVG_W} y={x2 * DB_SVG_H} textAnchor="middle" dominantBaseline="central"
                      fontSize={8} fontWeight={700} fill="white">{t}</text>
                  </g>
                ))}
              </svg>
            </div>
          </div>

          {/* Loss curve */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-1">Loss Curve</p>
            <svg viewBox={`0 0 ${LOSS_SVG_W} ${LOSS_SVG_H}`} className="w-full" style={{ aspectRatio: `${LOSS_SVG_W}/${LOSS_SVG_H}` }}>
              {/* Axes */}
              <line x1={30} y1={10} x2={30} y2={LOSS_SVG_H - 15} stroke="#cbd5e1" strokeWidth={1} />
              <line x1={30} y1={LOSS_SVG_H - 15} x2={LOSS_SVG_W - 10} y2={LOSS_SVG_H - 15} stroke="#cbd5e1" strokeWidth={1} />
              <text x={12} y={LOSS_SVG_H / 2} textAnchor="middle" fontSize={8} fill="#94a3b8" transform={`rotate(-90, 12, ${LOSS_SVG_H / 2})`}>Loss</text>
              <text x={LOSS_SVG_W / 2} y={LOSS_SVG_H - 2} textAnchor="middle" fontSize={8} fill="#94a3b8">Epoch</text>
              {/* Loss line */}
              {lossPath && <path d={lossPath} fill="none" stroke="#ef4444" strokeWidth={1.5} />}
              {lossHistory.length === 0 && (
                <text x={LOSS_SVG_W / 2} y={LOSS_SVG_H / 2} textAnchor="middle" fontSize={11} fill="#94a3b8">Press Train to start</text>
              )}
            </svg>
          </div>
        </div>

        <div className="w-full lg:w-60 space-y-3">
          {/* Controls */}
          <div className="flex gap-2">
            {!isTraining ? (
              <button onClick={startTraining}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-emerald-500 text-white hover:bg-emerald-600 transition-all">
                <Play className="w-3.5 h-3.5" /> Train
              </button>
            ) : (
              <button onClick={stopTraining}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-white hover:bg-amber-600 transition-all">
                <Pause className="w-3.5 h-3.5" /> Pause
              </button>
            )}
            <button onClick={resetAll}
              className="px-3 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-700">Speed</span>
              <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1 py-0.5 rounded">{speed}</span>
            </div>
            <input type="range" min={1} max={100} step={1} value={speed}
              onChange={e => setSpeed(parseInt(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-slate-700">Learning Rate</span>
              <span className="text-[10px] font-mono text-slate-600 bg-slate-100 px-1 py-0.5 rounded">{lr.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={2} step={0.01} value={lr}
              onChange={e => setLr(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[9px] text-slate-500 uppercase font-medium">Epoch</p>
              <p className="text-sm font-bold font-mono text-slate-800">{epoch}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[9px] text-slate-500 uppercase font-medium">Loss</p>
              <p className="text-sm font-bold font-mono text-red-600">{currentLoss.toFixed(4)}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
              <p className="text-[9px] text-slate-500 uppercase font-medium">Acc</p>
              <p className="text-sm font-bold font-mono text-emerald-600">{(accuracy * 100).toFixed(0)}%</p>
            </div>
          </div>

          {/* XOR data table */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">XOR Dataset</p>
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="text-slate-400">
                  <th className="text-left py-0.5">x1</th>
                  <th className="text-left py-0.5">x2</th>
                  <th className="text-left py-0.5">target</th>
                  <th className="text-left py-0.5">pred</th>
                </tr>
              </thead>
              <tbody>
                {XOR_DATA.map(([x1, x2, t], i) => {
                  const out = forwardPass([x1, x2], net, relu, sigmoid);
                  const pred = out[TRAIN_LAYERS.length - 1][0].a;
                  const correct = (pred > 0.5 ? 1 : 0) === t;
                  return (
                    <tr key={i} className={correct ? "text-emerald-600" : "text-red-500"}>
                      <td className="py-0.5">{x1}</td>
                      <td className="py-0.5">{x2}</td>
                      <td className="py-0.5">{t}</td>
                      <td className="py-0.5">{pred.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Observe</p>
            <ul className="text-[10px] text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Watch the decision boundary form a non-linear shape</li>
              <li>Notice loss drops sharply then levels off</li>
              <li>Edge colors/thickness change as weights evolve</li>
              <li>Try different learning rates and reset to compare</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main component
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export default function BackpropagationActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("forward");

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
      {activeTab === "forward" && <ForwardPassTab />}
      {activeTab === "backward" && <BackwardPassTab />}
      {activeTab === "chain" && <ChainRuleTab />}
      {activeTab === "gradient" && <GradientFlowTab />}
      {activeTab === "train" && <TrainWatchTab />}
    </div>
  );
}
