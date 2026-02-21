/**
 * K-Means Clustering Activity — comprehensive 3-tab interactive exploration
 *
 * Tabs:
 *   1. Step Through K-Means   – place centroids, assign, update, auto-run
 *   2. Choosing K (Elbow)     – elbow chart, silhouette scores, side-by-side
 *   3. Convergence Viz        – centroid trails, loss curve, playback controls
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Info,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  Target,
  LineChart,
  Shuffle,
  CheckCircle2,
  Zap,
  Eye,
} from "lucide-react";

// ════════════════════════════════════════════════════════════════════════════
// SHARED UTILITIES
// ════════════════════════════════════════════════════════════════════════════

/** Mulberry32 seeded PRNG — deterministic random numbers */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller Gaussian */
function gaussNoise(rand: () => number, std: number): number {
  const u1 = Math.max(rand(), 1e-10);
  const u2 = rand();
  return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

interface Pt {
  x: number;
  y: number;
}

interface Centroid extends Pt {
  prevX: number;
  prevY: number;
}

function dist(a: Pt, b: Pt): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function distSq(a: Pt, b: Pt): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

function nearest(p: Pt, centroids: Pt[]): number {
  let minD = Infinity;
  let idx = 0;
  for (let i = 0; i < centroids.length; i++) {
    const d = distSq(p, centroids[i]);
    if (d < minD) { minD = d; idx = i; }
  }
  return idx;
}

function computeWCSS(points: Pt[], assignments: number[], centroids: Pt[]): number {
  let wcss = 0;
  for (let i = 0; i < points.length; i++) {
    const c = centroids[assignments[i]];
    if (c) wcss += distSq(points[i], c);
  }
  return wcss;
}

function assignAll(points: Pt[], centroids: Pt[]): number[] {
  return points.map((p) => nearest(p, centroids));
}

function updateCentroids(points: Pt[], assignments: number[], k: number, old: Pt[]): Pt[] {
  const sums = Array.from({ length: k }, () => ({ sx: 0, sy: 0, n: 0 }));
  for (let i = 0; i < points.length; i++) {
    const c = assignments[i];
    sums[c].sx += points[i].x;
    sums[c].sy += points[i].y;
    sums[c].n++;
  }
  return sums.map((s, i) =>
    s.n > 0 ? { x: s.sx / s.n, y: s.sy / s.n } : { x: old[i].x, y: old[i].y },
  );
}

/** Run K-Means to convergence, return { centroids, assignments, wcss, iterations } */
function runKMeans(
  points: Pt[],
  initCentroids: Pt[],
  maxIter = 100,
): { centroids: Pt[]; assignments: number[]; wcss: number; iterations: number } {
  const k = initCentroids.length;
  let centroids = initCentroids.map((c) => ({ ...c }));
  let assignments = assignAll(points, centroids);
  let iterations = 0;
  for (let iter = 0; iter < maxIter; iter++) {
    const newCentroids = updateCentroids(points, assignments, k, centroids);
    const newAssignments = assignAll(points, newCentroids);
    iterations = iter + 1;
    let maxMove = 0;
    for (let i = 0; i < k; i++) {
      maxMove = Math.max(maxMove, dist(centroids[i], newCentroids[i]));
    }
    centroids = newCentroids;
    assignments = newAssignments;
    if (maxMove < 0.5) break;
  }
  const wcss = computeWCSS(points, assignments, centroids);
  return { centroids, assignments, wcss, iterations };
}

/** Random centroids within bounds */
function randomCentroids(k: number, rand: () => number, w: number, h: number, margin = 40): Pt[] {
  return Array.from({ length: k }, () => ({
    x: margin + rand() * (w - 2 * margin),
    y: margin + rand() * (h - 2 * margin),
  }));
}

/** Clamp a value between min and max */
function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

// ── Dataset generators ──────────────────────────────────────────────────

function generateBlobs(
  rand: () => number,
  centers: Pt[],
  perCluster: number,
  std: number,
  w: number,
  h: number,
): Pt[] {
  const points: Pt[] = [];
  for (let i = 0; i < centers.length * perCluster; i++) {
    const c = centers[i % centers.length];
    points.push({
      x: clamp(c.x + gaussNoise(rand, std), 10, w - 10),
      y: clamp(c.y + gaussNoise(rand, std), 10, h - 10),
    });
  }
  return points;
}

// ── Color palette ────────────────────────────────────────────────────────

const COLORS = [
  "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
] as const;

const COLORS_BG = [
  "rgba(239,68,68,0.12)", "rgba(59,130,246,0.12)", "rgba(34,197,94,0.12)",
  "rgba(245,158,11,0.12)", "rgba(139,92,246,0.12)", "rgba(236,72,153,0.12)",
  "rgba(20,184,166,0.12)", "rgba(249,115,22,0.12)", "rgba(99,102,241,0.12)",
  "rgba(132,204,22,0.12)",
] as const;

const COLORS_LIGHT = [
  "rgba(239,68,68,0.35)", "rgba(59,130,246,0.35)", "rgba(34,197,94,0.35)",
  "rgba(245,158,11,0.35)", "rgba(139,92,246,0.35)", "rgba(236,72,153,0.35)",
  "rgba(20,184,166,0.35)", "rgba(249,115,22,0.35)", "rgba(99,102,241,0.35)",
  "rgba(132,204,22,0.35)",
] as const;

// ── SVG helpers ──────────────────────────────────────────────────────────

function GridLines({ w, h, step = 50 }: { w: number; h: number; step?: number }) {
  const lines: React.ReactNode[] = [];
  for (let x = 0; x <= w; x += step) {
    lines.push(<line key={`v${x}`} x1={x} y1={0} x2={x} y2={h} stroke="#e2e8f0" strokeWidth={0.5} />);
  }
  for (let y = 0; y <= h; y += step) {
    lines.push(<line key={`h${y}`} x1={0} y1={y} x2={w} y2={y} stroke="#e2e8f0" strokeWidth={0.5} />);
  }
  return <>{lines}</>;
}

/** Voronoi background approximation via grid sampling */
function VoronoiBg({ w, h, centroids, cellSize = 10 }: { w: number; h: number; centroids: Pt[]; cellSize?: number }) {
  if (centroids.length === 0) return null;
  const cells: React.ReactNode[] = [];
  for (let gx = 0; gx < w; gx += cellSize) {
    for (let gy = 0; gy < h; gy += cellSize) {
      const idx = nearest({ x: gx + cellSize / 2, y: gy + cellSize / 2 }, centroids);
      cells.push(
        <rect key={`${gx}-${gy}`} x={gx} y={gy} width={cellSize} height={cellSize}
          fill={COLORS_BG[idx % COLORS_BG.length]} />,
      );
    }
  }
  return <>{cells}</>;
}

/** Centroid diamond marker */
function CentroidMarker({ c, idx, animate = true }: { c: Pt; idx: number; animate?: boolean }) {
  const color = COLORS[idx % COLORS.length];
  const s = 12;
  const diamond = `M 0 ${-s} L ${s} 0 L 0 ${s} L ${-s} 0 Z`;
  return (
    <g style={{
      transform: `translate(${c.x}px, ${c.y}px)`,
      transition: animate ? "transform 0.4s cubic-bezier(0.4,0,0.2,1)" : "none",
    }}>
      <circle cx={0} cy={0} r={18} fill={color} opacity={0.12} />
      <path d={diamond} fill={color} stroke="#fff" strokeWidth={2.5} />
      <circle cx={0} cy={0} r={2.5} fill="#fff" opacity={0.9} />
      <text x={0} y={-17} textAnchor="middle" fontSize={9} fontWeight={700} fill={color} fontFamily="sans-serif">
        C{idx + 1}
      </text>
    </g>
  );
}

/** Data point circle */
function DataDot({ p, color, r = 4.5, opacity = 0.85 }: { p: Pt; color: string; r?: number; opacity?: number }) {
  return <circle cx={p.x} cy={p.y} r={r} fill={color} stroke="#fff" strokeWidth={1} opacity={opacity} />;
}

/** Simple silhouette score (exact, O(n^2)) */
function silhouetteScore(points: Pt[], assignments: number[], k: number): number {
  if (points.length < 2 || k < 2) return 0;
  const n = points.length;
  let totalSil = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const ci = assignments[i];
    // a(i): mean distance to same cluster
    let sameSum = 0;
    let sameN = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && assignments[j] === ci) {
        sameSum += dist(points[i], points[j]);
        sameN++;
      }
    }
    if (sameN === 0) continue;
    const ai = sameSum / sameN;
    // b(i): min over other clusters of mean distance
    let bi = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci) continue;
      let otherSum = 0;
      let otherN = 0;
      for (let j = 0; j < n; j++) {
        if (assignments[j] === c) {
          otherSum += dist(points[i], points[j]);
          otherN++;
        }
      }
      if (otherN > 0) bi = Math.min(bi, otherSum / otherN);
    }
    if (!isFinite(bi)) continue;
    const si = (bi - ai) / Math.max(ai, bi);
    totalSil += si;
    counted++;
  }
  return counted > 0 ? totalSil / counted : 0;
}

// ════════════════════════════════════════════════════════════════════════════
// TAB DEFINITIONS
// ════════════════════════════════════════════════════════════════════════════

type TabKey = "stepthrough" | "elbow" | "convergence";

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabDef[] = [
  { key: "stepthrough", label: "Step Through K-Means", icon: <SkipForward className="w-3.5 h-3.5" /> },
  { key: "elbow", label: "Choosing K (Elbow)", icon: <LineChart className="w-3.5 h-3.5" /> },
  { key: "convergence", label: "Convergence Viz", icon: <Eye className="w-3.5 h-3.5" /> },
];

// ════════════════════════════════════════════════════════════════════════════
// TAB 1 — STEP THROUGH K-MEANS
// ════════════════════════════════════════════════════════════════════════════

const T1_W = 520;
const T1_H = 400;

function StepThroughTab() {
  const [seed, setSeed] = useState(42);
  const [k, setK] = useState(3);
  const [placedCentroids, setPlacedCentroids] = useState<Centroid[]>([]);
  const [assignments, setAssignments] = useState<number[]>([]);
  const [iteration, setIteration] = useState(0);
  const [converged, setConverged] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [showLines, setShowLines] = useState(true);
  const [phase, setPhase] = useState<"place" | "running">("place");
  const isRunningRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const rawPoints = useMemo(() => {
    const rand = mulberry32(seed);
    const centers = [
      { x: 130, y: 100 }, { x: 380, y: 110 }, { x: 250, y: 320 },
    ];
    return generateBlobs(rand, centers, 25, 38, T1_W, T1_H);
  }, [seed]);

  const wcss = useMemo(() => {
    if (assignments.length === 0 || placedCentroids.length === 0) return 0;
    return computeWCSS(rawPoints, assignments, placedCentroids);
  }, [rawPoints, assignments, placedCentroids]);

  // Click to place centroid
  const handleSvgClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (phase !== "place" || placedCentroids.length >= k) return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const scaleX = T1_W / rect.width;
      const scaleY = T1_H / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;
      setPlacedCentroids((prev) => [...prev, { x, y, prevX: x, prevY: y }]);
    },
    [phase, placedCentroids.length, k],
  );

  // Assign step: color points by nearest centroid
  const doAssign = useCallback(() => {
    if (placedCentroids.length < k) return;
    if (phase === "place") setPhase("running");
    const newAssignments = assignAll(rawPoints, placedCentroids);
    setAssignments(newAssignments);
  }, [rawPoints, placedCentroids, k, phase]);

  // Update step: move centroids to cluster means
  const doUpdate = useCallback(() => {
    if (assignments.length === 0) return;
    setPlacedCentroids((prev) => {
      const newC = updateCentroids(rawPoints, assignments, k, prev);
      let maxMove = 0;
      for (let i = 0; i < k; i++) {
        maxMove = Math.max(maxMove, dist(prev[i], newC[i]));
      }
      if (maxMove < 0.5) {
        setConverged(true);
        setIsRunning(false);
        isRunningRef.current = false;
      }
      return newC.map((c, i) => ({ ...c, prevX: prev[i].x, prevY: prev[i].y }));
    });
    setIteration((i) => i + 1);
  }, [rawPoints, assignments, k]);

  // Single full step (assign + update)
  const doStep = useCallback(() => {
    if (placedCentroids.length < k) return;
    if (phase === "place") setPhase("running");
    // assign then update
    const newAssignments = assignAll(rawPoints, placedCentroids);
    setAssignments(newAssignments);
    setPlacedCentroids((prev) => {
      const newC = updateCentroids(rawPoints, newAssignments, k, prev);
      let maxMove = 0;
      for (let i = 0; i < k; i++) {
        maxMove = Math.max(maxMove, dist(prev[i], newC[i]));
      }
      if (maxMove < 0.5) {
        setConverged(true);
        setIsRunning(false);
        isRunningRef.current = false;
      }
      return newC.map((c, i) => ({ ...c, prevX: prev[i].x, prevY: prev[i].y }));
    });
    setIteration((i) => i + 1);
  }, [rawPoints, placedCentroids, k, phase]);

  // Auto-run loop
  useEffect(() => {
    isRunningRef.current = isRunning;
    if (!isRunning) {
      if (timerRef.current) clearTimeout(timerRef.current);
      return;
    }
    function tick() {
      if (!isRunningRef.current) return;
      // inline step
      doStep();
      timerRef.current = setTimeout(tick, 400);
    }
    timerRef.current = setTimeout(tick, 400);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [isRunning, doStep]);

  // Reset
  const handleReset = useCallback(() => {
    setIsRunning(false);
    isRunningRef.current = false;
    setPlacedCentroids([]);
    setAssignments([]);
    setIteration(0);
    setConverged(false);
    setPhase("place");
  }, []);

  const handleNewData = useCallback(() => {
    setSeed((s) => s + 7);
    handleReset();
  }, [handleReset]);

  // Random place centroids
  const handleRandomPlace = useCallback(() => {
    handleReset();
    const rand = mulberry32(Date.now());
    const cents: Centroid[] = [];
    for (let i = 0; i < k; i++) {
      const x = 40 + rand() * (T1_W - 80);
      const y = 40 + rand() * (T1_H - 80);
      cents.push({ x, y, prevX: x, prevY: y });
    }
    setPlacedCentroids(cents);
  }, [k, handleReset]);

  // Voronoi background
  const voronoiCells = useMemo(() => {
    if (assignments.length === 0 || placedCentroids.length < k) return null;
    return <VoronoiBg w={T1_W} h={T1_H} centroids={placedCentroids} cellSize={10} />;
  }, [placedCentroids, assignments, k]);

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-sky-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-sky-900">Step Through K-Means</h3>
          <p className="text-xs text-sky-700 mt-1">
            Click on the canvas to place <strong>{k} centroids</strong>, then use the buttons to
            step through the algorithm. Watch each iteration: <strong>Assign</strong> colors every
            point by nearest centroid, <strong>Update</strong> moves centroids to cluster means.
            Repeat until convergence!
          </p>
        </div>
      </div>

      <div className="flex gap-5 flex-col lg:flex-row">
        {/* SVG Canvas */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${T1_W} ${T1_H}`}
              className={`w-full select-none ${phase === "place" && placedCentroids.length < k ? "cursor-crosshair" : ""}`}
              style={{ aspectRatio: `${T1_W}/${T1_H}` }}
              onClick={handleSvgClick}
            >
              <GridLines w={T1_W} h={T1_H} />
              {voronoiCells}

              {/* Lines from points to centroids */}
              {showLines && assignments.length > 0 && rawPoints.map((p, i) => {
                const c = placedCentroids[assignments[i]];
                if (!c) return null;
                return (
                  <line key={`ln-${i}`} x1={p.x} y1={p.y} x2={c.x} y2={c.y}
                    stroke={COLORS_LIGHT[assignments[i] % COLORS_LIGHT.length]}
                    strokeWidth={0.7} />
                );
              })}

              {/* Data points */}
              {rawPoints.map((p, i) => {
                const color = assignments.length > 0
                  ? COLORS[assignments[i] % COLORS.length]
                  : "#94a3b8";
                return <DataDot key={`dp-${i}`} p={p} color={color} />;
              })}

              {/* Centroids */}
              {placedCentroids.map((c, i) => (
                <CentroidMarker key={`c-${i}`} c={c} idx={i} />
              ))}

              {/* Placement hint */}
              {phase === "place" && placedCentroids.length < k && (
                <text x={T1_W / 2} y={T1_H - 15} textAnchor="middle" fontSize={12}
                  fill="#64748b" fontFamily="sans-serif">
                  Click to place centroid {placedCentroids.length + 1} of {k}
                </text>
              )}

              {/* Status */}
              {converged && (
                <text x={T1_W / 2} y={20} textAnchor="middle" fontSize={13}
                  fill="#16a34a" fontWeight={700} fontFamily="sans-serif">
                  Converged at iteration {iteration}!
                </text>
              )}
            </svg>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-72 space-y-3">
          {/* Metrics */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium">Iteration</p>
              <p className="text-xl font-bold text-slate-800 mt-0.5">{iteration}</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium">WCSS</p>
              <p className="text-lg font-bold text-slate-800 mt-0.5 font-mono">
                {wcss > 0 ? (wcss / 1000).toFixed(1) + "k" : "—"}
              </p>
            </div>
          </div>

          {/* Status indicator */}
          <div className="bg-white border border-slate-200 rounded-lg p-2.5 flex items-center justify-center gap-2">
            {converged ? (
              <><CheckCircle2 className="w-4 h-4 text-emerald-500" /><span className="text-sm font-semibold text-emerald-600">Converged</span></>
            ) : phase === "place" ? (
              <><Target className="w-4 h-4 text-sky-500" /><span className="text-sm font-semibold text-sky-600">Place Centroids ({placedCentroids.length}/{k})</span></>
            ) : (
              <><Play className="w-4 h-4 text-amber-500" /><span className="text-sm font-semibold text-amber-600">Running — Iteration {iteration}</span></>
            )}
          </div>

          {/* K selector */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <label className="text-xs font-semibold text-slate-700 mb-2 block">Clusters (K)</label>
            <div className="flex gap-2">
              {[2, 3, 4, 5].map((val) => (
                <button key={val} onClick={() => { setK(val); handleReset(); }}
                  className={`flex-1 py-1.5 rounded-lg text-sm font-bold transition-all ${k === val
                    ? "bg-sky-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>
                  {val}
                </button>
              ))}
            </div>
          </div>

          {/* Show lines toggle */}
          <label className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg p-2.5 cursor-pointer">
            <input type="checkbox" checked={showLines} onChange={(e) => setShowLines(e.target.checked)}
              className="accent-sky-500" />
            <span className="text-xs font-medium text-slate-700">Show lines to centroids</span>
          </label>

          {/* Action buttons */}
          <div className="space-y-2">
            <div className="flex gap-2">
              <button onClick={doAssign}
                disabled={placedCentroids.length < k || converged}
                className="flex-1 px-3 py-2 rounded-lg text-xs font-semibold bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-40 transition-all">
                Assign
              </button>
              <button onClick={doUpdate}
                disabled={assignments.length === 0 || converged}
                className="flex-1 px-3 py-2 rounded-lg text-xs font-semibold bg-emerald-500 text-white hover:bg-emerald-600 disabled:opacity-40 transition-all">
                Update
              </button>
            </div>
            <div className="flex gap-2">
              <button onClick={doStep}
                disabled={placedCentroids.length < k || converged || isRunning}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-indigo-500 text-white hover:bg-indigo-600 disabled:opacity-40 transition-all">
                <SkipForward className="w-3.5 h-3.5" /> Step
              </button>
              <button onClick={() => {
                if (placedCentroids.length < k) return;
                if (phase === "place") setPhase("running");
                setIsRunning((r) => !r);
              }}
                disabled={placedCentroids.length < k || converged}
                className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${isRunning
                  ? "bg-slate-800 text-white hover:bg-slate-700"
                  : "bg-amber-500 text-white hover:bg-amber-600"} disabled:opacity-40`}>
                {isRunning ? <><Pause className="w-3.5 h-3.5" /> Pause</> : <><Play className="w-3.5 h-3.5" /> Auto Run</>}
              </button>
            </div>
            <div className="flex gap-2">
              <button onClick={handleRandomPlace}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
                <Shuffle className="w-3.5 h-3.5" /> Random Place
              </button>
              <button onClick={handleReset}
                className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
                <RotateCcw className="w-3.5 h-3.5" /> Reset
              </button>
            </div>
            <button onClick={handleNewData}
              className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
              <Zap className="w-3.5 h-3.5" /> New Dataset
            </button>
          </div>

          {/* Cluster sizes */}
          {assignments.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-lg p-3">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide font-medium mb-2">Cluster Sizes</p>
              {Array.from({ length: k }, (_, i) => {
                const count = assignments.filter((a) => a === i).length;
                return (
                  <div key={i} className="flex items-center gap-2 text-xs mb-1">
                    <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                    <span className="text-slate-600 font-medium w-6">C{i + 1}</span>
                    <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-300"
                        style={{ width: `${(count / rawPoints.length) * 100}%`, backgroundColor: COLORS[i % COLORS.length] }} />
                    </div>
                    <span className="text-slate-700 font-mono font-semibold w-6 text-right">{count}</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Tips */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Tips:</p>
            <ul className="text-xs text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Place centroids far from clusters to see more iterations</li>
              <li>Place one centroid between two clusters — watch it split</li>
              <li>Use "Assign" then "Update" to see each half-step</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 2 — CHOOSING K (ELBOW METHOD)
// ════════════════════════════════════════════════════════════════════════════

const T2_SCATTER_W = 340;
const T2_SCATTER_H = 300;
const T2_CHART_W = 340;
const T2_CHART_H = 250;
const T2_PAD = { top: 25, right: 20, bottom: 35, left: 55 };

function ElbowTab() {
  const [seed] = useState(99);
  const [selectedK, setSelectedK] = useState(4);
  const [showSilhouette, setShowSilhouette] = useState(false);

  // Generate 4-blob data
  const data = useMemo(() => {
    const rand = mulberry32(seed);
    const centers = [
      { x: 80, y: 70 }, { x: 280, y: 60 },
      { x: 80, y: 240 }, { x: 270, y: 230 },
    ];
    return generateBlobs(rand, centers, 20, 30, T2_SCATTER_W, T2_SCATTER_H);
  }, [seed]);

  // Run K-Means for K=1..10, compute WCSS and silhouette
  const results = useMemo(() => {
    const res: { k: number; wcss: number; sil: number; centroids: Pt[]; assignments: number[] }[] = [];
    for (let kv = 1; kv <= 10; kv++) {
      // Run 3 trials, pick best
      let bestWCSS = Infinity;
      let bestResult: ReturnType<typeof runKMeans> | null = null;
      for (let trial = 0; trial < 3; trial++) {
        const rand = mulberry32(seed * 100 + kv * 17 + trial * 31);
        const initC = randomCentroids(kv, rand, T2_SCATTER_W, T2_SCATTER_H);
        const r = runKMeans(data, initC);
        if (r.wcss < bestWCSS) { bestWCSS = r.wcss; bestResult = r; }
      }
      const sil = kv >= 2 ? silhouetteScore(data, bestResult!.assignments, kv) : 0;
      res.push({ k: kv, wcss: bestResult!.wcss, sil, centroids: bestResult!.centroids, assignments: bestResult!.assignments });
    }
    return res;
  }, [data, seed]);

  const maxWCSS = Math.max(...results.map((r) => r.wcss));
  const maxSil = Math.max(...results.map((r) => r.sil), 0.01);

  const chartPlotW = T2_CHART_W - T2_PAD.left - T2_PAD.right;
  const chartPlotH = T2_CHART_H - T2_PAD.top - T2_PAD.bottom;

  // Detect elbow (biggest drop in WCSS rate)
  const elbowK = useMemo(() => {
    let maxDrop = 0;
    let elbow = 2;
    for (let i = 1; i < results.length - 1; i++) {
      const drop1 = results[i - 1].wcss - results[i].wcss;
      const drop2 = results[i].wcss - results[i + 1].wcss;
      const diff = drop1 - drop2;
      if (diff > maxDrop) { maxDrop = diff; elbow = results[i].k; }
    }
    return elbow;
  }, [results]);

  const selResult = results[selectedK - 1];

  // Build WCSS line path
  const wcssPath = results.map((r, i) => {
    const x = T2_PAD.left + (i / 9) * chartPlotW;
    const y = T2_PAD.top + (1 - r.wcss / maxWCSS) * chartPlotH;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  // Build silhouette line path
  const silPath = results.slice(1).map((r, i) => {
    const x = T2_PAD.left + ((i + 1) / 9) * chartPlotW;
    const y = T2_PAD.top + (1 - r.sil / maxSil) * chartPlotH;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  return (
    <div className="space-y-4">
      <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-sky-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-sky-900">Choosing K: The Elbow Method</h3>
          <p className="text-xs text-sky-700 mt-1">
            How do you pick the right number of clusters? Plot <strong>WCSS</strong> (within-cluster sum of squares)
            against K. The <strong>"elbow"</strong> — where the curve bends sharply — suggests the
            natural cluster count. This dataset has <strong>4 natural clusters</strong>. Click any K below
            to see the resulting clustering.
          </p>
        </div>
      </div>

      <div className="flex gap-4 flex-col xl:flex-row">
        {/* Scatter plot */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <div className="text-xs font-semibold text-slate-600 mb-2 text-center">
              Clustering with K = {selectedK}
              {selectedK === elbowK && <span className="ml-2 text-emerald-600">(Elbow point)</span>}
            </div>
            <svg viewBox={`0 0 ${T2_SCATTER_W} ${T2_SCATTER_H}`} className="w-full"
              style={{ aspectRatio: `${T2_SCATTER_W}/${T2_SCATTER_H}` }}>
              <GridLines w={T2_SCATTER_W} h={T2_SCATTER_H} />
              <VoronoiBg w={T2_SCATTER_W} h={T2_SCATTER_H} centroids={selResult.centroids} cellSize={8} />
              {data.map((p, i) => (
                <DataDot key={i} p={p} color={COLORS[selResult.assignments[i] % COLORS.length]} />
              ))}
              {selResult.centroids.map((c, i) => (
                <CentroidMarker key={i} c={c} idx={i} animate={false} />
              ))}
            </svg>
            <div className="text-xs text-slate-500 text-center mt-1">
              WCSS: <span className="font-mono font-semibold">{selResult.wcss.toFixed(0)}</span>
              {selectedK >= 2 && (
                <> | Silhouette: <span className="font-mono font-semibold">{selResult.sil.toFixed(3)}</span></>
              )}
            </div>
          </div>
        </div>

        {/* Elbow chart */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-600">
                {showSilhouette ? "Silhouette Score vs K" : "WCSS vs K (Elbow Chart)"}
              </span>
              <button onClick={() => setShowSilhouette((s) => !s)}
                className="text-[10px] px-2 py-1 rounded bg-slate-200 text-slate-600 hover:bg-slate-300 font-medium transition-all">
                {showSilhouette ? "Show WCSS" : "Show Silhouette"}
              </button>
            </div>
            <svg viewBox={`0 0 ${T2_CHART_W} ${T2_CHART_H}`} className="w-full"
              style={{ aspectRatio: `${T2_CHART_W}/${T2_CHART_H}` }}>

              {/* Axes */}
              <line x1={T2_PAD.left} y1={T2_PAD.top} x2={T2_PAD.left} y2={T2_PAD.top + chartPlotH}
                stroke="#cbd5e1" strokeWidth={1} />
              <line x1={T2_PAD.left} y1={T2_PAD.top + chartPlotH} x2={T2_PAD.left + chartPlotW} y2={T2_PAD.top + chartPlotH}
                stroke="#cbd5e1" strokeWidth={1} />

              {/* Y axis labels */}
              {[0, 0.25, 0.5, 0.75, 1].map((f) => {
                const y = T2_PAD.top + (1 - f) * chartPlotH;
                const val = showSilhouette ? (f * maxSil).toFixed(2) : (f * maxWCSS / 1000).toFixed(0) + "k";
                return (
                  <g key={f}>
                    <line x1={T2_PAD.left - 4} y1={y} x2={T2_PAD.left} y2={y} stroke="#94a3b8" strokeWidth={0.5} />
                    <text x={T2_PAD.left - 8} y={y + 3} textAnchor="end" fontSize={9} fill="#94a3b8" fontFamily="sans-serif">{val}</text>
                    <line x1={T2_PAD.left} y1={y} x2={T2_PAD.left + chartPlotW} y2={y} stroke="#f1f5f9" strokeWidth={0.5} />
                  </g>
                );
              })}

              {/* X axis labels */}
              {results.map((r, i) => {
                const x = T2_PAD.left + (i / 9) * chartPlotW;
                return (
                  <text key={i} x={x} y={T2_PAD.top + chartPlotH + 15} textAnchor="middle"
                    fontSize={10} fill="#64748b" fontFamily="sans-serif" fontWeight={r.k === selectedK ? 700 : 400}>
                    {r.k}
                  </text>
                );
              })}

              {/* Axis labels */}
              <text x={T2_PAD.left + chartPlotW / 2} y={T2_CHART_H - 5} textAnchor="middle"
                fontSize={10} fill="#64748b" fontFamily="sans-serif">K (Number of Clusters)</text>
              <text x={12} y={T2_PAD.top + chartPlotH / 2} textAnchor="middle"
                fontSize={10} fill="#64748b" fontFamily="sans-serif"
                transform={`rotate(-90, 12, ${T2_PAD.top + chartPlotH / 2})`}>
                {showSilhouette ? "Silhouette Score" : "WCSS"}
              </text>

              {/* Line */}
              {!showSilhouette && (
                <path d={wcssPath} fill="none" stroke="#3b82f6" strokeWidth={2.5} strokeLinejoin="round" />
              )}
              {showSilhouette && (
                <path d={silPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} strokeLinejoin="round" />
              )}

              {/* Dots (clickable) */}
              {results.map((r, i) => {
                if (showSilhouette && i === 0) return null; // no silhouette for K=1
                const x = T2_PAD.left + (i / 9) * chartPlotW;
                const y = showSilhouette
                  ? T2_PAD.top + (1 - r.sil / maxSil) * chartPlotH
                  : T2_PAD.top + (1 - r.wcss / maxWCSS) * chartPlotH;
                const isSelected = r.k === selectedK;
                const isElbow = r.k === elbowK;
                return (
                  <g key={`dot-${i}`} onClick={() => setSelectedK(r.k)} className="cursor-pointer">
                    {isElbow && !showSilhouette && (
                      <circle cx={x} cy={y} r={14} fill="none" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 2" />
                    )}
                    <circle cx={x} cy={y} r={isSelected ? 7 : 5}
                      fill={isSelected ? (showSilhouette ? "#8b5cf6" : "#3b82f6") : "#fff"}
                      stroke={showSilhouette ? "#8b5cf6" : "#3b82f6"} strokeWidth={2} />
                    {isElbow && !showSilhouette && (
                      <text x={x + 16} y={y - 8} fontSize={9} fill="#f59e0b" fontWeight={700} fontFamily="sans-serif">
                        Elbow
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Highlight selected K */}
              {(() => {
                const i = selectedK - 1;
                const x = T2_PAD.left + (i / 9) * chartPlotW;
                return (
                  <line x1={x} y1={T2_PAD.top} x2={x} y2={T2_PAD.top + chartPlotH}
                    stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" opacity={0.4} />
                );
              })()}
            </svg>
          </div>

          {/* K selector buttons */}
          <div className="mt-3">
            <p className="text-xs font-semibold text-slate-600 mb-1.5">Select K to visualize:</p>
            <div className="flex gap-1.5 flex-wrap">
              {results.map((r) => (
                <button key={r.k} onClick={() => setSelectedK(r.k)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all ${selectedK === r.k
                    ? "bg-sky-600 text-white" : r.k === elbowK
                      ? "bg-amber-100 text-amber-700 border border-amber-300 hover:bg-amber-200"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>
                  K={r.k}{r.k === elbowK ? " *" : ""}
                </button>
              ))}
            </div>
          </div>

          {/* Explanation */}
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 mt-3">
            <p className="text-xs text-emerald-800">
              <strong>Why the elbow at K={elbowK}?</strong> Beyond this point, adding more clusters
              reduces WCSS only marginally — the extra complexity is not worth the small improvement.
              The data has {4} natural groups, and the elbow method correctly identifies around K={elbowK} as optimal.
              {showSilhouette && (
                <> The <strong>silhouette score</strong> measures how well each point fits its own cluster
                  vs neighboring clusters. Higher is better (max 1.0).</>
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 3 — CONVERGENCE VISUALIZATION
// ════════════════════════════════════════════════════════════════════════════

const T4_SVG_W = 480;
const T4_SVG_H = 380;
const T4_CHART_W = 300;
const T4_CHART_H = 180;

interface HistoryFrame {
  centroids: Pt[];
  assignments: number[];
  wcss: number;
  changedPoints: number; // how many points changed assignment
}

function ConvergenceTab() {
  const [seed, setSeed] = useState(42);
  const [k, setK] = useState(3);
  const [speed, setSpeed] = useState(500);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const isPlayingRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Generate data
  const data = useMemo(() => {
    const rand = mulberry32(seed);
    const centers = [
      { x: 120, y: 90 }, { x: 380, y: 100 }, { x: 240, y: 310 },
    ];
    return generateBlobs(rand, centers, 25, 40, T4_SVG_W, T4_SVG_H);
  }, [seed]);

  // Precompute full history
  const history = useMemo<HistoryFrame[]>(() => {
    const rand = mulberry32(seed * 3 + k * 7);
    let centroids = randomCentroids(k, rand, T4_SVG_W, T4_SVG_H);
    const frames: HistoryFrame[] = [];
    let prevAssignments: number[] = [];

    // Frame 0: initial state before any assignment
    const initAssignments = assignAll(data, centroids);
    frames.push({ centroids: centroids.map((c) => ({ ...c })), assignments: initAssignments, wcss: computeWCSS(data, initAssignments, centroids), changedPoints: data.length });
    prevAssignments = initAssignments;

    for (let iter = 0; iter < 50; iter++) {
      const newCentroids = updateCentroids(data, prevAssignments, k, centroids);
      const newAssignments = assignAll(data, newCentroids);

      let changed = 0;
      for (let i = 0; i < data.length; i++) {
        if (prevAssignments[i] !== newAssignments[i]) changed++;
      }

      const wcss = computeWCSS(data, newAssignments, newCentroids);
      frames.push({ centroids: newCentroids.map((c) => ({ ...c })), assignments: newAssignments, wcss, changedPoints: changed });

      let maxMove = 0;
      for (let i = 0; i < k; i++) {
        maxMove = Math.max(maxMove, dist(centroids[i], newCentroids[i]));
      }

      centroids = newCentroids;
      prevAssignments = newAssignments;

      if (maxMove < 0.5) break;
    }
    return frames;
  }, [data, seed, k]);

  const maxFrames = history.length - 1;
  const frame = history[Math.min(currentFrame, maxFrames)];

  // Centroid trails up to current frame
  const trails = useMemo(() => {
    const paths: { points: Pt[]; color: string }[] = [];
    for (let c = 0; c < k; c++) {
      const pts: Pt[] = [];
      for (let f = 0; f <= Math.min(currentFrame, maxFrames); f++) {
        if (history[f].centroids[c]) {
          pts.push(history[f].centroids[c]);
        }
      }
      paths.push({ points: pts, color: COLORS[c % COLORS.length] });
    }
    return paths;
  }, [history, currentFrame, maxFrames, k]);

  // Points that changed this step
  const changedIndices = useMemo(() => {
    if (currentFrame === 0 || currentFrame > maxFrames) return new Set<number>();
    const prev = history[Math.max(0, currentFrame - 1)].assignments;
    const curr = frame.assignments;
    const changed = new Set<number>();
    for (let i = 0; i < data.length; i++) {
      if (prev[i] !== curr[i]) changed.add(i);
    }
    return changed;
  }, [history, currentFrame, maxFrames, frame, data.length]);

  // Playback
  useEffect(() => {
    isPlayingRef.current = isPlaying;
    if (!isPlaying) {
      if (timerRef.current) clearTimeout(timerRef.current);
      return;
    }
    function tick() {
      if (!isPlayingRef.current) return;
      setCurrentFrame((f) => {
        if (f >= maxFrames) {
          setIsPlaying(false);
          isPlayingRef.current = false;
          return f;
        }
        return f + 1;
      });
      timerRef.current = setTimeout(tick, speed);
    }
    timerRef.current = setTimeout(tick, speed);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [isPlaying, speed, maxFrames]);

  // WCSS chart
  const maxWCSS = Math.max(...history.map((h) => h.wcss), 1);
  const chartPad = { top: 15, right: 15, bottom: 25, left: 50 };
  const chartPW = T4_CHART_W - chartPad.left - chartPad.right;
  const chartPH = T4_CHART_H - chartPad.top - chartPad.bottom;

  const lossPath = history.map((h, i) => {
    const x = chartPad.left + (i / Math.max(maxFrames, 1)) * chartPW;
    const y = chartPad.top + (1 - h.wcss / maxWCSS) * chartPH;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    isPlayingRef.current = false;
    setSeed((s) => s + 13);
    setCurrentFrame(0);
  }, []);

  return (
    <div className="space-y-4">
      <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-sky-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-sky-900">Convergence Visualization</h3>
          <p className="text-xs text-sky-700 mt-1">
            Watch the entire K-Means convergence process. <strong>Centroid trails</strong> (dotted lines)
            show the path each centroid takes. <strong>Highlighted points</strong> (with rings) are those
            that changed cluster assignment. The <strong>loss curve</strong> shows WCSS decreasing each iteration.
          </p>
        </div>
      </div>

      <div className="flex gap-4 flex-col xl:flex-row">
        {/* Main visualization */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg viewBox={`0 0 ${T4_SVG_W} ${T4_SVG_H}`} className="w-full select-none"
              style={{ aspectRatio: `${T4_SVG_W}/${T4_SVG_H}` }}>
              <GridLines w={T4_SVG_W} h={T4_SVG_H} />
              <VoronoiBg w={T4_SVG_W} h={T4_SVG_H} centroids={frame.centroids} cellSize={10} />

              {/* Centroid trails */}
              {trails.map((trail, ci) => {
                if (trail.points.length < 2) return null;
                const pathD = trail.points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
                return (
                  <g key={`trail-${ci}`}>
                    <path d={pathD} fill="none" stroke={trail.color} strokeWidth={2}
                      strokeDasharray="4 3" opacity={0.6} />
                    {/* Breadcrumb dots */}
                    {trail.points.map((p, pi) => (
                      <circle key={pi} cx={p.x} cy={p.y} r={3} fill={trail.color}
                        opacity={pi === trail.points.length - 1 ? 1 : 0.35} />
                    ))}
                  </g>
                );
              })}

              {/* Data points */}
              {data.map((p, i) => {
                const color = COLORS[frame.assignments[i] % COLORS.length];
                const isChanged = changedIndices.has(i);
                return (
                  <g key={`dp-${i}`}>
                    {isChanged && (
                      <circle cx={p.x} cy={p.y} r={9} fill="none" stroke="#f59e0b"
                        strokeWidth={2} opacity={0.8} />
                    )}
                    <DataDot p={p} color={color} r={isChanged ? 5 : 4} />
                  </g>
                );
              })}

              {/* Final centroids */}
              {frame.centroids.map((c, i) => (
                <CentroidMarker key={i} c={c} idx={i} animate={false} />
              ))}

              {/* Info overlay */}
              <text x={10} y={T4_SVG_H - 10} fontSize={11} fill="#64748b" fontFamily="sans-serif">
                Iteration {currentFrame} / {maxFrames} — {frame.changedPoints} points changed
              </text>
            </svg>
          </div>
        </div>

        {/* Controls + loss curve */}
        <div className="w-full xl:w-80 space-y-3">
          {/* Playback controls */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-3">
            <div className="flex items-center gap-2">
              <button onClick={() => setCurrentFrame((f) => Math.max(0, f - 1))} disabled={currentFrame === 0}
                className="p-2 rounded-lg border border-slate-200 hover:bg-slate-50 disabled:opacity-40 transition-all">
                <SkipBack className="w-4 h-4" />
              </button>
              <button onClick={() => setIsPlaying((p) => !p)}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-sm font-semibold transition-all ${isPlaying
                  ? "bg-slate-800 text-white" : "bg-sky-500 text-white hover:bg-sky-600"}`}>
                {isPlaying ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Play</>}
              </button>
              <button onClick={() => setCurrentFrame((f) => Math.min(maxFrames, f + 1))} disabled={currentFrame >= maxFrames}
                className="p-2 rounded-lg border border-slate-200 hover:bg-slate-50 disabled:opacity-40 transition-all">
                <SkipForward className="w-4 h-4" />
              </button>
            </div>

            {/* Frame slider */}
            <div>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                <span>Iteration</span>
                <span className="font-mono font-semibold">{currentFrame} / {maxFrames}</span>
              </div>
              <input type="range" min={0} max={maxFrames} value={currentFrame}
                onChange={(e) => { setIsPlaying(false); setCurrentFrame(Number(e.target.value)); }}
                className="w-full accent-sky-500" />
            </div>

            {/* Speed slider */}
            <div>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                <span>Speed</span>
                <span className="font-mono font-semibold">{speed}ms</span>
              </div>
              <input type="range" min={100} max={1500} step={100} value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))} className="w-full accent-sky-500" />
            </div>
          </div>

          {/* K selector */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <label className="text-xs font-semibold text-slate-700 mb-2 block">Clusters (K)</label>
            <div className="flex gap-2">
              {[2, 3, 4, 5].map((val) => (
                <button key={val} onClick={() => { setK(val); setCurrentFrame(0); setIsPlaying(false); }}
                  className={`flex-1 py-1.5 rounded-lg text-sm font-bold transition-all ${k === val
                    ? "bg-sky-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>
                  {val}
                </button>
              ))}
            </div>
          </div>

          {/* Loss curve */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-xs font-semibold text-slate-700 mb-2">WCSS over Iterations</p>
            <svg viewBox={`0 0 ${T4_CHART_W} ${T4_CHART_H}`} className="w-full"
              style={{ aspectRatio: `${T4_CHART_W}/${T4_CHART_H}` }}>
              <line x1={chartPad.left} y1={chartPad.top} x2={chartPad.left} y2={chartPad.top + chartPH}
                stroke="#cbd5e1" strokeWidth={1} />
              <line x1={chartPad.left} y1={chartPad.top + chartPH} x2={chartPad.left + chartPW} y2={chartPad.top + chartPH}
                stroke="#cbd5e1" strokeWidth={1} />

              {/* Y labels */}
              {[0, 0.5, 1].map((f) => {
                const y = chartPad.top + (1 - f) * chartPH;
                return (
                  <g key={f}>
                    <line x1={chartPad.left} y1={y} x2={chartPad.left + chartPW} y2={y} stroke="#f1f5f9" strokeWidth={0.5} />
                    <text x={chartPad.left - 5} y={y + 3} textAnchor="end" fontSize={8} fill="#94a3b8" fontFamily="sans-serif">
                      {((f * maxWCSS) / 1000).toFixed(0)}k
                    </text>
                  </g>
                );
              })}

              {/* Loss line */}
              <path d={lossPath} fill="none" stroke="#ef4444" strokeWidth={2} strokeLinejoin="round" />

              {/* Current position marker */}
              {(() => {
                const x = chartPad.left + (currentFrame / Math.max(maxFrames, 1)) * chartPW;
                const y = chartPad.top + (1 - frame.wcss / maxWCSS) * chartPH;
                return (
                  <>
                    <line x1={x} y1={chartPad.top} x2={x} y2={chartPad.top + chartPH}
                      stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
                    <circle cx={x} cy={y} r={5} fill="#ef4444" stroke="#fff" strokeWidth={2} />
                  </>
                );
              })()}

              {/* X label */}
              <text x={chartPad.left + chartPW / 2} y={T4_CHART_H - 5} textAnchor="middle"
                fontSize={9} fill="#64748b" fontFamily="sans-serif">Iteration</text>
            </svg>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">WCSS</p>
              <p className="text-lg font-bold text-slate-800 font-mono">{(frame.wcss / 1000).toFixed(1)}k</p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-center">
              <p className="text-[10px] text-slate-500 uppercase font-medium">Changed</p>
              <p className="text-lg font-bold text-amber-600 font-mono">{frame.changedPoints}</p>
            </div>
          </div>

          <button onClick={handleReset}
            className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-all">
            <RotateCcw className="w-3.5 h-3.5" /> New Random Start
          </button>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════════════════

export default function KMeansClusteringActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("stepthrough");

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
        {activeTab === "stepthrough" && <StepThroughTab />}
        {activeTab === "elbow" && <ElbowTab />}
        {activeTab === "convergence" && <ConvergenceTab />}
      </div>
    </div>
  );
}
