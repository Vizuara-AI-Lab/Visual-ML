/**
 * KNN Playground — comprehensive 4-tab interactive K-Nearest Neighbors activity
 *
 * Tabs:
 *   1. Classify Points       – click to place query points, see K nearest neighbors vote
 *   2. K Value Effect         – decision boundary heatmap, accuracy vs K chart
 *   3. Distance Metrics       – compare Euclidean, Manhattan, Chebyshev iso-contours
 *   4. Weighted KNN           – uniform vs distance-weighted voting with bar charts
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Info,
  Crosshair,
  Trash2,
  MousePointer,
  Target,
  Layers,
  Ruler,
  Weight,
  RotateCcw,
  Plus,
} from "lucide-react";

// ════════════════════════════════════════════════════════════════════════════
// SEEDED PRNG — mulberry32
// ════════════════════════════════════════════════════════════════════════════

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianSample(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// ════════════════════════════════════════════════════════════════════════════
// SHARED TYPES & CONSTANTS
// ════════════════════════════════════════════════════════════════════════════

interface DataPoint {
  x: number;
  y: number;
  cls: number; // 0 = blue, 1 = orange
}

const CLASS_COLORS = ["#3b82f6", "#f97316"] as const;
const CLASS_NAMES = ["Blue", "Orange"] as const;
const CLASS_BG = ["bg-blue-500", "bg-orange-500"] as const;

const SVG_W = 500;
const SVG_H = 400;

// ════════════════════════════════════════════════════════════════════════════
// DISTANCE FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

function euclidean(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return Math.sqrt(dx * dx + dy * dy);
}

function manhattan(ax: number, ay: number, bx: number, by: number): number {
  return Math.abs(ax - bx) + Math.abs(ay - by);
}

function chebyshev(ax: number, ay: number, bx: number, by: number): number {
  return Math.max(Math.abs(ax - bx), Math.abs(ay - by));
}

type DistanceFn = (ax: number, ay: number, bx: number, by: number) => number;

const DISTANCE_FNS: Record<string, DistanceFn> = {
  euclidean,
  manhattan,
  chebyshev,
};

// ════════════════════════════════════════════════════════════════════════════
// DATA GENERATION
// ════════════════════════════════════════════════════════════════════════════

function generateTwoClassData(seed: number, count = 40): DataPoint[] {
  const rng = mulberry32(seed);
  const points: DataPoint[] = [];

  // Blue cluster — center around (150, 180)
  for (let i = 0; i < count / 2; i++) {
    points.push({
      x: 120 + gaussianSample(rng) * 55,
      y: 170 + gaussianSample(rng) * 55,
      cls: 0,
    });
  }
  // Orange cluster — center around (350, 230)
  for (let i = 0; i < count / 2; i++) {
    points.push({
      x: 340 + gaussianSample(rng) * 55,
      y: 220 + gaussianSample(rng) * 55,
      cls: 1,
    });
  }
  // A few "noisy" points for interest
  for (let i = 0; i < 6; i++) {
    points.push({
      x: 50 + rng() * 400,
      y: 50 + rng() * 300,
      cls: rng() > 0.5 ? 1 : 0,
    });
  }

  return points;
}

function generateOverlappingData(seed: number): DataPoint[] {
  const rng = mulberry32(seed);
  const points: DataPoint[] = [];

  // Two overlapping Gaussian blobs
  for (let i = 0; i < 30; i++) {
    points.push({
      x: 200 + gaussianSample(rng) * 80,
      y: 180 + gaussianSample(rng) * 70,
      cls: 0,
    });
  }
  for (let i = 0; i < 30; i++) {
    points.push({
      x: 290 + gaussianSample(rng) * 80,
      y: 220 + gaussianSample(rng) * 70,
      cls: 1,
    });
  }
  return points;
}

// ════════════════════════════════════════════════════════════════════════════
// KNN CLASSIFICATION HELPERS
// ════════════════════════════════════════════════════════════════════════════

function knnClassify(
  qx: number,
  qy: number,
  data: DataPoint[],
  k: number,
  distFn: DistanceFn = euclidean,
): { predicted: number; neighbors: { point: DataPoint; dist: number }[]; votes: number[] } {
  if (data.length === 0) return { predicted: 0, neighbors: [], votes: [0, 0] };
  const effectiveK = Math.min(k, data.length);

  const withDist = data.map((p) => ({
    point: p,
    dist: distFn(qx, qy, p.x, p.y),
  }));
  withDist.sort((a, b) => a.dist - b.dist);
  const neighbors = withDist.slice(0, effectiveK);

  const votes = [0, 0];
  for (const n of neighbors) votes[n.point.cls]++;
  const predicted = votes[0] >= votes[1] ? 0 : 1;

  return { predicted, neighbors, votes };
}

function knnClassifyWeighted(
  qx: number,
  qy: number,
  data: DataPoint[],
  k: number,
  weightFn: "uniform" | "1/d" | "1/d2" | "gaussian",
  distFn: DistanceFn = euclidean,
): { predicted: number; neighbors: { point: DataPoint; dist: number; weight: number }[]; weightedVotes: number[] } {
  if (data.length === 0) return { predicted: 0, neighbors: [], weightedVotes: [0, 0] };
  const effectiveK = Math.min(k, data.length);

  const withDist = data.map((p) => ({
    point: p,
    dist: distFn(qx, qy, p.x, p.y),
  }));
  withDist.sort((a, b) => a.dist - b.dist);
  const topK = withDist.slice(0, effectiveK);

  const neighbors = topK.map((n) => {
    let w = 1;
    const d = Math.max(n.dist, 0.001);
    if (weightFn === "1/d") w = 1 / d;
    else if (weightFn === "1/d2") w = 1 / (d * d);
    else if (weightFn === "gaussian") w = Math.exp(-(d * d) / (2 * 50 * 50));
    return { ...n, weight: w };
  });

  const weightedVotes = [0, 0];
  for (const n of neighbors) weightedVotes[n.point.cls] += n.weight;
  const predicted = weightedVotes[0] >= weightedVotes[1] ? 0 : 1;

  return { predicted, neighbors, weightedVotes };
}

// ════════════════════════════════════════════════════════════════════════════
// SVG HELPERS
// ════════════════════════════════════════════════════════════════════════════

function SvgGrid({ w, h, step = 50 }: { w: number; h: number; step?: number }) {
  const lines: React.ReactNode[] = [];
  for (let x = 0; x <= w; x += step) {
    lines.push(
      <line key={`v-${x}`} x1={x} y1={0} x2={x} y2={h} stroke="#e2e8f0" strokeWidth={0.5} />
    );
  }
  for (let y = 0; y <= h; y += step) {
    lines.push(
      <line key={`h-${y}`} x1={0} y1={y} x2={w} y2={y} stroke="#e2e8f0" strokeWidth={0.5} />
    );
  }
  return <>{lines}</>;
}

// ════════════════════════════════════════════════════════════════════════════
// TAB DEFINITIONS
// ════════════════════════════════════════════════════════════════════════════

type TabKey = "classify" | "kvalue" | "distance" | "weighted";

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabDef[] = [
  { key: "classify", label: "Classify Points", icon: <Crosshair className="w-3.5 h-3.5" /> },
  { key: "kvalue", label: "K Value Effect", icon: <Layers className="w-3.5 h-3.5" /> },
  { key: "distance", label: "Distance Metrics", icon: <Ruler className="w-3.5 h-3.5" /> },
  { key: "weighted", label: "Weighted KNN", icon: <Weight className="w-3.5 h-3.5" /> },
];

// ════════════════════════════════════════════════════════════════════════════
// TAB 1 — CLASSIFY POINTS
// ════════════════════════════════════════════════════════════════════════════

const INITIAL_TRAIN = generateTwoClassData(42, 40);

function ClassifyPointsTab() {
  const [points, setPoints] = useState<DataPoint[]>(() => [...INITIAL_TRAIN]);
  const [k, setK] = useState(5);
  const [mode, setMode] = useState<"classify" | "addBlue" | "addOrange">("classify");
  const [queryResult, setQueryResult] = useState<{
    query: { x: number; y: number };
    neighbors: { point: DataPoint; dist: number }[];
    predicted: number;
    votes: number[];
    searchRadius: number;
  } | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);

  const toSVGCoords = useCallback(
    (e: React.MouseEvent<SVGSVGElement>): { x: number; y: number } | null => {
      const svg = svgRef.current;
      if (!svg) return null;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      const svgPt = pt.matrixTransform(ctm.inverse());
      return { x: Math.round(svgPt.x), y: Math.round(svgPt.y) };
    },
    [],
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const coords = toSVGCoords(e);
      if (!coords) return;
      const x = Math.max(10, Math.min(SVG_W - 10, coords.x));
      const y = Math.max(10, Math.min(SVG_H - 10, coords.y));

      if (mode === "addBlue") {
        setPoints((prev) => [...prev, { x, y, cls: 0 }]);
        return;
      }
      if (mode === "addOrange") {
        setPoints((prev) => [...prev, { x, y, cls: 1 }]);
        return;
      }

      // classify mode
      if (points.length === 0) return;
      const { predicted, neighbors, votes } = knnClassify(x, y, points, k);
      const searchRadius = neighbors.length > 0 ? neighbors[neighbors.length - 1].dist : 0;
      setQueryResult({ query: { x, y }, neighbors, predicted, votes, searchRadius });
    },
    [mode, points, k, toSVGCoords],
  );

  // Re-classify when K changes and there is a query
  useEffect(() => {
    if (!queryResult) return;
    if (points.length === 0) return;
    const { x, y } = queryResult.query;
    const { predicted, neighbors, votes } = knnClassify(x, y, points, k);
    const searchRadius = neighbors.length > 0 ? neighbors[neighbors.length - 1].dist : 0;
    setQueryResult({ query: { x, y }, neighbors, predicted, votes, searchRadius });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k]);

  const clearAll = () => {
    setPoints([...INITIAL_TRAIN]);
    setQueryResult(null);
  };

  const counts = [0, 0];
  for (const p of points) counts[p.cls]++;

  return (
    <div className="space-y-4">
      {/* Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-blue-900">Classify New Points</h3>
          <p className="text-xs text-blue-700 mt-1">
            Click on the canvas in <strong>Classify</strong> mode to place a query point (shown as a
            diamond). The algorithm finds the <strong>K nearest neighbors</strong> and takes a
            majority vote to predict the class. Try changing K or adding more training points.
          </p>
        </div>
      </div>

      <div className="flex gap-5 flex-col lg:flex-row">
        {/* SVG Canvas */}
        <div className="flex-1 min-w-0">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="w-full mx-auto cursor-crosshair select-none"
              style={{ aspectRatio: `${SVG_W}/${SVG_H}` }}
              onClick={handleClick}
            >
              <SvgGrid w={SVG_W} h={SVG_H} />

              {/* Dashed lines to neighbors */}
              {queryResult &&
                queryResult.neighbors.map((n, i) => (
                  <line
                    key={`nline-${i}`}
                    x1={queryResult.query.x}
                    y1={queryResult.query.y}
                    x2={n.point.x}
                    y2={n.point.y}
                    stroke={CLASS_COLORS[n.point.cls]}
                    strokeWidth={1.5}
                    strokeDasharray="5,3"
                    opacity={0.55}
                  />
                ))}

              {/* Search radius circle */}
              {queryResult && (
                <circle
                  cx={queryResult.query.x}
                  cy={queryResult.query.y}
                  r={queryResult.searchRadius}
                  fill="none"
                  stroke={CLASS_COLORS[queryResult.predicted]}
                  strokeWidth={1.5}
                  strokeDasharray="4,4"
                  opacity={0.3}
                />
              )}

              {/* Data points */}
              {points.map((p, i) => {
                const isNeighbor =
                  queryResult?.neighbors.some(
                    (n) => n.point.x === p.x && n.point.y === p.y && n.point.cls === p.cls,
                  ) ?? false;

                return (
                  <g key={`pt-${i}`}>
                    {isNeighbor && (
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={14}
                        fill="none"
                        stroke={CLASS_COLORS[p.cls]}
                        strokeWidth={2.5}
                        opacity={0.5}
                      >
                        <animate
                          attributeName="r"
                          values="12;16;12"
                          dur="1.5s"
                          repeatCount="indefinite"
                        />
                      </circle>
                    )}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={7}
                      fill={CLASS_COLORS[p.cls]}
                      stroke="#fff"
                      strokeWidth={1.8}
                      opacity={isNeighbor ? 1 : 0.8}
                    />
                  </g>
                );
              })}

              {/* Query point — diamond shape */}
              {queryResult && (
                <g>
                  <polygon
                    points={`${queryResult.query.x},${queryResult.query.y - 13} ${queryResult.query.x + 10},${queryResult.query.y} ${queryResult.query.x},${queryResult.query.y + 13} ${queryResult.query.x - 10},${queryResult.query.y}`}
                    fill={CLASS_COLORS[queryResult.predicted]}
                    stroke="#fff"
                    strokeWidth={2.5}
                  />
                  <polygon
                    points={`${queryResult.query.x},${queryResult.query.y - 5} ${queryResult.query.x + 4},${queryResult.query.y} ${queryResult.query.x},${queryResult.query.y + 5} ${queryResult.query.x - 4},${queryResult.query.y}`}
                    fill="#fff"
                    opacity={0.7}
                  />
                </g>
              )}

              {/* Label */}
              <text x={10} y={SVG_H - 8} fontSize={11} fill="#94a3b8" fontFamily="sans-serif">
                {mode === "classify"
                  ? "Click to classify a point"
                  : mode === "addBlue"
                    ? "Click to add Blue points"
                    : "Click to add Orange points"}
              </text>
            </svg>
          </div>
        </div>

        {/* Controls */}
        <div className="w-full lg:w-72 space-y-3">
          {/* Mode selector */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Mode
            </p>
            <div className="flex gap-1.5 flex-wrap">
              <button
                onClick={() => setMode("classify")}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                  mode === "classify"
                    ? "bg-violet-600 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                <Crosshair className="w-3.5 h-3.5" />
                Classify
              </button>
              <button
                onClick={() => setMode("addBlue")}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                  mode === "addBlue"
                    ? "bg-blue-500 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                <Plus className="w-3.5 h-3.5" />
                Blue
              </button>
              <button
                onClick={() => setMode("addOrange")}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                  mode === "addOrange"
                    ? "bg-orange-500 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                <Plus className="w-3.5 h-3.5" />
                Orange
              </button>
            </div>
          </div>

          {/* K slider */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5 text-violet-500" />
                K Value
              </label>
              <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                {k}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={15}
              step={1}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-500"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>1 (sensitive)</span>
              <span>15 (smooth)</span>
            </div>
            {k % 2 === 0 && (
              <p className="text-[10px] text-amber-600 mt-1">
                Tip: Odd K avoids tie-breaking ambiguity.
              </p>
            )}
          </div>

          {/* Point counts */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Training Data
            </p>
            <div className="grid grid-cols-2 gap-3">
              {CLASS_NAMES.map((name, cls) => (
                <div key={cls} className="text-center">
                  <div
                    className="w-3 h-3 rounded-full mx-auto mb-1"
                    style={{ backgroundColor: CLASS_COLORS[cls] }}
                  />
                  <p className="text-lg font-bold text-slate-800">{counts[cls]}</p>
                  <p className="text-[10px] text-slate-500">{name}</p>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-slate-400 text-center mt-1">
              Total: {points.length} points
            </p>
          </div>

          {/* Prediction result */}
          {queryResult && (
            <div className="bg-white border border-slate-200 rounded-lg p-3">
              <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
                Prediction
              </p>
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-4 h-4 rounded-full border-2 border-white shadow"
                  style={{ backgroundColor: CLASS_COLORS[queryResult.predicted] }}
                />
                <span className="text-sm font-bold" style={{ color: CLASS_COLORS[queryResult.predicted] }}>
                  {CLASS_NAMES[queryResult.predicted]}
                </span>
              </div>
              <div className="space-y-1">
                {CLASS_NAMES.map((name, cls) => (
                  <div key={cls} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: CLASS_COLORS[cls] }}
                      />
                      <span className="text-slate-600">{name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${
                              queryResult.votes[cls] > 0
                                ? (queryResult.votes[cls] / Math.max(...queryResult.votes)) * 100
                                : 0
                            }%`,
                            backgroundColor: CLASS_COLORS[cls],
                          }}
                        />
                      </div>
                      <span className="text-slate-700 font-mono font-semibold w-4 text-right">
                        {queryResult.votes[cls]}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate-400 mt-2">
                ({queryResult.query.x}, {queryResult.query.y}) | K={queryResult.neighbors.length}
              </p>
            </div>
          )}

          {/* Reset */}
          <button
            onClick={clearAll}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-red-50 hover:text-red-600 hover:border-red-200 text-sm font-semibold transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>

          {/* Experiments */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium mb-1">Try these:</p>
            <ul className="text-xs text-amber-700 space-y-1 list-disc list-inside">
              <li>Place a query on the boundary between classes</li>
              <li>Compare K=1 vs K=15 on the same point</li>
              <li>Add outliers and see how they affect classification</li>
              <li>Create an island of one class inside the other</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 2 — K VALUE EFFECT
// ════════════════════════════════════════════════════════════════════════════

const HEATMAP_DATA = generateOverlappingData(99);
const HEATMAP_W = 460;
const HEATMAP_H = 360;
const ACC_CHART_W = 460;
const ACC_CHART_H = 200;
const ACC_PAD = { top: 20, right: 20, bottom: 35, left: 45 };

function KValueEffectTab() {
  const [k, setK] = useState(5);

  // Split data into train / test
  const { trainData, testData } = useMemo(() => {
    const rng = mulberry32(77);
    const shuffled = [...HEATMAP_DATA].sort(() => rng() - 0.5);
    const splitIdx = Math.floor(shuffled.length * 0.7);
    return { trainData: shuffled.slice(0, splitIdx), testData: shuffled.slice(splitIdx) };
  }, []);

  // Decision boundary heatmap grid
  const gridResolution = 28;
  const heatmapCells = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; color: string }[] = [];
    const cellW = HEATMAP_W / gridResolution;
    const cellH = HEATMAP_H / gridResolution;

    for (let gx = 0; gx < gridResolution; gx++) {
      for (let gy = 0; gy < gridResolution; gy++) {
        const qx = (gx + 0.5) * cellW;
        const qy = (gy + 0.5) * cellH;
        const { predicted, votes } = knnClassify(qx, qy, trainData, k);
        const confidence = Math.max(...votes) / k;
        const baseColor = predicted === 0 ? [59, 130, 246] : [249, 115, 22];
        const alpha = 0.12 + confidence * 0.28;
        cells.push({
          x: gx * cellW,
          y: gy * cellH,
          w: cellW + 0.5,
          h: cellH + 0.5,
          color: `rgba(${baseColor[0]},${baseColor[1]},${baseColor[2]},${alpha})`,
        });
      }
    }
    return cells;
  }, [k, trainData]);

  // Accuracy vs K chart data
  const accuracyData = useMemo(() => {
    const results: { k: number; trainAcc: number; testAcc: number }[] = [];
    for (let kv = 1; kv <= 25; kv++) {
      let trainCorrect = 0;
      for (const p of trainData) {
        // Leave-one-out for training: exclude the point itself
        const others = trainData.filter((o) => o !== p);
        const { predicted } = knnClassify(p.x, p.y, others, kv);
        if (predicted === p.cls) trainCorrect++;
      }

      let testCorrect = 0;
      for (const p of testData) {
        const { predicted } = knnClassify(p.x, p.y, trainData, kv);
        if (predicted === p.cls) testCorrect++;
      }

      results.push({
        k: kv,
        trainAcc: trainCorrect / trainData.length,
        testAcc: testCorrect / testData.length,
      });
    }
    return results;
  }, [trainData, testData]);

  const currentAccuracy = accuracyData.find((d) => d.k === k) || { trainAcc: 0, testAcc: 0 };
  const bestTestK = accuracyData.reduce((best, d) => (d.testAcc > best.testAcc ? d : best), accuracyData[0]);

  const accPlotW = ACC_CHART_W - ACC_PAD.left - ACC_PAD.right;
  const accPlotH = ACC_CHART_H - ACC_PAD.top - ACC_PAD.bottom;

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-indigo-900">How K Affects the Decision Boundary</h3>
          <p className="text-xs text-indigo-700 mt-1">
            A small K (like 1) creates a jagged, overfitting boundary that memorizes noise.
            A large K creates a smoother boundary but may underfit. The sweet spot balances
            bias and variance. Watch the decision regions and accuracy chart as you move the slider.
          </p>
        </div>
      </div>

      {/* K slider */}
      <div className="bg-white border border-slate-200 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
            <Target className="w-3.5 h-3.5 text-indigo-500" />
            K = {k}
          </label>
          <div className="flex gap-3 text-xs">
            <span className="text-blue-600 font-semibold">
              Train: {(currentAccuracy.trainAcc * 100).toFixed(1)}%
            </span>
            <span className="text-emerald-600 font-semibold">
              Test: {(currentAccuracy.testAcc * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        <input
          type="range"
          min={1}
          max={25}
          step={1}
          value={k}
          onChange={(e) => setK(parseInt(e.target.value, 10))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"
        />
        <div className="flex justify-between text-[10px] text-slate-400 mt-1">
          <span>K=1 (overfitting)</span>
          <span>K=25 (underfitting)</span>
        </div>
      </div>

      <div className="flex gap-5 flex-col xl:flex-row">
        {/* Decision boundary heatmap */}
        <div className="flex-1 min-w-0">
          <h4 className="text-xs font-semibold text-slate-600 mb-2">Decision Boundary (K={k})</h4>
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg
              viewBox={`0 0 ${HEATMAP_W} ${HEATMAP_H}`}
              className="w-full"
              style={{ aspectRatio: `${HEATMAP_W}/${HEATMAP_H}` }}
            >
              {/* Heatmap cells */}
              {heatmapCells.map((cell, i) => (
                <rect
                  key={`cell-${i}`}
                  x={cell.x}
                  y={cell.y}
                  width={cell.w}
                  height={cell.h}
                  fill={cell.color}
                />
              ))}

              {/* Training points */}
              {trainData.map((p, i) => (
                <circle
                  key={`train-${i}`}
                  cx={p.x}
                  cy={p.y}
                  r={5.5}
                  fill={CLASS_COLORS[p.cls]}
                  stroke="#fff"
                  strokeWidth={1.5}
                />
              ))}

              {/* Test points (squares) */}
              {testData.map((p, i) => {
                const { predicted } = knnClassify(p.x, p.y, trainData, k);
                const correct = predicted === p.cls;
                return (
                  <g key={`test-${i}`}>
                    <rect
                      x={p.x - 5}
                      y={p.y - 5}
                      width={10}
                      height={10}
                      fill={CLASS_COLORS[p.cls]}
                      stroke={correct ? "#fff" : "#ef4444"}
                      strokeWidth={correct ? 1.5 : 2.5}
                      rx={1}
                    />
                    {!correct && (
                      <line
                        x1={p.x - 6}
                        y1={p.y - 6}
                        x2={p.x + 6}
                        y2={p.y + 6}
                        stroke="#ef4444"
                        strokeWidth={1.8}
                      />
                    )}
                  </g>
                );
              })}

              {/* Legend */}
              <g transform="translate(10, 10)">
                <rect width={130} height={50} fill="rgba(255,255,255,0.85)" rx={4} />
                <circle cx={14} cy={16} r={4.5} fill={CLASS_COLORS[0]} stroke="#fff" strokeWidth={1} />
                <text x={24} y={20} fontSize={10} fill="#334155">Blue (train)</text>
                <circle cx={14} cy={34} r={4.5} fill={CLASS_COLORS[1]} stroke="#fff" strokeWidth={1} />
                <text x={24} y={38} fontSize={10} fill="#334155">Orange (train)</text>
                <rect x={85} y={12} width={8} height={8} fill={CLASS_COLORS[0]} stroke="#fff" strokeWidth={1} rx={1} />
                <text x={98} y={20} fontSize={10} fill="#334155">Test</text>
              </g>
            </svg>
          </div>
        </div>

        {/* Accuracy vs K chart */}
        <div className="flex-1 min-w-0">
          <h4 className="text-xs font-semibold text-slate-600 mb-2">
            Accuracy vs K
            <span className="ml-2 text-[10px] text-emerald-600 font-normal">
              (Best test: K={bestTestK.k}, {(bestTestK.testAcc * 100).toFixed(1)}%)
            </span>
          </h4>
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg
              viewBox={`0 0 ${ACC_CHART_W} ${ACC_CHART_H}`}
              className="w-full"
              style={{ aspectRatio: `${ACC_CHART_W}/${ACC_CHART_H}` }}
            >
              {/* Axes */}
              <line
                x1={ACC_PAD.left}
                y1={ACC_PAD.top}
                x2={ACC_PAD.left}
                y2={ACC_PAD.top + accPlotH}
                stroke="#cbd5e1"
                strokeWidth={1}
              />
              <line
                x1={ACC_PAD.left}
                y1={ACC_PAD.top + accPlotH}
                x2={ACC_PAD.left + accPlotW}
                y2={ACC_PAD.top + accPlotH}
                stroke="#cbd5e1"
                strokeWidth={1}
              />

              {/* Y-axis labels (0%, 50%, 100%) */}
              {[0, 0.25, 0.5, 0.75, 1].map((v) => {
                const yy = ACC_PAD.top + accPlotH - v * accPlotH;
                return (
                  <g key={`ytick-${v}`}>
                    <line
                      x1={ACC_PAD.left - 4}
                      y1={yy}
                      x2={ACC_PAD.left + accPlotW}
                      y2={yy}
                      stroke="#f1f5f9"
                      strokeWidth={0.5}
                    />
                    <text x={ACC_PAD.left - 8} y={yy + 3} fontSize={9} fill="#94a3b8" textAnchor="end">
                      {(v * 100).toFixed(0)}%
                    </text>
                  </g>
                );
              })}

              {/* X-axis labels */}
              {[1, 5, 10, 15, 20, 25].map((xv) => {
                const xx = ACC_PAD.left + ((xv - 1) / 24) * accPlotW;
                return (
                  <text
                    key={`xtick-${xv}`}
                    x={xx}
                    y={ACC_PAD.top + accPlotH + 14}
                    fontSize={9}
                    fill="#94a3b8"
                    textAnchor="middle"
                  >
                    {xv}
                  </text>
                );
              })}
              <text
                x={ACC_PAD.left + accPlotW / 2}
                y={ACC_PAD.top + accPlotH + 30}
                fontSize={10}
                fill="#64748b"
                textAnchor="middle"
              >
                K
              </text>

              {/* Train accuracy line */}
              <polyline
                points={accuracyData
                  .map((d) => {
                    const xx = ACC_PAD.left + ((d.k - 1) / 24) * accPlotW;
                    const yy = ACC_PAD.top + accPlotH - d.trainAcc * accPlotH;
                    return `${xx},${yy}`;
                  })
                  .join(" ")}
                fill="none"
                stroke="#3b82f6"
                strokeWidth={2}
                opacity={0.7}
              />

              {/* Test accuracy line */}
              <polyline
                points={accuracyData
                  .map((d) => {
                    const xx = ACC_PAD.left + ((d.k - 1) / 24) * accPlotW;
                    const yy = ACC_PAD.top + accPlotH - d.testAcc * accPlotH;
                    return `${xx},${yy}`;
                  })
                  .join(" ")}
                fill="none"
                stroke="#10b981"
                strokeWidth={2}
              />

              {/* Current K marker */}
              {(() => {
                const xx = ACC_PAD.left + ((k - 1) / 24) * accPlotW;
                return (
                  <line
                    x1={xx}
                    y1={ACC_PAD.top}
                    x2={xx}
                    y2={ACC_PAD.top + accPlotH}
                    stroke="#6366f1"
                    strokeWidth={2}
                    strokeDasharray="4,3"
                    opacity={0.6}
                  />
                );
              })()}

              {/* Dots on lines at current K */}
              {(() => {
                const xx = ACC_PAD.left + ((k - 1) / 24) * accPlotW;
                const trainY = ACC_PAD.top + accPlotH - currentAccuracy.trainAcc * accPlotH;
                const testY = ACC_PAD.top + accPlotH - currentAccuracy.testAcc * accPlotH;
                return (
                  <>
                    <circle cx={xx} cy={trainY} r={4} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
                    <circle cx={xx} cy={testY} r={4} fill="#10b981" stroke="#fff" strokeWidth={1.5} />
                  </>
                );
              })()}

              {/* Legend */}
              <g transform={`translate(${ACC_PAD.left + 8}, ${ACC_PAD.top + 4})`}>
                <line x1={0} y1={0} x2={16} y2={0} stroke="#3b82f6" strokeWidth={2} opacity={0.7} />
                <text x={20} y={4} fontSize={9} fill="#334155">Train</text>
                <line x1={52} y1={0} x2={68} y2={0} stroke="#10b981" strokeWidth={2} />
                <text x={72} y={4} fontSize={9} fill="#334155">Test</text>
              </g>
            </svg>
          </div>

          {/* Interpretation */}
          <div className="mt-3 bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-xs text-slate-700">
              {k <= 2 ? (
                <>
                  <strong className="text-red-600">Overfitting zone:</strong> K={k} is very small.
                  The boundary is jagged and follows noise. Training accuracy looks great, but test
                  accuracy may suffer.
                </>
              ) : k >= 18 ? (
                <>
                  <strong className="text-amber-600">Underfitting zone:</strong> K={k} is very large.
                  The boundary is very smooth, ignoring local structure. Both train and test accuracy
                  may degrade as the model becomes too simple.
                </>
              ) : (
                <>
                  <strong className="text-emerald-600">Balanced zone:</strong> K={k} provides a
                  good trade-off between a flexible boundary and noise robustness. Check the accuracy
                  chart to see if this is near the sweet spot.
                </>
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 3 — DISTANCE METRICS
// ════════════════════════════════════════════════════════════════════════════

const METRICS_DATA = generateTwoClassData(55, 30);
const METRIC_SVG_W = 340;
const METRIC_SVG_H = 300;

type MetricKey = "euclidean" | "manhattan" | "chebyshev";

const METRIC_LABELS: Record<MetricKey, string> = {
  euclidean: "Euclidean",
  manhattan: "Manhattan",
  chebyshev: "Chebyshev",
};

const METRIC_FORMULAS: Record<MetricKey, string> = {
  euclidean: "d = sqrt((x1-x2)^2 + (y1-y2)^2)",
  manhattan: "d = |x1-x2| + |y1-y2|",
  chebyshev: "d = max(|x1-x2|, |y1-y2|)",
};

const METRIC_COLORS: Record<MetricKey, string> = {
  euclidean: "#8b5cf6",
  manhattan: "#ec4899",
  chebyshev: "#14b8a6",
};

function DistanceMetricsTab() {
  const [queryPos, setQueryPos] = useState({ x: 250, y: 160 });
  const [k, setK] = useState(5);
  const [highlightMetric, setHighlightMetric] = useState<MetricKey>("euclidean");
  const [isDragging, setIsDragging] = useState(false);
  const svgRefs = useRef<Record<MetricKey, SVGSVGElement | null>>({
    euclidean: null,
    manhattan: null,
    chebyshev: null,
  });

  const metrics: MetricKey[] = ["euclidean", "manhattan", "chebyshev"];

  // Classifications per metric
  const results = useMemo(() => {
    const out: Record<MetricKey, ReturnType<typeof knnClassify>> = {} as any;
    for (const m of metrics) {
      out[m] = knnClassify(queryPos.x, queryPos.y, METRICS_DATA, k, DISTANCE_FNS[m]);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryPos.x, queryPos.y, k]);

  // Check if predictions differ
  const allSame =
    results.euclidean.predicted === results.manhattan.predicted &&
    results.manhattan.predicted === results.chebyshev.predicted;

  const handleMouseDown = useCallback(() => setIsDragging(true), []);
  const handleMouseUp = useCallback(() => setIsDragging(false), []);

  useEffect(() => {
    const handleGlobalUp = () => setIsDragging(false);
    window.addEventListener("mouseup", handleGlobalUp);
    return () => window.removeEventListener("mouseup", handleGlobalUp);
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>, metric: MetricKey) => {
      if (!isDragging) return;
      const svg = svgRefs.current[metric];
      if (!svg) return;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const svgPt = pt.matrixTransform(ctm.inverse());
      setQueryPos({
        x: Math.max(10, Math.min(METRIC_SVG_W - 10, Math.round(svgPt.x))),
        y: Math.max(10, Math.min(METRIC_SVG_H - 10, Math.round(svgPt.y))),
      });
    },
    [isDragging],
  );

  // Generate iso-distance contour points
  const isoContourPaths = useMemo(() => {
    const out: Record<MetricKey, string[]> = { euclidean: [], manhattan: [], chebyshev: [] };
    const radii = [40, 80, 120, 160];

    for (const r of radii) {
      // Euclidean: circle
      {
        const pts: string[] = [];
        for (let a = 0; a <= 360; a += 5) {
          const rad = (a * Math.PI) / 180;
          pts.push(`${queryPos.x + r * Math.cos(rad)},${queryPos.y + r * Math.sin(rad)}`);
        }
        out.euclidean.push(pts.join(" "));
      }

      // Manhattan: diamond (rotated square)
      {
        const pts = [
          `${queryPos.x},${queryPos.y - r}`,
          `${queryPos.x + r},${queryPos.y}`,
          `${queryPos.x},${queryPos.y + r}`,
          `${queryPos.x - r},${queryPos.y}`,
          `${queryPos.x},${queryPos.y - r}`,
        ];
        out.manhattan.push(pts.join(" "));
      }

      // Chebyshev: square
      {
        const pts = [
          `${queryPos.x - r},${queryPos.y - r}`,
          `${queryPos.x + r},${queryPos.y - r}`,
          `${queryPos.x + r},${queryPos.y + r}`,
          `${queryPos.x - r},${queryPos.y + r}`,
          `${queryPos.x - r},${queryPos.y - r}`,
        ];
        out.chebyshev.push(pts.join(" "));
      }
    }
    return out;
  }, [queryPos]);

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-purple-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-purple-900">Distance Metrics Comparison</h3>
          <p className="text-xs text-purple-700 mt-1">
            KNN depends on how you measure "distance." Different metrics define different shapes for
            equal-distance contours. <strong>Euclidean</strong> uses circles,{" "}
            <strong>Manhattan</strong> uses diamonds, and <strong>Chebyshev</strong> uses squares.
            Drag the query point to see how the selected neighbors change.
          </p>
        </div>
      </div>

      {/* K slider */}
      <div className="bg-white border border-slate-200 rounded-lg p-3 flex items-center gap-4">
        <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5 whitespace-nowrap">
          <Target className="w-3.5 h-3.5 text-purple-500" />
          K = {k}
        </label>
        <input
          type="range"
          min={1}
          max={15}
          step={1}
          value={k}
          onChange={(e) => setK(parseInt(e.target.value, 10))}
          className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
        />
        {!allSame && (
          <span className="text-xs font-semibold text-red-500 whitespace-nowrap">
            Predictions differ!
          </span>
        )}
      </div>

      {/* 3-panel comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {metrics.map((metric) => {
          const result = results[metric];
          const isHighlighted = highlightMetric === metric;
          const borderColor = isHighlighted ? METRIC_COLORS[metric] : "#e2e8f0";

          return (
            <div key={metric}>
              <button
                onClick={() => setHighlightMetric(metric)}
                className={`w-full text-left mb-2 px-2 py-1.5 rounded-lg border-2 transition-all ${
                  isHighlighted ? "bg-white shadow-sm" : "bg-slate-50 border-transparent"
                }`}
                style={{ borderColor }}
              >
                <h4 className="text-xs font-bold" style={{ color: METRIC_COLORS[metric] }}>
                  {METRIC_LABELS[metric]}
                </h4>
                <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                  {METRIC_FORMULAS[metric]}
                </p>
              </button>

              <div className="bg-slate-50 border border-slate-200 rounded-xl p-2">
                <svg
                  ref={(el) => { svgRefs.current[metric] = el; }}
                  viewBox={`0 0 ${METRIC_SVG_W} ${METRIC_SVG_H}`}
                  className="w-full cursor-grab select-none"
                  style={{ aspectRatio: `${METRIC_SVG_W}/${METRIC_SVG_H}` }}
                  onMouseDown={handleMouseDown}
                  onMouseUp={handleMouseUp}
                  onMouseMove={(e) => handleMouseMove(e, metric)}
                >
                  <SvgGrid w={METRIC_SVG_W} h={METRIC_SVG_H} step={40} />

                  {/* Iso-distance contours */}
                  {isoContourPaths[metric].map((pts, i) => (
                    <polyline
                      key={`iso-${i}`}
                      points={pts}
                      fill="none"
                      stroke={METRIC_COLORS[metric]}
                      strokeWidth={1}
                      opacity={0.2 + (3 - i) * 0.05}
                      strokeDasharray="3,3"
                    />
                  ))}

                  {/* Lines to neighbors */}
                  {result.neighbors.map((n, i) => (
                    <line
                      key={`nline-${i}`}
                      x1={queryPos.x}
                      y1={queryPos.y}
                      x2={n.point.x}
                      y2={n.point.y}
                      stroke={CLASS_COLORS[n.point.cls]}
                      strokeWidth={1.5}
                      strokeDasharray="4,3"
                      opacity={0.5}
                    />
                  ))}

                  {/* Data points */}
                  {METRICS_DATA.map((p, i) => {
                    const isNeighbor = result.neighbors.some(
                      (n) => n.point.x === p.x && n.point.y === p.y,
                    );
                    return (
                      <g key={`pt-${i}`}>
                        {isNeighbor && (
                          <circle
                            cx={p.x}
                            cy={p.y}
                            r={11}
                            fill="none"
                            stroke={METRIC_COLORS[metric]}
                            strokeWidth={2}
                            opacity={0.6}
                          />
                        )}
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={5}
                          fill={CLASS_COLORS[p.cls]}
                          stroke="#fff"
                          strokeWidth={1.2}
                          opacity={isNeighbor ? 1 : 0.6}
                        />
                      </g>
                    );
                  })}

                  {/* Query point */}
                  <circle
                    cx={queryPos.x}
                    cy={queryPos.y}
                    r={8}
                    fill={METRIC_COLORS[metric]}
                    stroke="#fff"
                    strokeWidth={2}
                    className="cursor-grab"
                  />
                  <circle
                    cx={queryPos.x}
                    cy={queryPos.y}
                    r={3}
                    fill="#fff"
                    opacity={0.8}
                  />
                </svg>
              </div>

              {/* Result */}
              <div className="mt-2 bg-white border border-slate-200 rounded-lg p-2 text-center">
                <span className="text-xs text-slate-500">Prediction: </span>
                <span
                  className="text-xs font-bold"
                  style={{ color: CLASS_COLORS[result.predicted] }}
                >
                  {CLASS_NAMES[result.predicted]}
                </span>
                <span className="text-[10px] text-slate-400 ml-2">
                  ({result.votes[0]}B / {result.votes[1]}O)
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* When predictions differ */}
      {!allSame && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-xs text-red-700 font-medium">
            The metric choice changes the prediction here! This happens because different metrics
            define "nearest" differently. In practice, the best metric depends on the structure of
            your data. Euclidean is the default but Manhattan works well for grid-like or sparse
            data.
          </p>
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 4 — WEIGHTED KNN
// ════════════════════════════════════════════════════════════════════════════

const WEIGHTED_DATA = generateTwoClassData(77, 36);
const WEIGHT_SVG_W = 460;
const WEIGHT_SVG_H = 340;
const BAR_CHART_W = 220;
const BAR_CHART_H = 160;

type WeightFnKey = "uniform" | "1/d" | "1/d2" | "gaussian";

const WEIGHT_FN_LABELS: Record<WeightFnKey, string> = {
  uniform: "Uniform (equal)",
  "1/d": "Inverse Distance (1/d)",
  "1/d2": "Inverse Sq. (1/d\u00B2)",
  gaussian: "Gaussian Kernel",
};

function WeightedKNNTab() {
  const [k, setK] = useState(7);
  const [weightFn, setWeightFn] = useState<WeightFnKey>("1/d");
  const [queryPos, setQueryPos] = useState({ x: 250, y: 200 });
  const [isDragging, setIsDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const up = () => setIsDragging(false);
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, []);

  const handleMouseDown = useCallback(() => setIsDragging(true), []);
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!isDragging) return;
      const svg = svgRef.current;
      if (!svg) return;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const svgPt = pt.matrixTransform(ctm.inverse());
      setQueryPos({
        x: Math.max(10, Math.min(WEIGHT_SVG_W - 10, Math.round(svgPt.x))),
        y: Math.max(10, Math.min(WEIGHT_SVG_H - 10, Math.round(svgPt.y))),
      });
    },
    [isDragging],
  );

  const uniformResult = useMemo(
    () => knnClassifyWeighted(queryPos.x, queryPos.y, WEIGHTED_DATA, k, "uniform"),
    [queryPos.x, queryPos.y, k],
  );

  const weightedResult = useMemo(
    () => knnClassifyWeighted(queryPos.x, queryPos.y, WEIGHTED_DATA, k, weightFn),
    [queryPos.x, queryPos.y, k, weightFn],
  );

  const predictionsDiffer = uniformResult.predicted !== weightedResult.predicted;

  // Max weight for normalization
  const maxWeight = Math.max(...weightedResult.neighbors.map((n) => n.weight), 0.001);

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-teal-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-teal-900">Uniform vs Weighted KNN</h3>
          <p className="text-xs text-teal-700 mt-1">
            Standard KNN gives every neighbor an equal vote. <strong>Weighted KNN</strong> gives
            closer neighbors more influence. This can be crucial when the K nearest neighbors are at
            very different distances. Drag the query point and compare the two voting schemes.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3 flex-wrap">
        <div className="bg-white border border-slate-200 rounded-lg p-3 flex-1 min-w-[200px]">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-teal-500" />
              K = {k}
            </label>
          </div>
          <input
            type="range"
            min={3}
            max={15}
            step={1}
            value={k}
            onChange={(e) => setK(parseInt(e.target.value, 10))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-teal-500"
          />
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-3 flex-1 min-w-[200px]">
          <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
            Weight Function
          </p>
          <div className="flex gap-1.5 flex-wrap">
            {(["uniform", "1/d", "1/d2", "gaussian"] as WeightFnKey[]).map((wf) => (
              <button
                key={wf}
                onClick={() => setWeightFn(wf)}
                className={`px-2.5 py-1.5 rounded-lg text-[11px] font-semibold transition-all ${
                  weightFn === wf
                    ? "bg-teal-600 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {wf === "1/d2" ? "1/d\u00B2" : wf}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex gap-5 flex-col xl:flex-row">
        {/* SVG Visualization */}
        <div className="flex-1 min-w-0">
          <h4 className="text-xs font-semibold text-slate-600 mb-2">
            Neighbor Weights (size = weight)
          </h4>
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${WEIGHT_SVG_W} ${WEIGHT_SVG_H}`}
              className="w-full cursor-grab select-none"
              style={{ aspectRatio: `${WEIGHT_SVG_W}/${WEIGHT_SVG_H}` }}
              onMouseDown={handleMouseDown}
              onMouseUp={() => setIsDragging(false)}
              onMouseMove={handleMouseMove}
            >
              <SvgGrid w={WEIGHT_SVG_W} h={WEIGHT_SVG_H} step={40} />

              {/* Lines to neighbors with varying opacity based on weight */}
              {weightedResult.neighbors.map((n, i) => {
                const normalizedWeight = n.weight / maxWeight;
                return (
                  <line
                    key={`wline-${i}`}
                    x1={queryPos.x}
                    y1={queryPos.y}
                    x2={n.point.x}
                    y2={n.point.y}
                    stroke={CLASS_COLORS[n.point.cls]}
                    strokeWidth={1 + normalizedWeight * 2}
                    opacity={0.2 + normalizedWeight * 0.5}
                  />
                );
              })}

              {/* All data points */}
              {WEIGHTED_DATA.map((p, i) => {
                const neighborInfo = weightedResult.neighbors.find(
                  (n) => n.point.x === p.x && n.point.y === p.y,
                );
                const isNeighbor = !!neighborInfo;
                const normalizedWeight = neighborInfo ? neighborInfo.weight / maxWeight : 0;

                return (
                  <g key={`pt-${i}`}>
                    {isNeighbor && (
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={6 + normalizedWeight * 12}
                        fill={CLASS_COLORS[p.cls]}
                        opacity={0.15 + normalizedWeight * 0.2}
                      />
                    )}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={isNeighbor ? 4 + normalizedWeight * 5 : 5}
                      fill={CLASS_COLORS[p.cls]}
                      stroke="#fff"
                      strokeWidth={1.5}
                      opacity={isNeighbor ? 0.8 + normalizedWeight * 0.2 : 0.5}
                    />
                    {isNeighbor && neighborInfo && (
                      <text
                        x={p.x}
                        y={p.y - 10 - normalizedWeight * 8}
                        fontSize={8}
                        fill="#475569"
                        textAnchor="middle"
                        fontFamily="monospace"
                      >
                        w={neighborInfo.weight.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Query point */}
              <circle
                cx={queryPos.x}
                cy={queryPos.y}
                r={9}
                fill="#475569"
                stroke="#fff"
                strokeWidth={2.5}
                className="cursor-grab"
              />
              <circle cx={queryPos.x} cy={queryPos.y} r={3.5} fill="#fff" opacity={0.8} />
            </svg>
          </div>
        </div>

        {/* Voting comparison */}
        <div className="w-full xl:w-72 space-y-3">
          {/* Uniform votes bar chart */}
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <h4 className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Uniform Votes
            </h4>
            <svg
              viewBox={`0 0 ${BAR_CHART_W} ${BAR_CHART_H}`}
              className="w-full"
              style={{ aspectRatio: `${BAR_CHART_W}/${BAR_CHART_H}` }}
            >
              {CLASS_NAMES.map((name, cls) => {
                const val = uniformResult.weightedVotes[cls];
                const maxVal = Math.max(...uniformResult.weightedVotes, 1);
                const barW = (val / maxVal) * (BAR_CHART_W - 80);
                const yy = cls * 55 + 20;
                return (
                  <g key={`ubar-${cls}`}>
                    <text x={10} y={yy + 14} fontSize={11} fill="#334155" fontWeight={600}>
                      {name}
                    </text>
                    <rect
                      x={65}
                      y={yy}
                      width={barW}
                      height={24}
                      fill={CLASS_COLORS[cls]}
                      rx={4}
                      opacity={0.8}
                    />
                    <text
                      x={70 + barW}
                      y={yy + 16}
                      fontSize={11}
                      fill="#334155"
                      fontWeight={700}
                    >
                      {val.toFixed(0)}
                    </text>
                  </g>
                );
              })}
              <text x={10} y={BAR_CHART_H - 8} fontSize={10} fill="#94a3b8">
                Prediction:{" "}
                <tspan fontWeight={700} fill={CLASS_COLORS[uniformResult.predicted]}>
                  {CLASS_NAMES[uniformResult.predicted]}
                </tspan>
              </text>
            </svg>
          </div>

          {/* Weighted votes bar chart */}
          <div
            className={`bg-white border rounded-lg p-3 ${
              predictionsDiffer ? "border-red-300 ring-2 ring-red-100" : "border-slate-200"
            }`}
          >
            <h4 className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">
              Weighted Votes ({WEIGHT_FN_LABELS[weightFn]})
            </h4>
            <svg
              viewBox={`0 0 ${BAR_CHART_W} ${BAR_CHART_H}`}
              className="w-full"
              style={{ aspectRatio: `${BAR_CHART_W}/${BAR_CHART_H}` }}
            >
              {CLASS_NAMES.map((name, cls) => {
                const val = weightedResult.weightedVotes[cls];
                const maxVal = Math.max(...weightedResult.weightedVotes, 0.001);
                const barW = (val / maxVal) * (BAR_CHART_W - 80);
                const yy = cls * 55 + 20;
                return (
                  <g key={`wbar-${cls}`}>
                    <text x={10} y={yy + 14} fontSize={11} fill="#334155" fontWeight={600}>
                      {name}
                    </text>
                    <rect
                      x={65}
                      y={yy}
                      width={barW}
                      height={24}
                      fill={CLASS_COLORS[cls]}
                      rx={4}
                      opacity={0.9}
                    />
                    <text
                      x={70 + barW}
                      y={yy + 16}
                      fontSize={11}
                      fill="#334155"
                      fontWeight={700}
                    >
                      {val.toFixed(2)}
                    </text>
                  </g>
                );
              })}
              <text x={10} y={BAR_CHART_H - 8} fontSize={10} fill="#94a3b8">
                Prediction:{" "}
                <tspan fontWeight={700} fill={CLASS_COLORS[weightedResult.predicted]}>
                  {CLASS_NAMES[weightedResult.predicted]}
                </tspan>
              </text>
            </svg>
          </div>

          {predictionsDiffer && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-2.5">
              <p className="text-xs text-red-700 font-medium">
                Weighting changed the prediction! The closer neighbors of one class outweigh the
                more numerous but distant neighbors of the other.
              </p>
            </div>
          )}

          {/* Weight function explanation */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-1">
              About {WEIGHT_FN_LABELS[weightFn]}
            </p>
            <p className="text-xs text-slate-600">
              {weightFn === "uniform"
                ? "Every neighbor gets weight = 1. This is standard KNN — pure majority vote."
                : weightFn === "1/d"
                  ? "Weight = 1/distance. A neighbor at distance 2 has half the vote power of one at distance 1."
                  : weightFn === "1/d2"
                    ? "Weight = 1/distance\u00B2. Even more aggressive distance decay. Very close neighbors dominate."
                    : "Weight = exp(-d\u00B2 / 2\u03C3\u00B2). A smooth Gaussian falloff. Neighbors beyond ~2\u03C3 have negligible influence."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════════════════

export default function KNNPlaygroundActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("classify");

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
        {activeTab === "classify" && <ClassifyPointsTab />}
        {activeTab === "kvalue" && <KValueEffectTab />}
        {activeTab === "distance" && <DistanceMetricsTab />}
        {activeTab === "weighted" && <WeightedKNNTab />}
      </div>
    </div>
  );
}
