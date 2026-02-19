/**
 * Decision Boundary Explorer — interactive SVG activity
 * Visualizes how different ML algorithms carve decision regions
 * on synthetic 2D datasets (Linear, Circles, Moons, XOR).
 */

import { useState, useMemo } from "react";
import { Info, Lightbulb, Grid3X3 } from "lucide-react";

// ── Types ───────────────────────────────────────────────────────────────
interface DataPoint {
  x: number;
  y: number;
  cls: number; // 0 or 1
}

type DatasetPattern = "linear" | "circles" | "moons" | "xor";
type AlgorithmName = "Linear" | "KNN (K=5)" | "Decision Tree" | "Neural Network";

// ── Constants ───────────────────────────────────────────────────────────
const SVG_SIZE = 400;
const HEATMAP_RES = 40;
const KNN_GRID_RES = 20; // coarser grid for KNN to avoid perf issues
const NUM_POINTS = 100;

const DATASET_OPTIONS: { key: DatasetPattern; label: string }[] = [
  { key: "linear", label: "Linear" },
  { key: "circles", label: "Circles" },
  { key: "moons", label: "Moons" },
  { key: "xor", label: "XOR" },
];

const ALGORITHMS: AlgorithmName[] = [
  "Linear",
  "KNN (K=5)",
  "Decision Tree",
  "Neural Network",
];

const BEST_FIT: Record<DatasetPattern, string> = {
  linear:
    "Linear classifier works best here — the data is linearly separable.",
  circles:
    "KNN and Neural Network handle the circular boundary well; Linear fails.",
  moons:
    "KNN and Neural Network capture the curved boundary; Decision Tree approximates with axis-aligned cuts.",
  xor:
    "Neural Network excels at XOR; Linear cannot separate this pattern at all.",
};

// ── Seeded random for reproducible datasets ─────────────────────────────
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// ── Dataset generators ──────────────────────────────────────────────────
// All points are normalized to roughly [0, 1] range for SVG mapping.

function generateLinear(rand: () => number): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < NUM_POINTS; i++) {
    const cls = i < NUM_POINTS / 2 ? 0 : 1;
    const cx = cls === 0 ? 0.3 : 0.7;
    const cy = cls === 0 ? 0.3 : 0.7;
    const x = cx + (rand() - 0.5) * 0.4;
    const y = cy + (rand() - 0.5) * 0.4;
    points.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
  }
  return points;
}

function generateCircles(rand: () => number): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < NUM_POINTS; i++) {
    const cls = i < NUM_POINTS / 2 ? 0 : 1;
    const angle = rand() * Math.PI * 2;
    const radius = cls === 0 ? rand() * 0.18 : 0.28 + rand() * 0.12;
    const x = 0.5 + Math.cos(angle) * radius;
    const y = 0.5 + Math.sin(angle) * radius;
    points.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
  }
  return points;
}

function generateMoons(rand: () => number): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < NUM_POINTS; i++) {
    const cls = i < NUM_POINTS / 2 ? 0 : 1;
    const angle = (i < NUM_POINTS / 2 ? i / (NUM_POINTS / 2) : (i - NUM_POINTS / 2) / (NUM_POINTS / 2)) * Math.PI;
    const noise = (rand() - 0.5) * 0.1;
    if (cls === 0) {
      const x = 0.35 + Math.cos(angle) * 0.25 + noise;
      const y = 0.45 - Math.sin(angle) * 0.25 + noise;
      points.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
    } else {
      const x = 0.65 - Math.cos(angle) * 0.25 + noise;
      const y = 0.55 + Math.sin(angle) * 0.25 + noise;
      points.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
    }
  }
  return points;
}

function generateXOR(rand: () => number): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < NUM_POINTS; i++) {
    const quadrant = i % 4;
    const cx = quadrant === 0 || quadrant === 3 ? 0.25 : 0.75;
    const cy = quadrant === 0 || quadrant === 1 ? 0.25 : 0.75;
    const cls = (quadrant === 0 || quadrant === 3) ? 0 : 1;
    const x = cx + (rand() - 0.5) * 0.3;
    const y = cy + (rand() - 0.5) * 0.3;
    points.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
  }
  return points;
}

function generateDataset(pattern: DatasetPattern): DataPoint[] {
  const rand = seededRandom(42);
  switch (pattern) {
    case "linear":
      return generateLinear(rand);
    case "circles":
      return generateCircles(rand);
    case "moons":
      return generateMoons(rand);
    case "xor":
      return generateXOR(rand);
  }
}

// ── Algorithm heuristic classifiers ─────────────────────────────────────
// Each returns a function (x, y) => predicted class (0 or 1)

function linearClassifier(data: DataPoint[]): (x: number, y: number) => number {
  // Find centroid of each class, then build perpendicular bisector
  let c0x = 0, c0y = 0, n0 = 0;
  let c1x = 0, c1y = 0, n1 = 0;
  for (const p of data) {
    if (p.cls === 0) { c0x += p.x; c0y += p.y; n0++; }
    else { c1x += p.x; c1y += p.y; n1++; }
  }
  c0x /= n0; c0y /= n0;
  c1x /= n1; c1y /= n1;

  // Normal vector from class 0 centroid to class 1 centroid
  const nx = c1x - c0x;
  const ny = c1y - c0y;
  // Midpoint
  const mx = (c0x + c1x) / 2;
  const my = (c0y + c1y) / 2;

  return (x: number, y: number) => {
    const dot = (x - mx) * nx + (y - my) * ny;
    return dot >= 0 ? 1 : 0;
  };
}

function knnClassifier(data: DataPoint[], k: number): (x: number, y: number) => number {
  return (x: number, y: number) => {
    // Compute squared distances to all data points
    const dists = data.map((p) => ({
      dist: (p.x - x) ** 2 + (p.y - y) ** 2,
      cls: p.cls,
    }));
    dists.sort((a, b) => a.dist - b.dist);
    const topK = dists.slice(0, k);
    const votes = topK.reduce((acc, d) => acc + d.cls, 0);
    return votes > k / 2 ? 1 : 0;
  };
}

function decisionTreeClassifier(data: DataPoint[]): (x: number, y: number) => number {
  // Simulate a shallow decision tree with 2-3 axis-aligned splits
  // based on median of each class per dimension at each level.

  type Region = { data: DataPoint[]; depth: number };
  type Leaf = { xMin: number; xMax: number; yMin: number; yMax: number; cls: number };

  const leaves: Leaf[] = [];

  function bestSplit(region: DataPoint[]): { dim: "x" | "y"; threshold: number } | null {
    if (region.length < 4) return null;

    let bestGini = Infinity;
    let bestDim: "x" | "y" = "x";
    let bestThresh = 0.5;

    for (const dim of ["x", "y"] as const) {
      const sorted = [...region].sort((a, b) => a[dim] - b[dim]);
      // Try a few candidate thresholds
      const candidates = [0.25, 0.5, 0.75].map(
        (f) => sorted[Math.floor(f * (sorted.length - 1))][dim]
      );
      for (const thresh of candidates) {
        const left = region.filter((p) => p[dim] < thresh);
        const right = region.filter((p) => p[dim] >= thresh);
        if (left.length === 0 || right.length === 0) continue;
        const giniLeft = 1 - (left.filter((p) => p.cls === 0).length / left.length) ** 2 - (left.filter((p) => p.cls === 1).length / left.length) ** 2;
        const giniRight = 1 - (right.filter((p) => p.cls === 0).length / right.length) ** 2 - (right.filter((p) => p.cls === 1).length / right.length) ** 2;
        const gini = (left.length * giniLeft + right.length * giniRight) / region.length;
        if (gini < bestGini) {
          bestGini = gini;
          bestDim = dim;
          bestThresh = thresh;
        }
      }
    }

    if (bestGini >= 0.5) return null; // not worth splitting
    return { dim: bestDim, threshold: bestThresh };
  }

  function buildTree(
    regionData: DataPoint[],
    depth: number,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number
  ) {
    const majority = regionData.filter((p) => p.cls === 1).length > regionData.length / 2 ? 1 : 0;

    if (depth >= 3 || regionData.length < 4) {
      leaves.push({ xMin, xMax, yMin, yMax, cls: majority });
      return;
    }

    const split = bestSplit(regionData);
    if (!split) {
      leaves.push({ xMin, xMax, yMin, yMax, cls: majority });
      return;
    }

    if (split.dim === "x") {
      buildTree(regionData.filter((p) => p.x < split.threshold), depth + 1, xMin, split.threshold, yMin, yMax);
      buildTree(regionData.filter((p) => p.x >= split.threshold), depth + 1, split.threshold, xMax, yMin, yMax);
    } else {
      buildTree(regionData.filter((p) => p.y < split.threshold), depth + 1, xMin, xMax, yMin, split.threshold);
      buildTree(regionData.filter((p) => p.y >= split.threshold), depth + 1, xMin, xMax, split.threshold, yMax);
    }
  }

  buildTree(data, 0, 0, 1, 0, 1);

  return (x: number, y: number) => {
    for (const leaf of leaves) {
      if (x >= leaf.xMin && x < leaf.xMax && y >= leaf.yMin && y < leaf.yMax) {
        return leaf.cls;
      }
    }
    // Fallback: majority class of all data
    return data.filter((p) => p.cls === 1).length > data.length / 2 ? 1 : 0;
  };
}

function neuralNetworkClassifier(data: DataPoint[]): (x: number, y: number) => number {
  // Approximate a smooth nonlinear boundary using a 2-hidden-unit network
  // with hand-tuned weights derived from the data centroids.
  // This creates a sigmoid-based smooth decision surface.

  let c0x = 0, c0y = 0, n0 = 0;
  let c1x = 0, c1y = 0, n1 = 0;
  for (const p of data) {
    if (p.cls === 0) { c0x += p.x; c0y += p.y; n0++; }
    else { c1x += p.x; c1y += p.y; n1++; }
  }
  c0x /= n0; c0y /= n0;
  c1x /= n1; c1y /= n1;

  // Compute variance along radial direction to set scale
  const midX = (c0x + c1x) / 2;
  const midY = (c0y + c1y) / 2;

  // Direction vector
  const dx = c1x - c0x;
  const dy = c1y - c0y;
  const len = Math.sqrt(dx * dx + dy * dy) || 0.01;

  // Compute variance of data projected onto the perpendicular
  const perpX = -dy / len;
  const perpY = dx / len;

  // Two hidden neurons that capture both the main separating direction
  // and a perpendicular curvature
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

  // Scale factor for sharpness based on class separation
  const sharpness = 12 / (len + 0.01);

  return (x: number, y: number) => {
    // Centered coordinates
    const cx = x - midX;
    const cy = y - midY;

    // Neuron 1: primary direction (class 0 -> class 1)
    const z1 = sharpness * (cx * (dx / len) + cy * (dy / len));
    const h1 = sigmoid(z1);

    // Neuron 2: radial component (captures circular/nonlinear boundaries)
    const r = Math.sqrt(cx * cx + cy * cy);
    const z2 = sharpness * (r - len * 0.8);
    const h2 = sigmoid(z2);

    // Neuron 3: perpendicular interaction (captures XOR-like patterns)
    const projPerp = cx * perpX + cy * perpY;
    const projMain = cx * (dx / len) + cy * (dy / len);
    const z3 = sharpness * 0.8 * projPerp * projMain;
    const h3 = sigmoid(z3);

    // Combine hidden units with learned-ish weights
    const output = sigmoid(4 * h1 - 2 + 2 * h2 - 1 + 3 * h3 - 1.5);
    return output >= 0.5 ? 1 : 0;
  };
}

// ── Build heatmap grid (predicted class for each cell) ──────────────────
function buildHeatmap(
  classifier: (x: number, y: number) => number,
  resolution: number
): number[][] {
  const grid: number[][] = [];
  for (let row = 0; row < resolution; row++) {
    const rowData: number[] = [];
    for (let col = 0; col < resolution; col++) {
      const x = (col + 0.5) / resolution;
      const y = (row + 0.5) / resolution;
      rowData.push(classifier(x, y));
    }
    grid.push(rowData);
  }
  return grid;
}

// ── SVG Panel component ─────────────────────────────────────────────────
interface PanelProps {
  algorithmName: AlgorithmName;
  data: DataPoint[];
  heatmap: number[][];
  resolution: number;
}

function BoundaryPanel({ algorithmName, data, heatmap, resolution }: PanelProps) {
  const cellSize = SVG_SIZE / resolution;

  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
      {/* Panel header */}
      <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
        <Grid3X3 className="w-3.5 h-3.5 text-violet-500" />
        <span className="text-xs font-semibold text-slate-700">{algorithmName}</span>
      </div>

      {/* SVG */}
      <div className="p-2">
        <svg
          viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
          className="w-full"
          style={{ aspectRatio: "1 / 1" }}
        >
          {/* Background heatmap */}
          {heatmap.map((row, ri) =>
            row.map((cls, ci) => (
              <rect
                key={`${ri}-${ci}`}
                x={ci * cellSize}
                y={ri * cellSize}
                width={cellSize + 0.5}
                height={cellSize + 0.5}
                fill={cls === 1 ? "#3b82f6" : "#ef4444"}
                opacity={0.15}
              />
            ))
          )}

          {/* Data points */}
          {data.map((p, i) => (
            <circle
              key={i}
              cx={p.x * SVG_SIZE}
              cy={p.y * SVG_SIZE}
              r={4}
              fill={p.cls === 1 ? "#2563eb" : "#dc2626"}
              stroke="#fff"
              strokeWidth={1}
              opacity={0.85}
            />
          ))}
        </svg>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────
export default function DecisionBoundaryActivity() {
  const [pattern, setPattern] = useState<DatasetPattern>("linear");

  // Generate dataset
  const data = useMemo(() => generateDataset(pattern), [pattern]);

  // Build classifiers and heatmaps for all 4 algorithms
  const panels = useMemo(() => {
    const linearCls = linearClassifier(data);
    const knnCls = knnClassifier(data, 5);
    const treeCls = decisionTreeClassifier(data);
    const nnCls = neuralNetworkClassifier(data);

    return [
      {
        name: "Linear" as AlgorithmName,
        heatmap: buildHeatmap(linearCls, HEATMAP_RES),
        resolution: HEATMAP_RES,
      },
      {
        name: "KNN (K=5)" as AlgorithmName,
        heatmap: buildHeatmap(knnCls, KNN_GRID_RES),
        resolution: KNN_GRID_RES,
      },
      {
        name: "Decision Tree" as AlgorithmName,
        heatmap: buildHeatmap(treeCls, HEATMAP_RES),
        resolution: HEATMAP_RES,
      },
      {
        name: "Neural Network" as AlgorithmName,
        heatmap: buildHeatmap(nnCls, HEATMAP_RES),
        resolution: HEATMAP_RES,
      },
    ];
  }, [data]);

  return (
    <div className="space-y-5">
      {/* Info banner */}
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-4 flex gap-3">
        <Info className="w-5 h-5 text-violet-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-violet-900">
            What Are Decision Boundaries?
          </h3>
          <p className="text-xs text-violet-700 mt-1">
            A decision boundary is the region in feature space where a classifier
            switches its prediction from one class to another. Different algorithms
            produce different boundary shapes: linear classifiers draw straight
            lines, KNN creates irregular local regions, decision trees produce
            axis-aligned rectangles, and neural networks can learn smooth nonlinear
            curves. The background color shows each algorithm's predicted class
            across the space (
            <span className="inline-block w-2.5 h-2.5 rounded-sm bg-red-400 align-middle" /> class 0,{" "}
            <span className="inline-block w-2.5 h-2.5 rounded-sm bg-blue-400 align-middle" /> class 1).
          </p>
        </div>
      </div>

      {/* Dataset selector */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-slate-600 mr-1">Dataset:</span>
        {DATASET_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            onClick={() => setPattern(opt.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              pattern === opt.key
                ? "bg-violet-600 text-white shadow-sm"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* 2x2 grid of algorithm panels */}
      <div className="grid grid-cols-2 gap-4">
        {panels.map((panel) => (
          <BoundaryPanel
            key={panel.name}
            algorithmName={panel.name}
            data={data}
            heatmap={panel.heatmap}
            resolution={panel.resolution}
          />
        ))}
      </div>

      {/* Side note: best fit for this pattern */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-3">
        <Lightbulb className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
        <div>
          <p className="text-xs font-semibold text-amber-800 mb-0.5">
            Best fit for "{DATASET_OPTIONS.find((o) => o.key === pattern)?.label}" pattern
          </p>
          <p className="text-xs text-amber-700">{BEST_FIT[pattern]}</p>
        </div>
      </div>
    </div>
  );
}
