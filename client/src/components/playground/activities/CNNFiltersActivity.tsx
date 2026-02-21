/**
 * CNN Convolution Filters Activity — comprehensive 5-tab interactive explorer
 *
 * Tab 1: Apply Filters — paint an image, choose/edit kernels, step through convolution
 * Tab 2: Edge Detection Deep Dive — Sobel-X, Sobel-Y, Laplacian side by side
 * Tab 3: Stride & Padding — visualize output size changes
 * Tab 4: Pooling Operations — max / average / min pooling comparison
 * Tab 5: Feature Maps & Channels — multi-filter & multi-channel concepts
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Layers,
  Play,
  SkipForward,
  RotateCcw,
  Grid3X3,
  SlidersHorizontal,
  Box,
  Pause,
  Eye,
  EyeOff,
  ChevronRight,
  Minus,
  Plus,
  Scan,
} from "lucide-react";

// ── Seeded PRNG (mulberry32) ─────────────────────────────────────────────
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Types ────────────────────────────────────────────────────────────────
type PaddingMode = "valid" | "same";
type PoolingType = "max" | "average" | "min";
type TabId = "apply" | "edge" | "stride" | "pooling" | "features";

interface TabDef {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

interface FilterPreset {
  name: string;
  kernel: number[][];
  description: string;
}

interface ImagePreset {
  name: string;
  generator: () => number[][];
}

// ── Constants ────────────────────────────────────────────────────────────
const CELL = 32;
const CELL_SM = 28;
const CELL_XS = 24;
const CELL_TINY = 20;
const KERNEL_CELL = 42;
const GAP = 1;

const TABS: TabDef[] = [
  { id: "apply", label: "Apply Filters", icon: <Grid3X3 className="w-4 h-4" /> },
  { id: "edge", label: "Edge Detection", icon: <Scan className="w-4 h-4" /> },
  { id: "stride", label: "Stride & Padding", icon: <SlidersHorizontal className="w-4 h-4" /> },
  { id: "pooling", label: "Pooling", icon: <Layers className="w-4 h-4" /> },
  { id: "features", label: "Feature Maps", icon: <Box className="w-4 h-4" /> },
];

// ── Filter presets ───────────────────────────────────────────────────────
const FILTER_PRESETS: FilterPreset[] = [
  {
    name: "Identity",
    kernel: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
    description: "Passes input through unchanged",
  },
  {
    name: "Edge (H)",
    kernel: [
      [-1, -1, -1],
      [0, 0, 0],
      [1, 1, 1],
    ],
    description: "Detects horizontal edges",
  },
  {
    name: "Edge (V)",
    kernel: [
      [-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1],
    ],
    description: "Detects vertical edges",
  },
  {
    name: "Sharpen",
    kernel: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
    description: "Enhances contrast at edges",
  },
  {
    name: "Blur",
    kernel: [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
    description: "Box blur — averages the neighborhood (÷9)",
  },
  {
    name: "Emboss",
    kernel: [
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2],
    ],
    description: "Creates a raised/embossed effect",
  },
];

// ── Edge-detection kernels ───────────────────────────────────────────────
const SOBEL_X: number[][] = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
];
const SOBEL_Y: number[][] = [
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1],
];
const LAPLACIAN: number[][] = [
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0],
];

// ── Image pattern generators (8x8) ──────────────────────────────────────
function makeHorizontalLines(): number[][] {
  return Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, () => (r % 2 === 0 ? 220 : 30))
  );
}
function makeVerticalLines(): number[][] {
  return Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, (_, c) => (c % 2 === 0 ? 220 : 30))
  );
}
function makeDiagonal(): number[][] {
  return Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => (Math.abs(r - c) <= 1 ? 230 : 20))
  );
}
function makeCheckerboard(): number[][] {
  return Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => ((r + c) % 2 === 0 ? 240 : 15))
  );
}
function makeCircle(): number[][] {
  const cx = 3.5,
    cy = 3.5,
    radius = 2.8;
  return Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => {
      const dist = Math.sqrt((r - cy) ** 2 + (c - cx) ** 2);
      return Math.abs(dist - radius) < 1.0 ? 230 : dist < radius ? 120 : 15;
    })
  );
}
function makeSquare(): number[][] {
  const grid: number[][] = Array.from({ length: 8 }, () => Array(8).fill(0));
  for (let i = 1; i <= 6; i++) {
    grid[1][i] = 200;
    grid[6][i] = 200;
    grid[i][1] = 200;
    grid[i][6] = 200;
  }
  for (let r = 2; r <= 5; r++)
    for (let c = 2; c <= 5; c++) grid[r][c] = 80;
  return grid;
}

const IMAGE_PRESETS_8: ImagePreset[] = [
  { name: "Square", generator: makeSquare },
  { name: "H-Lines", generator: makeHorizontalLines },
  { name: "V-Lines", generator: makeVerticalLines },
  { name: "Diagonal", generator: makeDiagonal },
  { name: "Checker", generator: makeCheckerboard },
  { name: "Circle", generator: makeCircle },
];

// ── 10x10 generators for stride/padding tab ──────────────────────────────
function make10x10Square(): number[][] {
  const grid: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));
  for (let i = 2; i <= 7; i++) {
    grid[2][i] = 200;
    grid[7][i] = 200;
    grid[i][2] = 200;
    grid[i][7] = 200;
  }
  for (let r = 3; r <= 6; r++)
    for (let c = 3; c <= 6; c++) grid[r][c] = 100;
  return grid;
}

function make10x10Cross(): number[][] {
  const grid: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));
  for (let i = 0; i < 10; i++) {
    grid[4][i] = 200;
    grid[5][i] = 200;
    grid[i][4] = 200;
    grid[i][5] = 200;
  }
  return grid;
}

// ── 8x8 random feature map for pooling tab ───────────────────────────────
function makeRandomFeatureMap(seed: number): number[][] {
  const rng = mulberry32(seed);
  return Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, () => Math.round(rng() * 255))
  );
}

// ── Convolution computation ──────────────────────────────────────────────
function computeConvolution(
  input: number[][],
  kernel: number[][],
  stride: number,
  padding: PaddingMode
): { output: number[][]; paddedInput: number[][] } {
  const inputH = input.length;
  const inputW = input[0].length;
  const kH = kernel.length;
  const kW = kernel[0].length;
  const padY = padding === "same" ? Math.floor(kH / 2) : 0;
  const padX = padding === "same" ? Math.floor(kW / 2) : 0;

  const paddedH = inputH + 2 * padY;
  const paddedW = inputW + 2 * padX;
  const paddedInput: number[][] = Array.from({ length: paddedH }, () =>
    Array(paddedW).fill(0)
  );
  for (let r = 0; r < inputH; r++)
    for (let c = 0; c < inputW; c++)
      paddedInput[r + padY][c + padX] = input[r][c];

  const outH = Math.floor((paddedH - kH) / stride) + 1;
  const outW = Math.floor((paddedW - kW) / stride) + 1;
  const output: number[][] = Array.from({ length: outH }, () => Array(outW).fill(0));

  for (let or_ = 0; or_ < outH; or_++) {
    for (let oc = 0; oc < outW; oc++) {
      let sum = 0;
      const startR = or_ * stride;
      const startC = oc * stride;
      for (let kr = 0; kr < kH; kr++)
        for (let kc = 0; kc < kW; kc++)
          sum += paddedInput[startR + kr][startC + kc] * kernel[kr][kc];
      output[or_][oc] = sum;
    }
  }
  return { output, paddedInput };
}

// Simple convolution (no stride/padding options) returning just output
function convolve(input: number[][], kernel: number[][]): number[][] {
  const H = input.length;
  const W = input[0].length;
  const kH = kernel.length;
  const kW = kernel[0].length;
  const oH = H - kH + 1;
  const oW = W - kW + 1;
  const out: number[][] = Array.from({ length: oH }, () => Array(oW).fill(0));
  for (let r = 0; r < oH; r++)
    for (let c = 0; c < oW; c++) {
      let s = 0;
      for (let kr = 0; kr < kH; kr++)
        for (let kc = 0; kc < kW; kc++)
          s += input[r + kr][c + kc] * kernel[kr][kc];
      out[r][c] = s;
    }
  return out;
}

// ── Pooling computation ──────────────────────────────────────────────────
function computePooling(
  input: number[][],
  poolSize: number,
  stride: number,
  type: PoolingType
): number[][] {
  const H = input.length;
  const W = input[0].length;
  const oH = Math.floor((H - poolSize) / stride) + 1;
  const oW = Math.floor((W - poolSize) / stride) + 1;
  const out: number[][] = Array.from({ length: oH }, () => Array(oW).fill(0));

  for (let r = 0; r < oH; r++) {
    for (let c = 0; c < oW; c++) {
      const vals: number[] = [];
      for (let pr = 0; pr < poolSize; pr++)
        for (let pc = 0; pc < poolSize; pc++)
          vals.push(input[r * stride + pr][c * stride + pc]);
      if (type === "max") out[r][c] = Math.max(...vals);
      else if (type === "min") out[r][c] = Math.min(...vals);
      else out[r][c] = Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
    }
  }
  return out;
}

// ── Color helpers ────────────────────────────────────────────────────────
function grayFill(v: number): string {
  const clamped = Math.max(0, Math.min(255, Math.round(v)));
  return `rgb(${clamped},${clamped},${clamped})`;
}

function grayTextColor(v: number): string {
  return v > 140 ? "#1e293b" : "#e2e8f0";
}

function divergingFill(value: number, maxAbs: number): string {
  if (maxAbs === 0) return "#ffffff";
  const n = value / maxAbs;
  if (n > 0.01) {
    const i = Math.min(1, Math.abs(n));
    return `rgb(${Math.round(255 - i * 196)},${Math.round(255 - i * 125)},255)`;
  }
  if (n < -0.01) {
    const i = Math.min(1, Math.abs(n));
    return `rgb(255,${Math.round(255 - i * 187)},${Math.round(255 - i * 187)})`;
  }
  return "#ffffff";
}

function kernelValueColor(v: number): string {
  if (v > 0) return "#1e40af";
  if (v < 0) return "#dc2626";
  return "#64748b";
}

function kernelCellBg(v: number): string {
  if (v > 0) return "#dbeafe";
  if (v < 0) return "#fee2e2";
  return "#f8fafc";
}

function heatFill(v: number, min: number, max: number): string {
  if (max === min) return "#94a3b8";
  const t = (v - min) / (max - min);
  const r = Math.round(255 * Math.min(1, t * 2));
  const g = Math.round(255 * Math.min(1, (1 - t) * 2));
  return `rgb(${r},${g},60)`;
}

function maxAbsOf(grid: number[][]): number {
  let m = 0;
  for (const row of grid) for (const v of row) m = Math.max(m, Math.abs(v));
  return m;
}

function minMaxOf(grid: number[][]): [number, number] {
  let mi = Infinity,
    ma = -Infinity;
  for (const row of grid)
    for (const v of row) {
      mi = Math.min(mi, v);
      ma = Math.max(ma, v);
    }
  return [mi, ma];
}

// ── Shared SVG grid renderers ────────────────────────────────────────────

/** Render a grayscale pixel grid (for input images) */
function PixelGrid({
  grid,
  cellSize,
  onCellClick,
  highlightRect,
  showValues = true,
  paddingCells,
  isPaddingCell,
}: {
  grid: number[][];
  cellSize: number;
  onCellClick?: (r: number, c: number) => void;
  highlightRect?: { r: number; c: number; h: number; w: number } | null;
  showValues?: boolean;
  paddingCells?: number;
  isPaddingCell?: (r: number, c: number) => boolean;
}) {
  const H = grid.length;
  const W = grid[0]?.length ?? 0;
  const svgW = W * (cellSize + GAP) + GAP;
  const svgH = H * (cellSize + GAP) + GAP;

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} width={svgW} height={svgH} className="block">
      {grid.map((row, r) =>
        row.map((val, c) => {
          const x = GAP + c * (cellSize + GAP);
          const y = GAP + r * (cellSize + GAP);
          const isPad = isPaddingCell ? isPaddingCell(r, c) : false;
          const inHighlight =
            highlightRect &&
            r >= highlightRect.r &&
            r < highlightRect.r + highlightRect.h &&
            c >= highlightRect.c &&
            c < highlightRect.c + highlightRect.w;

          return (
            <g key={`${r}-${c}`}>
              <rect
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                rx={2}
                fill={isPad ? "#f1f5f9" : grayFill(val)}
                stroke={inHighlight ? "#f59e0b" : isPad ? "#cbd5e1" : "#94a3b8"}
                strokeWidth={inHighlight ? 2 : isPad ? 0.5 : 0.5}
                strokeDasharray={isPad ? "2,1" : undefined}
                className={onCellClick && !isPad ? "cursor-pointer hover:opacity-80" : ""}
                onClick={onCellClick && !isPad ? () => onCellClick(r, c) : undefined}
              />
              {showValues && cellSize >= 20 && (
                <text
                  x={x + cellSize / 2}
                  y={y + cellSize / 2 + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={cellSize >= 28 ? 9 : 7}
                  fontFamily="monospace"
                  fontWeight={500}
                  fill={isPad ? "#94a3b8" : grayTextColor(val)}
                  pointerEvents="none"
                >
                  {Math.round(val)}
                </text>
              )}
            </g>
          );
        })
      )}
      {highlightRect && (
        <rect
          x={GAP + highlightRect.c * (cellSize + GAP) - 1}
          y={GAP + highlightRect.r * (cellSize + GAP) - 1}
          width={highlightRect.w * (cellSize + GAP) - GAP + 2}
          height={highlightRect.h * (cellSize + GAP) - GAP + 2}
          rx={3}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={2.5}
        />
      )}
    </svg>
  );
}

/** Render a diverging-color output grid */
function OutputGrid({
  grid,
  cellSize,
  maxAbs,
  highlightCell,
  revealUpTo,
}: {
  grid: number[][];
  cellSize: number;
  maxAbs: number;
  highlightCell?: { r: number; c: number } | null;
  revealUpTo?: number | null;
}) {
  const H = grid.length;
  const W = grid[0]?.length ?? 0;
  const svgW = W * (cellSize + GAP) + GAP;
  const svgH = H * (cellSize + GAP) + GAP;

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} width={svgW} height={svgH} className="block">
      {grid.map((row, r) =>
        row.map((val, c) => {
          const x = GAP + c * (cellSize + GAP);
          const y = GAP + r * (cellSize + GAP);
          const idx = r * W + c;
          const isRevealed = revealUpTo === null || revealUpTo === undefined || idx <= revealUpTo;
          const isActive =
            highlightCell && r === highlightCell.r && c === highlightCell.c;

          return (
            <g key={`${r}-${c}`}>
              <rect
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                rx={2}
                fill={isRevealed ? divergingFill(val, maxAbs) : "#f8fafc"}
                stroke={isActive ? "#f59e0b" : isRevealed ? "#94a3b8" : "#e2e8f0"}
                strokeWidth={isActive ? 2.5 : 0.5}
              />
              {cellSize >= 20 && (
                <text
                  x={x + cellSize / 2}
                  y={y + cellSize / 2 + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={cellSize >= 28 ? 8 : 7}
                  fontFamily="monospace"
                  fontWeight={500}
                  fill={isRevealed ? "#334155" : "#cbd5e1"}
                  pointerEvents="none"
                >
                  {isRevealed ? Math.round(val) : "?"}
                </text>
              )}
            </g>
          );
        })
      )}
    </svg>
  );
}

/** Render an editable kernel */
function KernelGrid({
  kernel,
  cellSize,
  onCellClick,
  highlight,
  label,
}: {
  kernel: number[][];
  cellSize: number;
  onCellClick?: (r: number, c: number, dir: 1 | -1) => void;
  highlight?: boolean;
  label?: string;
}) {
  const H = kernel.length;
  const W = kernel[0].length;
  const svgW = W * (cellSize + GAP) + GAP;
  const svgH = H * (cellSize + GAP) + GAP;

  return (
    <div>
      {label && (
        <p className="text-[10px] text-slate-500 font-semibold mb-1 text-center uppercase tracking-wide">
          {label}
        </p>
      )}
      <svg viewBox={`0 0 ${svgW} ${svgH}`} width={svgW} height={svgH} className="block">
        {kernel.map((row, r) =>
          row.map((v, c) => {
            const x = GAP + c * (cellSize + GAP);
            const y = GAP + r * (cellSize + GAP);
            return (
              <g key={`${r}-${c}`} className={onCellClick ? "cursor-pointer" : ""}>
                <rect
                  x={x}
                  y={y}
                  width={cellSize}
                  height={cellSize}
                  rx={3}
                  fill={kernelCellBg(v)}
                  stroke={highlight ? "#f59e0b" : "#a78bfa"}
                  strokeWidth={highlight ? 2 : 1}
                  onClick={onCellClick ? () => onCellClick(r, c, 1) : undefined}
                  onContextMenu={
                    onCellClick
                      ? (e) => {
                          e.preventDefault();
                          onCellClick(r, c, -1);
                        }
                      : undefined
                  }
                />
                <text
                  x={x + cellSize / 2}
                  y={y + cellSize / 2 + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={cellSize >= 36 ? 14 : cellSize >= 28 ? 11 : 9}
                  fontFamily="monospace"
                  fontWeight={700}
                  fill={kernelValueColor(v)}
                  pointerEvents="none"
                >
                  {v}
                </text>
              </g>
            );
          })
        )}
      </svg>
    </div>
  );
}

/** A small value-color grid used in pooling and feature map tabs */
function ValueGrid({
  grid,
  cellSize,
  colorFn,
  highlightRect,
  highlightCell,
  showValues = true,
}: {
  grid: number[][];
  cellSize: number;
  colorFn: (v: number) => string;
  highlightRect?: { r: number; c: number; h: number; w: number } | null;
  highlightCell?: { r: number; c: number } | null;
  showValues?: boolean;
}) {
  const H = grid.length;
  const W = grid[0]?.length ?? 0;
  const svgW = W * (cellSize + GAP) + GAP;
  const svgH = H * (cellSize + GAP) + GAP;

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} width={svgW} height={svgH} className="block">
      {grid.map((row, r) =>
        row.map((val, c) => {
          const x = GAP + c * (cellSize + GAP);
          const y = GAP + r * (cellSize + GAP);
          const inHl =
            highlightRect &&
            r >= highlightRect.r &&
            r < highlightRect.r + highlightRect.h &&
            c >= highlightRect.c &&
            c < highlightRect.c + highlightRect.w;
          const isCell =
            highlightCell && r === highlightCell.r && c === highlightCell.c;

          return (
            <g key={`${r}-${c}`}>
              <rect
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                rx={2}
                fill={colorFn(val)}
                stroke={inHl || isCell ? "#f59e0b" : "#94a3b8"}
                strokeWidth={inHl || isCell ? 2 : 0.5}
              />
              {showValues && cellSize >= 18 && (
                <text
                  x={x + cellSize / 2}
                  y={y + cellSize / 2 + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={cellSize >= 28 ? 9 : 7}
                  fontFamily="monospace"
                  fontWeight={600}
                  fill="#1e293b"
                  pointerEvents="none"
                >
                  {Math.round(val)}
                </text>
              )}
            </g>
          );
        })
      )}
      {highlightRect && (
        <rect
          x={GAP + highlightRect.c * (cellSize + GAP) - 1}
          y={GAP + highlightRect.r * (cellSize + GAP) - 1}
          width={highlightRect.w * (cellSize + GAP) - GAP + 2}
          height={highlightRect.h * (cellSize + GAP) - GAP + 2}
          rx={3}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={2.5}
        />
      )}
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 1: Apply Filters
// ═══════════════════════════════════════════════════════════════════════════
function ApplyFiltersTab() {
  const [inputGrid, setInputGrid] = useState<number[][]>(makeSquare);
  const [kernel, setKernel] = useState<number[][]>(FILTER_PRESETS[0].kernel.map((r) => [...r]));
  const [activePreset, setActivePreset] = useState("Identity");
  const [paintValue, setPaintValue] = useState(255);
  const [stepIndex, setStepIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { output, paddedInput } = useMemo(
    () => computeConvolution(inputGrid, kernel, 1, "valid"),
    [inputGrid, kernel]
  );

  const outputH = output.length;
  const outputW = output[0]?.length ?? 0;
  const totalSteps = outputH * outputW;
  const maxAbs = useMemo(() => maxAbsOf(output), [output]);

  // Step info for current animation position
  const stepInfo = useMemo(() => {
    if (stepIndex === null || stepIndex >= totalSteps) return null;
    const outRow = Math.floor(stepIndex / outputW);
    const outCol = stepIndex % outputW;
    const products: number[][] = [];
    let sum = 0;
    for (let kr = 0; kr < 3; kr++) {
      const row: number[] = [];
      for (let kc = 0; kc < 3; kc++) {
        const iv = paddedInput[outRow + kr]?.[outCol + kc] ?? 0;
        const kv = kernel[kr][kc];
        const prod = iv * kv;
        row.push(prod);
        sum += prod;
      }
      products.push(row);
    }
    return { outRow, outCol, startR: outRow, startC: outCol, products, sum };
  }, [stepIndex, totalSteps, outputW, paddedInput, kernel]);

  // Auto-play cleanup
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const stopPlaying = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const handleAutoPlay = useCallback(() => {
    if (isPlaying) {
      stopPlaying();
      return;
    }
    setIsPlaying(true);
    setStepIndex(0);
    let cur = 0;
    timerRef.current = setInterval(() => {
      cur++;
      if (cur >= totalSteps) {
        stopPlaying();
        setStepIndex(null);
        return;
      }
      setStepIndex(cur);
    }, 350);
  }, [isPlaying, totalSteps, stopPlaying]);

  const stepForward = useCallback(() => {
    setStepIndex((prev) => {
      if (prev === null) return 0;
      return prev >= totalSteps - 1 ? 0 : prev + 1;
    });
  }, [totalSteps]);

  const handleInputClick = useCallback(
    (r: number, c: number) => {
      setInputGrid((prev) => {
        const next = prev.map((row) => [...row]);
        next[r][c] = next[r][c] === paintValue ? 0 : paintValue;
        return next;
      });
    },
    [paintValue]
  );

  const handleKernelClick = useCallback((r: number, c: number, dir: 1 | -1) => {
    setKernel((prev) => {
      const next = prev.map((row) => [...row]);
      const v = next[r][c];
      if (dir === 1) next[r][c] = v >= 5 ? -3 : v + 1;
      else next[r][c] = v <= -3 ? 5 : v - 1;
      return next;
    });
    setActivePreset("");
  }, []);

  const loadPreset = useCallback((p: FilterPreset) => {
    setKernel(p.kernel.map((r) => [...r]));
    setActivePreset(p.name);
    setStepIndex(null);
    stopPlaying();
  }, [stopPlaying]);

  const loadImage = useCallback((gen: () => number[][]) => {
    setInputGrid(gen());
    setStepIndex(null);
    stopPlaying();
  }, [stopPlaying]);

  const kernelSum = useMemo(() => kernel.flat().reduce((a, b) => a + b, 0), [kernel]);

  return (
    <div className="space-y-4">
      {/* Image presets */}
      <div className="flex flex-wrap gap-1.5 items-center">
        <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide mr-1">
          Image:
        </span>
        {IMAGE_PRESETS_8.map((p) => (
          <button
            key={p.name}
            onClick={() => loadImage(p.generator)}
            className="px-2.5 py-1 rounded text-[11px] font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors"
          >
            {p.name}
          </button>
        ))}
        <button
          onClick={() => {
            setInputGrid(Array.from({ length: 8 }, () => Array(8).fill(0)));
            setStepIndex(null);
            stopPlaying();
          }}
          className="px-2.5 py-1 rounded text-[11px] font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors flex items-center gap-1"
        >
          <RotateCcw className="w-3 h-3" /> Clear
        </button>
      </div>

      <div className="flex gap-5 flex-col lg:flex-row">
        {/* Input + kernel + output grids */}
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap gap-4 items-start justify-center">
            {/* Input */}
            <div>
              <p className="text-[10px] text-slate-500 font-semibold mb-1 uppercase tracking-wide flex items-center gap-1">
                <Grid3X3 className="w-3 h-3" /> Input (8x8)
              </p>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
                <PixelGrid
                  grid={inputGrid}
                  cellSize={CELL}
                  onCellClick={handleInputClick}
                  highlightRect={
                    stepInfo
                      ? { r: stepInfo.startR, c: stepInfo.startC, h: 3, w: 3 }
                      : null
                  }
                  showValues
                />
                <div className="mt-2 flex items-center gap-1.5">
                  <span className="text-[9px] text-slate-400 font-medium">Paint:</span>
                  {[0, 80, 150, 220, 255].map((v) => (
                    <button
                      key={v}
                      onClick={() => setPaintValue(v)}
                      className={`w-5 h-5 rounded border-2 transition-all ${
                        paintValue === v ? "border-amber-400 scale-110" : "border-slate-300"
                      }`}
                      style={{ backgroundColor: grayFill(v) }}
                      title={`${v}`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Convolution symbol */}
            <div className="flex items-center self-center text-slate-400 text-2xl font-bold">*</div>

            {/* Kernel */}
            <div>
              <p className="text-[10px] text-slate-500 font-semibold mb-1 uppercase tracking-wide flex items-center gap-1">
                <Layers className="w-3 h-3" /> Kernel (3x3)
              </p>
              <div className="bg-violet-50/50 border border-violet-200 rounded-lg p-2">
                <KernelGrid
                  kernel={kernel}
                  cellSize={KERNEL_CELL}
                  onCellClick={handleKernelClick}
                  highlight={stepInfo !== null}
                />
                <p className="text-[9px] text-slate-400 mt-1 text-center">
                  Click +1 / right-click -1
                </p>
                {kernelSum !== 0 && kernelSum !== 1 && (
                  <p className="text-[9px] text-amber-600 text-center mt-0.5">
                    Sum = {kernelSum}
                    {activePreset === "Blur" && " (divide by 9 for average)"}
                  </p>
                )}
              </div>
            </div>

            {/* Equals symbol */}
            <div className="flex items-center self-center text-slate-400 text-2xl font-bold">=</div>

            {/* Output */}
            <div>
              <p className="text-[10px] text-slate-500 font-semibold mb-1 uppercase tracking-wide flex items-center gap-1">
                <Grid3X3 className="w-3 h-3" /> Output ({outputW}x{outputH})
              </p>
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
                <OutputGrid
                  grid={output}
                  cellSize={CELL}
                  maxAbs={maxAbs}
                  highlightCell={stepInfo ? { r: stepInfo.outRow, c: stepInfo.outCol } : null}
                  revealUpTo={stepIndex}
                />
                <div className="flex items-center justify-center gap-3 mt-2">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-sm bg-red-400" />
                    <span className="text-[9px] text-slate-500">Neg</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-sm bg-white border border-slate-200" />
                    <span className="text-[9px] text-slate-500">Zero</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-sm bg-blue-400" />
                    <span className="text-[9px] text-slate-500">Pos</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Step-through detail */}
          {stepInfo && (
            <div className="mt-4 bg-amber-50 border border-amber-200 rounded-lg p-3">
              <p className="text-[11px] text-amber-800 font-semibold mb-2">
                Step {(stepIndex ?? 0) + 1} / {totalSteps} — Output[{stepInfo.outRow}][{stepInfo.outCol}]
              </p>
              <div className="flex flex-wrap items-center gap-1 text-xs font-mono text-slate-700">
                <span className="text-amber-700 font-bold">
                  out[{stepInfo.outRow}][{stepInfo.outCol}]
                </span>
                <span>=</span>
                {stepInfo.products.flatMap((row, kr) =>
                  row.map((prod, kc) => {
                    const iv = paddedInput[stepInfo.startR + kr]?.[stepInfo.startC + kc] ?? 0;
                    const kv = kernel[kr][kc];
                    const idx = kr * 3 + kc;
                    return (
                      <span key={`${kr}-${kc}`} className="whitespace-nowrap">
                        {idx > 0 && <span className="text-slate-400 mx-0.5">+</span>}
                        <span className="text-slate-500">({Math.round(iv)}</span>
                        <span className="text-violet-600 font-semibold">x{kv}</span>
                        <span className="text-slate-500">)</span>
                      </span>
                    );
                  })
                )}
                <span>=</span>
                <span className="text-emerald-700 font-bold text-sm">{Math.round(stepInfo.sum)}</span>
              </div>

              <div className="mt-3 flex items-center gap-4">
                <div>
                  <p className="text-[9px] text-slate-500 font-medium mb-1">Element-wise products:</p>
                  <div className="grid grid-cols-3 gap-1">
                    {stepInfo.products.flat().map((prod, i) => (
                      <div
                        key={i}
                        className="w-12 h-7 flex items-center justify-center rounded text-[10px] font-mono font-semibold"
                        style={{
                          backgroundColor: prod > 0 ? "#dbeafe" : prod < 0 ? "#fee2e2" : "#f1f5f9",
                          color: prod > 0 ? "#1e40af" : prod < 0 ? "#dc2626" : "#64748b",
                        }}
                      >
                        {Math.round(prod)}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="text-center">
                  <p className="text-[9px] text-slate-500 font-medium mb-1">Sum</p>
                  <div
                    className="w-14 h-14 rounded-lg flex items-center justify-center text-sm font-bold font-mono border-2"
                    style={{
                      backgroundColor: stepInfo.sum > 0 ? "#dbeafe" : stepInfo.sum < 0 ? "#fee2e2" : "#f1f5f9",
                      borderColor: stepInfo.sum > 0 ? "#3b82f6" : stepInfo.sum < 0 ? "#ef4444" : "#94a3b8",
                      color: stepInfo.sum > 0 ? "#1e40af" : stepInfo.sum < 0 ? "#dc2626" : "#334155",
                    }}
                  >
                    {Math.round(stepInfo.sum)}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar: presets + animation */}
        <div className="w-full lg:w-60 space-y-3">
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[10px] text-slate-500 uppercase tracking-wide font-semibold mb-2">
              Filter Presets
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {FILTER_PRESETS.map((p) => (
                <button
                  key={p.name}
                  onClick={() => loadPreset(p)}
                  className={`px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all text-left ${
                    activePreset === p.name
                      ? "bg-violet-500 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {p.name}
                </button>
              ))}
            </div>
            {activePreset && (
              <p className="text-[9px] text-slate-500 mt-2 italic">
                {FILTER_PRESETS.find((p) => p.name === activePreset)?.description}
              </p>
            )}
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <p className="text-[10px] text-slate-500 uppercase tracking-wide font-semibold mb-2">
              Animation
            </p>
            <div className="flex gap-2">
              <button
                onClick={handleAutoPlay}
                className={`flex-1 flex items-center justify-center gap-1 px-2 py-2 rounded-lg text-[11px] font-semibold transition-all ${
                  isPlaying ? "bg-amber-500 text-white" : "bg-violet-500 text-white hover:bg-violet-600"
                }`}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {isPlaying ? "Stop" : "Play"}
              </button>
              <button
                onClick={stepForward}
                disabled={isPlaying}
                className="flex-1 flex items-center justify-center gap-1 px-2 py-2 rounded-lg text-[11px] font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-40 transition-all"
              >
                <SkipForward className="w-3.5 h-3.5" /> Step
              </button>
            </div>
            {stepIndex !== null && (
              <div className="mt-2">
                <div className="flex justify-between text-[9px] text-slate-500 mb-0.5">
                  <span>Step {stepIndex + 1}/{totalSteps}</span>
                  <span>{Math.round(((stepIndex + 1) / totalSteps) * 100)}%</span>
                </div>
                <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-violet-500 rounded-full transition-all"
                    style={{ width: `${((stepIndex + 1) / totalSteps) * 100}%` }}
                  />
                </div>
              </div>
            )}
            <button
              onClick={() => {
                setStepIndex(null);
                stopPlaying();
              }}
              className="w-full mt-2 flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg text-[10px] font-medium bg-slate-50 text-slate-500 hover:bg-slate-100 transition-all"
            >
              <RotateCcw className="w-3 h-3" /> Reset
            </button>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-[10px] text-slate-500 uppercase tracking-wide font-semibold mb-1">
              Convolution Info
            </p>
            <div className="space-y-1 text-[11px] text-slate-600">
              <div className="flex justify-between">
                <span>Input</span><span className="font-mono font-bold">8x8</span>
              </div>
              <div className="flex justify-between">
                <span>Kernel</span><span className="font-mono font-bold">3x3</span>
              </div>
              <div className="flex justify-between">
                <span>Padding</span><span className="font-mono font-bold">valid (0)</span>
              </div>
              <div className="flex justify-between border-t border-slate-200 pt-1">
                <span className="font-medium">Output</span>
                <span className="font-mono font-bold text-emerald-600">{outputW}x{outputH}</span>
              </div>
            </div>
            <div className="mt-2 p-1.5 bg-white rounded text-[9px] font-mono text-slate-400 text-center">
              O = (W - K + 2P)/S + 1 = (8-3+0)/1 + 1 = {outputW}
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-2.5">
            <p className="text-[11px] text-amber-800 font-semibold mb-1">Try this:</p>
            <ul className="text-[10px] text-amber-700 space-y-0.5 list-disc list-inside">
              <li>Compare Edge(H) vs Edge(V) on H-Lines</li>
              <li>Apply Sharpen to the Square pattern</li>
              <li>Paint your own pattern and experiment</li>
              <li>Use Step to see each multiplication</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 2: Edge Detection Deep Dive
// ═══════════════════════════════════════════════════════════════════════════
function EdgeDetectionTab() {
  const [imageKey, setImageKey] = useState("square");
  const [showSobelX, setShowSobelX] = useState(true);
  const [showSobelY, setShowSobelY] = useState(true);
  const [showLaplacian, setShowLaplacian] = useState(true);

  const imageOptions: { key: string; label: string; gen: () => number[][] }[] = useMemo(
    () => [
      { key: "square", label: "Square", gen: makeSquare },
      { key: "hlines", label: "H-Lines", gen: makeHorizontalLines },
      { key: "vlines", label: "V-Lines", gen: makeVerticalLines },
      { key: "diagonal", label: "Diagonal", gen: makeDiagonal },
      { key: "circle", label: "Circle", gen: makeCircle },
      { key: "checker", label: "Checker", gen: makeCheckerboard },
    ],
    []
  );

  const inputGrid = useMemo(() => {
    const opt = imageOptions.find((o) => o.key === imageKey);
    return opt ? opt.gen() : makeSquare();
  }, [imageKey, imageOptions]);

  const sobelXOut = useMemo(() => convolve(inputGrid, SOBEL_X), [inputGrid]);
  const sobelYOut = useMemo(() => convolve(inputGrid, SOBEL_Y), [inputGrid]);
  const laplacianOut = useMemo(() => convolve(inputGrid, LAPLACIAN), [inputGrid]);

  // Edge magnitude: sqrt(Gx^2 + Gy^2)
  const edgeMagnitude = useMemo(() => {
    const H = sobelXOut.length;
    const W = sobelXOut[0].length;
    return Array.from({ length: H }, (_, r) =>
      Array.from({ length: W }, (_, c) =>
        Math.sqrt(sobelXOut[r][c] ** 2 + sobelYOut[r][c] ** 2)
      )
    );
  }, [sobelXOut, sobelYOut]);

  const maxAbsSX = useMemo(() => maxAbsOf(sobelXOut), [sobelXOut]);
  const maxAbsSY = useMemo(() => maxAbsOf(sobelYOut), [sobelYOut]);
  const maxAbsLap = useMemo(() => maxAbsOf(laplacianOut), [laplacianOut]);
  const maxAbsMag = useMemo(() => maxAbsOf(edgeMagnitude), [edgeMagnitude]);

  const explanations: Record<string, string> = {
    "Sobel-X":
      "Detects vertical edges (intensity changes left-right). The kernel has negative values on the left and positive on the right, computing the horizontal gradient.",
    "Sobel-Y":
      "Detects horizontal edges (intensity changes top-bottom). The kernel has negative values on top and positive on bottom, computing the vertical gradient.",
    Laplacian:
      "Detects edges in all directions by computing the second derivative. It highlights regions of rapid intensity change regardless of orientation.",
  };

  return (
    <div className="space-y-4">
      {/* Image selector */}
      <div className="flex flex-wrap gap-1.5 items-center">
        <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide mr-1">
          Input Image:
        </span>
        {imageOptions.map((opt) => (
          <button
            key={opt.key}
            onClick={() => setImageKey(opt.key)}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
              imageKey === opt.key
                ? "bg-violet-500 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Toggle buttons */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide mr-1">
          Show:
        </span>
        {[
          { label: "Sobel-X", active: showSobelX, toggle: () => setShowSobelX((v) => !v), color: "blue" },
          { label: "Sobel-Y", active: showSobelY, toggle: () => setShowSobelY((v) => !v), color: "green" },
          { label: "Laplacian", active: showLaplacian, toggle: () => setShowLaplacian((v) => !v), color: "purple" },
        ].map((btn) => (
          <button
            key={btn.label}
            onClick={btn.toggle}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors flex items-center gap-1 ${
              btn.active
                ? btn.color === "blue"
                  ? "bg-blue-500 text-white"
                  : btn.color === "green"
                  ? "bg-green-500 text-white"
                  : "bg-purple-500 text-white"
                : "bg-slate-100 text-slate-500"
            }`}
          >
            {btn.active ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            {btn.label}
          </button>
        ))}
      </div>

      {/* Grids */}
      <div className="flex flex-wrap gap-4 items-start justify-center">
        {/* Input */}
        <div>
          <p className="text-[10px] text-slate-500 font-semibold mb-1 text-center uppercase tracking-wide">
            Input (8x8)
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
            <PixelGrid grid={inputGrid} cellSize={CELL_SM} showValues />
          </div>
        </div>

        <div className="flex items-center self-center">
          <ChevronRight className="w-5 h-5 text-slate-400" />
        </div>

        {/* Sobel-X */}
        {showSobelX && (
          <div>
            <p className="text-[10px] text-blue-600 font-semibold mb-1 text-center uppercase tracking-wide">
              Sobel-X (6x6)
            </p>
            <div className="bg-blue-50/50 border border-blue-200 rounded-lg p-2">
              <OutputGrid grid={sobelXOut} cellSize={CELL_SM} maxAbs={maxAbsSX} />
            </div>
            <div className="mt-1">
              <KernelGrid kernel={SOBEL_X} cellSize={22} label="Kernel" />
            </div>
          </div>
        )}

        {/* Sobel-Y */}
        {showSobelY && (
          <div>
            <p className="text-[10px] text-green-600 font-semibold mb-1 text-center uppercase tracking-wide">
              Sobel-Y (6x6)
            </p>
            <div className="bg-green-50/50 border border-green-200 rounded-lg p-2">
              <OutputGrid grid={sobelYOut} cellSize={CELL_SM} maxAbs={maxAbsSY} />
            </div>
            <div className="mt-1">
              <KernelGrid kernel={SOBEL_Y} cellSize={22} label="Kernel" />
            </div>
          </div>
        )}

        {/* Laplacian */}
        {showLaplacian && (
          <div>
            <p className="text-[10px] text-purple-600 font-semibold mb-1 text-center uppercase tracking-wide">
              Laplacian (6x6)
            </p>
            <div className="bg-purple-50/50 border border-purple-200 rounded-lg p-2">
              <OutputGrid grid={laplacianOut} cellSize={CELL_SM} maxAbs={maxAbsLap} />
            </div>
            <div className="mt-1">
              <KernelGrid kernel={LAPLACIAN} cellSize={22} label="Kernel" />
            </div>
          </div>
        )}
      </div>

      {/* Combined edge magnitude */}
      {showSobelX && showSobelY && (
        <div className="bg-gradient-to-r from-blue-50 to-green-50 border border-slate-200 rounded-lg p-4">
          <p className="text-xs font-semibold text-slate-700 mb-2">
            Combined Edge Magnitude: sqrt(Gx² + Gy²)
          </p>
          <div className="flex flex-wrap gap-4 items-center justify-center">
            <div>
              <p className="text-[9px] text-slate-500 text-center mb-1">Sobel-X²</p>
              <OutputGrid
                grid={sobelXOut.map((row) => row.map((v) => v * v))}
                cellSize={CELL_XS}
                maxAbs={maxAbsSX * maxAbsSX}
              />
            </div>
            <span className="text-slate-400 text-lg font-bold self-center">+</span>
            <div>
              <p className="text-[9px] text-slate-500 text-center mb-1">Sobel-Y²</p>
              <OutputGrid
                grid={sobelYOut.map((row) => row.map((v) => v * v))}
                cellSize={CELL_XS}
                maxAbs={maxAbsSY * maxAbsSY}
              />
            </div>
            <span className="text-slate-400 text-lg font-bold self-center">=</span>
            <div>
              <p className="text-[9px] text-emerald-600 font-bold text-center mb-1">Edge Map</p>
              <OutputGrid grid={edgeMagnitude} cellSize={CELL_XS} maxAbs={maxAbsMag} />
            </div>
          </div>
          <p className="text-[10px] text-slate-500 mt-3 text-center">
            The edge magnitude combines both horizontal and vertical gradients, detecting edges in all orientations.
          </p>
        </div>
      )}

      {/* Explanations */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {showSobelX && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-[11px] text-blue-800 font-semibold mb-1">Sobel-X</p>
            <p className="text-[10px] text-blue-700">{explanations["Sobel-X"]}</p>
          </div>
        )}
        {showSobelY && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <p className="text-[11px] text-green-800 font-semibold mb-1">Sobel-Y</p>
            <p className="text-[10px] text-green-700">{explanations["Sobel-Y"]}</p>
          </div>
        )}
        {showLaplacian && (
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
            <p className="text-[11px] text-purple-800 font-semibold mb-1">Laplacian</p>
            <p className="text-[10px] text-purple-700">{explanations.Laplacian}</p>
          </div>
        )}
      </div>

      {/* Which edges each kernel detects */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <p className="text-xs font-semibold text-slate-700 mb-2">
          Why do these kernels detect edges?
        </p>
        <div className="space-y-2 text-[11px] text-slate-600">
          <p>
            <strong>Edge detection kernels</strong> are designed so that when the kernel overlaps a
            region of uniform intensity, the positive and negative weights cancel out (sum near zero).
            But when the kernel straddles a boundary between light and dark pixels, one side produces
            large positive products while the other produces large negative products — the
            imbalance yields a large output.
          </p>
          <p>
            <strong>Sobel-X</strong> has negative values on the left, positive on the right.
            It responds strongly to vertical edges (where left differs from right).{" "}
            <strong>Sobel-Y</strong> has negative values on top, positive on bottom — it detects
            horizontal edges. The <strong>Laplacian</strong> uses a center-surround pattern to
            detect edges in any direction by measuring how different a pixel is from its neighbors.
          </p>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 3: Stride & Padding
// ═══════════════════════════════════════════════════════════════════════════
function StridePaddingTab() {
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState<PaddingMode>("valid");
  const [stepIndex, setStepIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [imageKey, setImageKey] = useState("square");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const kernel: number[][] = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
  ]; // Sobel-X for demonstration

  const inputGrid = useMemo(
    () => (imageKey === "square" ? make10x10Square() : make10x10Cross()),
    [imageKey]
  );

  const { output, paddedInput } = useMemo(
    () => computeConvolution(inputGrid, kernel, stride, padding),
    [inputGrid, kernel, stride, padding]
  );

  const padAmount = padding === "same" ? 1 : 0;
  const paddedH = inputGrid.length + 2 * padAmount;
  const paddedW = inputGrid[0].length + 2 * padAmount;
  const outputH = output.length;
  const outputW = output[0]?.length ?? 0;
  const totalSteps = outputH * outputW;
  const maxAbs = useMemo(() => maxAbsOf(output), [output]);

  // Also compute stride=1 and stride=2 for comparison
  const out1 = useMemo(() => computeConvolution(inputGrid, kernel, 1, padding).output, [inputGrid, kernel, padding]);
  const out2 = useMemo(() => computeConvolution(inputGrid, kernel, 2, padding).output, [inputGrid, kernel, padding]);
  const maxAbs1 = useMemo(() => maxAbsOf(out1), [out1]);
  const maxAbs2 = useMemo(() => maxAbsOf(out2), [out2]);

  const stepInfo = useMemo(() => {
    if (stepIndex === null || stepIndex >= totalSteps) return null;
    const outRow = Math.floor(stepIndex / outputW);
    const outCol = stepIndex % outputW;
    return {
      outRow,
      outCol,
      startR: outRow * stride,
      startC: outCol * stride,
    };
  }, [stepIndex, totalSteps, outputW, stride]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const stopPlaying = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const handleAutoPlay = useCallback(() => {
    if (isPlaying) {
      stopPlaying();
      return;
    }
    setIsPlaying(true);
    setStepIndex(0);
    let cur = 0;
    timerRef.current = setInterval(() => {
      cur++;
      if (cur >= totalSteps) {
        stopPlaying();
        setStepIndex(null);
        return;
      }
      setStepIndex(cur);
    }, 300);
  }, [isPlaying, totalSteps, stopPlaying]);

  // Reset animation when stride/padding changes
  useEffect(() => {
    setStepIndex(null);
    stopPlaying();
  }, [stride, padding, stopPlaying]);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 font-semibold uppercase">Image:</span>
          {[
            { key: "square", label: "Square" },
            { key: "cross", label: "Cross" },
          ].map((opt) => (
            <button
              key={opt.key}
              onClick={() => setImageKey(opt.key)}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
                imageKey === opt.key
                  ? "bg-slate-800 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 font-semibold uppercase">Stride:</span>
          {[1, 2, 3].map((s) => (
            <button
              key={s}
              onClick={() => setStride(s)}
              className={`w-8 h-8 rounded-lg text-[11px] font-bold transition-all ${
                stride === s
                  ? "bg-violet-500 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 font-semibold uppercase">Padding:</span>
          {(["valid", "same"] as PaddingMode[]).map((p) => (
            <button
              key={p}
              onClick={() => setPadding(p)}
              className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-all capitalize ${
                padding === p
                  ? "bg-violet-500 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {p === "valid" ? "Valid (none)" : "Same (zero pad)"}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={handleAutoPlay}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-all ${
              isPlaying ? "bg-amber-500 text-white" : "bg-violet-500 text-white hover:bg-violet-600"
            }`}
          >
            {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
            {isPlaying ? "Stop" : "Animate"}
          </button>
          <button
            onClick={() =>
              setStepIndex((prev) =>
                prev === null ? 0 : prev >= totalSteps - 1 ? 0 : prev + 1
              )
            }
            disabled={isPlaying}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-40 transition-all"
          >
            <SkipForward className="w-3 h-3" /> Step
          </button>
        </div>
      </div>

      {/* Output size formula */}
      <div className="bg-gradient-to-r from-violet-50 to-indigo-50 border border-violet-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-violet-800 mb-2">Output Size Formula</p>
        <div className="font-mono text-sm text-violet-700 text-center">
          O = floor((W - F + 2P) / S) + 1
        </div>
        <div className="font-mono text-xs text-violet-600 text-center mt-1">
          = floor(({inputGrid[0].length} - 3 + 2x{padAmount}) / {stride}) + 1 ={" "}
          <span className="font-bold text-emerald-600 text-sm">{outputW}</span>
        </div>
        <div className="flex justify-center gap-6 mt-3 text-[11px] text-slate-600">
          <span>
            W={inputGrid[0].length} (input)
          </span>
          <span>F=3 (kernel)</span>
          <span>P={padAmount} (padding)</span>
          <span>S={stride} (stride)</span>
        </div>
        <div className="mt-2 text-center">
          <span className="inline-block bg-white border border-violet-200 rounded-lg px-4 py-2 font-bold text-violet-800 text-sm">
            Input: {paddedW}x{paddedH}
            {padding === "same" && <span className="text-slate-400 font-normal"> (padded)</span>}
            {" "} → Output: {outputW}x{outputH}
          </span>
        </div>
      </div>

      {/* Main grids */}
      <div className="flex flex-wrap gap-5 items-start justify-center">
        {/* Padded input */}
        <div>
          <p className="text-[10px] text-slate-500 font-semibold mb-1 text-center uppercase tracking-wide">
            {padding === "same" ? `Padded Input (${paddedW}x${paddedH})` : `Input (${inputGrid[0].length}x${inputGrid.length})`}
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
            <PixelGrid
              grid={paddedInput}
              cellSize={CELL_SM}
              highlightRect={
                stepInfo ? { r: stepInfo.startR, c: stepInfo.startC, h: 3, w: 3 } : null
              }
              showValues
              isPaddingCell={
                padding === "same"
                  ? (r, c) =>
                      r < padAmount ||
                      r >= paddedH - padAmount ||
                      c < padAmount ||
                      c >= paddedW - padAmount
                  : undefined
              }
            />
            {padding === "same" && (
              <p className="text-[9px] text-slate-400 mt-1 text-center">
                Dashed border = zero-padded cells
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center self-center">
          <ChevronRight className="w-5 h-5 text-slate-400" />
        </div>

        {/* Kernel */}
        <div>
          <p className="text-[10px] text-slate-500 font-semibold mb-1 text-center uppercase tracking-wide">
            Kernel (Sobel-X)
          </p>
          <div className="bg-violet-50/50 border border-violet-200 rounded-lg p-2">
            <KernelGrid kernel={kernel} cellSize={36} highlight={stepInfo !== null} />
          </div>
          <p className="text-[9px] text-slate-400 mt-1 text-center">Stride = {stride}</p>
        </div>

        <div className="flex items-center self-center">
          <ChevronRight className="w-5 h-5 text-slate-400" />
        </div>

        {/* Output */}
        <div>
          <p className="text-[10px] text-slate-500 font-semibold mb-1 text-center uppercase tracking-wide">
            Output ({outputW}x{outputH})
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
            <OutputGrid
              grid={output}
              cellSize={CELL_SM}
              maxAbs={maxAbs}
              highlightCell={stepInfo ? { r: stepInfo.outRow, c: stepInfo.outCol } : null}
              revealUpTo={stepIndex}
            />
          </div>
        </div>
      </div>

      {/* Step progress */}
      {stepIndex !== null && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500">
            Step {stepIndex + 1}/{totalSteps}
          </span>
          <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-violet-500 rounded-full transition-all"
              style={{ width: `${((stepIndex + 1) / totalSteps) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Side-by-side stride comparison */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-slate-700 mb-3">
          Stride Comparison (same kernel, same {padding} padding)
        </p>
        <div className="flex flex-wrap gap-6 items-start justify-center">
          <div>
            <p className="text-[10px] text-blue-600 font-semibold mb-1 text-center">
              Stride=1 → {out1[0]?.length ?? 0}x{out1.length}
            </p>
            <OutputGrid grid={out1} cellSize={CELL_XS} maxAbs={maxAbs1} />
          </div>
          <div>
            <p className="text-[10px] text-violet-600 font-semibold mb-1 text-center">
              Stride=2 → {out2[0]?.length ?? 0}x{out2.length}
            </p>
            <OutputGrid grid={out2} cellSize={CELL_XS} maxAbs={maxAbs2} />
          </div>
        </div>
        <p className="text-[10px] text-slate-500 mt-3 text-center">
          Larger stride = smaller output = more aggressive downsampling. Stride=2 reduces each
          spatial dimension by roughly half.
        </p>
      </div>

      {/* Explanation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <p className="text-[11px] text-blue-800 font-semibold mb-1">What is Stride?</p>
          <p className="text-[10px] text-blue-700">
            Stride controls how many pixels the kernel moves at each step. With stride=1 the kernel
            shifts one pixel, overlapping heavily. With stride=2, it jumps two pixels, producing a
            smaller output. Larger strides reduce computation and spatial dimensions, acting as a
            form of downsampling.
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-3">
          <p className="text-[11px] text-green-800 font-semibold mb-1">What is Padding?</p>
          <p className="text-[10px] text-green-700">
            Padding adds extra pixels (usually zeros) around the input border. "Valid" means no
            padding, so the output shrinks. "Same" adds enough zeros so that (with stride=1) the
            output is the same size as the input. Padding preserves spatial dimensions and ensures
            border pixels are fully covered by the kernel.
          </p>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 4: Pooling Operations
// ═══════════════════════════════════════════════════════════════════════════
function PoolingTab() {
  const [poolSize, setPoolSize] = useState(2);
  const [poolStride, setPoolStride] = useState(2);
  const [stepIndex, setStepIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [seed, setSeed] = useState(42);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const inputGrid = useMemo(() => makeRandomFeatureMap(seed), [seed]);
  const [minVal, maxVal] = useMemo(() => minMaxOf(inputGrid), [inputGrid]);

  const maxPool = useMemo(
    () => computePooling(inputGrid, poolSize, poolStride, "max"),
    [inputGrid, poolSize, poolStride]
  );
  const avgPool = useMemo(
    () => computePooling(inputGrid, poolSize, poolStride, "average"),
    [inputGrid, poolSize, poolStride]
  );
  const minPool = useMemo(
    () => computePooling(inputGrid, poolSize, poolStride, "min"),
    [inputGrid, poolSize, poolStride]
  );

  const outH = maxPool.length;
  const outW = maxPool[0]?.length ?? 0;
  const totalSteps = outH * outW;

  // Global average pooling
  const globalAvg = useMemo(() => {
    let sum = 0;
    for (const row of inputGrid) for (const v of row) sum += v;
    return Math.round(sum / (inputGrid.length * inputGrid[0].length));
  }, [inputGrid]);

  const colorFn = useCallback(
    (v: number) => heatFill(v, minVal, maxVal),
    [minVal, maxVal]
  );

  const stepInfo = useMemo(() => {
    if (stepIndex === null || stepIndex >= totalSteps) return null;
    const outRow = Math.floor(stepIndex / outW);
    const outCol = stepIndex % outW;
    return {
      outRow,
      outCol,
      startR: outRow * poolStride,
      startC: outCol * poolStride,
    };
  }, [stepIndex, totalSteps, outW, poolStride]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const stopPlaying = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const handleAutoPlay = useCallback(() => {
    if (isPlaying) {
      stopPlaying();
      return;
    }
    setIsPlaying(true);
    setStepIndex(0);
    let cur = 0;
    timerRef.current = setInterval(() => {
      cur++;
      if (cur >= totalSteps) {
        stopPlaying();
        setStepIndex(null);
        return;
      }
      setStepIndex(cur);
    }, 500);
  }, [isPlaying, totalSteps, stopPlaying]);

  useEffect(() => {
    setStepIndex(null);
    stopPlaying();
  }, [poolSize, poolStride, seed, stopPlaying]);

  // Window values for current step
  const windowVals = useMemo(() => {
    if (!stepInfo) return [];
    const vals: number[] = [];
    for (let pr = 0; pr < poolSize; pr++)
      for (let pc = 0; pc < poolSize; pc++)
        vals.push(inputGrid[stepInfo.startR + pr]?.[stepInfo.startC + pc] ?? 0);
    return vals;
  }, [stepInfo, inputGrid, poolSize]);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 font-semibold uppercase">Pool Size:</span>
          {[2, 3].map((s) => (
            <button
              key={s}
              onClick={() => setPoolSize(s)}
              className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-all ${
                poolSize === s
                  ? "bg-violet-500 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {s}x{s}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 font-semibold uppercase">Stride:</span>
          {[1, 2, 3].map((s) => (
            <button
              key={s}
              onClick={() => setPoolStride(s)}
              className={`w-8 h-8 rounded-lg text-[11px] font-bold transition-all ${
                poolStride === s
                  ? "bg-violet-500 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        <button
          onClick={() => setSeed((prev) => prev + 1)}
          className="px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 flex items-center gap-1"
        >
          <RotateCcw className="w-3 h-3" /> New Data
        </button>

        <div className="flex items-center gap-1.5">
          <button
            onClick={handleAutoPlay}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-all ${
              isPlaying ? "bg-amber-500 text-white" : "bg-violet-500 text-white hover:bg-violet-600"
            }`}
          >
            {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
            {isPlaying ? "Stop" : "Animate"}
          </button>
          <button
            onClick={() =>
              setStepIndex((prev) =>
                prev === null ? 0 : prev >= totalSteps - 1 ? 0 : prev + 1
              )
            }
            disabled={isPlaying}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200 disabled:opacity-40 transition-all"
          >
            <SkipForward className="w-3 h-3" /> Step
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-wrap gap-5 items-start justify-center">
        {/* Input feature map */}
        <div>
          <p className="text-[10px] text-slate-500 font-semibold mb-1 text-center uppercase tracking-wide">
            Input Feature Map (8x8)
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
            <ValueGrid
              grid={inputGrid}
              cellSize={CELL}
              colorFn={colorFn}
              highlightRect={
                stepInfo
                  ? { r: stepInfo.startR, c: stepInfo.startC, h: poolSize, w: poolSize }
                  : null
              }
              showValues
            />
          </div>
        </div>

        <div className="flex items-center self-center">
          <ChevronRight className="w-5 h-5 text-slate-400" />
        </div>

        {/* Pooling outputs */}
        <div className="space-y-3">
          {/* Max pool */}
          <div>
            <p className="text-[10px] text-red-600 font-semibold mb-1 text-center uppercase tracking-wide">
              Max Pool ({outW}x{outH})
            </p>
            <div className="bg-red-50/50 border border-red-200 rounded-lg p-2">
              <ValueGrid
                grid={maxPool}
                cellSize={CELL}
                colorFn={colorFn}
                highlightCell={stepInfo ? { r: stepInfo.outRow, c: stepInfo.outCol } : null}
                showValues
              />
            </div>
          </div>

          {/* Avg pool */}
          <div>
            <p className="text-[10px] text-blue-600 font-semibold mb-1 text-center uppercase tracking-wide">
              Average Pool ({outW}x{outH})
            </p>
            <div className="bg-blue-50/50 border border-blue-200 rounded-lg p-2">
              <ValueGrid
                grid={avgPool}
                cellSize={CELL}
                colorFn={colorFn}
                highlightCell={stepInfo ? { r: stepInfo.outRow, c: stepInfo.outCol } : null}
                showValues
              />
            </div>
          </div>

          {/* Min pool */}
          <div>
            <p className="text-[10px] text-green-600 font-semibold mb-1 text-center uppercase tracking-wide">
              Min Pool ({outW}x{outH})
            </p>
            <div className="bg-green-50/50 border border-green-200 rounded-lg p-2">
              <ValueGrid
                grid={minPool}
                cellSize={CELL}
                colorFn={colorFn}
                highlightCell={stepInfo ? { r: stepInfo.outRow, c: stepInfo.outCol } : null}
                showValues
              />
            </div>
          </div>
        </div>
      </div>

      {/* Step detail */}
      {stepInfo && windowVals.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
          <p className="text-[11px] text-amber-800 font-semibold mb-2">
            Step {(stepIndex ?? 0) + 1}/{totalSteps} — Window at ({stepInfo.startR},{stepInfo.startC})
          </p>
          <div className="flex flex-wrap gap-4 items-center">
            <div>
              <p className="text-[9px] text-slate-500 mb-1 font-medium">Window values:</p>
              <div className={`grid gap-1 ${poolSize === 2 ? "grid-cols-2" : "grid-cols-3"}`}>
                {windowVals.map((v, i) => (
                  <div
                    key={i}
                    className="w-10 h-8 flex items-center justify-center rounded text-[10px] font-mono font-bold"
                    style={{ backgroundColor: colorFn(v) }}
                  >
                    {v}
                  </div>
                ))}
              </div>
            </div>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="text-red-600 font-bold">Max:</span>
                <span className="bg-red-100 px-2 py-0.5 rounded font-bold">
                  {Math.max(...windowVals)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-600 font-bold">Avg:</span>
                <span className="bg-blue-100 px-2 py-0.5 rounded font-bold">
                  {Math.round(windowVals.reduce((a, b) => a + b, 0) / windowVals.length)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600 font-bold">Min:</span>
                <span className="bg-green-100 px-2 py-0.5 rounded font-bold">
                  {Math.min(...windowVals)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step progress */}
      {stepIndex !== null && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500">
            Step {stepIndex + 1}/{totalSteps}
          </span>
          <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-violet-500 rounded-full transition-all"
              style={{ width: `${((stepIndex + 1) / totalSteps) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Global Average Pooling */}
      <div className="bg-gradient-to-r from-indigo-50 to-violet-50 border border-indigo-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-indigo-800 mb-2">
          Global Average Pooling (GAP)
        </p>
        <div className="flex items-center justify-center gap-4">
          <div>
            <p className="text-[9px] text-slate-500 text-center mb-1">8x8 Feature Map</p>
            <ValueGrid grid={inputGrid} cellSize={CELL_TINY} colorFn={colorFn} showValues={false} />
          </div>
          <div className="text-center">
            <ChevronRight className="w-5 h-5 text-slate-400 mx-auto" />
            <p className="text-[9px] text-slate-400">GAP</p>
          </div>
          <div className="text-center">
            <div
              className="w-16 h-16 rounded-lg flex items-center justify-center text-lg font-bold font-mono border-2 border-indigo-300"
              style={{ backgroundColor: colorFn(globalAvg) }}
            >
              {globalAvg}
            </div>
            <p className="text-[9px] text-indigo-600 font-semibold mt-1">1x1 output</p>
          </div>
        </div>
        <p className="text-[10px] text-slate-500 mt-3 text-center">
          Global Average Pooling averages the entire feature map into a single value. It is often
          used before the final classification layer, replacing fully connected layers to reduce
          parameters and overfitting.
        </p>
      </div>

      {/* Explanation cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-[11px] text-red-800 font-semibold mb-1">Max Pooling</p>
          <p className="text-[10px] text-red-700">
            Takes the maximum value in each window. This preserves the strongest activations
            (features) while providing translation invariance. It is the most commonly used pooling
            operation in CNNs because it retains the most prominent features detected by the
            filters.
          </p>
        </div>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <p className="text-[11px] text-blue-800 font-semibold mb-1">Average Pooling</p>
          <p className="text-[10px] text-blue-700">
            Computes the mean of all values in the window. This produces smoother downsampled
            feature maps. It is useful when you want to preserve overall feature magnitude rather
            than just the strongest signal.
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-3">
          <p className="text-[11px] text-green-800 font-semibold mb-1">Min Pooling</p>
          <p className="text-[10px] text-green-700">
            Takes the minimum value. Rarely used in practice, but useful for detecting the absence
            of features or finding the weakest signal in a region. Included here for comparison to
            help understand how pooling operations differ.
          </p>
        </div>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <p className="text-[11px] text-slate-700 font-semibold mb-1">Output size:</p>
        <p className="text-[10px] text-slate-600 font-mono text-center">
          O = floor((8 - {poolSize}) / {poolStride}) + 1 = {outW} x {outH}
        </p>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 5: Feature Maps & Channels
// ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_FEATURE_FILTERS: number[][][] = [
  // Horizontal edge
  [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
  ],
  // Vertical edge
  [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
  ],
  // Blur
  [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
  ],
  // Sharpen
  [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
];

const FILTER_NAMES = ["Edge (H)", "Edge (V)", "Blur", "Sharpen"];
const FILTER_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#8b5cf6"];

function FeatureMapsTab() {
  const [filters, setFilters] = useState<number[][][]>(
    DEFAULT_FEATURE_FILTERS.map((f) => f.map((r) => [...r]))
  );
  const [activeFilter, setActiveFilter] = useState(0);
  const [imageKey, setImageKey] = useState("square");
  const [showStack, setShowStack] = useState(true);
  const [inputChannels, setInputChannels] = useState(3);
  const [outputChannels, setOutputChannels] = useState(8);

  const imageOptions: { key: string; label: string; gen: () => number[][] }[] = useMemo(
    () => [
      { key: "square", label: "Square", gen: makeSquare },
      { key: "circle", label: "Circle", gen: makeCircle },
      { key: "checker", label: "Checker", gen: makeCheckerboard },
      { key: "diagonal", label: "Diagonal", gen: makeDiagonal },
    ],
    []
  );

  const inputGrid = useMemo(() => {
    const opt = imageOptions.find((o) => o.key === imageKey);
    return opt ? opt.gen() : makeSquare();
  }, [imageKey, imageOptions]);

  // Compute 4 output feature maps
  const featureMaps = useMemo(
    () => filters.map((kernel) => convolve(inputGrid, kernel)),
    [inputGrid, filters]
  );

  const featureMapMaxAbs = useMemo(
    () => featureMaps.map((fm) => maxAbsOf(fm)),
    [featureMaps]
  );

  const handleFilterCellClick = useCallback(
    (filterIdx: number, r: number, c: number, dir: 1 | -1) => {
      setFilters((prev) => {
        const next = prev.map((f) => f.map((row) => [...row]));
        const v = next[filterIdx][r][c];
        if (dir === 1) next[filterIdx][r][c] = v >= 5 ? -3 : v + 1;
        else next[filterIdx][r][c] = v <= -3 ? 5 : v - 1;
        return next;
      });
    },
    []
  );

  const resetFilters = useCallback(() => {
    setFilters(DEFAULT_FEATURE_FILTERS.map((f) => f.map((r) => [...r])));
  }, []);

  const totalFilters = inputChannels * outputChannels;

  return (
    <div className="space-y-4">
      {/* Image selector */}
      <div className="flex flex-wrap gap-1.5 items-center">
        <span className="text-[10px] text-slate-500 font-semibold uppercase mr-1">Image:</span>
        {imageOptions.map((opt) => (
          <button
            key={opt.key}
            onClick={() => setImageKey(opt.key)}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
              imageKey === opt.key
                ? "bg-violet-500 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {opt.label}
          </button>
        ))}
        <button
          onClick={resetFilters}
          className="px-2.5 py-1 rounded text-[11px] font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 flex items-center gap-1 ml-2"
        >
          <RotateCcw className="w-3 h-3" /> Reset Filters
        </button>
      </div>

      {/* Single-channel demonstration */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-slate-700 mb-3">
          One Input Channel → 4 Filters → 4 Feature Maps
        </p>

        <div className="flex flex-wrap gap-4 items-start justify-center">
          {/* Input */}
          <div className="text-center">
            <p className="text-[10px] text-slate-500 font-semibold mb-1 uppercase tracking-wide">
              Input (8x8)
            </p>
            <div className="bg-white border border-slate-200 rounded-lg p-2 inline-block">
              <PixelGrid grid={inputGrid} cellSize={CELL_SM} showValues />
            </div>
          </div>

          <div className="flex items-center self-center">
            <ChevronRight className="w-5 h-5 text-slate-400" />
          </div>

          {/* 4 Filters */}
          <div className="text-center space-y-2">
            <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide">
              4 Filters (3x3)
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {filters.map((kernel, fi) => (
                <div key={fi}>
                  <button
                    onClick={() => setActiveFilter(fi)}
                    className={`text-[9px] font-semibold mb-0.5 block w-full text-center px-1 py-0.5 rounded transition-colors ${
                      activeFilter === fi
                        ? "text-white"
                        : "text-slate-500 hover:bg-slate-100"
                    }`}
                    style={activeFilter === fi ? { backgroundColor: FILTER_COLORS[fi] } : {}}
                  >
                    {FILTER_NAMES[fi]}
                  </button>
                  <div
                    className="border rounded-lg p-1 inline-block"
                    style={{
                      borderColor: activeFilter === fi ? FILTER_COLORS[fi] : "#e2e8f0",
                    }}
                  >
                    <KernelGrid
                      kernel={kernel}
                      cellSize={22}
                      onCellClick={(r, c, dir) => handleFilterCellClick(fi, r, c, dir)}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[8px] text-slate-400">Click +1 / right-click -1 to edit</p>
          </div>

          <div className="flex items-center self-center">
            <ChevronRight className="w-5 h-5 text-slate-400" />
          </div>

          {/* 4 Feature maps */}
          <div className="text-center space-y-2">
            <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide">
              4 Feature Maps (6x6)
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {featureMaps.map((fm, fi) => (
                <div key={fi}>
                  <p
                    className="text-[9px] font-semibold mb-0.5 text-center"
                    style={{ color: FILTER_COLORS[fi] }}
                  >
                    {FILTER_NAMES[fi]}
                  </p>
                  <div
                    className="border rounded-lg p-1 inline-block"
                    style={{
                      borderColor: activeFilter === fi ? FILTER_COLORS[fi] : "#e2e8f0",
                      backgroundColor: activeFilter === fi ? `${FILTER_COLORS[fi]}08` : "transparent",
                    }}
                  >
                    <OutputGrid grid={fm} cellSize={CELL_SM} maxAbs={featureMapMaxAbs[fi]} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <p className="text-[10px] text-slate-500 mt-3 text-center">
          Each filter "looks for" a different pattern. The edge filters respond to boundaries, the
          blur filter smooths, and the sharpen filter enhances detail. This is how CNNs learn to
          extract hierarchical features.
        </p>
      </div>

      {/* Stacked feature map visualization */}
      <div className="bg-gradient-to-r from-violet-50 to-indigo-50 border border-violet-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-violet-800">
            Stacked Feature Maps (Depth View)
          </p>
          <button
            onClick={() => setShowStack((v) => !v)}
            className="text-[10px] text-violet-600 font-semibold flex items-center gap-1 hover:text-violet-800 transition-colors"
          >
            {showStack ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
            {showStack ? "Hide" : "Show"}
          </button>
        </div>

        {showStack && (
          <div className="flex justify-center">
            <svg viewBox="0 0 320 220" width={320} height={220} className="block">
              {/* Draw 4 stacked feature map layers */}
              {[3, 2, 1, 0].map((fi) => {
                const fm = featureMaps[fi];
                const ma = featureMapMaxAbs[fi];
                const offsetX = fi * 20;
                const offsetY = fi * 20;
                const size = 6;
                const cellSz = 18;
                const baseX = 50 + offsetX;
                const baseY = 20 + offsetY;

                return (
                  <g key={fi} opacity={0.85}>
                    {/* Background rectangle */}
                    <rect
                      x={baseX - 2}
                      y={baseY - 2}
                      width={size * cellSz + 4}
                      height={size * cellSz + 4}
                      rx={4}
                      fill="white"
                      stroke={FILTER_COLORS[fi]}
                      strokeWidth={activeFilter === fi ? 2 : 1}
                    />
                    {/* Grid cells */}
                    {fm.map((row, r) =>
                      row.map((val, c) => (
                        <rect
                          key={`${r}-${c}`}
                          x={baseX + c * cellSz}
                          y={baseY + r * cellSz}
                          width={cellSz - 1}
                          height={cellSz - 1}
                          rx={1}
                          fill={divergingFill(val, ma)}
                        />
                      ))
                    )}
                    {/* Label */}
                    <text
                      x={baseX + size * cellSz + 6}
                      y={baseY + (size * cellSz) / 2}
                      fontSize={9}
                      fontWeight={600}
                      fill={FILTER_COLORS[fi]}
                      dominantBaseline="central"
                    >
                      {FILTER_NAMES[fi]}
                    </text>
                  </g>
                );
              })}

              {/* Arrow and label */}
              <text x={160} y={210} textAnchor="middle" fontSize={10} fill="#64748b" fontWeight={500}>
                4 feature maps stacked = output depth of 4
              </text>
            </svg>
          </div>
        )}
      </div>

      {/* Multi-channel concept */}
      <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-amber-800 mb-3">
          Multi-Channel Convolution Concept
        </p>

        <div className="flex flex-wrap gap-4 items-center justify-center mb-4">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-600 font-semibold">Input Channels:</span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setInputChannels((v) => Math.max(1, v - 1))}
                className="w-6 h-6 rounded bg-slate-100 flex items-center justify-center hover:bg-slate-200 transition-colors"
              >
                <Minus className="w-3 h-3 text-slate-600" />
              </button>
              <span className="w-8 text-center text-sm font-bold text-slate-700">
                {inputChannels}
              </span>
              <button
                onClick={() => setInputChannels((v) => Math.min(8, v + 1))}
                className="w-6 h-6 rounded bg-slate-100 flex items-center justify-center hover:bg-slate-200 transition-colors"
              >
                <Plus className="w-3 h-3 text-slate-600" />
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-600 font-semibold">Output Channels:</span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setOutputChannels((v) => Math.max(1, v - 1))}
                className="w-6 h-6 rounded bg-slate-100 flex items-center justify-center hover:bg-slate-200 transition-colors"
              >
                <Minus className="w-3 h-3 text-slate-600" />
              </button>
              <span className="w-8 text-center text-sm font-bold text-slate-700">
                {outputChannels}
              </span>
              <button
                onClick={() => setOutputChannels((v) => Math.min(16, v + 1))}
                className="w-6 h-6 rounded bg-slate-100 flex items-center justify-center hover:bg-slate-200 transition-colors"
              >
                <Plus className="w-3 h-3 text-slate-600" />
              </button>
            </div>
          </div>
        </div>

        {/* Visual diagram */}
        <div className="flex justify-center">
          <svg viewBox="0 0 520 250" width={520} height={250} className="block">
            {/* Input channels stack */}
            {Array.from({ length: Math.min(inputChannels, 5) }, (_, i) => (
              <g key={`in-${i}`}>
                <rect
                  x={20 + i * 8}
                  y={40 + i * 8}
                  width={60}
                  height={60}
                  rx={4}
                  fill={i === 0 ? "#fee2e2" : i === 1 ? "#dcfce7" : i === 2 ? "#dbeafe" : "#fef3c7"}
                  stroke="#94a3b8"
                  strokeWidth={0.5}
                  opacity={0.9}
                />
              </g>
            ))}
            <text x={50} y={125} textAnchor="middle" fontSize={10} fontWeight={600} fill="#334155">
              Input
            </text>
            <text x={50} y={138} textAnchor="middle" fontSize={9} fill="#64748b">
              {inputChannels}ch x HxW
            </text>

            {/* Arrow */}
            <line x1={110} y1={75} x2={160} y2={75} stroke="#94a3b8" strokeWidth={1.5} />
            <polygon points="158,71 166,75 158,79" fill="#94a3b8" />

            {/* Filters (3D block) */}
            {Array.from({ length: Math.min(outputChannels, 6) }, (_, o) => (
              <g key={`filter-${o}`}>
                <rect
                  x={170 + o * 5}
                  y={35 + o * 5}
                  width={40}
                  height={40}
                  rx={3}
                  fill="#f5f3ff"
                  stroke="#8b5cf6"
                  strokeWidth={0.5}
                  opacity={0.85}
                />
              </g>
            ))}
            <text x={210} y={125} textAnchor="middle" fontSize={10} fontWeight={600} fill="#5b21b6">
              Filters
            </text>
            <text x={210} y={138} textAnchor="middle" fontSize={9} fill="#7c3aed">
              {outputChannels} x {inputChannels} x 3x3
            </text>
            <text x={210} y={151} textAnchor="middle" fontSize={8} fill="#94a3b8">
              ({totalFilters} kernels total)
            </text>

            {/* Arrow */}
            <line x1={260} y1={75} x2={310} y2={75} stroke="#94a3b8" strokeWidth={1.5} />
            <polygon points="308,71 316,75 308,79" fill="#94a3b8" />

            {/* Output channels stack */}
            {Array.from({ length: Math.min(outputChannels, 6) }, (_, i) => (
              <g key={`out-${i}`}>
                <rect
                  x={320 + i * 8}
                  y={40 + i * 8}
                  width={60}
                  height={60}
                  rx={4}
                  fill={`hsl(${(i * 360) / outputChannels}, 60%, 90%)`}
                  stroke="#94a3b8"
                  strokeWidth={0.5}
                  opacity={0.9}
                />
              </g>
            ))}
            <text x={370} y={125} textAnchor="middle" fontSize={10} fontWeight={600} fill="#334155">
              Output
            </text>
            <text x={370} y={138} textAnchor="middle" fontSize={9} fill="#64748b">
              {outputChannels}ch x H'xW'
            </text>

            {/* Title */}
            <text x={260} y={20} textAnchor="middle" fontSize={11} fontWeight={700} fill="#334155">
              Conv2D Layer
            </text>

            {/* Explanation line at bottom */}
            <text x={260} y={180} textAnchor="middle" fontSize={9} fill="#64748b">
              Each output channel = sum of (input_ch * kernel) across all input channels
            </text>

            {/* Formula */}
            <rect x={120} y={195} width={280} height={40} rx={6} fill="white" stroke="#e2e8f0" />
            <text x={260} y={210} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#334155" fontWeight={600}>
              Total 3x3 kernels = {inputChannels} x {outputChannels} = {totalFilters}
            </text>
            <text x={260} y={225} textAnchor="middle" fontSize={9} fontFamily="monospace" fill="#64748b">
              Total params = {totalFilters} x 9 + {outputChannels} bias = {totalFilters * 9 + outputChannels}
            </text>
          </svg>
        </div>
      </div>

      {/* Explanation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
          <p className="text-[11px] text-violet-800 font-semibold mb-1">
            How Multi-Channel Convolution Works
          </p>
          <p className="text-[10px] text-violet-700">
            For an input with C_in channels (e.g., 3 for RGB), each output filter has C_in separate
            3x3 kernels. The filter convolves each kernel with the corresponding input channel, then
            sums all results to produce one output feature map. With C_out filters, you get C_out
            output channels.
          </p>
        </div>
        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
          <p className="text-[11px] text-indigo-800 font-semibold mb-1">
            Parameter Count
          </p>
          <p className="text-[10px] text-indigo-700">
            A Conv2D layer with {inputChannels} input channels and {outputChannels} output channels
            using 3x3 kernels has {inputChannels} x {outputChannels} = {totalFilters} separate 3x3
            kernels. That is {totalFilters * 9} weight parameters plus {outputChannels} bias terms ={" "}
            <strong>{totalFilters * 9 + outputChannels}</strong> total learnable parameters.
          </p>
        </div>
      </div>

      {/* Why multiple filters matter */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <p className="text-xs font-semibold text-slate-700 mb-2">
          Why Do CNNs Use Multiple Filters?
        </p>
        <div className="space-y-2 text-[11px] text-slate-600">
          <p>
            A single filter can only detect one type of pattern (e.g., horizontal edges). By
            using many filters, a CNN layer can detect multiple patterns simultaneously. In
            early layers, filters learn simple features like edges and corners. In deeper
            layers, filters combine these simple features to detect complex patterns like
            textures, object parts, and eventually entire objects.
          </p>
          <p>
            The set of feature maps produced by a convolutional layer forms a
            <strong> tensor</strong> with shape (height x width x num_filters). This tensor
            becomes the input to the next layer, allowing the network to build increasingly
            abstract representations.
          </p>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════
export default function CNNFiltersActivity() {
  const [activeTab, setActiveTab] = useState<TabId>("apply");

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 bg-slate-100 rounded-xl p-1 overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
              activeTab === tab.id
                ? "bg-white text-violet-700 shadow-sm"
                : "text-slate-500 hover:text-slate-700 hover:bg-white/50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "apply" && <ApplyFiltersTab />}
        {activeTab === "edge" && <EdgeDetectionTab />}
        {activeTab === "stride" && <StridePaddingTab />}
        {activeTab === "pooling" && <PoolingTab />}
        {activeTab === "features" && <FeatureMapsTab />}
      </div>
    </div>
  );
}
