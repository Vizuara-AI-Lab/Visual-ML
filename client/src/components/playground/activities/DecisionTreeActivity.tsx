/**
 * Decision Tree Explorer — comprehensive 3-tab interactive activity
 *
 * Tabs:
 *   1. Build a Tree      – click scatter plot to place splits, watch tree grow
 *   2. Depth & Pruning    – control tree depth, see overfitting, prune leaves
 *   3. Feature Importance – bar charts, permutation importance, feature removal
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Info,
  GitBranch,
  RotateCcw,
  Scissors,
  BarChart3,
  Layers,
  MousePointer,
  Play,
  Undo2,
  ChevronDown,
  ChevronUp,
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

function gaussRng(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// ════════════════════════════════════════════════════════════════════════════
// TYPES
// ════════════════════════════════════════════════════════════════════════════

interface Pt {
  x: number;
  y: number;
  cls: number;
}

interface Pt4 {
  f: number[];
  cls: number;
}

interface Split {
  feat: number;
  thresh: number;
}

interface TNode {
  idx: number[];
  split: Split | null;
  left: TNode | null;
  right: TNode | null;
  depth: number;
}

// ════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ════════════════════════════════════════════════════════════════════════════

const W = 420;
const H = 340;
const PAD = 35;
const CLS_C = ["#3b82f6", "#f97316"] as const;
const CLS_BG = ["rgba(59,130,246,0.12)", "rgba(249,115,22,0.12)"] as const;

// ════════════════════════════════════════════════════════════════════════════
// DATA GENERATION
// ════════════════════════════════════════════════════════════════════════════

function genData2D(seed: number, n = 40): Pt[] {
  const r = mulberry32(seed);
  const pts: Pt[] = [];
  for (let i = 0; i < n / 4; i++) {
    pts.push({ x: 0.2 + gaussRng(r) * 0.12, y: 0.75 + gaussRng(r) * 0.1, cls: 0 });
  }
  for (let i = 0; i < n / 4; i++) {
    pts.push({ x: 0.75 + gaussRng(r) * 0.1, y: 0.25 + gaussRng(r) * 0.12, cls: 0 });
  }
  for (let i = 0; i < n / 4; i++) {
    pts.push({ x: 0.7 + gaussRng(r) * 0.12, y: 0.75 + gaussRng(r) * 0.1, cls: 1 });
  }
  for (let i = 0; i < n / 4; i++) {
    pts.push({ x: 0.25 + gaussRng(r) * 0.1, y: 0.3 + gaussRng(r) * 0.12, cls: 1 });
  }
  return pts.map((p) => ({
    x: Math.max(0.02, Math.min(0.98, p.x)),
    y: Math.max(0.02, Math.min(0.98, p.y)),
    cls: p.cls,
  }));
}

function genData4F(seed: number, n = 80): Pt4[] {
  const r = mulberry32(seed);
  const pts: Pt4[] = [];
  for (let i = 0; i < n; i++) {
    const f0 = r();
    const f1 = r();
    const f2 = r();
    const f3 = r();
    const score = 2.5 * f0 + 1.5 * f1 - 0.3 * f2 + 0.1 * f3 + gaussRng(r) * 0.3;
    pts.push({ f: [f0, f1, f2, f3], cls: score > 2.0 ? 1 : 0 });
  }
  return pts;
}

// ════════════════════════════════════════════════════════════════════════════
// TREE UTILITIES
// ════════════════════════════════════════════════════════════════════════════

function gini(idx: number[], data: { cls: number }[]): number {
  if (idx.length === 0) return 0;
  const c0 = idx.filter((i) => data[i].cls === 0).length;
  const p0 = c0 / idx.length;
  const p1 = 1 - p0;
  return 1 - p0 * p0 - p1 * p1;
}

function entropy(idx: number[], data: { cls: number }[]): number {
  if (idx.length === 0) return 0;
  const c0 = idx.filter((i) => data[i].cls === 0).length;
  const p0 = c0 / idx.length;
  const p1 = 1 - p0;
  let e = 0;
  if (p0 > 0) e -= p0 * Math.log2(p0);
  if (p1 > 0) e -= p1 * Math.log2(p1);
  return e;
}

function misclass(idx: number[], data: { cls: number }[]): number {
  if (idx.length === 0) return 0;
  const c0 = idx.filter((i) => data[i].cls === 0).length;
  const p0 = c0 / idx.length;
  return 1 - Math.max(p0, 1 - p0);
}

function majority(idx: number[], data: { cls: number }[]): number {
  if (idx.length === 0) return 0;
  const c0 = idx.filter((i) => data[i].cls === 0).length;
  return c0 >= idx.length - c0 ? 0 : 1;
}

function weightedMetric(
  lIdx: number[],
  rIdx: number[],
  data: { cls: number }[],
  fn: (i: number[], d: { cls: number }[]) => number,
): number {
  const t = lIdx.length + rIdx.length;
  if (t === 0) return 0;
  return (lIdx.length / t) * fn(lIdx, data) + (rIdx.length / t) * fn(rIdx, data);
}

function accuracy(idx: number[], data: { cls: number }[], tree: TNode): number {
  if (idx.length === 0) return 1;
  let correct = 0;
  for (const i of idx) {
    const pred = predictNode(tree, data, i);
    if (pred === data[i].cls) correct++;
  }
  return correct / idx.length;
}

function predictNode(node: TNode, data: { cls: number; [k: string]: any }[], i: number): number {
  if (!node.split) return majority(node.idx, data);
  const val = getFeatureVal(data, i, node.split.feat);
  if (val < node.split.thresh) {
    return node.left ? predictNode(node.left, data, i) : majority(node.idx, data);
  }
  return node.right ? predictNode(node.right, data, i) : majority(node.idx, data);
}

function getFeatureVal(data: any[], i: number, feat: number): number {
  const d = data[i];
  if ("f" in d) return d.f[feat];
  if (feat === 0) return d.x;
  return d.y;
}

function bestSplit2D(
  idx: number[],
  data: Pt[],
  criterion: "gini" | "entropy" = "gini",
): Split | null {
  if (idx.length < 2) return null;
  const fn = criterion === "gini" ? gini : entropy;
  let bestScore = Infinity;
  let bestSplit: Split | null = null;
  for (let feat = 0; feat < 2; feat++) {
    const vals = idx.map((i) => (feat === 0 ? data[i].x : data[i].y)).sort((a, b) => a - b);
    const thresholds: number[] = [];
    for (let j = 0; j < vals.length - 1; j++) {
      if (vals[j] !== vals[j + 1]) thresholds.push((vals[j] + vals[j + 1]) / 2);
    }
    for (const th of thresholds) {
      const lI = idx.filter((i) => (feat === 0 ? data[i].x : data[i].y) < th);
      const rI = idx.filter((i) => (feat === 0 ? data[i].x : data[i].y) >= th);
      if (lI.length === 0 || rI.length === 0) continue;
      const sc = weightedMetric(lI, rI, data, fn);
      if (sc < bestScore) {
        bestScore = sc;
        bestSplit = { feat, thresh: th };
      }
    }
  }
  return bestSplit;
}

function bestSplit4F(
  idx: number[],
  data: Pt4[],
  criterion: "gini" | "entropy" = "gini",
): Split | null {
  if (idx.length < 2) return null;
  const fn = criterion === "gini" ? gini : entropy;
  let bestScore = Infinity;
  let bestSplit: Split | null = null;
  for (let feat = 0; feat < 4; feat++) {
    const vals = idx.map((i) => data[i].f[feat]).sort((a, b) => a - b);
    const thresholds: number[] = [];
    for (let j = 0; j < vals.length - 1; j++) {
      if (vals[j] !== vals[j + 1]) thresholds.push((vals[j] + vals[j + 1]) / 2);
    }
    for (const th of thresholds) {
      const lI = idx.filter((i) => data[i].f[feat] < th);
      const rI = idx.filter((i) => data[i].f[feat] >= th);
      if (lI.length === 0 || rI.length === 0) continue;
      const sc = weightedMetric(lI, rI, data, fn);
      if (sc < bestScore) {
        bestScore = sc;
        bestSplit = { feat, thresh: th };
      }
    }
  }
  return bestSplit;
}

function buildTree2D(data: Pt[], idx: number[], maxDepth: number, depth = 0): TNode {
  const node: TNode = { idx, split: null, left: null, right: null, depth };
  if (depth >= maxDepth || idx.length < 2) return node;
  const g = gini(idx, data);
  if (g < 0.01) return node;
  const sp = bestSplit2D(idx, data);
  if (!sp) return node;
  const lI = idx.filter((i) => (sp.feat === 0 ? data[i].x : data[i].y) < sp.thresh);
  const rI = idx.filter((i) => (sp.feat === 0 ? data[i].x : data[i].y) >= sp.thresh);
  if (lI.length === 0 || rI.length === 0) return node;
  node.split = sp;
  node.left = buildTree2D(data, lI, maxDepth, depth + 1);
  node.right = buildTree2D(data, rI, maxDepth, depth + 1);
  return node;
}

function buildTree4F(data: Pt4[], idx: number[], maxDepth: number, depth = 0, featureSubset?: number[]): TNode {
  const node: TNode = { idx, split: null, left: null, right: null, depth };
  if (depth >= maxDepth || idx.length < 2) return node;
  const g = gini(idx, data);
  if (g < 0.01) return node;

  const feats = featureSubset || [0, 1, 2, 3];
  const fn = gini;
  let bestScore = Infinity;
  let bestSp: Split | null = null;
  for (const feat of feats) {
    const vals = idx.map((i) => data[i].f[feat]).sort((a, b) => a - b);
    const thresholds: number[] = [];
    for (let j = 0; j < vals.length - 1; j++) {
      if (vals[j] !== vals[j + 1]) thresholds.push((vals[j] + vals[j + 1]) / 2);
    }
    for (const th of thresholds) {
      const lI = idx.filter((i) => data[i].f[feat] < th);
      const rI = idx.filter((i) => data[i].f[feat] >= th);
      if (lI.length === 0 || rI.length === 0) continue;
      const sc = weightedMetric(lI, rI, data, fn);
      if (sc < bestScore) {
        bestScore = sc;
        bestSp = { feat, thresh: th };
      }
    }
  }
  if (!bestSp) return node;
  const lI = idx.filter((i) => data[i].f[bestSp!.feat] < bestSp!.thresh);
  const rI = idx.filter((i) => data[i].f[bestSp!.feat] >= bestSp!.thresh);
  if (lI.length === 0 || rI.length === 0) return node;
  node.split = bestSp;
  node.left = buildTree4F(data, lI, maxDepth, depth + 1, featureSubset);
  node.right = buildTree4F(data, rI, maxDepth, depth + 1, featureSubset);
  return node;
}

function treeDepth(node: TNode): number {
  if (!node.split) return node.depth;
  return Math.max(
    node.left ? treeDepth(node.left) : node.depth,
    node.right ? treeDepth(node.right) : node.depth,
  );
}

function countNodes(node: TNode): number {
  let c = 1;
  if (node.left) c += countNodes(node.left);
  if (node.right) c += countNodes(node.right);
  return c;
}

function countLeaves(node: TNode): number {
  if (!node.split) return 1;
  return (node.left ? countLeaves(node.left) : 0) + (node.right ? countLeaves(node.right) : 0);
}

function pruneTree(node: TNode, testIdx: number[], data: { cls: number }[]): TNode {
  if (!node.split) return { ...node };
  const left = node.left ? pruneTree(node.left, testIdx, data) : null;
  const right = node.right ? pruneTree(node.right, testIdx, data) : null;
  const pruned: TNode = { ...node, left, right };
  const accWithChildren = accuracy(testIdx, data, pruned);
  const leaf: TNode = { idx: node.idx, split: null, left: null, right: null, depth: node.depth };
  const accAsLeaf = accuracy(testIdx, data, leaf);
  if (accAsLeaf >= accWithChildren) return leaf;
  return pruned;
}

function cloneNode(n: TNode): TNode {
  return {
    idx: [...n.idx],
    split: n.split ? { ...n.split } : null,
    left: n.left ? cloneNode(n.left) : null,
    right: n.right ? cloneNode(n.right) : null,
    depth: n.depth,
  };
}

// ════════════════════════════════════════════════════════════════════════════
// SVG HELPERS
// ════════════════════════════════════════════════════════════════════════════

function sx(v: number, w = W): number {
  return PAD + v * (w - 2 * PAD);
}
function sy(v: number, h = H): number {
  return h - PAD - v * (h - 2 * PAD);
}

interface Region {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  cls: number;
}

function collectRegions(
  node: TNode,
  data: { cls: number }[],
  x0: number,
  x1: number,
  y0: number,
  y1: number,
): Region[] {
  if (!node.split) {
    return [{ x0, x1, y0, y1, cls: majority(node.idx, data) }];
  }
  const regions: Region[] = [];
  const f = node.split.feat;
  const th = node.split.thresh;
  if (f === 0) {
    if (node.left) regions.push(...collectRegions(node.left, data, x0, th, y0, y1));
    if (node.right) regions.push(...collectRegions(node.right, data, th, x1, y0, y1));
  } else {
    if (node.left) regions.push(...collectRegions(node.left, data, x0, x1, y0, th));
    if (node.right) regions.push(...collectRegions(node.right, data, x0, x1, th, y1));
  }
  return regions;
}

interface SLine {
  feat: number;
  thresh: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

function collectSplitLines(
  node: TNode,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
): SLine[] {
  if (!node.split) return [];
  const lines: SLine[] = [{ feat: node.split.feat, thresh: node.split.thresh, x0, x1, y0, y1 }];
  const f = node.split.feat;
  const th = node.split.thresh;
  if (f === 0) {
    if (node.left) lines.push(...collectSplitLines(node.left, x0, th, y0, y1));
    if (node.right) lines.push(...collectSplitLines(node.right, th, x1, y0, y1));
  } else {
    if (node.left) lines.push(...collectSplitLines(node.left, x0, x1, y0, th));
    if (node.right) lines.push(...collectSplitLines(node.right, x0, x1, th, y1));
  }
  return lines;
}

interface LayoutN {
  cx: number;
  cy: number;
  isLeaf: boolean;
  label: string;
  sub: string;
  cls: number;
  n: number;
  id: string;
  leftChild?: LayoutN;
  rightChild?: LayoutN;
}

function layoutTreeSVG(
  node: TNode,
  data: { cls: number }[],
  xMin: number,
  xMax: number,
  y: number,
  yStep: number,
  featNames: string[],
): LayoutN {
  const mc = majority(node.idx, data);
  const g = gini(node.idx, data);
  const ln: LayoutN = {
    cx: (xMin + xMax) / 2,
    cy: y,
    isLeaf: !node.split,
    label: node.split ? `${featNames[node.split.feat]}<${node.split.thresh.toFixed(2)}` : `${["A", "B"][mc]}`,
    sub: `n=${node.idx.length} G=${g.toFixed(2)}`,
    cls: mc,
    n: node.idx.length,
    id: `${node.depth}-${xMin.toFixed(0)}-${xMax.toFixed(0)}`,
  };
  if (node.split) {
    if (node.left) ln.leftChild = layoutTreeSVG(node.left, data, xMin, (xMin + xMax) / 2, y + yStep, yStep, featNames);
    if (node.right) ln.rightChild = layoutTreeSVG(node.right, data, (xMin + xMax) / 2, xMax, y + yStep, yStep, featNames);
  }
  return ln;
}

function flattenLN(n: LayoutN): LayoutN[] {
  const r: LayoutN[] = [n];
  if (n.leftChild) r.push(...flattenLN(n.leftChild));
  if (n.rightChild) r.push(...flattenLN(n.rightChild));
  return r;
}

function TreeDiagram({
  tree,
  data,
  w,
  h,
  featNames,
}: {
  tree: TNode;
  data: { cls: number }[];
  w: number;
  h: number;
  featNames: string[];
}) {
  const depth = treeDepth(tree);
  const yStep = depth > 0 ? (h - 70) / (depth + 1) : 60;
  const root = layoutTreeSVG(tree, data, 20, w - 20, 30, yStep, featNames);
  const all = flattenLN(root);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ maxHeight: h }}>
      {all.map((ln) => (
        <g key={`e-${ln.id}`}>
          {ln.leftChild && (
            <line x1={ln.cx} y1={ln.cy + 14} x2={ln.leftChild.cx} y2={ln.leftChild.cy - 14} stroke="#cbd5e1" strokeWidth={1.5} />
          )}
          {ln.rightChild && (
            <line x1={ln.cx} y1={ln.cy + 14} x2={ln.rightChild.cx} y2={ln.rightChild.cy - 14} stroke="#cbd5e1" strokeWidth={1.5} />
          )}
          {ln.leftChild && (
            <text x={(ln.cx + ln.leftChild.cx) / 2 - 10} y={(ln.cy + ln.leftChild.cy) / 2} fontSize={7} fill="#10b981" fontWeight={600}>
              Yes
            </text>
          )}
          {ln.rightChild && (
            <text x={(ln.cx + ln.rightChild.cx) / 2 + 3} y={(ln.cy + ln.rightChild.cy) / 2} fontSize={7} fill="#ef4444" fontWeight={600}>
              No
            </text>
          )}
        </g>
      ))}
      {all.map((ln) => {
        const col = ln.isLeaf ? CLS_C[ln.cls] : "#059669";
        const bw = ln.isLeaf ? 48 : 64;
        const bh = 28;
        return (
          <g key={`n-${ln.id}`}>
            <rect x={ln.cx - bw / 2} y={ln.cy - bh / 2} width={bw} height={bh} rx={6} fill={col} opacity={0.13} stroke={col} strokeWidth={1} />
            <text x={ln.cx} y={ln.cy - 2} textAnchor="middle" fontSize={7.5} fill={col} fontWeight={700}>
              {ln.label}
            </text>
            <text x={ln.cx} y={ln.cy + 9} textAnchor="middle" fontSize={6.5} fill="#64748b">
              {ln.sub}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ScatterWithRegions({
  data,
  tree,
  w,
  h,
  highlightIdx,
  onClickSVG,
  previewLine,
}: {
  data: Pt[];
  tree: TNode | null;
  w: number;
  h: number;
  highlightIdx?: Set<number>;
  onClickSVG?: (x: number, y: number, feat: number) => void;
  previewLine?: { feat: number; thresh: number; x0: number; x1: number; y0: number; y1: number } | null;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const regions = useMemo(() => (tree ? collectRegions(tree, data, 0, 1, 0, 1) : []), [tree, data]);
  const splitL = useMemo(() => (tree ? collectSplitLines(tree, 0, 1, 0, 1) : []), [tree]);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (!onClickSVG || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const scaleX = w / rect.width;
      const scaleY = h / rect.height;
      const px = (e.clientX - rect.left) * scaleX;
      const py = (e.clientY - rect.top) * scaleY;
      const dataX = (px - PAD) / (w - 2 * PAD);
      const dataY = 1 - (py - PAD) / (h - 2 * PAD);
      if (dataX < 0 || dataX > 1 || dataY < 0 || dataY > 1) return;
      const midX = (w - 2 * PAD) / 2 + PAD;
      const feat = Math.abs(px - sx(dataX, w)) < Math.abs(py - sy(dataY, h)) ? 1 : 0;
      onClickSVG(dataX, dataY, feat);
    },
    [onClickSVG, w, h],
  );

  return (
    <svg ref={svgRef} viewBox={`0 0 ${w} ${h}`} className="w-full cursor-crosshair" style={{ maxHeight: h }} onClick={handleClick}>
      {regions.map((rg, i) => (
        <rect
          key={`r-${i}`}
          x={sx(rg.x0, w)}
          y={sy(rg.y1, h)}
          width={sx(rg.x1, w) - sx(rg.x0, w)}
          height={sy(rg.y0, h) - sy(rg.y1, h)}
          fill={CLS_C[rg.cls]}
          opacity={0.08}
        />
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={`g-${v}`}>
          <line x1={sx(v, w)} y1={sy(0, h)} x2={sx(v, w)} y2={sy(1, h)} stroke="#e2e8f0" strokeWidth={0.5} />
          <line x1={sx(0, w)} y1={sy(v, h)} x2={sx(1, w)} y2={sy(v, h)} stroke="#e2e8f0" strokeWidth={0.5} />
        </g>
      ))}
      {splitL.map((sl, i) =>
        sl.feat === 0 ? (
          <line key={`s-${i}`} x1={sx(sl.thresh, w)} y1={sy(sl.y0, h)} x2={sx(sl.thresh, w)} y2={sy(sl.y1, h)} stroke="#059669" strokeWidth={1.8} strokeDasharray="5,3" opacity={0.7} />
        ) : (
          <line key={`s-${i}`} x1={sx(sl.x0, w)} y1={sy(sl.thresh, h)} x2={sx(sl.x1, w)} y2={sy(sl.thresh, h)} stroke="#059669" strokeWidth={1.8} strokeDasharray="5,3" opacity={0.7} />
        ),
      )}
      {previewLine &&
        (previewLine.feat === 0 ? (
          <line x1={sx(previewLine.thresh, w)} y1={sy(previewLine.y0, h)} x2={sx(previewLine.thresh, w)} y2={sy(previewLine.y1, h)} stroke="#f59e0b" strokeWidth={2} opacity={0.85} />
        ) : (
          <line x1={sx(previewLine.x0, w)} y1={sy(previewLine.thresh, h)} x2={sx(previewLine.x1, w)} y2={sy(previewLine.thresh, h)} stroke="#f59e0b" strokeWidth={2} opacity={0.85} />
        ))}
      <text x={w / 2} y={h - 5} textAnchor="middle" fontSize={10} fill="#64748b" fontWeight={600}>
        X1
      </text>
      <text x={8} y={h / 2} textAnchor="middle" fontSize={10} fill="#64748b" fontWeight={600} transform={`rotate(-90,8,${h / 2})`}>
        X2
      </text>
      {[0, 0.5, 1].map((v) => (
        <g key={`t-${v}`}>
          <text x={sx(v, w)} y={h - PAD + 15} textAnchor="middle" fontSize={8} fill="#94a3b8">
            {v.toFixed(1)}
          </text>
          <text x={PAD - 6} y={sy(v, h) + 3} textAnchor="end" fontSize={8} fill="#94a3b8">
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      {data.map((p, i) => (
        <circle
          key={i}
          cx={sx(p.x, w)}
          cy={sy(p.y, h)}
          r={highlightIdx && highlightIdx.has(i) ? 5.5 : 4}
          fill={CLS_C[p.cls]}
          stroke={highlightIdx && highlightIdx.has(i) ? "#fbbf24" : "#fff"}
          strokeWidth={highlightIdx && highlightIdx.has(i) ? 2 : 1.2}
          opacity={highlightIdx ? (highlightIdx.has(i) ? 1 : 0.35) : 0.85}
        />
      ))}
    </svg>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB DEFINITIONS
// ════════════════════════════════════════════════════════════════════════════

type TabKey = "build" | "depth" | "importance";

interface TabDef {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabDef[] = [
  { key: "build", label: "Build a Tree", icon: <MousePointer className="w-3.5 h-3.5" /> },
  { key: "depth", label: "Depth & Pruning", icon: <Layers className="w-3.5 h-3.5" /> },
  { key: "importance", label: "Feature Importance", icon: <BarChart3 className="w-3.5 h-3.5" /> },
];

// ════════════════════════════════════════════════════════════════════════════
// TAB 1 — BUILD A TREE MANUALLY
// ════════════════════════════════════════════════════════════════════════════

const TAB1_DATA = genData2D(42, 40);

interface ManualSplit {
  feat: number;
  thresh: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

function findLeafBounds(
  node: TNode,
  target: TNode,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
): { x0: number; x1: number; y0: number; y1: number } | null {
  if (node === target) return { x0, x1, y0, y1 };
  if (node.split) {
    const f = node.split.feat;
    const th = node.split.thresh;
    if (f === 0) {
      if (node.left) {
        const r = findLeafBounds(node.left, target, x0, th, y0, y1);
        if (r) return r;
      }
      if (node.right) {
        const r = findLeafBounds(node.right, target, th, x1, y0, y1);
        if (r) return r;
      }
    } else {
      if (node.left) {
        const r = findLeafBounds(node.left, target, x0, x1, y0, th);
        if (r) return r;
      }
      if (node.right) {
        const r = findLeafBounds(node.right, target, x0, x1, th, y1);
        if (r) return r;
      }
    }
  }
  return null;
}

function collectLeaves(node: TNode): TNode[] {
  if (!node.split) return [node];
  const leaves: TNode[] = [];
  if (node.left) leaves.push(...collectLeaves(node.left));
  if (node.right) leaves.push(...collectLeaves(node.right));
  return leaves;
}

function findLeafForPoint(node: TNode, data: Pt[], i: number): TNode {
  if (!node.split) return node;
  const val = node.split.feat === 0 ? data[i].x : data[i].y;
  if (val < node.split.thresh) return node.left ? findLeafForPoint(node.left, data, i) : node;
  return node.right ? findLeafForPoint(node.right, data, i) : node;
}

function BuildTreeTab() {
  const data = TAB1_DATA;
  const allIdx = useMemo(() => data.map((_, i) => i), []);

  const [tree, setTree] = useState<TNode>(() => ({
    idx: allIdx,
    split: null,
    left: null,
    right: null,
    depth: 0,
  }));

  const [splitMode, setSplitMode] = useState<"vertical" | "horizontal">("vertical");
  const [history, setHistory] = useState<TNode[]>([]);

  const acc = useMemo(() => accuracy(allIdx, data, tree), [tree, allIdx]);
  const giniVal = useMemo(() => gini(allIdx, data), [allIdx]);
  const nLeaves = useMemo(() => countLeaves(tree), [tree]);

  const handleClickSVG = useCallback(
    (dataX: number, dataY: number, _feat: number) => {
      const feat = splitMode === "vertical" ? 0 : 1;
      const thresh = feat === 0 ? dataX : dataY;

      setTree((prev) => {
        const sampleIdx = Math.floor(data.length / 2);
        const targetLeaf = findLeafForPoint(prev, data, sampleIdx);

        const leaves = collectLeaves(prev);
        let bestLeaf: TNode | null = null;
        let bestDist = Infinity;
        for (const leaf of leaves) {
          if (leaf.idx.length < 2 || leaf.depth >= 8) continue;
          const bounds = findLeafBounds(prev, leaf, 0, 1, 0, 1);
          if (!bounds) continue;
          if (feat === 0) {
            if (thresh >= bounds.x0 && thresh <= bounds.x1) {
              const centerX = (bounds.x0 + bounds.x1) / 2;
              const centerY = (bounds.y0 + bounds.y1) / 2;
              const dist = Math.abs(thresh - centerX) + Math.abs(dataY - centerY) * 0.01;
              if (dist < bestDist) {
                bestDist = dist;
                bestLeaf = leaf;
              }
            }
          } else {
            if (thresh >= bounds.y0 && thresh <= bounds.y1) {
              const centerX = (bounds.x0 + bounds.x1) / 2;
              const centerY = (bounds.y0 + bounds.y1) / 2;
              const dist = Math.abs(thresh - centerY) + Math.abs(dataX - centerX) * 0.01;
              if (dist < bestDist) {
                bestDist = dist;
                bestLeaf = leaf;
              }
            }
          }
        }

        if (!bestLeaf) return prev;

        const lI = bestLeaf.idx.filter((i) => (feat === 0 ? data[i].x : data[i].y) < thresh);
        const rI = bestLeaf.idx.filter((i) => (feat === 0 ? data[i].x : data[i].y) >= thresh);
        if (lI.length === 0 || rI.length === 0) return prev;

        setHistory((h) => [...h, cloneNode(prev)]);

        const newTree = cloneNode(prev);
        const leaves2 = collectLeaves(newTree);
        for (const leaf of leaves2) {
          if (
            leaf.idx.length === bestLeaf.idx.length &&
            leaf.depth === bestLeaf.depth &&
            leaf.idx[0] === bestLeaf.idx[0]
          ) {
            leaf.split = { feat, thresh };
            leaf.left = { idx: lI, split: null, left: null, right: null, depth: leaf.depth + 1 };
            leaf.right = { idx: rI, split: null, left: null, right: null, depth: leaf.depth + 1 };
            break;
          }
        }
        return newTree;
      });
    },
    [splitMode, data],
  );

  const handleUndo = useCallback(() => {
    setHistory((h) => {
      if (h.length === 0) return h;
      const prev = h[h.length - 1];
      setTree(prev);
      return h.slice(0, -1);
    });
  }, []);

  const handleReset = useCallback(() => {
    setTree({ idx: allIdx, split: null, left: null, right: null, depth: 0 });
    setHistory([]);
  }, [allIdx]);

  const handleAuto = useCallback(() => {
    setHistory((h) => [...h, cloneNode(tree)]);
    const newTree = buildTree2D(data, allIdx, 4);
    setTree(newTree);
  }, [tree, data, allIdx]);

  const previewLine = null;

  return (
    <div className="space-y-4">
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
        <p className="text-xs text-emerald-700">
          Click on the scatter plot to place split lines. Choose <strong>vertical</strong> (splits on X1) or <strong>horizontal</strong> (splits on X2). Each click finds the best leaf to split at that position. The tree diagram grows as you add splits. Try to separate the blue and orange points!
        </p>
      </div>

      <div className="flex gap-3 items-center flex-wrap">
        <div className="flex gap-1">
          <button
            onClick={() => setSplitMode("vertical")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${splitMode === "vertical" ? "bg-emerald-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}
          >
            | Vertical (X1)
          </button>
          <button
            onClick={() => setSplitMode("horizontal")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${splitMode === "horizontal" ? "bg-emerald-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}
          >
            — Horizontal (X2)
          </button>
        </div>
        <button onClick={handleUndo} disabled={history.length === 0} className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-100 text-amber-700 hover:bg-amber-200 disabled:opacity-40 transition-all">
          <Undo2 className="w-3.5 h-3.5" /> Undo
        </button>
        <button onClick={handleAuto} className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold bg-violet-100 text-violet-700 hover:bg-violet-200 transition-all">
          <Play className="w-3.5 h-3.5" /> Auto Build
        </button>
        <button onClick={handleReset} className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-100 text-red-700 hover:bg-red-200 transition-all">
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-emerald-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-emerald-700 font-semibold uppercase">Accuracy</p>
          <p className="text-xl font-bold text-emerald-800">{(acc * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-slate-500 font-semibold uppercase">Leaves</p>
          <p className="text-xl font-bold text-slate-800">{nLeaves}</p>
        </div>
        <div className="bg-amber-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-amber-600 font-semibold uppercase">Root Gini</p>
          <p className="text-xl font-bold text-amber-800">{giniVal.toFixed(3)}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
            <Scissors className="w-3.5 h-3.5 text-emerald-500" />
            <span className="text-xs font-semibold text-slate-700">Feature Space — Click to Split</span>
          </div>
          <div className="p-2">
            <ScatterWithRegions data={data} tree={tree} w={W} h={H} onClickSVG={handleClickSVG} previewLine={previewLine} />
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
            <GitBranch className="w-3.5 h-3.5 text-emerald-500" />
            <span className="text-xs font-semibold text-slate-700">Decision Tree</span>
          </div>
          <div className="p-2">
            <TreeDiagram tree={tree} data={data} w={W} h={H} featNames={["X1", "X2"]} />
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-slate-700 mb-2">Leaf Breakdown</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {collectLeaves(tree).map((leaf, i) => {
            const g = gini(leaf.idx, data);
            const mc = majority(leaf.idx, data);
            return (
              <div key={i} className="rounded-lg p-2 text-center" style={{ backgroundColor: CLS_BG[mc] }}>
                <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ backgroundColor: CLS_C[mc] }} />
                <p className="text-[10px] font-bold text-slate-700">n={leaf.idx.length}</p>
                <p className="text-[10px] text-slate-500">Gini={g.toFixed(3)}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 2 — DEPTH & PRUNING
// ════════════════════════════════════════════════════════════════════════════

function DepthPruningTab() {
  const rng = useMemo(() => mulberry32(99), []);
  const allData = useMemo(() => genData2D(99, 60), []);
  const trainIdx = useMemo(() => {
    const idx: number[] = [];
    for (let i = 0; i < allData.length; i++) {
      if (i % 3 !== 0) idx.push(i);
    }
    return idx;
  }, [allData]);
  const testIdx = useMemo(() => {
    const idx: number[] = [];
    for (let i = 0; i < allData.length; i++) {
      if (i % 3 === 0) idx.push(i);
    }
    return idx;
  }, [allData]);
  const trainData = allData;

  const [maxDepth, setMaxDepth] = useState(4);
  const [showPruned, setShowPruned] = useState(false);

  const tree = useMemo(() => buildTree2D(trainData, trainIdx, maxDepth), [trainData, trainIdx, maxDepth]);
  const prunedTree = useMemo(() => pruneTree(cloneNode(tree), testIdx, trainData), [tree, testIdx, trainData]);

  const displayTree = showPruned ? prunedTree : tree;

  const depthCurve = useMemo(() => {
    const pts: { d: number; trainAcc: number; testAcc: number }[] = [];
    for (let d = 1; d <= 10; d++) {
      const t = buildTree2D(trainData, trainIdx, d);
      pts.push({
        d,
        trainAcc: accuracy(trainIdx, trainData, t),
        testAcc: accuracy(testIdx, trainData, t),
      });
    }
    return pts;
  }, [trainData, trainIdx, testIdx]);

  const trainAcc = accuracy(trainIdx, trainData, displayTree);
  const testAcc = accuracy(testIdx, trainData, displayTree);

  const chartW2 = 380;
  const chartH2 = 200;
  const cPad2 = 45;

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-purple-500 shrink-0 mt-0.5" />
        <p className="text-xs text-purple-700">
          A deeper tree fits training data better but may <strong>overfit</strong> — memorizing noise instead of learning patterns. Use the depth slider to see how the decision boundary changes. Notice how training accuracy keeps rising but test accuracy peaks then drops. <strong>Pruning</strong> removes unhelpful leaves to improve generalization.
        </p>
      </div>

      <div className="flex gap-4 items-end flex-wrap">
        <div className="flex-1 min-w-50">
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs font-semibold text-slate-700">Max Depth</label>
            <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">{maxDepth}</span>
          </div>
          <input type="range" min={1} max={10} step={1} value={maxDepth} onChange={(e) => { setMaxDepth(parseInt(e.target.value)); setShowPruned(false); }} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500" />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>1 (underfit)</span>
            <span>10 (overfit)</span>
          </div>
        </div>
        <button
          onClick={() => setShowPruned(!showPruned)}
          className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${showPruned ? "bg-purple-600 text-white" : "bg-purple-100 text-purple-700 hover:bg-purple-200"}`}
        >
          {showPruned ? "Showing Pruned" : "Prune Tree"}
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-emerald-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-emerald-600 font-semibold uppercase">Train Acc</p>
          <p className="text-lg font-bold text-emerald-800">{(trainAcc * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-blue-600 font-semibold uppercase">Test Acc</p>
          <p className="text-lg font-bold text-blue-800">{(testAcc * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-slate-500 font-semibold uppercase">Nodes</p>
          <p className="text-lg font-bold text-slate-800">{countNodes(displayTree)}</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-slate-500 font-semibold uppercase">Leaves</p>
          <p className="text-lg font-bold text-slate-800">{countLeaves(displayTree)}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
            <span className="text-xs font-semibold text-slate-700">
              Decision Regions {showPruned ? "(Pruned)" : `(Depth ${maxDepth})`}
            </span>
          </div>
          <div className="p-2">
            <ScatterWithRegions
              data={allData}
              tree={displayTree}
              w={W}
              h={H}
              highlightIdx={new Set(testIdx)}
            />
          </div>
          <div className="px-3 pb-2 flex gap-3 text-[10px] text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full border-2 border-amber-400 bg-transparent inline-block" /> Test points (highlighted)
            </span>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
            <span className="text-xs font-semibold text-slate-700">Tree Structure</span>
          </div>
          <div className="p-2">
            <TreeDiagram tree={displayTree} data={allData} w={W} h={H} featNames={["X1", "X2"]} />
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
          <span className="text-xs font-semibold text-slate-700">Train vs Test Accuracy by Depth (Overfitting Chart)</span>
        </div>
        <div className="p-3">
          <svg viewBox={`0 0 ${chartW2} ${chartH2}`} className="w-full" style={{ maxHeight: 220 }}>
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={`gy-${v}`}>
                <line x1={cPad2} y1={cPad2 / 2 + (1 - v) * (chartH2 - cPad2 * 1.2)} x2={chartW2 - 20} y2={cPad2 / 2 + (1 - v) * (chartH2 - cPad2 * 1.2)} stroke="#f1f5f9" strokeWidth={0.5} />
                <text x={cPad2 - 5} y={cPad2 / 2 + (1 - v) * (chartH2 - cPad2 * 1.2) + 3} textAnchor="end" fontSize={8} fill="#94a3b8">{(v * 100).toFixed(0)}%</text>
              </g>
            ))}
            {depthCurve.map((pt) => {
              const xPos = cPad2 + ((pt.d - 1) / 9) * (chartW2 - cPad2 - 20);
              return (
                <g key={`gx-${pt.d}`}>
                  <text x={xPos} y={chartH2 - 5} textAnchor="middle" fontSize={8} fill="#94a3b8">{pt.d}</text>
                  {pt.d === maxDepth && (
                    <line x1={xPos} y1={cPad2 / 2} x2={xPos} y2={chartH2 - cPad2 + 10} stroke="#a855f7" strokeWidth={1.5} strokeDasharray="4,2" opacity={0.5} />
                  )}
                </g>
              );
            })}
            <text x={chartW2 / 2} y={chartH2 + 2} textAnchor="middle" fontSize={9} fill="#64748b">Max Depth</text>
            {/* Train accuracy line */}
            <polyline
              points={depthCurve
                .map((pt) => {
                  const xPos = cPad2 + ((pt.d - 1) / 9) * (chartW2 - cPad2 - 20);
                  const yPos = cPad2 / 2 + (1 - pt.trainAcc) * (chartH2 - cPad2 * 1.2);
                  return `${xPos},${yPos}`;
                })
                .join(" ")}
              fill="none"
              stroke="#10b981"
              strokeWidth={2}
            />
            {depthCurve.map((pt) => {
              const xPos = cPad2 + ((pt.d - 1) / 9) * (chartW2 - cPad2 - 20);
              const yPos = cPad2 / 2 + (1 - pt.trainAcc) * (chartH2 - cPad2 * 1.2);
              return <circle key={`tr-${pt.d}`} cx={xPos} cy={yPos} r={pt.d === maxDepth ? 5 : 3} fill="#10b981" stroke="#fff" strokeWidth={1} />;
            })}
            {/* Test accuracy line */}
            <polyline
              points={depthCurve
                .map((pt) => {
                  const xPos = cPad2 + ((pt.d - 1) / 9) * (chartW2 - cPad2 - 20);
                  const yPos = cPad2 / 2 + (1 - pt.testAcc) * (chartH2 - cPad2 * 1.2);
                  return `${xPos},${yPos}`;
                })
                .join(" ")}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={2}
            />
            {depthCurve.map((pt) => {
              const xPos = cPad2 + ((pt.d - 1) / 9) * (chartW2 - cPad2 - 20);
              const yPos = cPad2 / 2 + (1 - pt.testAcc) * (chartH2 - cPad2 * 1.2);
              return <circle key={`te-${pt.d}`} cx={xPos} cy={yPos} r={pt.d === maxDepth ? 5 : 3} fill="#3b82f6" stroke="#fff" strokeWidth={1} />;
            })}
            {/* Legend */}
            <line x1={chartW2 - 110} y1={15} x2={chartW2 - 95} y2={15} stroke="#10b981" strokeWidth={2} />
            <text x={chartW2 - 92} y={18} fontSize={8} fill="#10b981" fontWeight={600}>Train</text>
            <line x1={chartW2 - 60} y1={15} x2={chartW2 - 45} y2={15} stroke="#3b82f6" strokeWidth={2} />
            <text x={chartW2 - 42} y={18} fontSize={8} fill="#3b82f6" fontWeight={600}>Test</text>
          </svg>
        </div>
      </div>

      {showPruned && (
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
          <p className="text-xs text-purple-800 font-medium mb-1">Pruning Result</p>
          <p className="text-xs text-purple-700">
            Before pruning: {countLeaves(tree)} leaves, {countNodes(tree)} nodes.
            After pruning: {countLeaves(prunedTree)} leaves, {countNodes(prunedTree)} nodes.
            Pruning removed {countLeaves(tree) - countLeaves(prunedTree)} leaves that did not improve test accuracy.
          </p>
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// TAB 3 — FEATURE IMPORTANCE
// ════════════════════════════════════════════════════════════════════════════

const FEAT_NAMES = ["Height", "Weight", "Age", "Score"];

function FeatureImportanceTab() {
  const data = useMemo(() => genData4F(123, 80), []);
  const allIdx = useMemo(() => data.map((_, i) => i), []);
  const trainIdx = useMemo(() => allIdx.filter((_, i) => i % 4 !== 0), [allIdx]);
  const testIdx = useMemo(() => allIdx.filter((_, i) => i % 4 === 0), [allIdx]);

  const [removedFeats, setRemovedFeats] = useState<Set<number>>(new Set());
  const [showPermutation, setShowPermutation] = useState(false);

  const activeFeats = useMemo(() => [0, 1, 2, 3].filter((f) => !removedFeats.has(f)), [removedFeats]);

  const tree = useMemo(
    () => buildTree4F(data, trainIdx, 5, 0, activeFeats.length > 0 ? activeFeats : undefined),
    [data, trainIdx, activeFeats],
  );

  const trainAcc = useMemo(() => {
    let correct = 0;
    for (const i of trainIdx) {
      if (predictNode4F(tree, data, i) === data[i].cls) correct++;
    }
    return correct / trainIdx.length;
  }, [tree, data, trainIdx]);

  const testAcc = useMemo(() => {
    let correct = 0;
    for (const i of testIdx) {
      if (predictNode4F(tree, data, i) === data[i].cls) correct++;
    }
    return correct / testIdx.length;
  }, [tree, data, testIdx]);

  const treeImportance = useMemo(() => {
    const imp = [0, 0, 0, 0];
    const walk = (node: TNode) => {
      if (!node.split) return;
      const parentG = gini(node.idx, data);
      const lG = node.left ? gini(node.left.idx, data) : 0;
      const rG = node.right ? gini(node.right.idx, data) : 0;
      const lN = node.left ? node.left.idx.length : 0;
      const rN = node.right ? node.right.idx.length : 0;
      const wG = (lN * lG + rN * rG) / node.idx.length;
      imp[node.split.feat] += (parentG - wG) * node.idx.length;
      if (node.left) walk(node.left);
      if (node.right) walk(node.right);
    };
    walk(tree);
    const total = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map((v) => v / total);
  }, [tree, data]);

  const permImportance = useMemo(() => {
    const baseAcc = testAcc;
    const imp: number[] = [];
    for (let f = 0; f < 4; f++) {
      if (removedFeats.has(f)) {
        imp.push(0);
        continue;
      }
      const rng = mulberry32(300 + f);
      const shuffled = testIdx.map((i) => data[i].f[f]);
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      const tempData = data.map((d, di) => {
        const tIdx = testIdx.indexOf(di);
        if (tIdx === -1) return d;
        return { ...d, f: d.f.map((v, fi) => (fi === f ? shuffled[tIdx] : v)) };
      });
      let correct = 0;
      for (const i of testIdx) {
        if (predictNode4F(tree, tempData, i) === data[i].cls) correct++;
      }
      const permAcc = correct / testIdx.length;
      imp.push(Math.max(0, baseAcc - permAcc));
    }
    const total = imp.reduce((a, b) => a + b, 0) || 1;
    return imp.map((v) => v / total);
  }, [tree, data, testIdx, testAcc, removedFeats]);

  const removalAccs = useMemo(() => {
    const accs: { feat: number; acc: number }[] = [];
    for (let f = 0; f < 4; f++) {
      const feats = [0, 1, 2, 3].filter((x) => x !== f);
      const t = buildTree4F(data, trainIdx, 5, 0, feats);
      let correct = 0;
      for (const i of testIdx) {
        if (predictNode4F(t, data, i) === data[i].cls) correct++;
      }
      accs.push({ feat: f, acc: correct / testIdx.length });
    }
    return accs;
  }, [data, trainIdx, testIdx]);

  function toggleRemoveFeat(f: number) {
    setRemovedFeats((prev) => {
      const next = new Set(prev);
      if (next.has(f)) next.delete(f);
      else if (next.size < 3) next.add(f);
      return next;
    });
  }

  const barW = 360;
  const barH = 160;
  const bPad = 50;

  const maxImp = Math.max(...treeImportance, ...permImportance, 0.01);

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-3 flex gap-3">
        <Info className="w-5 h-5 text-teal-500 shrink-0 mt-0.5" />
        <p className="text-xs text-teal-700">
          Feature importance tells you which features the tree relies on most. <strong>Tree-based importance</strong> measures total Gini reduction from each feature. <strong>Permutation importance</strong> shuffles one feature at a time and measures accuracy drop. Try removing features to see how accuracy changes.
        </p>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium mb-2">Dataset Preview (first 8 rows)</p>
        <div className="overflow-x-auto">
          <table className="text-[10px] w-full">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="py-1 px-2 text-left text-slate-500">#</th>
                {FEAT_NAMES.map((name, f) => (
                  <th key={f} className={`py-1 px-2 text-left ${removedFeats.has(f) ? "text-slate-300 line-through" : "text-slate-700"}`}>{name}</th>
                ))}
                <th className="py-1 px-2 text-left text-slate-700">Class</th>
              </tr>
            </thead>
            <tbody>
              {data.slice(0, 8).map((d, i) => (
                <tr key={i} className="border-b border-slate-100">
                  <td className="py-1 px-2 text-slate-400">{i + 1}</td>
                  {d.f.map((v, f) => (
                    <td key={f} className={`py-1 px-2 font-mono ${removedFeats.has(f) ? "text-slate-300" : "text-slate-600"}`}>{v.toFixed(3)}</td>
                  ))}
                  <td className="py-1 px-2">
                    <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: CLS_C[d.cls] }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex gap-3 items-center flex-wrap">
        <span className="text-xs font-semibold text-slate-600">Toggle Features:</span>
        {FEAT_NAMES.map((name, f) => (
          <button
            key={f}
            onClick={() => toggleRemoveFeat(f)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${removedFeats.has(f) ? "bg-red-100 text-red-500 line-through" : "bg-emerald-100 text-emerald-700"}`}
          >
            {name}
          </button>
        ))}
        <button
          onClick={() => setShowPermutation(!showPermutation)}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${showPermutation ? "bg-violet-600 text-white" : "bg-violet-100 text-violet-700"}`}
        >
          {showPermutation ? "Showing Permutation" : "Show Permutation Imp."}
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-emerald-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-emerald-600 font-semibold uppercase">Train Acc</p>
          <p className="text-lg font-bold text-emerald-800">{(trainAcc * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-blue-600 font-semibold uppercase">Test Acc</p>
          <p className="text-lg font-bold text-blue-800">{(testAcc * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-slate-500 font-semibold uppercase">Active Feats</p>
          <p className="text-lg font-bold text-slate-800">{4 - removedFeats.size}</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <p className="text-[10px] text-slate-500 font-semibold uppercase">Tree Leaves</p>
          <p className="text-lg font-bold text-slate-800">{countLeaves(tree)}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
            <span className="text-xs font-semibold text-slate-700">
              {showPermutation ? "Permutation Importance" : "Tree-Based Importance"} (Gini Reduction)
            </span>
          </div>
          <div className="p-3">
            <svg viewBox={`0 0 ${barW} ${barH}`} className="w-full" style={{ maxHeight: barH }}>
              {FEAT_NAMES.map((name, f) => {
                const val = showPermutation ? permImportance[f] : treeImportance[f];
                const barMaxW = barW - bPad - 30;
                const barHeight = 22;
                const yPos = 10 + f * (barHeight + 12);
                const barWidth = (val / Math.max(maxImp, 0.001)) * barMaxW;
                const isRemoved = removedFeats.has(f);
                const isTop = !isRemoved && val === Math.max(...(showPermutation ? permImportance : treeImportance).filter((_, i) => !removedFeats.has(i)));
                return (
                  <g key={f}>
                    <text x={bPad - 5} y={yPos + barHeight / 2 + 3} textAnchor="end" fontSize={9} fill={isRemoved ? "#cbd5e1" : "#475569"} fontWeight={600}>
                      {name}
                    </text>
                    <rect x={bPad} y={yPos} width={barMaxW} height={barHeight} rx={4} fill="#f1f5f9" />
                    <rect x={bPad} y={yPos} width={Math.max(barWidth, 0)} height={barHeight} rx={4} fill={isRemoved ? "#e2e8f0" : isTop ? "#f59e0b" : "#6366f1"} opacity={isRemoved ? 0.4 : 0.85} />
                    <text x={bPad + Math.max(barWidth, 0) + 5} y={yPos + barHeight / 2 + 3} fontSize={8} fill="#64748b" fontWeight={600}>
                      {(val * 100).toFixed(1)}%
                    </text>
                    {isTop && !isRemoved && (
                      <text x={bPad + barWidth / 2} y={yPos + barHeight / 2 + 3} textAnchor="middle" fontSize={7} fill="#fff" fontWeight={700}>
                        MOST IMPORTANT
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
            <span className="text-xs font-semibold text-slate-700">Accuracy When Removing Each Feature</span>
          </div>
          <div className="p-3">
            <svg viewBox={`0 0 ${barW} ${barH}`} className="w-full" style={{ maxHeight: barH }}>
              {removalAccs.map((ra, f) => {
                const barMaxW = barW - bPad - 40;
                const barHeight = 22;
                const yPos = 10 + f * (barHeight + 12);
                const barWidth = ra.acc * barMaxW;
                const drop = testAcc - ra.acc;
                return (
                  <g key={f}>
                    <text x={bPad - 5} y={yPos + barHeight / 2 + 3} textAnchor="end" fontSize={9} fill="#475569" fontWeight={600}>
                      w/o {FEAT_NAMES[f]}
                    </text>
                    <rect x={bPad} y={yPos} width={barMaxW} height={barHeight} rx={4} fill="#f1f5f9" />
                    <rect x={bPad} y={yPos} width={Math.max(barWidth, 0)} height={barHeight} rx={4} fill={drop > 0.05 ? "#ef4444" : drop > 0.01 ? "#f97316" : "#10b981"} opacity={0.75} />
                    <text x={bPad + Math.max(barWidth, 0) + 5} y={yPos + barHeight / 2 + 3} fontSize={8} fill="#64748b" fontWeight={600}>
                      {(ra.acc * 100).toFixed(1)}% ({drop > 0 ? "-" : "+"}{(Math.abs(drop) * 100).toFixed(1)}%)
                    </text>
                  </g>
                );
              })}
              <line x1={bPad + testAcc * (barW - bPad - 40)} y1={0} x2={bPad + testAcc * (barW - bPad - 40)} y2={barH} stroke="#6366f1" strokeWidth={1.5} strokeDasharray="4,2" opacity={0.6} />
              <text x={bPad + testAcc * (barW - bPad - 40)} y={barH - 2} textAnchor="middle" fontSize={7} fill="#6366f1" fontWeight={600}>
                baseline {(testAcc * 100).toFixed(1)}%
              </text>
            </svg>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
            <span className="text-xs font-semibold text-slate-700">Decision Tree Structure</span>
          </div>
          <div className="p-2">
            <TreeDiagram tree={tree} data={data} w={W} h={280} featNames={FEAT_NAMES} />
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
          <p className="text-[11px] text-slate-500 uppercase tracking-wide font-medium">Importance Comparison</p>
          <div className="space-y-2">
            {FEAT_NAMES.map((name, f) => (
              <div key={f} className="bg-slate-50 rounded-lg p-2.5">
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-xs font-bold ${removedFeats.has(f) ? "text-slate-300 line-through" : "text-slate-700"}`}>{name}</span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-[10px]">
                  <div>
                    <p className="text-indigo-500 font-semibold">Tree-based</p>
                    <div className="h-2 bg-slate-200 rounded-full overflow-hidden mt-1">
                      <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${treeImportance[f] * 100}%` }} />
                    </div>
                    <p className="text-slate-500 mt-0.5">{(treeImportance[f] * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-violet-500 font-semibold">Permutation</p>
                    <div className="h-2 bg-slate-200 rounded-full overflow-hidden mt-1">
                      <div className="h-full bg-violet-500 rounded-full" style={{ width: `${permImportance[f] * 100}%` }} />
                    </div>
                    <p className="text-slate-500 mt-0.5">{(permImportance[f] * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-slate-500 italic">
            Tree-based importance can overestimate features with many possible splits. Permutation importance is more reliable but can underestimate correlated features.
          </p>
        </div>
      </div>
    </div>
  );
}

function predictNode4F(node: TNode, data: Pt4[], i: number): number {
  if (!node.split) return majority(node.idx, data);
  const val = data[i].f[node.split.feat];
  if (val < node.split.thresh) return node.left ? predictNode4F(node.left, data, i) : majority(node.idx, data);
  return node.right ? predictNode4F(node.right, data, i) : majority(node.idx, data);
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════════════════

export default function DecisionTreeActivity() {
  const [activeTab, setActiveTab] = useState<TabKey>("build");

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
                ? "bg-emerald-50 text-emerald-700 border-emerald-500"
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
        {activeTab === "build" && <BuildTreeTab />}
        {activeTab === "depth" && <DepthPruningTab />}
        {activeTab === "importance" && <FeatureImportanceTab />}
      </div>
    </div>
  );
}
