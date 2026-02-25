/**
 * ForestTreePathAnimation — Real SVG tree visualizations for RF prediction.
 *
 * Each tree is rendered as an actual tree diagram (like DecisionTreeAnimation)
 * with the prediction path highlighted. Trees appear one by one, then votes
 * converge to the final ensemble prediction.
 */

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, TreePine } from "lucide-react";

/* ═══════════════════════════════════════════════ */
/*  Types                                          */
/* ═══════════════════════════════════════════════ */

interface TreeNodeData {
  id: number;
  type: "internal" | "leaf";
  depth: number;
  n_samples: number;
  impurity: number;
  on_path: boolean;
  feature?: string;
  threshold?: number;
  left_child?: number;
  right_child?: number;
  class_label?: string;
  value?: number;
}

interface PerTreePrediction {
  tree_index: number;
  prediction: string;
  numeric_value?: number;
  probabilities?: Record<string, number>;
  tree_structure?: TreeNodeData[];
  tree_depth?: number;
  n_leaves?: number;
}

interface LayoutNode {
  id: number;
  parentId: number | null;
  type: "internal" | "leaf";
  x: number;
  y: number;
  onPath: boolean;
  feature?: string;
  threshold?: number;
  classLabel?: string;
  samples: number;
  color?: string;
  side?: "left" | "right";
  value?: number;
}

interface ForestTreePathAnimationProps {
  perTreePredictions: PerTreePrediction[];
  classColorMap: Record<string, string>;
  finalPrediction: string;
  voteSummary?: Record<string, number>;
  regressionMean?: number;
  isRegression: boolean;
}

/* ═══════════════════════════════════════════════ */
/*  Constants                                      */
/* ═══════════════════════════════════════════════ */

const TREE_COLORS = [
  "#0d9488", "#7c3aed", "#ea580c", "#2563eb", "#db2777",
  "#16a34a", "#f59e0b", "#6366f1", "#84cc16", "#e11d48",
];

// Mini-tree SVG dimensions
const MINI_W = 260;
const MINI_H = 220;
const NODE_W = 72;
const NODE_H_INT = 28;
const NODE_H_LEAF = 24;
const Y_TOP = 20;
const Y_GAP = 44;

/* ═══════════════════════════════════════════════ */
/*  Layout helper — same approach as DT animation  */
/* ═══════════════════════════════════════════════ */

function buildMiniLayout(
  nodes: TreeNodeData[],
  classColorMap: Record<string, string>,
): LayoutNode[] {
  if (!nodes || nodes.length === 0) return [];

  const byId = new Map<number, TreeNodeData>();
  for (const n of nodes) byId.set(n.id, n);

  // Parent + side maps
  const parentMap = new Map<number, number>();
  const sideMap = new Map<number, "left" | "right">();
  for (const n of nodes) {
    if (n.type === "internal") {
      if (n.left_child !== undefined) {
        parentMap.set(n.left_child, n.id);
        sideMap.set(n.left_child, "left");
      }
      if (n.right_child !== undefined) {
        parentMap.set(n.right_child, n.id);
        sideMap.set(n.right_child, "right");
      }
    }
  }

  // Subtree widths
  const subtreeW = new Map<number, number>();
  function computeW(nid: number): number {
    const node = byId.get(nid);
    if (!node || node.type === "leaf") {
      subtreeW.set(nid, 1);
      return 1;
    }
    let w = 0;
    if (node.left_child !== undefined && byId.has(node.left_child))
      w += computeW(node.left_child);
    if (node.right_child !== undefined && byId.has(node.right_child))
      w += computeW(node.right_child);
    if (w === 0) w = 1;
    subtreeW.set(nid, w);
    return w;
  }

  const root = nodes.find((n) => n.depth === 0);
  if (!root) return [];
  computeW(root.id);

  // X positions
  const totalW = subtreeW.get(root.id) || 1;
  const unitW = (MINI_W - 20) / totalW;
  const xPos = new Map<number, number>();

  function assignX(nid: number, left: number) {
    const node = byId.get(nid);
    if (!node) return;
    const w = subtreeW.get(nid) || 1;
    xPos.set(nid, left + (w * unitW) / 2);
    if (node.type === "internal") {
      let off = left;
      if (node.left_child !== undefined && byId.has(node.left_child)) {
        const lw = subtreeW.get(node.left_child) || 1;
        assignX(node.left_child, off);
        off += lw * unitW;
      }
      if (node.right_child !== undefined && byId.has(node.right_child)) {
        assignX(node.right_child, off);
      }
    }
  }
  assignX(root.id, 10);

  // BFS order
  const bfs: number[] = [];
  const queue = [root.id];
  while (queue.length > 0) {
    const id = queue.shift()!;
    bfs.push(id);
    const node = byId.get(id);
    if (node?.type === "internal") {
      if (node.left_child !== undefined && byId.has(node.left_child))
        queue.push(node.left_child);
      if (node.right_child !== undefined && byId.has(node.right_child))
        queue.push(node.right_child);
    }
  }

  return bfs.map((nid) => {
    const n = byId.get(nid)!;
    const y = Y_TOP + n.depth * Y_GAP;
    const x = xPos.get(nid) || MINI_W / 2;
    const layout: LayoutNode = {
      id: nid,
      parentId: parentMap.get(nid) ?? null,
      type: n.type,
      x,
      y,
      onPath: n.on_path,
      samples: n.n_samples,
      side: sideMap.get(nid),
    };
    if (n.type === "internal") {
      layout.feature = n.feature;
      layout.threshold = n.threshold;
    } else {
      if (n.class_label) {
        layout.classLabel = n.class_label;
        layout.color = classColorMap[n.class_label] || "#94a3b8";
      } else if (n.value !== undefined) {
        layout.classLabel = String(n.value);
        layout.color = "#3b82f6";
      }
    }
    return layout;
  });
}

/* ═══════════════════════════════════════════════ */
/*  MiniTreeSVG — one tree rendered as SVG         */
/* ═══════════════════════════════════════════════ */

function MiniTreeSVG({
  layoutNodes,
  visible,
  treeColor,
}: {
  layoutNodes: LayoutNode[];
  visible: boolean;
  treeColor: string;
}) {
  if (layoutNodes.length === 0) return null;
  const maxY = Math.max(...layoutNodes.map((n) => n.y));
  const svgH = Math.max(MINI_H, maxY + 40);
  const nodeById = new Map(layoutNodes.map((n) => [n.id, n]));

  return (
    <svg
      viewBox={`0 0 ${MINI_W} ${svgH}`}
      className="w-full h-auto"
      style={{ minHeight: 120 }}
    >
      {/* Grid background */}
      <defs>
        <pattern id={`mg-${treeColor.replace("#", "")}`} width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f8fafc" strokeWidth={0.5} />
        </pattern>
      </defs>
      <rect width={MINI_W} height={svgH} fill={`url(#mg-${treeColor.replace("#", "")})`} rx={8} />

      {/* Edges */}
      {layoutNodes.map((node) => {
        if (node.parentId === null) return null;
        const parent = nodeById.get(node.parentId);
        if (!parent) return null;
        const pH = parent.type === "internal" ? NODE_H_INT : NODE_H_LEAF;
        const cH = node.type === "internal" ? NODE_H_INT : NODE_H_LEAF;
        const bothOnPath = node.onPath && parent.onPath;

        return (
          <g key={`e-${node.id}`}>
            <line
              x1={parent.x}
              y1={parent.y + pH / 2}
              x2={node.x}
              y2={node.y - cH / 2}
              stroke={bothOnPath ? treeColor : "#cbd5e1"}
              strokeWidth={bothOnPath ? 2.5 : 1.5}
              opacity={visible ? (bothOnPath ? 1 : 0.4) : 0}
              style={{ transition: "opacity 0.4s ease" }}
            />
            {visible && (
              <text
                x={(parent.x + node.x) / 2 + (node.side === "left" ? -8 : 8)}
                y={(parent.y + pH / 2 + node.y - cH / 2) / 2}
                textAnchor="middle"
                fontSize={8}
                fontWeight={700}
                fill={bothOnPath ? (node.side === "left" ? "#16a34a" : "#dc2626") : "#94a3b8"}
                opacity={bothOnPath ? 1 : 0.5}
              >
                {node.side === "left" ? "Y" : "N"}
              </text>
            )}
          </g>
        );
      })}

      {/* Nodes */}
      {layoutNodes.map((node) => {
        const h = node.type === "internal" ? NODE_H_INT : NODE_H_LEAF;
        const w = NODE_W;
        const onPath = node.onPath;

        if (node.type === "internal") {
          const featureLabel = (node.feature || "").length > 8
            ? (node.feature || "").slice(0, 7) + "…"
            : node.feature || "";
          return (
            <g
              key={`n-${node.id}`}
              opacity={visible ? 1 : 0}
              style={{ transition: "opacity 0.4s ease" }}
            >
              <rect
                x={node.x - w / 2}
                y={node.y - h / 2}
                width={w}
                height={h}
                rx={6}
                fill={onPath ? "#eff6ff" : "#f8fafc"}
                stroke={onPath ? "#3b82f6" : "#e2e8f0"}
                strokeWidth={onPath ? 2 : 1}
              />
              <text
                x={node.x}
                y={node.y + 1}
                textAnchor="middle"
                fontSize={9}
                fontWeight={onPath ? 700 : 500}
                fill={onPath ? "#1e40af" : "#94a3b8"}
              >
                {featureLabel} &lt; {node.threshold}
              </text>
            </g>
          );
        }

        // Leaf
        const lw = w - 10;
        return (
          <g
            key={`n-${node.id}`}
            opacity={visible ? 1 : 0}
            style={{ transition: "opacity 0.4s ease" }}
          >
            <rect
              x={node.x - lw / 2}
              y={node.y - h / 2}
              width={lw}
              height={h}
              rx={5}
              fill={onPath ? (node.color ? `${node.color}25` : "#f0fdf4") : "#f8fafc"}
              stroke={onPath ? (node.color || "#22c55e") : "#e2e8f0"}
              strokeWidth={onPath ? 2 : 1}
            />
            <text
              x={node.x}
              y={node.y + 1}
              textAnchor="middle"
              fontSize={9}
              fontWeight={onPath ? 800 : 500}
              fill={onPath ? (node.color || "#166534") : "#94a3b8"}
            >
              {(node.classLabel || "").length > 10
                ? (node.classLabel || "").slice(0, 9) + "…"
                : node.classLabel}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ═══════════════════════════════════════════════ */
/*  Main Component                                 */
/* ═══════════════════════════════════════════════ */

export default function ForestTreePathAnimation({
  perTreePredictions,
  classColorMap,
  finalPrediction,
  voteSummary,
  regressionMean,
  isRegression,
}: ForestTreePathAnimationProps) {
  const [playing, setPlaying] = useState(false);
  const [visibleTreeCount, setVisibleTreeCount] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animKey = useRef(0);

  const treesWithData = useMemo(
    () => perTreePredictions.filter((t) => t.tree_structure && t.tree_structure.length > 0),
    [perTreePredictions],
  );

  // Pre-compute layouts for all trees
  const treeLayouts = useMemo(
    () => treesWithData.map((t) => buildMiniLayout(t.tree_structure!, classColorMap)),
    [treesWithData, classColorMap],
  );

  // Reset on data change
  useEffect(() => {
    animKey.current += 1;
    setVisibleTreeCount(0);
    setShowResult(false);
    setPlaying(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  }, [perTreePredictions]);

  const reset = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    animKey.current += 1;
    setVisibleTreeCount(0);
    setShowResult(false);
    setPlaying(false);
  }, []);

  // Animation: reveal trees one by one
  useEffect(() => {
    if (!playing) return;
    const key = animKey.current;

    const step = () => {
      if (animKey.current !== key) return;
      setVisibleTreeCount((prev) => {
        const next = prev + 1;
        if (next >= treesWithData.length) {
          timerRef.current = setTimeout(() => {
            if (animKey.current !== key) return;
            setShowResult(true);
            setPlaying(false);
          }, 500);
          return treesWithData.length;
        }
        timerRef.current = setTimeout(step, 400);
        return next;
      });
    };

    timerRef.current = setTimeout(step, 300);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [playing, treesWithData.length]);

  const handlePlayPause = () => {
    if (showResult) {
      reset();
      setTimeout(() => setPlaying(true), 50);
    } else if (playing) {
      setPlaying(false);
    } else {
      setPlaying(true);
    }
  };

  const showAll = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setPlaying(false);
    setVisibleTreeCount(treesWithData.length);
    setShowResult(true);
  };

  if (treesWithData.length === 0) return null;

  const allRevealed = visibleTreeCount >= treesWithData.length;

  return (
    <div className="space-y-4">
      {/* Header + controls */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
          <TreePine size={15} className="text-teal-600" />
          Forest Decision Paths ({treesWithData.length} trees)
        </h4>
        <div className="flex items-center gap-2">
          <button
            onClick={handlePlayPause}
            className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-teal-50 text-teal-700 hover:bg-teal-100 transition font-medium"
          >
            {playing ? <Pause size={12} /> : <Play size={12} />}
            {showResult ? "Replay" : playing ? "Pause" : "Animate"}
          </button>
          {(visibleTreeCount > 0 || showResult) && (
            <button
              onClick={reset}
              className="inline-flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition"
            >
              <RotateCcw size={12} />
            </button>
          )}
          {!showResult && visibleTreeCount === 0 && (
            <button
              onClick={showAll}
              className="text-xs px-2.5 py-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition"
            >
              Show All
            </button>
          )}
        </div>
      </div>

      {/* Info text */}
      <p className="text-xs text-gray-500">
        Each tree independently decides by following splits from root to leaf.
        The <span className="font-semibold text-blue-600">highlighted path</span> shows
        the route your input takes. Non-path nodes are dimmed.
      </p>

      {/* Tree grid — 2 cols on sm, 5 cols on lg */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        {treesWithData.map((tree, idx) => {
          const isVisible = idx < visibleTreeCount || showResult || visibleTreeCount === 0;
          const isActive = idx === visibleTreeCount - 1 && playing;
          const treeColor = TREE_COLORS[idx % TREE_COLORS.length];
          const predColor = isRegression
            ? treeColor
            : classColorMap[tree.prediction] || treeColor;
          const layout = treeLayouts[idx];

          return (
            <div
              key={idx}
              className={`rounded-xl border overflow-hidden transition-all duration-300 ${
                isActive
                  ? "border-teal-400 shadow-lg shadow-teal-100/60 scale-[1.02]"
                  : isVisible
                  ? "border-gray-200 shadow-sm"
                  : "border-gray-100 opacity-30"
              }`}
              style={{ backgroundColor: isVisible ? "#fff" : "#fafafa" }}
            >
              {/* Header */}
              <div
                className="px-3 py-1.5 flex items-center justify-between border-b"
                style={{
                  backgroundColor: treeColor + "0a",
                  borderBottomColor: treeColor + "20",
                }}
              >
                <div className="flex items-center gap-1.5">
                  <TreePine size={11} style={{ color: treeColor }} />
                  <span className="text-[11px] font-bold" style={{ color: treeColor }}>
                    Tree {idx + 1}
                  </span>
                </div>
                {isVisible && (
                  <span
                    className="text-[10px] font-bold px-2 py-0.5 rounded-full transition-all duration-300"
                    style={{
                      backgroundColor: predColor + "18",
                      color: predColor,
                      border: `1px solid ${predColor}30`,
                    }}
                  >
                    {tree.prediction}
                  </span>
                )}
              </div>

              {/* Tree SVG */}
              <div className="p-1">
                <MiniTreeSVG
                  layoutNodes={layout}
                  visible={isVisible}
                  treeColor={treeColor}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      {playing && (
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 rounded-full bg-gray-100 overflow-hidden">
            <div
              className="h-full rounded-full bg-teal-500 transition-all duration-300"
              style={{ width: `${(visibleTreeCount / treesWithData.length) * 100}%` }}
            />
          </div>
          <span className="text-[10px] font-medium text-gray-400 tabular-nums">
            {visibleTreeCount}/{treesWithData.length}
          </span>
        </div>
      )}

      {/* Vote summary badges */}
      {!isRegression && voteSummary && (allRevealed || showResult) && (
        <div className="flex flex-wrap gap-2 justify-center">
          {Object.entries(voteSummary)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, count]) => (
              <span
                key={cls}
                className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-full"
                style={{
                  backgroundColor: (classColorMap[cls] || "#6b7280") + "15",
                  color: classColorMap[cls] || "#6b7280",
                  border: `1.5px solid ${classColorMap[cls] || "#6b7280"}40`,
                }}
              >
                {cls}: {count} {count === 1 ? "tree" : "trees"}
              </span>
            ))}
        </div>
      )}

      {/* Ensemble result */}
      {(showResult || visibleTreeCount === 0) && (
        <div
          className={`rounded-xl border-2 p-4 text-center transition-all duration-500 ${
            showResult ? "opacity-100 scale-100" : "opacity-60 scale-[0.98]"
          }`}
          style={{
            borderColor: isRegression
              ? "#2563eb"
              : classColorMap[finalPrediction] || "#0d9488",
            backgroundColor: isRegression
              ? "#eff6ff"
              : (classColorMap[finalPrediction] || "#0d9488") + "08",
          }}
        >
          <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">
            {isRegression ? "Ensemble Average" : "Majority Vote"}
          </p>
          <p
            className="text-2xl font-extrabold"
            style={{
              color: isRegression
                ? "#2563eb"
                : classColorMap[finalPrediction] || "#0d9488",
            }}
          >
            {isRegression && regressionMean != null
              ? regressionMean.toFixed(4)
              : finalPrediction}
          </p>
          {!isRegression && voteSummary && (
            <div className="flex justify-center gap-3 mt-2">
              {Object.entries(voteSummary)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, count]) => (
                  <span
                    key={cls}
                    className="text-[10px] font-semibold"
                    style={{ color: classColorMap[cls] || "#6b7280" }}
                  >
                    {cls}: {count}
                  </span>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
