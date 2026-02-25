import { useState, useRef, useCallback, useMemo } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  Info,
  TreePine,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BackendTreeNode {
  id: number;
  type: "internal" | "leaf";
  depth: number;
  n_samples: number;
  impurity: number;
  feature?: string;
  threshold?: number;
  left_child?: number;
  right_child?: number;
  class_label?: string;
  class_distribution?: number[];
  value?: number;
}

interface LayoutNode {
  id: number;
  parentId: number | null;
  type: "internal" | "leaf";
  x: number;
  y: number;
  feature?: string;
  threshold?: number;
  gini?: number;
  classLabel?: string;
  samples: number;
  color?: string;
  side?: "left" | "right";
  /** For regression leaves */
  value?: number;
}

interface DecisionTreeAnimationProps {
  result?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Colour palette for up to 10 distinct classes
// ---------------------------------------------------------------------------

const CLASS_COLORS = [
  "#22c55e", // green
  "#f97316", // orange
  "#a855f7", // purple
  "#3b82f6", // blue
  "#ef4444", // red
  "#14b8a6", // teal
  "#f59e0b", // amber
  "#ec4899", // pink
  "#6366f1", // indigo
  "#84cc16", // lime
];

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

const NODE_W = 140;
const NODE_H_INTERNAL = 58;
const NODE_H_LEAF = 48;
const SVG_W = 600;
const SVG_H = 420;
const Y_PADDING_TOP = 45;
const Y_LEVEL_GAP = 100;

function buildLayoutNodes(backendNodes: BackendTreeNode[]): LayoutNode[] {
  if (!backendNodes || backendNodes.length === 0) return [];

  // Build a lookup map by id
  const byId = new Map<number, BackendTreeNode>();
  for (const n of backendNodes) byId.set(n.id, n);

  // Find max depth
  const maxDepth = Math.max(...backendNodes.map((n) => n.depth));

  // Collect unique class labels for color mapping
  const classLabels: string[] = [];
  for (const n of backendNodes) {
    if (n.type === "leaf" && n.class_label && !classLabels.includes(n.class_label)) {
      classLabels.push(n.class_label);
    }
  }
  const classColorMap = new Map<string, string>();
  classLabels.forEach((label, i) => {
    classColorMap.set(label, CLASS_COLORS[i % CLASS_COLORS.length]);
  });

  // Build parent map + side map from parent's left_child/right_child
  const parentMap = new Map<number, number>(); // childId -> parentId
  const sideMap = new Map<number, "left" | "right">();
  for (const n of backendNodes) {
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

  // Group nodes by depth for positioning
  const byDepth = new Map<number, BackendTreeNode[]>();
  for (const n of backendNodes) {
    const list = byDepth.get(n.depth) || [];
    list.push(n);
    byDepth.set(n.depth, list);
  }

  // Compute x positions using subtree size approach
  // First compute subtree widths
  const subtreeWidth = new Map<number, number>();
  function computeWidth(nodeId: number): number {
    const node = byId.get(nodeId);
    if (!node || node.type === "leaf") {
      subtreeWidth.set(nodeId, 1);
      return 1;
    }
    let w = 0;
    if (node.left_child !== undefined && byId.has(node.left_child)) {
      w += computeWidth(node.left_child);
    }
    if (node.right_child !== undefined && byId.has(node.right_child)) {
      w += computeWidth(node.right_child);
    }
    if (w === 0) w = 1;
    subtreeWidth.set(nodeId, w);
    return w;
  }

  // Root is the first node (id of depth 0)
  const rootNode = backendNodes.find((n) => n.depth === 0);
  if (!rootNode) return [];
  computeWidth(rootNode.id);

  // Assign x positions
  const xPos = new Map<number, number>();
  const totalWidth = subtreeWidth.get(rootNode.id) || 1;
  const unitWidth = (SVG_W - 40) / totalWidth; // leave some padding

  function assignX(nodeId: number, leftBound: number) {
    const node = byId.get(nodeId);
    if (!node) return;
    const w = subtreeWidth.get(nodeId) || 1;
    const cx = leftBound + (w * unitWidth) / 2;
    xPos.set(nodeId, cx);

    if (node.type === "internal") {
      let offset = leftBound;
      if (node.left_child !== undefined && byId.has(node.left_child)) {
        const lw = subtreeWidth.get(node.left_child) || 1;
        assignX(node.left_child, offset);
        offset += lw * unitWidth;
      }
      if (node.right_child !== undefined && byId.has(node.right_child)) {
        assignX(node.right_child, offset);
      }
    }
  }
  assignX(rootNode.id, 20);

  // BFS order for animation
  const bfsOrder: number[] = [];
  const queue = [rootNode.id];
  while (queue.length > 0) {
    const id = queue.shift()!;
    bfsOrder.push(id);
    const node = byId.get(id);
    if (node && node.type === "internal") {
      if (node.left_child !== undefined && byId.has(node.left_child)) queue.push(node.left_child);
      if (node.right_child !== undefined && byId.has(node.right_child)) queue.push(node.right_child);
    }
  }

  // Build LayoutNode array in BFS order
  return bfsOrder.map((nodeId) => {
    const n = byId.get(nodeId)!;
    const y = Y_PADDING_TOP + n.depth * Y_LEVEL_GAP;
    const x = xPos.get(nodeId) || SVG_W / 2;

    const layout: LayoutNode = {
      id: nodeId,
      parentId: parentMap.get(nodeId) ?? null,
      type: n.type,
      x,
      y,
      samples: n.n_samples,
      gini: n.impurity,
      side: sideMap.get(nodeId),
    };

    if (n.type === "internal") {
      layout.feature = n.feature;
      layout.threshold = n.threshold;
    } else {
      if (n.class_label) {
        layout.classLabel = n.class_label;
        layout.color = classColorMap.get(n.class_label) || "#94a3b8";
      } else if (n.value !== undefined) {
        layout.classLabel = `${n.value}`;
        layout.color = "#3b82f6";
      }
    }

    return layout;
  });
}

// Speed presets (ms per step)
const SPEED_MAP: Record<string, number> = {
  slow: 1400,
  medium: 800,
  fast: 400,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DecisionTreeAnimation = ({ result }: DecisionTreeAnimationProps) => {
  // Extract tree data from result
  const treeNodes = useMemo(() => {
    const fullMeta = (result?.metadata as Record<string, unknown>)?.full_training_metadata as Record<string, unknown> | undefined;
    const backendNodes = fullMeta?.tree_structure as BackendTreeNode[] | undefined;
    if (backendNodes && backendNodes.length > 0) {
      return buildLayoutNodes(backendNodes);
    }
    return [];
  }, [result]);

  const taskType = (result?.task_type as string) || "classification";
  const isRegression = taskType === "regression";

  // Extract class names for legend
  const classInfo = useMemo(() => {
    const classes: { label: string; color: string }[] = [];
    const seen = new Set<string>();
    for (const n of treeNodes) {
      if (n.type === "leaf" && n.classLabel && n.color && !seen.has(n.classLabel)) {
        seen.add(n.classLabel);
        classes.push({ label: n.classLabel, color: n.color });
      }
    }
    return classes;
  }, [treeNodes]);

  // How many nodes are currently visible
  const [visibleCount, setVisibleCount] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<string>("medium");

  const rafRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number>(0);

  // Computed SVG height based on tree depth
  const svgH = useMemo(() => {
    if (treeNodes.length === 0) return SVG_H;
    const maxY = Math.max(...treeNodes.map((n) => n.y));
    return Math.max(SVG_H, maxY + 80);
  }, [treeNodes]);

  // ------- Animation loop -------

  const stopLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const tick = useCallback(
    (timestamp: number) => {
      const interval = SPEED_MAP[speed] ?? 800;
      if (timestamp - lastStepTimeRef.current >= interval) {
        lastStepTimeRef.current = timestamp;
        setVisibleCount((prev) => {
          const next = prev + 1;
          if (next >= treeNodes.length) {
            setTimeout(() => stopLoop(), 0);
            return treeNodes.length;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    },
    [speed, stopLoop, treeNodes.length],
  );

  const startLoop = useCallback(() => {
    setIsPlaying(true);
    lastStepTimeRef.current = performance.now();
    setVisibleCount((prev) => {
      if (prev >= treeNodes.length) return 0;
      return prev;
    });
    rafRef.current = requestAnimationFrame(tick);
  }, [tick, treeNodes.length]);

  // ------- Control handlers -------

  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      stopLoop();
    } else {
      startLoop();
    }
  }, [isPlaying, startLoop, stopLoop]);

  const handleStep = useCallback(() => {
    if (isPlaying) stopLoop();
    setVisibleCount((prev) => Math.min(prev + 1, treeNodes.length));
  }, [isPlaying, stopLoop, treeNodes.length]);

  const handleReset = useCallback(() => {
    stopLoop();
    setVisibleCount(0);
  }, [stopLoop]);

  // ------- No data fallback -------

  if (treeNodes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-400 gap-3">
        <TreePine className="h-10 w-10" />
        <p className="text-sm font-medium">No tree structure data available</p>
        <p className="text-xs text-gray-400">Run the pipeline to see the trained decision tree</p>
      </div>
    );
  }

  // ------- Render helpers -------

  const nodeById = new Map(treeNodes.map((n) => [n.id, n]));

  const renderEdge = (node: LayoutNode, index: number) => {
    if (node.parentId === null) return null;
    const parent = nodeById.get(node.parentId);
    if (!parent) return null;
    const parentH = parent.type === "internal" ? NODE_H_INTERNAL : NODE_H_LEAF;
    const visible = index < visibleCount;

    return (
      <g key={`edge-${node.id}`}>
        <line
          x1={parent.x}
          y1={parent.y + parentH / 2}
          x2={node.x}
          y2={node.y - (node.type === "internal" ? NODE_H_INTERNAL : NODE_H_LEAF) / 2}
          stroke="#94a3b8"
          strokeWidth={2}
          strokeDasharray={visible ? "none" : "4 4"}
          style={{
            opacity: visible ? 1 : 0,
            transition: "opacity 0.5s ease",
          }}
        />
        {visible && (
          <text
            x={(parent.x + node.x) / 2 + (node.side === "left" ? -12 : 12)}
            y={
              (parent.y +
                parentH / 2 +
                node.y -
                (node.type === "internal" ? NODE_H_INTERNAL : NODE_H_LEAF) / 2) /
              2
            }
            textAnchor="middle"
            fontSize={11}
            fontWeight={600}
            fill={node.side === "left" ? "#16a34a" : "#dc2626"}
            style={{
              opacity: visible ? 1 : 0,
              transition: "opacity 0.6s ease 0.2s",
            }}
          >
            {node.side === "left" ? "Yes" : "No"}
          </text>
        )}
      </g>
    );
  };

  const renderNode = (node: LayoutNode, index: number) => {
    const visible = index < visibleCount;
    const h = node.type === "internal" ? NODE_H_INTERNAL : NODE_H_LEAF;

    if (node.type === "internal") {
      return (
        <g
          key={`node-${node.id}`}
          style={{
            opacity: visible ? 1 : 0,
            transform: visible ? "scale(1)" : "scale(0.7)",
            transformOrigin: `${node.x}px ${node.y}px`,
            transition: "opacity 0.45s ease, transform 0.45s ease",
          }}
        >
          <rect
            x={node.x - NODE_W / 2}
            y={node.y - h / 2}
            width={NODE_W}
            height={h}
            rx={10}
            ry={10}
            fill="#eff6ff"
            stroke="#3b82f6"
            strokeWidth={2}
          />
          <text
            x={node.x}
            y={node.y - 10}
            textAnchor="middle"
            fontSize={12}
            fontWeight={700}
            fill="#1e40af"
          >
            {node.feature} &lt; {node.threshold}
          </text>
          <text
            x={node.x}
            y={node.y + 10}
            textAnchor="middle"
            fontSize={10}
            fill="#64748b"
          >
            {isRegression ? "MSE" : "Gini"} = {node.gini?.toFixed(3)}
          </text>
        </g>
      );
    }

    // Leaf node
    return (
      <g
        key={`node-${node.id}`}
        style={{
          opacity: visible ? 1 : 0,
          transform: visible ? "scale(1)" : "scale(0.7)",
          transformOrigin: `${node.x}px ${node.y}px`,
          transition: "opacity 0.45s ease, transform 0.45s ease",
        }}
      >
        <rect
          x={node.x - NODE_W / 2 + 10}
          y={node.y - h / 2}
          width={NODE_W - 20}
          height={h}
          rx={8}
          ry={8}
          fill={node.color ? `${node.color}18` : "#f1f5f9"}
          stroke={node.color ?? "#94a3b8"}
          strokeWidth={2}
        />
        <text
          x={node.x}
          y={node.y - 5}
          textAnchor="middle"
          fontSize={12}
          fontWeight={700}
          fill={node.color ?? "#334155"}
        >
          {node.classLabel}
        </text>
        <text
          x={node.x}
          y={node.y + 12}
          textAnchor="middle"
          fontSize={10}
          fill="#64748b"
        >
          n = {node.samples}
        </text>
      </g>
    );
  };

  // ------- Metrics summary -------
  const metrics = result?.training_metrics as Record<string, number> | undefined;

  // ------- JSX -------

  return (
    <div className="flex gap-6 flex-col lg:flex-row">
      {/* ---- SVG visualization (left) ---- */}
      <div className="flex-1 min-w-0">
        {/* Info banner */}
        <div className="flex items-start gap-3 rounded-lg border border-green-200 bg-green-50 px-4 py-3 mb-4">
          <Info className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
          <p className="text-sm text-green-800">
            <span className="font-semibold">Decision Tree</span> trained on{" "}
            <strong>{result?.training_samples as number ?? "?"}</strong> samples with{" "}
            <strong>{result?.n_features as number ?? "?"}</strong> features.
            {metrics && !isRegression && (
              <> Training accuracy: <strong>{(metrics.accuracy * 100).toFixed(1)}%</strong>.</>
            )}
            {metrics && isRegression && (
              <> Training R²: <strong>{metrics.r2?.toFixed(4)}</strong>.</>
            )}
            {" "}Watch the tree grow node-by-node to see how it partitions the data.
          </p>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-2 shadow-sm">
          <svg
            viewBox={`0 0 ${SVG_W} ${svgH}`}
            className="w-full h-auto"
            style={{ minHeight: 280 }}
          >
            {/* Background grid */}
            <defs>
              <pattern
                id="dt-grid"
                width="30"
                height="30"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M 30 0 L 0 0 0 30"
                  fill="none"
                  stroke="#f1f5f9"
                  strokeWidth={0.5}
                />
              </pattern>
            </defs>
            <rect width={SVG_W} height={svgH} fill="url(#dt-grid)" />

            {/* Edges */}
            {treeNodes.map((node, i) => renderEdge(node, i))}

            {/* Nodes */}
            {treeNodes.map((node, i) => renderNode(node, i))}

            {/* Empty-state prompt */}
            {visibleCount === 0 && (
              <text
                x={SVG_W / 2}
                y={svgH / 2}
                textAnchor="middle"
                fontSize={14}
                fill="#94a3b8"
              >
                Press Play or Step to start building the tree
              </text>
            )}
          </svg>
        </div>

        {/* Progress indicator */}
        <div className="mt-2 flex items-center gap-2 px-1">
          <div className="h-1.5 flex-1 rounded-full bg-gray-100 overflow-hidden">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-300"
              style={{
                width: `${(visibleCount / treeNodes.length) * 100}%`,
              }}
            />
          </div>
          <span className="text-xs font-medium text-gray-500 tabular-nums">
            {visibleCount}/{treeNodes.length}
          </span>
        </div>
      </div>

      {/* ---- Controls panel (right) ---- */}
      <div className="w-64 shrink-0 space-y-5">
        {/* Header */}
        <div className="flex items-center gap-2 text-gray-800">
          <TreePine className="h-5 w-5 text-green-600" />
          <h3 className="text-base font-semibold">Tree Builder</h3>
        </div>

        {/* Playback buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={handlePlayPause}
            className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium shadow-sm transition
              ${
                isPlaying
                  ? "bg-amber-500 text-white hover:bg-amber-600"
                  : "bg-blue-600 text-white hover:bg-blue-700"
              }`}
          >
            {isPlaying ? (
              <>
                <Pause className="h-4 w-4" /> Pause
              </>
            ) : (
              <>
                <Play className="h-4 w-4" /> Play
              </>
            )}
          </button>

          <button
            onClick={handleStep}
            disabled={visibleCount >= treeNodes.length}
            className="inline-flex items-center justify-center gap-1 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <SkipForward className="h-4 w-4" /> Step
          </button>

          <button
            onClick={handleReset}
            className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white p-2 text-gray-500 shadow-sm transition hover:bg-gray-50 hover:text-gray-700"
            title="Reset"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>

        {/* Speed slider */}
        <div className="space-y-1.5">
          <label className="block text-xs font-semibold uppercase tracking-wide text-gray-500">
            Speed
          </label>
          <div className="flex rounded-lg border border-gray-200 overflow-hidden">
            {(["slow", "medium", "fast"] as const).map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`flex-1 py-1.5 text-xs font-medium capitalize transition
                  ${
                    speed === s
                      ? "bg-blue-600 text-white"
                      : "bg-white text-gray-600 hover:bg-gray-50"
                  }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Legend */}
        {!isRegression && classInfo.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Legend
            </h4>
            <div className="space-y-1.5 text-sm">
              <div className="flex items-center gap-2">
                <span className="inline-block h-3 w-5 rounded border-2 border-blue-500 bg-blue-50" />
                <span className="text-gray-600">Internal node (split)</span>
              </div>
              {classInfo.map((c) => (
                <div key={c.label} className="flex items-center gap-2">
                  <span
                    className="inline-block h-3 w-5 rounded border-2"
                    style={{
                      borderColor: c.color,
                      backgroundColor: `${c.color}20`,
                    }}
                  />
                  <span className="text-gray-600">{c.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tree stats */}
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-1">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
            Tree Info
          </h4>
          <div className="text-sm text-gray-700 space-y-0.5">
            <p>Depth: <span className="font-medium">{result?.tree_depth as number ?? "?"}</span></p>
            <p>Leaves: <span className="font-medium">{result?.n_leaves as number ?? "?"}</span></p>
            <p>Task: <span className="font-medium capitalize">{taskType}</span></p>
          </div>
        </div>

        {/* Current-node detail card */}
        {visibleCount > 0 && visibleCount <= treeNodes.length && (
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-1">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Last Added Node
            </h4>
            {(() => {
              const node = treeNodes[visibleCount - 1];
              if (node.type === "internal") {
                return (
                  <>
                    <p className="text-sm font-medium text-blue-700">
                      {node.feature} &lt; {node.threshold}
                    </p>
                    <p className="text-xs text-gray-500">
                      {isRegression ? "MSE" : "Gini"}: {node.gini?.toFixed(3)} | n = {node.samples}
                    </p>
                  </>
                );
              }
              return (
                <>
                  <p
                    className="text-sm font-medium"
                    style={{ color: node.color }}
                  >
                    {node.classLabel}
                  </p>
                  <p className="text-xs text-gray-500">
                    Samples: {node.samples}
                  </p>
                </>
              );
            })()}
          </div>
        )}

        {/* Completion badge */}
        {visibleCount >= treeNodes.length && (
          <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-center text-sm font-medium text-green-700">
            Tree complete! All {treeNodes.length} nodes placed.
          </div>
        )}
      </div>
    </div>
  );
};

export default DecisionTreeAnimation;
