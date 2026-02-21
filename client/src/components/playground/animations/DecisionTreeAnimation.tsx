import { useState, useRef, useCallback } from "react";
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

interface TreeNode {
  id: number;
  parentId: number | null;
  /** "internal" nodes carry a split; "leaf" nodes carry a class label */
  type: "internal" | "leaf";
  /** Display position (SVG coordinates) */
  x: number;
  y: number;
  /** Internal-node fields */
  feature?: string;
  threshold?: number;
  gini?: number;
  /** Leaf-node fields */
  classLabel?: string;
  samples?: number;
  /** Colour used for leaf badges */
  color?: string;
  /** Which side of the parent ("left" | "right") – used for edge labels */
  side?: "left" | "right";
}

// ---------------------------------------------------------------------------
// Pre-defined Iris decision-tree structure (7 nodes, 3 levels)
// ---------------------------------------------------------------------------

const TREE_NODES: TreeNode[] = [
  // Level 0 – root
  {
    id: 0,
    parentId: null,
    type: "internal",
    x: 300,
    y: 45,
    feature: "Petal Length",
    threshold: 2.5,
    gini: 0.667,
  },
  // Level 1
  {
    id: 1,
    parentId: 0,
    type: "leaf",
    x: 130,
    y: 150,
    classLabel: "Setosa",
    samples: 50,
    color: "#22c55e",
    side: "left",
  },
  {
    id: 2,
    parentId: 0,
    type: "internal",
    x: 470,
    y: 150,
    feature: "Petal Width",
    threshold: 1.8,
    gini: 0.5,
    side: "right",
  },
  // Level 2
  {
    id: 3,
    parentId: 2,
    type: "internal",
    x: 340,
    y: 260,
    feature: "Petal Length",
    threshold: 4.9,
    gini: 0.168,
    side: "left",
  },
  {
    id: 4,
    parentId: 2,
    type: "leaf",
    x: 530,
    y: 260,
    classLabel: "Virginica",
    samples: 43,
    color: "#a855f7",
    side: "right",
  },
  // Level 3
  {
    id: 5,
    parentId: 3,
    type: "leaf",
    x: 260,
    y: 355,
    classLabel: "Versicolor",
    samples: 47,
    color: "#f97316",
    side: "left",
  },
  {
    id: 6,
    parentId: 3,
    type: "leaf",
    x: 420,
    y: 355,
    classLabel: "Virginica",
    samples: 3,
    color: "#a855f7",
    side: "right",
  },
];

// Node dimensions
const NODE_W = 140;
const NODE_H_INTERNAL = 58;
const NODE_H_LEAF = 48;

// Speed presets (ms per step)
const SPEED_MAP: Record<string, number> = {
  slow: 1400,
  medium: 800,
  fast: 400,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DecisionTreeAnimation = () => {
  // How many nodes are currently visible (0 = nothing, 7 = full tree)
  const [visibleCount, setVisibleCount] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<string>("medium");

  // Refs for requestAnimationFrame loop
  const rafRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number>(0);

  // ------- Animation loop (rAF-based) -------

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
          if (next >= TREE_NODES.length) {
            // Tree is fully built – stop on next frame
            setTimeout(() => stopLoop(), 0);
            return TREE_NODES.length;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    },
    [speed, stopLoop],
  );

  const startLoop = useCallback(() => {
    setIsPlaying(true);
    lastStepTimeRef.current = performance.now();
    // If already complete, reset first
    setVisibleCount((prev) => {
      if (prev >= TREE_NODES.length) return 0;
      return prev;
    });
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

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
    setVisibleCount((prev) => Math.min(prev + 1, TREE_NODES.length));
  }, [isPlaying, stopLoop]);

  const handleReset = useCallback(() => {
    stopLoop();
    setVisibleCount(0);
  }, [stopLoop]);

  // ------- Render helpers -------

  /** Render a single edge from child to parent */
  const renderEdge = (node: TreeNode, index: number) => {
    if (node.parentId === null) return null;
    const parent = TREE_NODES.find((n) => n.id === node.parentId)!;
    const parentH =
      parent.type === "internal" ? NODE_H_INTERNAL : NODE_H_LEAF;
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
        {/* Yes / No label */}
        {visible && (
          <text
            x={(parent.x + node.x) / 2 + (node.side === "left" ? -12 : 12)}
            y={(parent.y + parentH / 2 + node.y - (node.type === "internal" ? NODE_H_INTERNAL : NODE_H_LEAF) / 2) / 2}
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

  /** Render a single tree node */
  const renderNode = (node: TreeNode, index: number) => {
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
          {/* Feature + threshold */}
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
          {/* Gini */}
          <text
            x={node.x}
            y={node.y + 10}
            textAnchor="middle"
            fontSize={10}
            fill="#64748b"
          >
            Gini = {node.gini?.toFixed(3)}
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
        {/* Class label */}
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
        {/* Samples */}
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

  // ------- JSX -------

  return (
    <div className="flex gap-6 flex-col lg:flex-row">
      {/* ---- SVG visualization (left) ---- */}
      <div className="flex-1 min-w-0">
        {/* Info banner */}
        <div className="flex items-start gap-3 rounded-lg border border-green-200 bg-green-50 px-4 py-3 mb-4">
          <Info className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
          <p className="text-sm text-green-800">
            <span className="font-semibold">Decision Trees</span> learn by
            recursively splitting the data on the feature and threshold that
            best separates the classes, measured by <em>Gini impurity</em>.
            Each internal node represents a question; each leaf holds a
            prediction. Watch the tree grow node-by-node to see how it
            partitions the Iris dataset.
          </p>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-2 shadow-sm">
          <svg
            viewBox="0 0 600 400"
            className="w-full h-auto"
            style={{ minHeight: 280 }}
          >
            {/* Background grid (subtle) */}
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
            <rect width="600" height="400" fill="url(#dt-grid)" />

            {/* Edges (rendered first so nodes sit on top) */}
            {TREE_NODES.map((node, i) => renderEdge(node, i))}

            {/* Nodes */}
            {TREE_NODES.map((node, i) => renderNode(node, i))}

            {/* Empty-state prompt */}
            {visibleCount === 0 && (
              <text
                x={300}
                y={200}
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
                width: `${(visibleCount / TREE_NODES.length) * 100}%`,
              }}
            />
          </div>
          <span className="text-xs font-medium text-gray-500 tabular-nums">
            {visibleCount}/{TREE_NODES.length}
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
            disabled={visibleCount >= TREE_NODES.length}
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
        <div className="space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
            Legend
          </h4>
          <div className="space-y-1.5 text-sm">
            <div className="flex items-center gap-2">
              <span className="inline-block h-3 w-5 rounded border-2 border-blue-500 bg-blue-50" />
              <span className="text-gray-600">Internal node (split)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block h-3 w-5 rounded border-2 border-green-500 bg-green-50" />
              <span className="text-gray-600">Setosa</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block h-3 w-5 rounded border-2 border-orange-500 bg-orange-50" />
              <span className="text-gray-600">Versicolor</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block h-3 w-5 rounded border-2 border-purple-500 bg-purple-50" />
              <span className="text-gray-600">Virginica</span>
            </div>
          </div>
        </div>

        {/* Current-node detail card */}
        {visibleCount > 0 && visibleCount <= TREE_NODES.length && (
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-1">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Last Added Node
            </h4>
            {(() => {
              const node = TREE_NODES[visibleCount - 1];
              if (node.type === "internal") {
                return (
                  <>
                    <p className="text-sm font-medium text-blue-700">
                      {node.feature} &lt; {node.threshold}
                    </p>
                    <p className="text-xs text-gray-500">
                      Gini impurity: {node.gini?.toFixed(3)}
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
        {visibleCount >= TREE_NODES.length && (
          <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-center text-sm font-medium text-green-700">
            Tree complete! All {TREE_NODES.length} nodes placed.
          </div>
        )}
      </div>
    </div>
  );
};

export default DecisionTreeAnimation;
