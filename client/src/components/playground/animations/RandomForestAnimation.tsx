/**
 * RandomForestAnimation - Educational SVG animation showing how Random Forest works.
 *
 * Phases:
 *   1. Trees fade in one by one (left to right)
 *   2. Each tree pulses with a glow ("thinking")
 *   3. Individual predictions appear beneath each tree
 *   4. Voting particles travel from each tree to the final prediction box
 *   5. Final majority-vote prediction appears with emphasis
 *
 * Controls: Play, Reset, tree-count slider (3 or 5).
 * No external animation library -- uses requestAnimationFrame + SVG.
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { Play, RotateCcw, Info, TreePine, Vote } from "lucide-react";

/* ---------- colour palette ---------- */
const TREE_COLORS = [
  { header: "#0d9488", fill: "#ccfbf1", stroke: "#0d9488" }, // teal
  { header: "#7c3aed", fill: "#ede9fe", stroke: "#7c3aed" }, // violet
  { header: "#ea580c", fill: "#ffedd5", stroke: "#ea580c" }, // orange
  { header: "#2563eb", fill: "#dbeafe", stroke: "#2563eb" }, // blue
  { header: "#db2777", fill: "#fce7f3", stroke: "#db2777" }, // pink
];

const CLASS_A_COLOR = "#0d9488";
const CLASS_B_COLOR = "#e11d48";

/* ---------- helpers ---------- */
interface TreePrediction {
  label: "A" | "B";
  color: string;
}

function getPredictions(count: number): TreePrediction[] {
  if (count === 3) {
    return [
      { label: "A", color: CLASS_A_COLOR },
      { label: "B", color: CLASS_B_COLOR },
      { label: "A", color: CLASS_A_COLOR },
    ];
  }
  // 5 trees: A, B, A, A, B => majority A
  return [
    { label: "A", color: CLASS_A_COLOR },
    { label: "B", color: CLASS_B_COLOR },
    { label: "A", color: CLASS_A_COLOR },
    { label: "A", color: CLASS_A_COLOR },
    { label: "B", color: CLASS_B_COLOR },
  ];
}

function majorityVote(preds: TreePrediction[]): TreePrediction {
  const countA = preds.filter((p) => p.label === "A").length;
  return countA > preds.length / 2
    ? { label: "A", color: CLASS_A_COLOR }
    : { label: "B", color: CLASS_B_COLOR };
}

/* ---------- animation timing constants (ms) ---------- */
const PHASE_TREE_APPEAR_EACH = 500;
const PHASE_THINK_DURATION = 600;
const PHASE_PREDICT_DELAY = 300;
const PHASE_PARTICLE_DURATION = 900;
const PHASE_FINAL_DELAY = 400;

/* ---------- sub-components (pure SVG) ---------- */

/** A simplified 3-node decision tree drawn in SVG. */
function MiniTree({
  cx,
  cy,
  palette,
  opacity,
  glowing,
}: {
  cx: number;
  cy: number;
  palette: (typeof TREE_COLORS)[0];
  opacity: number;
  glowing: boolean;
}) {
  const w = 90;
  const h = 120;
  const x = cx - w / 2;
  const y = cy;

  return (
    <g opacity={opacity}>
      {/* glow filter behind the tree */}
      {glowing && (
        <rect
          x={x - 6}
          y={y - 6}
          width={w + 12}
          height={h + 12}
          rx={10}
          fill="none"
          stroke={palette.header}
          strokeWidth={3}
          opacity={0.55}
        >
          <animate
            attributeName="opacity"
            values="0.25;0.7;0.25"
            dur="0.6s"
            repeatCount="indefinite"
          />
        </rect>
      )}

      {/* card background */}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill={palette.fill}
        stroke={palette.stroke}
        strokeWidth={1.5}
      />

      {/* header bar */}
      <rect
        x={x}
        y={y}
        width={w}
        height={24}
        rx={8}
        fill={palette.header}
      />
      {/* flatten bottom corners of header */}
      <rect
        x={x}
        y={y + 16}
        width={w}
        height={8}
        fill={palette.header}
      />

      <text
        x={cx}
        y={y + 16}
        textAnchor="middle"
        fill="white"
        fontSize={11}
        fontWeight={600}
      >
        Tree
      </text>

      {/* simplified 3-node diagram: root -> left, right */}
      {/* root node */}
      <circle cx={cx} cy={y + 46} r={8} fill={palette.header} opacity={0.85} />
      {/* left child */}
      <line
        x1={cx}
        y1={y + 54}
        x2={cx - 20}
        y2={y + 76}
        stroke={palette.stroke}
        strokeWidth={1.2}
      />
      <circle
        cx={cx - 20}
        cy={y + 82}
        r={7}
        fill={palette.header}
        opacity={0.6}
      />
      {/* right child */}
      <line
        x1={cx}
        y1={y + 54}
        x2={cx + 20}
        y2={y + 76}
        stroke={palette.stroke}
        strokeWidth={1.2}
      />
      <circle
        cx={cx + 20}
        cy={y + 82}
        r={7}
        fill={palette.header}
        opacity={0.6}
      />

      {/* leaf labels */}
      <text
        x={cx - 20}
        y={y + 102}
        textAnchor="middle"
        fontSize={8}
        fill={palette.stroke}
      >
        L
      </text>
      <text
        x={cx + 20}
        y={y + 102}
        textAnchor="middle"
        fontSize={8}
        fill={palette.stroke}
      >
        R
      </text>
    </g>
  );
}

/* ---------- main component ---------- */

type AnimPhase =
  | "idle"
  | "trees"
  | "thinking"
  | "predicting"
  | "voting"
  | "final"
  | "done";

export default function RandomForestAnimation() {
  const [treeCount, setTreeCount] = useState<3 | 5>(3);
  const [phase, setPhase] = useState<AnimPhase>("idle");
  const [treesVisible, setTreesVisible] = useState<number>(0);
  const [thinkingIndex, setThinkingIndex] = useState<number>(-1);
  const [predictionsVisible, setPredictionsVisible] = useState<number>(0);
  const [particleProgress, setParticleProgress] = useState<number>(0); // 0-1
  const [showFinal, setShowFinal] = useState(false);

  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number>(0);
  const phaseRef = useRef<AnimPhase>("idle");

  const predictions = getPredictions(treeCount);
  const finalPrediction = majorityVote(predictions);

  /* keep phaseRef in sync */
  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  /* clean up on unmount */
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  /* ---- layout helpers ---- */
  const SVG_W = 700;
  const SVG_H = 450;
  const treeSpacing = SVG_W / (treeCount + 1);
  const treeCenters = Array.from(
    { length: treeCount },
    (_, i) => treeSpacing * (i + 1)
  );
  const treeTopY = 40;
  const predictionY = treeTopY + 140;
  const voteBoxY = 340;
  const voteBoxCx = SVG_W / 2;

  /* ---- animation loop ---- */
  const tick = useCallback(
    (now: number) => {
      const elapsed = now - startRef.current;
      const currentPhase = phaseRef.current;

      if (currentPhase === "trees") {
        const idx = Math.floor(elapsed / PHASE_TREE_APPEAR_EACH);
        if (idx < treeCount) {
          setTreesVisible(idx + 1);
        } else {
          setTreesVisible(treeCount);
          startRef.current = now;
          setPhase("thinking");
          phaseRef.current = "thinking";
        }
      } else if (currentPhase === "thinking") {
        const idx = Math.floor(elapsed / PHASE_THINK_DURATION);
        if (idx < treeCount) {
          setThinkingIndex(idx);
        } else {
          setThinkingIndex(-1);
          startRef.current = now;
          setPhase("predicting");
          phaseRef.current = "predicting";
        }
      } else if (currentPhase === "predicting") {
        const idx = Math.floor(elapsed / PHASE_PREDICT_DELAY);
        if (idx < treeCount) {
          setPredictionsVisible(idx + 1);
        } else {
          setPredictionsVisible(treeCount);
          startRef.current = now;
          setPhase("voting");
          phaseRef.current = "voting";
        }
      } else if (currentPhase === "voting") {
        const progress = Math.min(elapsed / PHASE_PARTICLE_DURATION, 1);
        setParticleProgress(progress);
        if (progress >= 1) {
          startRef.current = now;
          setPhase("final");
          phaseRef.current = "final";
        }
      } else if (currentPhase === "final") {
        if (elapsed >= PHASE_FINAL_DELAY) {
          setShowFinal(true);
          setPhase("done");
          phaseRef.current = "done";
        }
      }

      if (
        phaseRef.current !== "idle" &&
        phaseRef.current !== "done"
      ) {
        rafRef.current = requestAnimationFrame(tick);
      }
    },
    [treeCount]
  );

  const play = useCallback(() => {
    // reset all state
    setTreesVisible(0);
    setThinkingIndex(-1);
    setPredictionsVisible(0);
    setParticleProgress(0);
    setShowFinal(false);
    setPhase("trees");
    phaseRef.current = "trees";
    startRef.current = performance.now();
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const reset = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    setPhase("idle");
    phaseRef.current = "idle";
    setTreesVisible(0);
    setThinkingIndex(-1);
    setPredictionsVisible(0);
    setParticleProgress(0);
    setShowFinal(false);
  }, []);

  /* ---- render ---- */
  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col gap-4">
      {/* ---- SVG canvas ---- */}
      <div className="w-full rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          className="w-full h-auto"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* background */}
          <rect width={SVG_W} height={SVG_H} fill="#f8fafc" rx={12} />

          {/* title */}
          <text
            x={SVG_W / 2}
            y={28}
            textAnchor="middle"
            fontSize={16}
            fontWeight={700}
            fill="#1e293b"
          >
            Random Forest — Ensemble Voting
          </text>

          {/* ---- Trees ---- */}
          {treeCenters.map((cx, i) => (
            <MiniTree
              key={i}
              cx={cx}
              cy={treeTopY}
              palette={TREE_COLORS[i % TREE_COLORS.length]}
              opacity={i < treesVisible ? 1 : 0.1}
              glowing={thinkingIndex === i}
            />
          ))}

          {/* ---- Individual predictions ---- */}
          {treeCenters.map((cx, i) => {
            if (i >= predictionsVisible) return null;
            const pred = predictions[i];
            return (
              <g key={`pred-${i}`}>
                <rect
                  x={cx - 32}
                  y={predictionY}
                  width={64}
                  height={26}
                  rx={6}
                  fill={pred.color}
                  opacity={0.15}
                  stroke={pred.color}
                  strokeWidth={1.2}
                />
                <text
                  x={cx}
                  y={predictionY + 17}
                  textAnchor="middle"
                  fontSize={12}
                  fontWeight={600}
                  fill={pred.color}
                >
                  Class {pred.label}
                </text>
              </g>
            );
          })}

          {/* ---- Voting particles ---- */}
          {phase === "voting" &&
            treeCenters.map((cx, i) => {
              const pred = predictions[i];
              const startX = cx;
              const startY = predictionY + 30;
              const endX = voteBoxCx;
              const endY = voteBoxY;
              const curX = startX + (endX - startX) * particleProgress;
              const curY = startY + (endY - startY) * particleProgress;
              return (
                <circle
                  key={`particle-${i}`}
                  cx={curX}
                  cy={curY}
                  r={5}
                  fill={pred.color}
                  opacity={0.9}
                >
                  <animate
                    attributeName="r"
                    values="4;6;4"
                    dur="0.4s"
                    repeatCount="indefinite"
                  />
                </circle>
              );
            })}

          {/* ---- Voting connector lines (faded) ---- */}
          {(phase === "voting" || phase === "final" || phase === "done") &&
            treeCenters.map((cx, i) => (
              <line
                key={`line-${i}`}
                x1={cx}
                y1={predictionY + 26}
                x2={voteBoxCx}
                y2={voteBoxY}
                stroke="#94a3b8"
                strokeWidth={1}
                strokeDasharray="4 3"
                opacity={0.4}
              />
            ))}

          {/* ---- Final prediction box ---- */}
          <g>
            <rect
              x={voteBoxCx - 80}
              y={voteBoxY}
              width={160}
              height={50}
              rx={10}
              fill={showFinal ? finalPrediction.color : "#e2e8f0"}
              opacity={showFinal ? 0.18 : 0.5}
              stroke={showFinal ? finalPrediction.color : "#94a3b8"}
              strokeWidth={showFinal ? 2 : 1}
            />
            {showFinal ? (
              <>
                <text
                  x={voteBoxCx}
                  y={voteBoxY + 22}
                  textAnchor="middle"
                  fontSize={11}
                  fill="#64748b"
                >
                  Majority Vote
                </text>
                <text
                  x={voteBoxCx}
                  y={voteBoxY + 40}
                  textAnchor="middle"
                  fontSize={16}
                  fontWeight={700}
                  fill={finalPrediction.color}
                >
                  Class {finalPrediction.label}
                </text>
              </>
            ) : (
              <text
                x={voteBoxCx}
                y={voteBoxY + 30}
                textAnchor="middle"
                fontSize={12}
                fill="#94a3b8"
              >
                Final Prediction
              </text>
            )}
          </g>

          {/* vote tally when done */}
          {showFinal && (
            <text
              x={voteBoxCx}
              y={voteBoxY + 66}
              textAnchor="middle"
              fontSize={10}
              fill="#64748b"
            >
              ({predictions.filter((p) => p.label === "A").length}A /{" "}
              {predictions.filter((p) => p.label === "B").length}B)
            </text>
          )}
        </svg>
      </div>

      {/* ---- Controls ---- */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        <button
          onClick={play}
          disabled={
            phase !== "idle" && phase !== "done"
          }
          className="inline-flex items-center gap-1.5 rounded-lg bg-teal-600 px-4 py-2 text-sm font-medium text-white shadow hover:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Play size={16} />
          Play
        </button>
        <button
          onClick={reset}
          className="inline-flex items-center gap-1.5 rounded-lg bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 shadow hover:bg-gray-300 transition-colors"
        >
          <RotateCcw size={16} />
          Reset
        </button>

        {/* tree count slider */}
        <div className="flex items-center gap-2 ml-4">
          <TreePine size={16} className="text-teal-600" />
          <label className="text-sm font-medium text-gray-700">
            Trees:
          </label>
          <input
            type="range"
            min={3}
            max={5}
            step={2}
            value={treeCount}
            onChange={(e) => {
              const v = Number(e.target.value) as 3 | 5;
              setTreeCount(v);
              reset();
            }}
            className="w-24 accent-teal-600"
          />
          <span className="text-sm font-semibold text-teal-700 w-4 text-center">
            {treeCount}
          </span>
        </div>
      </div>

      {/* ---- Info banner ---- */}
      <div className="flex items-start gap-3 rounded-xl border border-cyan-200 bg-cyan-50 p-4">
        <Info size={20} className="mt-0.5 shrink-0 text-cyan-600" />
        <div className="text-sm text-cyan-900 space-y-1">
          <p className="font-semibold flex items-center gap-1.5">
            <Vote size={15} className="text-cyan-700" />
            How Random Forest Works
          </p>
          <ul className="list-disc list-inside space-y-0.5 text-cyan-800">
            <li>
              <span className="font-medium">Ensemble of trees:</span> Multiple
              decision trees are trained, each on a random subset of the data
              (bootstrap sampling).
            </li>
            <li>
              <span className="font-medium">Independent predictions:</span> Every
              tree makes its own prediction without knowing what the others chose.
            </li>
            <li>
              <span className="font-medium">Majority voting:</span> The final
              prediction is the class that receives the most votes, making the
              model more robust than any single tree.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
