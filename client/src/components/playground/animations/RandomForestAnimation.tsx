/**
 * RandomForestAnimation - Dynamic SVG animation showing how Random Forest works.
 *
 * Shows real data from the trained model:
 * - Actual n_estimators, class names, metrics
 * - Feature importances bar chart
 * - Ensemble voting animation with real class labels
 *
 * Phases:
 *   1. Trees fade in one by one
 *   2. Each tree pulses with a glow ("thinking")
 *   3. Individual predictions appear beneath each tree
 *   4. Voting particles travel to the final prediction box
 *   5. Final majority-vote prediction appears
 */

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { Play, RotateCcw, Info, TreePine, Vote, BarChart3 } from "lucide-react";

interface RandomForestAnimationProps {
  result?: Record<string, unknown>;
}

/* ---------- colour palette ---------- */
const TREE_COLORS = [
  { header: "#0d9488", fill: "#ccfbf1", stroke: "#0d9488" },
  { header: "#7c3aed", fill: "#ede9fe", stroke: "#7c3aed" },
  { header: "#ea580c", fill: "#ffedd5", stroke: "#ea580c" },
  { header: "#2563eb", fill: "#dbeafe", stroke: "#2563eb" },
  { header: "#db2777", fill: "#fce7f3", stroke: "#db2777" },
];

const CLASS_COLORS = [
  "#0d9488", "#e11d48", "#7c3aed", "#ea580c", "#2563eb",
  "#db2777", "#16a34a", "#f59e0b", "#6366f1", "#84cc16",
];

/* ---------- helpers ---------- */
interface FeatureImportance {
  feature: string;
  importance: number;
}

/* ---------- animation timing constants (ms) ---------- */
const PHASE_TREE_APPEAR_EACH = 500;
const PHASE_THINK_DURATION = 600;
const PHASE_PREDICT_DELAY = 300;
const PHASE_PARTICLE_DURATION = 900;
const PHASE_FINAL_DELAY = 400;

/* ---------- sub-components (pure SVG) ---------- */

function MiniTree({
  cx,
  cy,
  palette,
  opacity,
  glowing,
  label,
}: {
  cx: number;
  cy: number;
  palette: (typeof TREE_COLORS)[0];
  opacity: number;
  glowing: boolean;
  label: string;
}) {
  const w = 90;
  const h = 120;
  const x = cx - w / 2;
  const y = cy;

  return (
    <g opacity={opacity}>
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

      <rect x={x} y={y} width={w} height={h} rx={8} fill={palette.fill} stroke={palette.stroke} strokeWidth={1.5} />
      <rect x={x} y={y} width={w} height={24} rx={8} fill={palette.header} />
      <rect x={x} y={y + 16} width={w} height={8} fill={palette.header} />

      <text x={cx} y={y + 16} textAnchor="middle" fill="white" fontSize={11} fontWeight={600}>
        {label}
      </text>

      {/* simplified 3-node diagram */}
      <circle cx={cx} cy={y + 46} r={8} fill={palette.header} opacity={0.85} />
      <line x1={cx} y1={y + 54} x2={cx - 20} y2={y + 76} stroke={palette.stroke} strokeWidth={1.2} />
      <circle cx={cx - 20} cy={y + 82} r={7} fill={palette.header} opacity={0.6} />
      <line x1={cx} y1={y + 54} x2={cx + 20} y2={y + 76} stroke={palette.stroke} strokeWidth={1.2} />
      <circle cx={cx + 20} cy={y + 82} r={7} fill={palette.header} opacity={0.6} />

      <text x={cx - 20} y={y + 102} textAnchor="middle" fontSize={8} fill={palette.stroke}>L</text>
      <text x={cx + 20} y={y + 102} textAnchor="middle" fontSize={8} fill={palette.stroke}>R</text>
    </g>
  );
}

/* ---------- main component ---------- */

type AnimPhase = "idle" | "trees" | "thinking" | "predicting" | "voting" | "final" | "done";

export default function RandomForestAnimation({ result }: RandomForestAnimationProps) {
  // Extract data from result
  const fullMeta = useMemo(() => {
    return (result?.metadata as Record<string, unknown>)?.full_training_metadata as Record<string, unknown> | undefined;
  }, [result]);

  const taskType = (result?.task_type as string) || "classification";
  const isRegression = taskType === "regression";
  const nEstimators = (result?.n_estimators as number) || (fullMeta?.n_estimators as number) || 5;
  const classNames = (fullMeta?.class_names as string[]) || [];
  const metrics = result?.training_metrics as Record<string, number> | undefined;
  const featureImportances = (fullMeta?.feature_importances as FeatureImportance[]) || [];
  const topFeatures = featureImportances.slice(0, 5);
  const trainingSamples = (result?.training_samples as number) || (fullMeta?.n_samples as number) || 0;
  const nFeatures = (result?.n_features as number) || (fullMeta?.n_features as number) || 0;

  // How many trees to show in the animation (max 5 for visual clarity)
  const displayTreeCount = Math.min(nEstimators, 5);

  // Build class color map
  const classColorMap = useMemo(() => {
    const map = new Map<string, string>();
    classNames.forEach((name, i) => {
      map.set(String(name), CLASS_COLORS[i % CLASS_COLORS.length]);
    });
    return map;
  }, [classNames]);

  // Generate simulated per-tree predictions based on the majority class
  // In a real forest, most trees agree with the majority vote
  const predictions = useMemo(() => {
    if (isRegression || classNames.length === 0) {
      // For regression or no class names, just show generic labels
      return Array.from({ length: displayTreeCount }, (_, i) => ({
        label: isRegression ? `Tree ${i + 1}` : `Pred ${i + 1}`,
        color: TREE_COLORS[i % TREE_COLORS.length].header,
      }));
    }

    // For classification: simulate majority voting
    // Most trees should predict the class with the highest accuracy
    const majorityClass = classNames[0]; // first class
    return Array.from({ length: displayTreeCount }, (_, i) => {
      // Make ~70% of trees agree on majority, rest random
      const cls = i < Math.ceil(displayTreeCount * 0.7)
        ? majorityClass
        : classNames[Math.min(i % classNames.length, classNames.length - 1)];
      return {
        label: String(cls),
        color: classColorMap.get(String(cls)) || CLASS_COLORS[i % CLASS_COLORS.length],
      };
    });
  }, [displayTreeCount, classNames, classColorMap, isRegression]);

  // Compute majority vote
  const finalPrediction = useMemo(() => {
    if (isRegression) {
      return { label: "Average", color: "#2563eb" };
    }
    const counts = new Map<string, number>();
    for (const p of predictions) {
      counts.set(p.label, (counts.get(p.label) || 0) + 1);
    }
    let maxLabel = predictions[0]?.label || "?";
    let maxCount = 0;
    for (const [label, count] of counts) {
      if (count > maxCount) {
        maxCount = count;
        maxLabel = label;
      }
    }
    return {
      label: maxLabel,
      color: classColorMap.get(maxLabel) || CLASS_COLORS[0],
    };
  }, [predictions, classColorMap, isRegression]);

  const [phase, setPhase] = useState<AnimPhase>("idle");
  const [treesVisible, setTreesVisible] = useState<number>(0);
  const [thinkingIndex, setThinkingIndex] = useState<number>(-1);
  const [predictionsVisible, setPredictionsVisible] = useState<number>(0);
  const [particleProgress, setParticleProgress] = useState<number>(0);
  const [showFinal, setShowFinal] = useState(false);

  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number>(0);
  const phaseRef = useRef<AnimPhase>("idle");

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  /* ---- layout helpers ---- */
  const SVG_W = 700;
  const SVG_H = 450;
  const treeSpacing = SVG_W / (displayTreeCount + 1);
  const treeCenters = Array.from(
    { length: displayTreeCount },
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
        if (idx < displayTreeCount) {
          setTreesVisible(idx + 1);
        } else {
          setTreesVisible(displayTreeCount);
          startRef.current = now;
          setPhase("thinking");
          phaseRef.current = "thinking";
        }
      } else if (currentPhase === "thinking") {
        const idx = Math.floor(elapsed / PHASE_THINK_DURATION);
        if (idx < displayTreeCount) {
          setThinkingIndex(idx);
        } else {
          setThinkingIndex(-1);
          startRef.current = now;
          setPhase("predicting");
          phaseRef.current = "predicting";
        }
      } else if (currentPhase === "predicting") {
        const idx = Math.floor(elapsed / PHASE_PREDICT_DELAY);
        if (idx < displayTreeCount) {
          setPredictionsVisible(idx + 1);
        } else {
          setPredictionsVisible(displayTreeCount);
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

      if (phaseRef.current !== "idle" && phaseRef.current !== "done") {
        rafRef.current = requestAnimationFrame(tick);
      }
    },
    [displayTreeCount]
  );

  const play = useCallback(() => {
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

  // No data fallback
  const hasData = !!result && (nEstimators > 0 || trainingSamples > 0);

  if (!hasData) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-400 gap-3">
        <TreePine className="h-10 w-10" />
        <p className="text-sm font-medium">No random forest data available</p>
        <p className="text-xs text-gray-400">Run the pipeline to see the trained ensemble</p>
      </div>
    );
  }

  // Vote tally
  const voteTally = new Map<string, number>();
  for (const p of predictions) {
    voteTally.set(p.label, (voteTally.get(p.label) || 0) + 1);
  }
  const tallyStr = Array.from(voteTally.entries())
    .map(([label, count]) => `${count} ${label}`)
    .join(" / ");

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
          <rect width={SVG_W} height={SVG_H} fill="#f8fafc" rx={12} />

          {/* title */}
          <text x={SVG_W / 2} y={28} textAnchor="middle" fontSize={16} fontWeight={700} fill="#1e293b">
            Random Forest — {nEstimators} Trees{displayTreeCount < nEstimators ? ` (showing ${displayTreeCount})` : ""}
          </text>

          {/* Trees */}
          {treeCenters.map((cx, i) => (
            <MiniTree
              key={i}
              cx={cx}
              cy={treeTopY}
              palette={TREE_COLORS[i % TREE_COLORS.length]}
              opacity={i < treesVisible ? 1 : 0.1}
              glowing={thinkingIndex === i}
              label={`Tree ${i + 1}`}
            />
          ))}

          {/* Individual predictions */}
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
                  fontSize={11}
                  fontWeight={600}
                  fill={pred.color}
                >
                  {pred.label.length > 8 ? pred.label.slice(0, 7) + "…" : pred.label}
                </text>
              </g>
            );
          })}

          {/* Voting particles */}
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
                <circle key={`particle-${i}`} cx={curX} cy={curY} r={5} fill={pred.color} opacity={0.9}>
                  <animate attributeName="r" values="4;6;4" dur="0.4s" repeatCount="indefinite" />
                </circle>
              );
            })}

          {/* Voting connector lines */}
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

          {/* Final prediction box */}
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
                <text x={voteBoxCx} y={voteBoxY + 22} textAnchor="middle" fontSize={11} fill="#64748b">
                  {isRegression ? "Average Prediction" : "Majority Vote"}
                </text>
                <text
                  x={voteBoxCx}
                  y={voteBoxY + 40}
                  textAnchor="middle"
                  fontSize={16}
                  fontWeight={700}
                  fill={finalPrediction.color}
                >
                  {finalPrediction.label}
                </text>
              </>
            ) : (
              <text x={voteBoxCx} y={voteBoxY + 30} textAnchor="middle" fontSize={12} fill="#94a3b8">
                Final Prediction
              </text>
            )}
          </g>

          {/* vote tally */}
          {showFinal && !isRegression && (
            <text x={voteBoxCx} y={voteBoxY + 66} textAnchor="middle" fontSize={10} fill="#64748b">
              ({tallyStr})
            </text>
          )}
        </svg>
      </div>

      {/* ---- Controls + Metrics Row ---- */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        <button
          onClick={play}
          disabled={phase !== "idle" && phase !== "done"}
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
      </div>

      {/* ---- Metrics + Feature Importances ---- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Training Summary */}
        <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
          <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <TreePine size={16} className="text-teal-600" />
            Training Summary
          </h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="rounded-lg bg-gray-50 p-2">
              <p className="text-xs text-gray-500">Trees</p>
              <p className="font-semibold text-gray-800">{nEstimators}</p>
            </div>
            <div className="rounded-lg bg-gray-50 p-2">
              <p className="text-xs text-gray-500">Samples</p>
              <p className="font-semibold text-gray-800">{trainingSamples.toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-gray-50 p-2">
              <p className="text-xs text-gray-500">Features</p>
              <p className="font-semibold text-gray-800">{nFeatures}</p>
            </div>
            <div className="rounded-lg bg-gray-50 p-2">
              <p className="text-xs text-gray-500">
                {isRegression ? "R² Score" : "Accuracy"}
              </p>
              <p className="font-semibold text-gray-800">
                {metrics
                  ? isRegression
                    ? metrics.r2?.toFixed(4)
                    : `${(metrics.accuracy * 100).toFixed(1)}%`
                  : "—"}
              </p>
            </div>
          </div>
          {!isRegression && classNames.length > 0 && (
            <div className="pt-1">
              <p className="text-xs text-gray-500 mb-1">Classes</p>
              <div className="flex flex-wrap gap-1.5">
                {classNames.map((cls, i) => (
                  <span
                    key={String(cls)}
                    className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
                    style={{
                      backgroundColor: `${CLASS_COLORS[i % CLASS_COLORS.length]}15`,
                      color: CLASS_COLORS[i % CLASS_COLORS.length],
                      border: `1px solid ${CLASS_COLORS[i % CLASS_COLORS.length]}40`,
                    }}
                  >
                    {String(cls)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Feature Importances */}
        {topFeatures.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
            <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              <BarChart3 size={16} className="text-amber-600" />
              Top Feature Importances
            </h4>
            <div className="space-y-2">
              {topFeatures.map((fi, i) => {
                const maxImp = topFeatures[0]?.importance || 1;
                const pct = maxImp > 0 ? (fi.importance / maxImp) * 100 : 0;
                return (
                  <div key={fi.feature} className="space-y-0.5">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600 truncate max-w-[140px]" title={fi.feature}>
                        {fi.feature}
                      </span>
                      <span className="font-medium text-gray-700">
                        {(fi.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${pct}%`,
                          backgroundColor: TREE_COLORS[i % TREE_COLORS.length].header,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
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
              <span className="font-medium">Ensemble of {nEstimators} trees:</span> Each trained on a
              random subset of the {trainingSamples.toLocaleString()} training samples.
            </li>
            <li>
              <span className="font-medium">Independent predictions:</span> Every tree makes its own
              prediction without knowing what the others chose.
            </li>
            <li>
              <span className="font-medium">{isRegression ? "Averaging" : "Majority voting"}:</span>{" "}
              The final prediction is the {isRegression ? "average of all tree predictions" : "class that receives the most votes"},
              making the model more robust than any single tree.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
