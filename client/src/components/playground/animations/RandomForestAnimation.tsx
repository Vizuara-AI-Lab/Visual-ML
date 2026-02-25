/**
 * RandomForestAnimation — Visual explainer for how Random Forest ensembles work.
 *
 * The entire story is told INSIDE the SVG so it's impossible to miss.
 * A phase banner at the top narrates each step as it happens.
 *
 * Works for any dataset: adapts to classification (majority vote)
 * or regression (average of predictions).
 */

import { useState, useRef, useCallback, useEffect, useMemo, Fragment } from "react";
import {
  Play,
  RotateCcw,
  TreePine,
  BarChart3,
  Database,
  ArrowDown,
  Sigma,
  Vote,
} from "lucide-react";

interface RandomForestAnimationProps {
  result?: Record<string, unknown>;
}

/* ── palettes ── */
const TREE_COLORS = [
  { header: "#0d9488", fill: "#f0fdfa", stroke: "#0d9488", glow: "#99f6e4" },
  { header: "#7c3aed", fill: "#f5f3ff", stroke: "#7c3aed", glow: "#c4b5fd" },
  { header: "#ea580c", fill: "#fff7ed", stroke: "#ea580c", glow: "#fdba74" },
  { header: "#2563eb", fill: "#eff6ff", stroke: "#2563eb", glow: "#93c5fd" },
  { header: "#db2777", fill: "#fdf2f8", stroke: "#db2777", glow: "#f9a8d4" },
];

const CLASS_COLORS = [
  "#0d9488", "#e11d48", "#7c3aed", "#ea580c", "#2563eb",
  "#db2777", "#16a34a", "#f59e0b", "#6366f1", "#84cc16",
];

/* ── types ── */
interface FeatureImportance { feature: string; importance: number }
interface TreePrediction { label: string; numericValue?: number; color: string }

/* ── timing (ms) ── */
const T_TREE = 600;
const T_THINK = 750;
const T_PREDICT = 400;
const T_PARTICLE = 1200;
const T_FINAL = 600;

/* ── phases ── */
type AnimPhase = "idle" | "trees" | "thinking" | "predicting" | "voting" | "final" | "done";

const STEP_LABELS: { phases: AnimPhase[]; label: string; icon: React.ComponentType<{ size?: number; className?: string }> }[] = [
  { phases: ["trees"], label: "Build", icon: TreePine },
  { phases: ["thinking"], label: "Analyze", icon: Database },
  { phases: ["predicting"], label: "Predict", icon: ArrowDown },
  { phases: ["voting"], label: "Combine", icon: Sigma },
  { phases: ["final", "done"], label: "Result", icon: Vote },
];

function stepIdx(p: AnimPhase) {
  if (p === "idle") return -1;
  const i = STEP_LABELS.findIndex((s) => s.phases.includes(p));
  return i >= 0 ? i : 4;
}

/* ── SVG inline banner text per phase ── */
function phaseBanner(
  p: AnimPhase,
  isReg: boolean,
  nTrees: number,
  nSamples: number,
): { title: string; subtitle: string } {
  switch (p) {
    case "idle":
      return {
        title: "Random Forest Ensemble",
        subtitle: `${nTrees} trees trained on ${nSamples.toLocaleString()} samples — press Play to see how it works`,
      };
    case "trees":
      return {
        title: "Step 1 — Building the Forest",
        subtitle: `Each of the ${nTrees} trees gets a random ~63% sample of the training data (bootstrap sampling)`,
      };
    case "thinking":
      return {
        title: "Step 2 — Trees Analyze Data",
        subtitle: "Each tree learns patterns independently, using random features at every split",
      };
    case "predicting":
      return {
        title: isReg ? "Step 3 — Each Tree Predicts a Value" : "Step 3 — Each Tree Votes",
        subtitle: isReg
          ? "Every tree outputs its own numeric prediction — they differ because each saw different data"
          : "Every tree votes for a class — votes differ because each tree learned from different data",
      };
    case "voting":
      return {
        title: isReg ? "Step 4 — Averaging Predictions" : "Step 4 — Counting Votes",
        subtitle: isReg
          ? "All predictions are collected and averaged — this cancels out individual tree errors"
          : "Votes are tallied — the class with the most votes wins (majority voting)",
      };
    case "final":
    case "done":
      return {
        title: "Final Result",
        subtitle: isReg
          ? "The average is more accurate and stable than any single tree's prediction"
          : "The majority vote is more reliable than any single tree — errors get outvoted",
      };
    default:
      return { title: "", subtitle: "" };
  }
}

/* ── SVG mini-tree card ── */
function MiniTree({
  cx, cy, palette, opacity, glowing, label, samplePct,
}: {
  cx: number; cy: number;
  palette: (typeof TREE_COLORS)[0];
  opacity: number; glowing: boolean;
  label: string; samplePct: string;
}) {
  const w = 88;
  const h = 115;
  const x = cx - w / 2;
  const y = cy;

  return (
    <g opacity={opacity} style={{ transition: "opacity 0.4s" }}>
      {glowing && (
        <>
          <rect x={x - 6} y={y - 6} width={w + 12} height={h + 12} rx={14} fill={palette.glow} opacity={0.25}>
            <animate attributeName="opacity" values="0.1;0.35;0.1" dur="0.8s" repeatCount="indefinite" />
          </rect>
          <rect x={x - 4} y={y - 4} width={w + 8} height={h + 8} rx={12} fill="none" stroke={palette.header} strokeWidth={2} opacity={0.5}>
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="0.8s" repeatCount="indefinite" />
          </rect>
        </>
      )}

      {/* card */}
      <rect x={x} y={y} width={w} height={h} rx={10} fill={palette.fill} stroke={palette.stroke} strokeWidth={1.2} />
      {/* header */}
      <rect x={x} y={y} width={w} height={24} rx={10} fill={palette.header} />
      <rect x={x} y={y + 14} width={w} height={10} fill={palette.header} />
      <text x={cx} y={y + 16} textAnchor="middle" fill="white" fontSize={11} fontWeight={600}>{label}</text>

      {/* tree diagram — root → 2 children → 2 leaves */}
      <circle cx={cx} cy={y + 42} r={6} fill={palette.header} opacity={0.85} />
      <line x1={cx} y1={y + 48} x2={cx - 16} y2={y + 64} stroke={palette.stroke} strokeWidth={1} opacity={0.5} />
      <line x1={cx} y1={y + 48} x2={cx + 16} y2={y + 64} stroke={palette.stroke} strokeWidth={1} opacity={0.5} />
      <circle cx={cx - 16} cy={y + 68} r={5} fill={palette.header} opacity={0.5} />
      <circle cx={cx + 16} cy={y + 68} r={5} fill={palette.header} opacity={0.5} />
      <line x1={cx - 16} y1={y + 73} x2={cx - 24} y2={y + 83} stroke={palette.stroke} strokeWidth={0.7} opacity={0.35} />
      <line x1={cx - 16} y1={y + 73} x2={cx - 8} y2={y + 83} stroke={palette.stroke} strokeWidth={0.7} opacity={0.35} />
      <rect x={cx - 28} y={y + 83} width={8} height={6} rx={2} fill={palette.header} opacity={0.3} />
      <rect x={cx - 12} y={y + 83} width={8} height={6} rx={2} fill={palette.header} opacity={0.3} />

      {/* data sample badge */}
      <rect x={cx - 26} y={y + h - 18} width={52} height={14} rx={7} fill={palette.glow} opacity={0.4} stroke={palette.stroke} strokeWidth={0.6} />
      <text x={cx} y={y + h - 8} textAnchor="middle" fontSize={8} fontWeight={600} fill={palette.header}>{samplePct} data</text>
    </g>
  );
}

/* ════════════════════════════════════════════ */
/*  MAIN                                       */
/* ════════════════════════════════════════════ */

export default function RandomForestAnimation({ result }: RandomForestAnimationProps) {
  /* ── data extraction ── */
  const fullMeta = useMemo(() => (result?.metadata as Record<string, unknown>)?.full_training_metadata as Record<string, unknown> | undefined, [result]);

  const taskType = (result?.task_type as string) || "classification";
  const isReg = taskType === "regression";
  const nEstimators = (result?.n_estimators as number) || (fullMeta?.n_estimators as number) || 5;
  const classNames = (fullMeta?.class_names as string[]) || [];
  const metrics = result?.training_metrics as Record<string, number> | undefined;
  const featureImportances = (fullMeta?.feature_importances as FeatureImportance[]) || [];
  const topFeatures = featureImportances.slice(0, 5);
  const trainSamples = (result?.training_samples as number) || (fullMeta?.n_samples as number) || 0;
  const nFeatures = (result?.n_features as number) || (fullMeta?.n_features as number) || 0;

  const treeCount = Math.min(nEstimators, 5);

  const classColorMap = useMemo(() => {
    const m = new Map<string, string>();
    classNames.forEach((n, i) => m.set(String(n), CLASS_COLORS[i % CLASS_COLORS.length]));
    return m;
  }, [classNames]);

  /* ── real sample predictions from backend ── */
  const samplePreds = useMemo(() => fullMeta?.sample_predictions as {
    tree_predictions?: { tree_index: number; prediction: string; numeric_value?: number }[];
    ensemble_prediction?: string;
    ensemble_numeric?: number;
    actual_label?: string;
    actual_value?: number;
    task_type?: string;
  } | undefined, [fullMeta]);

  /* ── predictions (use real data when available, fallback for older results) ── */
  const predictions: TreePrediction[] = useMemo(() => {
    const realPreds = samplePreds?.tree_predictions;
    if (realPreds && realPreds.length > 0) {
      return realPreds.slice(0, treeCount).map((tp, i) => {
        const label = tp.prediction;
        const numVal = tp.numeric_value;
        const color = isReg
          ? TREE_COLORS[i % TREE_COLORS.length].header
          : classColorMap.get(label) || CLASS_COLORS[i % CLASS_COLORS.length];
        return { label, numericValue: numVal, color };
      });
    }
    // Fallback: no real data available
    if (isReg) {
      return Array.from({ length: treeCount }, (_, i) => ({
        label: "?", numericValue: 0, color: TREE_COLORS[i % TREE_COLORS.length].header,
      }));
    }
    if (classNames.length === 0) {
      return Array.from({ length: treeCount }, (_, i) => ({
        label: `Tree ${i + 1}`, color: TREE_COLORS[i % TREE_COLORS.length].header,
      }));
    }
    return Array.from({ length: treeCount }, (_, i) => ({
      label: classNames[i % classNames.length],
      color: classColorMap.get(classNames[i % classNames.length]) || CLASS_COLORS[i % CLASS_COLORS.length],
    }));
  }, [treeCount, classNames, classColorMap, isReg, samplePreds]);

  const finalPred = useMemo(() => {
    // Use real ensemble prediction if available
    if (samplePreds?.ensemble_prediction) {
      const label = samplePreds.ensemble_prediction;
      const color = isReg ? "#2563eb" : classColorMap.get(label) || CLASS_COLORS[0];
      return { label, color };
    }
    // Fallback: compute from per-tree predictions
    if (isReg) {
      const avg = predictions.reduce((s, p) => s + (p.numericValue ?? 0), 0) / predictions.length;
      return { label: avg.toFixed(1), color: "#2563eb" };
    }
    const c = new Map<string, number>();
    for (const p of predictions) c.set(p.label, (c.get(p.label) || 0) + 1);
    let best = predictions[0]?.label || "?";
    let bestN = 0;
    for (const [l, n] of c) { if (n > bestN) { best = l; bestN = n; } }
    return { label: best, color: classColorMap.get(best) || CLASS_COLORS[0] };
  }, [predictions, classColorMap, isReg, samplePreds]);

  const tallyStr = useMemo(() => {
    if (isReg) return predictions.map((p) => p.label).join(" + ") + ` ÷ ${predictions.length}`;
    const t = new Map<string, number>();
    for (const p of predictions) t.set(p.label, (t.get(p.label) || 0) + 1);
    return Array.from(t.entries()).sort((a, b) => b[1] - a[1]).map(([l, n]) => `${l}: ${n}`).join(" · ");
  }, [predictions, isReg]);

  const samplePcts = ["~63%", "~65%", "~61%", "~64%", "~62%"].slice(0, treeCount);

  /* ── animation state ── */
  const [phase, setPhase] = useState<AnimPhase>("idle");
  const [treesVis, setTreesVis] = useState(0);
  const [thinkIdx, setThinkIdx] = useState(-1);
  const [predsVis, setPredsVis] = useState(0);
  const [partProg, setPartProg] = useState(0);
  const [showFinal, setShowFinal] = useState(false);

  const raf = useRef<number | null>(null);
  const t0 = useRef(0);
  const phRef = useRef<AnimPhase>("idle");

  useEffect(() => { phRef.current = phase; }, [phase]);
  useEffect(() => () => { if (raf.current !== null) cancelAnimationFrame(raf.current); }, []);

  /* ── layout ── */
  const W = 720;
  const H = 540;
  const spacing = W / (treeCount + 1);
  const centers = Array.from({ length: treeCount }, (_, i) => spacing * (i + 1));
  const bannerH = 56;
  const dataY = bannerH + 14;
  const treeY = dataY + 42;
  const predY = treeY + 132;
  const voteY = predY + 95;
  const cx0 = W / 2;

  /* ── tick ── */
  const tick = useCallback((now: number) => {
    const dt = now - t0.current;
    const cp = phRef.current;
    if (cp === "trees") {
      const idx = Math.floor(dt / T_TREE);
      if (idx < treeCount) { setTreesVis(idx + 1); } else { setTreesVis(treeCount); t0.current = now; setPhase("thinking"); phRef.current = "thinking"; }
    } else if (cp === "thinking") {
      const idx = Math.floor(dt / T_THINK);
      if (idx < treeCount) { setThinkIdx(idx); } else { setThinkIdx(-1); t0.current = now; setPhase("predicting"); phRef.current = "predicting"; }
    } else if (cp === "predicting") {
      const idx = Math.floor(dt / T_PREDICT);
      if (idx < treeCount) { setPredsVis(idx + 1); } else { setPredsVis(treeCount); t0.current = now; setPhase("voting"); phRef.current = "voting"; }
    } else if (cp === "voting") {
      const p = Math.min(dt / T_PARTICLE, 1);
      setPartProg(p);
      if (p >= 1) { t0.current = now; setPhase("final"); phRef.current = "final"; }
    } else if (cp === "final") {
      if (dt >= T_FINAL) { setShowFinal(true); setPhase("done"); phRef.current = "done"; }
    }
    if (phRef.current !== "idle" && phRef.current !== "done") raf.current = requestAnimationFrame(tick);
  }, [treeCount]);

  const play = useCallback(() => {
    setTreesVis(0); setThinkIdx(-1); setPredsVis(0); setPartProg(0); setShowFinal(false);
    setPhase("trees"); phRef.current = "trees"; t0.current = performance.now();
    if (raf.current !== null) cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(tick);
  }, [tick]);

  const reset = useCallback(() => {
    if (raf.current !== null) cancelAnimationFrame(raf.current);
    setPhase("idle"); phRef.current = "idle";
    setTreesVis(0); setThinkIdx(-1); setPredsVis(0); setPartProg(0); setShowFinal(false);
  }, []);

  if (!(result && (nEstimators > 0 || trainSamples > 0))) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-400 gap-3">
        <TreePine className="h-10 w-10" />
        <p className="text-sm font-medium">No random forest data available</p>
        <p className="text-xs">Run the pipeline to see the trained ensemble</p>
      </div>
    );
  }

  const curStep = stepIdx(phase);
  const banner = phaseBanner(phase, isReg, nEstimators, trainSamples);

  /* ── render ── */
  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col gap-5">

      {/* ═══════ SVG ═══════ */}
      <div className="w-full rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="rf-bg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f8fafc" />
              <stop offset="100%" stopColor="#f1f5f9" />
            </linearGradient>
            <linearGradient id="rf-banner" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#0d9488" stopOpacity={0.08} />
              <stop offset="100%" stopColor="#7c3aed" stopOpacity={0.06} />
            </linearGradient>
          </defs>
          <rect width={W} height={H} fill="url(#rf-bg)" />

          {/* ── Phase banner (inside SVG) ── */}
          <rect x={0} y={0} width={W} height={bannerH} fill="url(#rf-banner)" />
          <line x1={0} y1={bannerH} x2={W} y2={bannerH} stroke="#e2e8f0" strokeWidth={0.5} />

          <text x={20} y={23} fontSize={14} fontWeight={700} fill="#0f172a">
            {banner.title}
          </text>
          <text x={20} y={42} fontSize={11} fill="#64748b">
            {banner.subtitle}
          </text>

          {/* step dots (top-right) */}
          {STEP_LABELS.map((_, i) => {
            const dotX = W - 20 - (STEP_LABELS.length - 1 - i) * 18;
            const active = i === curStep;
            const done = i < curStep;
            return (
              <g key={`dot-${i}`}>
                <circle
                  cx={dotX}
                  cy={bannerH / 2}
                  r={active ? 6 : 4.5}
                  fill={done ? "#0d9488" : active ? "#0d9488" : "#cbd5e1"}
                  opacity={done || active ? 1 : 0.5}
                />
                {active && (
                  <circle cx={dotX} cy={bannerH / 2} r={9} fill="none" stroke="#0d9488" strokeWidth={1.5} opacity={0.4}>
                    <animate attributeName="r" values="8;11;8" dur="1.2s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.4;0.15;0.4" dur="1.2s" repeatCount="indefinite" />
                  </circle>
                )}
              </g>
            );
          })}

          {/* ── Training Data pill ── */}
          <rect x={cx0 - 85} y={dataY} width={170} height={26} rx={13} fill="#e0e7ff" stroke="#818cf8" strokeWidth={0.8} />
          <text x={cx0} y={dataY + 17} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="#4338ca">
            Training Data · {trainSamples.toLocaleString()} rows
          </text>

          {/* arrows from data → trees */}
          {centers.map((cx, i) => (
            <line key={`da-${i}`} x1={cx0} y1={dataY + 26} x2={cx} y2={treeY - 3} stroke="#818cf8" strokeWidth={0.8} strokeDasharray="3 3" opacity={i < treesVis ? 0.45 : 0.08} style={{ transition: "opacity 0.3s" }} />
          ))}

          {/* ── Trees ── */}
          {centers.map((cx, i) => (
            <MiniTree key={i} cx={cx} cy={treeY} palette={TREE_COLORS[i % TREE_COLORS.length]} opacity={i < treesVis ? 1 : 0.06} glowing={thinkIdx === i} label={`Tree ${i + 1}`} samplePct={samplePcts[i]} />
          ))}

          {treeCount < nEstimators && (
            <text x={cx0} y={treeY + 125} textAnchor="middle" fontSize={9.5} fill="#94a3b8" fontStyle="italic">
              Showing {treeCount} of {nEstimators} trees
            </text>
          )}

          {/* ── "predicts:" label row ── */}
          {predsVis > 0 && (
            <text x={cx0} y={predY - 8} textAnchor="middle" fontSize={10} fill="#94a3b8" fontWeight={500}>
              {isReg ? "Each tree's predicted value:" : "Each tree's vote:"}
            </text>
          )}

          {/* ── Prediction pills ── */}
          {centers.map((cx, i) => {
            if (i >= predsVis) return null;
            const pr = predictions[i];
            const pw = Math.max(pr.label.length * 8.5 + 20, 58);
            return (
              <g key={`pr-${i}`}>
                <rect x={cx - pw / 2} y={predY} width={pw} height={28} rx={14} fill={pr.color} opacity={0.12} stroke={pr.color} strokeWidth={1.2} />
                <text x={cx} y={predY + 18} textAnchor="middle" fontSize={12} fontWeight={700} fill={pr.color}>
                  {pr.label.length > 9 ? pr.label.slice(0, 8) + "…" : pr.label}
                </text>
              </g>
            );
          })}

          {/* ── Connector lines ── */}
          {(phase === "voting" || phase === "final" || phase === "done") && centers.map((cx, i) => (
            <line key={`cl-${i}`} x1={cx} y1={predY + 28} x2={cx0} y2={voteY} stroke={predictions[i].color} strokeWidth={0.7} strokeDasharray="4 3" opacity={0.3} />
          ))}

          {/* ── Particles (ease-in-out) ── */}
          {phase === "voting" && centers.map((cx, i) => {
            const pr = predictions[i];
            const t = partProg < 0.5 ? 2 * partProg * partProg : 1 - Math.pow(-2 * partProg + 2, 2) / 2;
            const px = cx + (cx0 - cx) * t;
            const py = (predY + 32) + (voteY - predY - 32) * t;
            return (
              <circle key={`pt-${i}`} cx={px} cy={py} r={5} fill={pr.color} opacity={0.85}>
                <animate attributeName="r" values="4;7;4" dur="0.5s" repeatCount="indefinite" />
              </circle>
            );
          })}

          {/* ── Aggregation label ── */}
          {(phase === "voting" || phase === "final" || phase === "done") && (
            <text x={cx0} y={voteY - 12} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="#475569">
              {isReg ? "Average of all predictions ↓" : "Majority vote ↓"}
            </text>
          )}

          {/* ── Final box ── */}
          <rect
            x={cx0 - 95}
            y={voteY}
            width={190}
            height={showFinal ? 62 : 52}
            rx={16}
            fill={showFinal ? finalPred.color : "#e2e8f0"}
            opacity={showFinal ? 0.12 : 0.35}
            stroke={showFinal ? finalPred.color : "#94a3b8"}
            strokeWidth={showFinal ? 2 : 0.8}
            style={{ transition: "all 0.4s" }}
          />
          {showFinal ? (
            <>
              <text x={cx0} y={voteY + 20} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={500}>
                {isReg ? "Average Prediction" : "Majority Vote Winner"}
              </text>
              <text x={cx0} y={voteY + 44} textAnchor="middle" fontSize={20} fontWeight={800} fill={finalPred.color}>
                {finalPred.label}
              </text>
            </>
          ) : (
            <text x={cx0} y={voteY + 30} textAnchor="middle" fontSize={12} fill="#94a3b8">
              Final Prediction
            </text>
          )}

          {/* tally / formula */}
          {showFinal && (
            <text x={cx0} y={voteY + (showFinal ? 78 : 68)} textAnchor="middle" fontSize={9} fill="#94a3b8">
              ({tallyStr})
            </text>
          )}
        </svg>
      </div>

      {/* ═══════ Controls ═══════ */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={play}
          disabled={phase !== "idle" && phase !== "done"}
          className="inline-flex items-center gap-2 rounded-xl bg-teal-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          <Play size={16} />
          {phase === "done" ? "Replay" : "Play Animation"}
        </button>
        <button
          onClick={reset}
          className="inline-flex items-center gap-2 rounded-xl bg-gray-100 px-4 py-2.5 text-sm font-medium text-gray-600 hover:bg-gray-200 transition-all"
        >
          <RotateCcw size={15} />
          Reset
        </button>
      </div>

      {/* ═══════ Step Progress (below SVG) ═══════ */}
      <div className="flex items-center justify-center gap-0">
        {STEP_LABELS.map((step, i) => {
          const Icon = step.icon;
          const active = i === curStep;
          const done = i < curStep;
          return (
            <Fragment key={i}>
              {i > 0 && <div className={`h-0.5 w-10 transition-colors duration-300 ${done ? "bg-teal-400" : "bg-gray-200"}`} />}
              <div className="flex flex-col items-center gap-1">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${done ? "bg-teal-500 text-white" : active ? "bg-teal-500 text-white ring-[3px] ring-teal-200" : "bg-gray-100 text-gray-400"}`}>
                  <Icon size={15} />
                </div>
                <span className={`text-[10px] font-semibold transition-colors duration-300 ${done || active ? "text-teal-700" : "text-gray-400"}`}>{step.label}</span>
              </div>
            </Fragment>
          );
        })}
      </div>

      {/* ═══════ Summary Cards ═══════ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
          <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <TreePine size={16} className="text-teal-600" />
            Training Summary
          </h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {[
              { k: "Trees", v: nEstimators },
              { k: "Samples", v: trainSamples.toLocaleString() },
              { k: "Features", v: nFeatures },
              { k: isReg ? "R² Score" : "Accuracy", v: metrics ? (isReg ? metrics.r2?.toFixed(4) : `${(metrics.accuracy * 100).toFixed(1)}%`) : "—" },
            ].map((m) => (
              <div key={m.k} className="rounded-lg bg-gray-50 p-2.5">
                <p className="text-[11px] text-gray-500">{m.k}</p>
                <p className="font-semibold text-gray-800">{m.v}</p>
              </div>
            ))}
          </div>
          {!isReg && classNames.length > 0 && (
            <div className="pt-1">
              <p className="text-[11px] text-gray-500 mb-1.5">Classes</p>
              <div className="flex flex-wrap gap-1.5">
                {classNames.map((cls, i) => (
                  <span key={String(cls)} className="inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium" style={{ backgroundColor: `${CLASS_COLORS[i % CLASS_COLORS.length]}12`, color: CLASS_COLORS[i % CLASS_COLORS.length], border: `1px solid ${CLASS_COLORS[i % CLASS_COLORS.length]}35` }}>
                    {String(cls)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {topFeatures.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
            <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              <BarChart3 size={16} className="text-amber-600" />
              Top Feature Importances
            </h4>
            <div className="space-y-2.5">
              {topFeatures.map((fi, i) => {
                const pct = (topFeatures[0]?.importance || 1) > 0 ? (fi.importance / (topFeatures[0]?.importance || 1)) * 100 : 0;
                return (
                  <div key={fi.feature} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600 truncate max-w-[140px]" title={fi.feature}>{fi.feature}</span>
                      <span className="font-semibold text-gray-700">{(fi.importance * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: TREE_COLORS[i % TREE_COLORS.length].header }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ═══════ Why It Works ═══════ */}
      <div className="rounded-xl border border-teal-200/50 bg-gradient-to-r from-teal-50/60 to-cyan-50/40 p-4 space-y-2.5">
        <h4 className="text-sm font-semibold text-teal-900 flex items-center gap-2">
          <Vote size={16} className="text-teal-700" />
          Why This Works for Any Dataset
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[13px] text-teal-800">
          <div>
            <p className="font-semibold text-teal-900 mb-0.5">Diversity</p>
            <p>Each tree sees different data and features, so they learn different patterns and make different mistakes.</p>
          </div>
          <div>
            <p className="font-semibold text-teal-900 mb-0.5">Independence</p>
            <p>Trees predict without knowing what others chose. One tree's error doesn't spread to others.</p>
          </div>
          <div>
            <p className="font-semibold text-teal-900 mb-0.5">{isReg ? "Averaging" : "Voting"}</p>
            <p>{isReg ? "Averaging smooths out individual errors — outlier predictions get diluted by the majority." : "Majority voting corrects mistakes — a few wrong votes are outnumbered by the correct ones."}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
