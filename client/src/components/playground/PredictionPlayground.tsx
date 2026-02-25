/**
 * PredictionPlayground — Interactive prediction interface for RF and DT models.
 *
 * Auto-generates input sliders from training feature stats, calls the
 * /ml/predict/interactive endpoint, and renders model-specific visualizations:
 *   - Random Forest: per-tree voting bars + vote tally + confidence
 *   - Decision Tree: decision path stepper + leaf prediction
 */

import { useState, useEffect, useRef, useCallback, useMemo, lazy, Suspense } from "react";
import {
  Play,
  RotateCcw,
  Shuffle,
  TreePine,
  Trees,
  ChevronRight,
  ChevronDown,
  Target,
  BarChart3,
  Loader2,
  AlertCircle,
  Sparkles,
} from "lucide-react";
import apiClient from "../../lib/axios";

const ForestTreePathAnimation = lazy(
  () => import("./animations/ForestTreePathAnimation")
);

/* ═══════════════════════════════════════════════ */
/*  Types                                          */
/* ═══════════════════════════════════════════════ */

interface FeatureStat {
  name: string;
  min: number;
  max: number;
  mean: number;
  median: number;
}

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

interface PathNode {
  node_id: number;
  type: "split" | "leaf";
  feature?: string;
  threshold?: number;
  value?: number;
  direction?: "left" | "right";
  samples?: number;
  impurity?: number;
  prediction?: string;
  numeric_value?: number;
  probabilities?: Record<string, number>;
}

interface PredictionResult {
  prediction: string | number;
  probabilities: Record<string, number>;
  confidence: number;
  model_type: string;
  task_type: string;
  class_names: string[];
  feature_stats: FeatureStat[];
  per_tree_predictions?: PerTreePrediction[];
  vote_summary?: Record<string, number>;
  regression_mean?: number;
  decision_path?: PathNode[];
}

interface PredictionPlaygroundProps {
  result: any;
  variant: "random_forest" | "decision_tree";
}

/* ═══════════════════════════════════════════════ */
/*  Constants                                      */
/* ═══════════════════════════════════════════════ */

const CLASS_COLORS = [
  "#0d9488", "#e11d48", "#7c3aed", "#ea580c", "#2563eb",
  "#db2777", "#16a34a", "#f59e0b", "#6366f1", "#84cc16",
];

const TREE_COLORS = [
  "#0d9488", "#7c3aed", "#ea580c", "#2563eb", "#db2777",
  "#16a34a", "#f59e0b", "#6366f1", "#84cc16", "#e11d48",
];

/* ═══════════════════════════════════════════════ */
/*  Component                                      */
/* ═══════════════════════════════════════════════ */

export default function PredictionPlayground({ result, variant }: PredictionPlaygroundProps) {
  const fullMeta = useMemo(
    () =>
      (result?.metadata as Record<string, unknown>)?.full_training_metadata as
        | Record<string, unknown>
        | undefined,
    [result],
  );

  const featureStats: FeatureStat[] = useMemo(() => {
    const raw = fullMeta?.feature_stats as FeatureStat[] | undefined;
    if (raw && raw.length > 0) return raw;
    // Fallback: build from feature_names with dummy ranges
    const names = (fullMeta?.feature_names as string[]) || [];
    return names.map((n) => ({ name: n, min: 0, max: 100, mean: 50, median: 50 }));
  }, [fullMeta]);

  const modelPath = result?.model_path as string;
  const taskType = (result?.task_type as string) || "classification";
  const isReg = taskType === "regression";

  // Feature values state — initialized to medians
  const [featureValues, setFeatureValues] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    for (const fs of featureStats) {
      init[fs.name] = fs.median;
    }
    return init;
  });

  // Reset featureValues when featureStats change (different model loaded)
  useEffect(() => {
    const init: Record<string, number> = {};
    for (const fs of featureStats) {
      init[fs.name] = fs.median;
    }
    setFeatureValues(init);
  }, [featureStats]);

  const [predResult, setPredResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Class → color map
  const classColorMap = useMemo(() => {
    const names = predResult?.class_names || (fullMeta?.class_names as string[]) || [];
    const m: Record<string, string> = {};
    names.forEach((c, i) => {
      m[String(c)] = CLASS_COLORS[i % CLASS_COLORS.length];
    });
    return m;
  }, [predResult, fullMeta]);

  // Predict function
  const predict = useCallback(
    async (features: Record<string, number>) => {
      if (!modelPath) return;
      setLoading(true);
      setError(null);
      try {
        const resp = await apiClient.post("/ml/predict/interactive", {
          model_path: modelPath,
          model_type: variant,
          task_type: taskType,
          features,
        });
        setPredResult(resp.data);
      } catch (err: any) {
        const msg = err?.response?.data?.detail || err?.message || "Prediction failed";
        setError(typeof msg === "string" ? msg : JSON.stringify(msg));
      } finally {
        setLoading(false);
      }
    },
    [modelPath, variant, taskType],
  );

  // Debounced auto-predict on slider change
  const handleFeatureChange = useCallback(
    (name: string, value: number) => {
      setFeatureValues((prev) => {
        const next = { ...prev, [name]: value };
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => predict(next), 400);
        return next;
      });
    },
    [predict],
  );

  const handleRandomize = () => {
    const newVals: Record<string, number> = {};
    for (const fs of featureStats) {
      const range = fs.max - fs.min;
      newVals[fs.name] = Math.round((fs.min + Math.random() * range) * 100) / 100;
    }
    setFeatureValues(newVals);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => predict(newVals), 300);
  };

  const handleResetMedian = () => {
    const newVals: Record<string, number> = {};
    for (const fs of featureStats) {
      newVals[fs.name] = fs.median;
    }
    setFeatureValues(newVals);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => predict(newVals), 300);
  };

  if (!modelPath || featureStats.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-400 gap-3">
        <AlertCircle className="h-10 w-10" />
        <p className="text-sm font-medium">No model data available</p>
        <p className="text-xs">Run the pipeline first to train a model</p>
      </div>
    );
  }

  const isRF = variant === "random_forest";
  const accentColor = isRF ? "teal" : "green";

  return (
    <div className="space-y-5">
      {/* Info banner */}
      <div className={`flex items-start gap-3 rounded-lg border border-${accentColor}-200 bg-${accentColor}-50 px-4 py-3`}>
        <Sparkles className={`mt-0.5 h-5 w-5 shrink-0 text-${accentColor}-600`} />
        <div className="text-sm text-gray-700">
          <p className="font-semibold">Interactive Prediction Playground</p>
          <p className="mt-0.5">
            Adjust the feature sliders below and see how the {isRF ? "forest's trees vote" : "decision tree traces a path"} to
            reach a prediction. Values auto-predict as you move sliders.
          </p>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-5">
        {/* ═══ LEFT: Feature Inputs ═══ */}
        <div className="lg:w-[380px] shrink-0 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-gray-700">Feature Values</h4>
            <div className="flex gap-2">
              <button
                onClick={handleRandomize}
                className="inline-flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition"
              >
                <Shuffle size={12} /> Random
              </button>
              <button
                onClick={handleResetMedian}
                className="inline-flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition"
              >
                <RotateCcw size={12} /> Median
              </button>
            </div>
          </div>

          <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
            {featureStats.map((fs) => {
              const val = featureValues[fs.name] ?? fs.median;
              const range = fs.max - fs.min;
              const step = range > 0 ? Math.max(range / 200, 0.01) : 0.01;
              return (
                <div key={fs.name} className="rounded-lg border border-gray-200 bg-white px-3 py-2.5">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-medium text-gray-700 truncate max-w-[180px]" title={fs.name}>
                      {fs.name}
                    </span>
                    <input
                      type="number"
                      value={val}
                      step={step}
                      onChange={(e) => handleFeatureChange(fs.name, parseFloat(e.target.value) || 0)}
                      className="w-20 text-right text-xs font-mono bg-gray-50 border border-gray-200 rounded px-1.5 py-1 focus:outline-none focus:ring-1 focus:ring-teal-400"
                    />
                  </div>
                  <input
                    type="range"
                    min={fs.min}
                    max={fs.max}
                    step={step}
                    value={val}
                    onChange={(e) => handleFeatureChange(fs.name, parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer accent-teal-500"
                  />
                  <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                    <span>{fs.min}</span>
                    <span className="text-gray-500">avg {fs.mean}</span>
                    <span>{fs.max}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <button
            onClick={() => predict(featureValues)}
            disabled={loading}
            className={`w-full inline-flex items-center justify-center gap-2 rounded-xl bg-${accentColor}-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-${accentColor}-700 disabled:opacity-40 transition-all`}
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            {loading ? "Predicting..." : "Predict"}
          </button>
        </div>

        {/* ═══ RIGHT: Results ═══ */}
        <div className="flex-1 min-w-0 space-y-4">
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <AlertCircle className="inline w-4 h-4 mr-1.5 -mt-0.5" />
              {error}
            </div>
          )}

          {!predResult && !loading && !error && (
            <div className="flex flex-col items-center justify-center py-16 text-gray-400 gap-3">
              {isRF ? <Trees className="h-12 w-12" /> : <TreePine className="h-12 w-12" />}
              <p className="text-sm font-medium">Adjust features and click Predict</p>
              <p className="text-xs">or move any slider — it predicts automatically</p>
            </div>
          )}

          {predResult && (
            <>
              {/* Hero prediction */}
              <div
                className="rounded-xl border-2 p-5 text-center"
                style={{
                  borderColor: isReg ? "#2563eb" : (classColorMap[String(predResult.prediction)] || "#0d9488"),
                  backgroundColor: isReg ? "#eff6ff" : ((classColorMap[String(predResult.prediction)] || "#0d9488") + "10"),
                }}
              >
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                  {isReg ? "Predicted Value" : "Predicted Class"}
                </p>
                <p
                  className="text-3xl font-extrabold"
                  style={{ color: isReg ? "#2563eb" : (classColorMap[String(predResult.prediction)] || "#0d9488") }}
                >
                  {String(predResult.prediction)}
                </p>
                {!isReg && predResult.confidence > 0 && (
                  <div className="mt-3 max-w-xs mx-auto">
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                      <span>Confidence</span>
                      <span className="font-semibold text-gray-700">{(predResult.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${predResult.confidence * 100}%`,
                          backgroundColor: classColorMap[String(predResult.prediction)] || "#0d9488",
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Probability distribution (classification) */}
              {!isReg && Object.keys(predResult.probabilities).length > 0 && (
                <div className="rounded-xl border border-gray-200 bg-white p-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    <BarChart3 size={15} className="text-gray-500" />
                    Class Probabilities
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(predResult.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([cls, prob]) => (
                        <div key={cls} className="flex items-center gap-3">
                          <span
                            className="text-xs font-semibold w-20 truncate text-right"
                            style={{ color: classColorMap[cls] || "#6b7280" }}
                            title={cls}
                          >
                            {cls}
                          </span>
                          <div className="flex-1 h-5 rounded-full bg-gray-100 overflow-hidden relative">
                            <div
                              className="h-full rounded-full transition-all duration-300"
                              style={{
                                width: `${Math.max(prob * 100, 1)}%`,
                                backgroundColor: classColorMap[cls] || "#6b7280",
                                opacity: 0.8,
                              }}
                            />
                            <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] font-bold text-gray-700">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* RF: Per-tree voting with SVG tree path animation */}
              {isRF && predResult.per_tree_predictions && predResult.per_tree_predictions.length > 0 && (
                predResult.per_tree_predictions[0]?.tree_structure ? (
                  <Suspense
                    fallback={
                      <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-6 w-6 animate-spin text-teal-500" />
                      </div>
                    }
                  >
                    <ForestTreePathAnimation
                      perTreePredictions={predResult.per_tree_predictions}
                      classColorMap={classColorMap}
                      finalPrediction={String(predResult.prediction)}
                      voteSummary={predResult.vote_summary}
                      regressionMean={predResult.regression_mean}
                      isRegression={isReg}
                    />
                  </Suspense>
                ) : (
                  <div className="rounded-xl border border-gray-200 bg-white p-4">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                      <Trees size={15} className="text-teal-600" />
                      {isReg ? "Per-Tree Predictions" : "Per-Tree Voting"} ({predResult.per_tree_predictions.length} trees)
                    </h4>

                    {!isReg && predResult.vote_summary && (
                      <div className="mb-3 flex flex-wrap gap-2">
                        {Object.entries(predResult.vote_summary)
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

                    <div className="space-y-1.5">
                      {predResult.per_tree_predictions.map((tp, i) => {
                        const color = isReg
                          ? TREE_COLORS[i % TREE_COLORS.length]
                          : classColorMap[tp.prediction] || TREE_COLORS[i % TREE_COLORS.length];

                        if (isReg) {
                          const allVals = predResult.per_tree_predictions!.map((t) => t.numeric_value || 0);
                          const mn = Math.min(...allVals);
                          const mx = Math.max(...allVals);
                          const rng = mx - mn || 1;
                          const pct = ((tp.numeric_value || 0) - mn) / rng * 100;
                          return (
                            <div key={i} className="flex items-center gap-2">
                              <span className="text-[10px] font-semibold text-gray-500 w-12 shrink-0">Tree {i + 1}</span>
                              <div className="flex-1 h-4 rounded-full bg-gray-100 relative">
                                <div
                                  className="absolute top-0.5 w-3 h-3 rounded-full"
                                  style={{ left: `calc(${pct}% - 6px)`, backgroundColor: color }}
                                />
                              </div>
                              <span className="text-xs font-mono font-semibold w-16 text-right" style={{ color }}>
                                {tp.numeric_value?.toFixed(2)}
                              </span>
                            </div>
                          );
                        }

                        return (
                          <div key={i} className="flex items-center gap-2">
                            <span className="text-[10px] font-semibold text-gray-500 w-12 shrink-0">Tree {i + 1}</span>
                            <span
                              className="text-xs font-semibold px-2.5 py-1 rounded-full"
                              style={{
                                backgroundColor: color + "15",
                                color: color,
                                border: `1px solid ${color}40`,
                              }}
                            >
                              {tp.prediction}
                            </span>
                            {tp.probabilities && (
                              <div className="flex-1 flex gap-0.5 h-3 rounded-full overflow-hidden bg-gray-100">
                                {Object.entries(tp.probabilities)
                                  .sort(([, a], [, b]) => b - a)
                                  .map(([cls, p]) => (
                                    <div
                                      key={cls}
                                      className="h-full first:rounded-l-full last:rounded-r-full"
                                      style={{
                                        width: `${p * 100}%`,
                                        backgroundColor: classColorMap[cls] || "#d1d5db",
                                        opacity: 0.7,
                                      }}
                                      title={`${cls}: ${(p * 100).toFixed(1)}%`}
                                    />
                                  ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    {isReg && predResult.regression_mean != null && (
                      <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between">
                        <span className="text-xs font-medium text-gray-500">Ensemble Average</span>
                        <span className="text-sm font-bold text-blue-600">{predResult.regression_mean.toFixed(4)}</span>
                      </div>
                    )}
                  </div>
                )
              )}

              {/* DT: Decision Path */}
              {!isRF && predResult.decision_path && predResult.decision_path.length > 0 && (
                <div className="rounded-xl border border-gray-200 bg-white p-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    <TreePine size={15} className="text-green-600" />
                    Decision Path ({predResult.decision_path.length} nodes)
                  </h4>
                  <div className="relative pl-6">
                    {predResult.decision_path.map((node, i) => {
                      const isLast = i === predResult.decision_path!.length - 1;
                      const isLeaf = node.type === "leaf";
                      return (
                        <div key={i} className="relative pb-4 last:pb-0">
                          {/* connector */}
                          {!isLast && (
                            <div className="absolute left-[-14px] top-6 bottom-0 w-0.5 bg-green-300" />
                          )}
                          {/* dot */}
                          <div
                            className={`absolute left-[-18px] top-1.5 w-3 h-3 rounded-full border-2 ${
                              isLeaf ? "bg-green-500 border-green-500" : "bg-blue-500 border-blue-500"
                            }`}
                          />
                          {/* card */}
                          <div
                            className={`rounded-lg border px-3 py-2.5 ${
                              isLeaf ? "border-green-300 bg-green-50" : "border-gray-200 bg-gray-50"
                            }`}
                          >
                            {node.type === "split" ? (
                              <>
                                <div className="flex items-center gap-2 text-xs">
                                  <span className="font-semibold text-blue-600">Split</span>
                                  <span className="text-gray-400">node {node.node_id}</span>
                                  <span className="text-gray-400">|</span>
                                  <span className="text-gray-500">{node.samples} samples</span>
                                </div>
                                <p className="text-sm font-medium text-gray-900 mt-1">
                                  Is <span className="font-bold text-blue-700">{node.feature}</span> &le; {node.threshold}?
                                </p>
                                <div className="mt-1.5 flex items-center gap-2 text-xs">
                                  <span className="font-mono bg-white border rounded px-1.5 py-0.5 text-gray-700">
                                    {node.feature} = {node.value}
                                  </span>
                                  <ChevronRight className="w-3 h-3 text-green-500" />
                                  <span className="font-semibold text-green-700">
                                    {node.direction === "left" ? "Yes (left)" : "No (right)"}
                                  </span>
                                </div>
                              </>
                            ) : (
                              <>
                                <div className="flex items-center gap-2 text-xs">
                                  <Target className="w-3.5 h-3.5 text-green-600" />
                                  <span className="font-semibold text-green-700">Leaf</span>
                                  <span className="text-gray-400">|</span>
                                  <span className="text-gray-500">{node.samples} samples</span>
                                </div>
                                <p className="text-lg font-bold text-green-800 mt-1">
                                  {node.prediction}
                                </p>
                                {node.probabilities && (
                                  <div className="flex gap-2 mt-1.5 flex-wrap">
                                    {Object.entries(node.probabilities)
                                      .sort(([, a], [, b]) => b - a)
                                      .map(([cls, p]) => (
                                        <span
                                          key={cls}
                                          className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
                                          style={{
                                            backgroundColor: (classColorMap[cls] || "#6b7280") + "15",
                                            color: classColorMap[cls] || "#6b7280",
                                          }}
                                        >
                                          {cls}: {(p * 100).toFixed(1)}%
                                        </span>
                                      ))}
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
