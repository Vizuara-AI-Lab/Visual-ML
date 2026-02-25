/**
 * RandomForestExplorer - Interactive learning activities for Random Forest
 * Tabbed component: Results | Ensemble Voting | Feature Importance | Tree Comparison | Metrics Explained | Quiz | How It Works
 */

import { useState, lazy, Suspense } from "react";
import {
  ClipboardList,
  TreePine,
  BarChart3,
  GitCompare,
  Gauge,
  HelpCircle,
  Cog,
  CheckCircle,
  XCircle,
  Trophy,
  Target,
  Timer,
  Hash,
  Database,
  Trees,
  Layers,
  Leaf,
} from "lucide-react";

const RandomForestAnimation = lazy(
  () => import("./animations/RandomForestAnimation")
);

interface RandomForestExplorerProps {
  result: any;
}

type ExplorerTab =
  | "results"
  | "ensemble_voting"
  | "feature_importance"
  | "tree_comparison"
  | "metric_explainer"
  | "quiz"
  | "how_it_works";

export const RandomForestExplorer = ({
  result,
}: RandomForestExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "ensemble_voting",
      label: "Ensemble Voting",
      icon: Trees,
      available: true,
    },
    {
      id: "feature_importance",
      label: "Feature Importance",
      icon: BarChart3,
      available: !!result.feature_importance_analysis?.features?.length,
    },
    {
      id: "tree_comparison",
      label: "Tree Comparison",
      icon: GitCompare,
      available: !!result.tree_comparison?.trees?.length,
    },
    {
      id: "metric_explainer",
      label: "Metrics Explained",
      icon: Gauge,
      available: !!result.metric_explainer?.metrics?.length,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available: !!result.quiz_questions && result.quiz_questions.length > 0,
    },
    {
      id: "how_it_works",
      label: "How It Works",
      icon: Cog,
      available: true,
    },
  ];

  const isRegression = result.task_type === "regression";
  const metrics = result.training_metrics || {};

  return (
    <div className="space-y-4">
      {/* Tab navigation */}
      <div className="flex border-b border-gray-200 overflow-x-auto">
        {tabs
          .filter((t) => t.available)
          .map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-teal-500 text-teal-700"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
      </div>

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === "results" && (
          <ResultsTab result={result} isRegression={isRegression} metrics={metrics} />
        )}
        {activeTab === "ensemble_voting" && (
          <Suspense
            fallback={
              <div className="flex items-center justify-center py-12">
                <div className="w-6 h-6 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" />
              </div>
            }
          >
            <RandomForestAnimation result={result} />
          </Suspense>
        )}
        {activeTab === "feature_importance" && (
          <FeatureImportanceTab data={result.feature_importance_analysis} />
        )}
        {activeTab === "tree_comparison" && (
          <TreeComparisonTab data={result.tree_comparison} />
        )}
        {activeTab === "metric_explainer" && (
          <MetricExplainerTab data={result.metric_explainer} />
        )}
        {activeTab === "quiz" && (
          <QuizTab questions={result.quiz_questions || []} />
        )}
        {activeTab === "how_it_works" && <HowItWorksTab nEstimators={result.n_estimators} />}
      </div>
    </div>
  );
};

// ======================== Results Tab ========================

function ResultsTab({
  result,
  isRegression,
  metrics,
}: {
  result: any;
  isRegression: boolean;
  metrics: any;
}) {
  const primaryMetric = isRegression ? metrics.r2 : metrics.accuracy;
  const primaryLabel = isRegression ? "R² Score" : "Accuracy";
  const primaryPct = primaryMetric != null ? primaryMetric * 100 : null;

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 border border-teal-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-teal-500 rounded-full flex items-center justify-center">
            <CheckCircle className="text-white w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-teal-900">
              Random Forest Training Complete
            </h3>
            <p className="text-sm text-teal-700">
              Ensemble of {result.n_estimators} trees trained on{" "}
              {result.training_samples?.toLocaleString()} samples with{" "}
              {result.n_features} features ({isRegression ? "regression" : "classification"})
            </p>
          </div>
        </div>
      </div>

      {/* Hero Metric */}
      {primaryPct != null && (
        <div className="flex justify-center">
          <div className="relative w-36 h-36">
            <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#e5e7eb" strokeWidth="8" />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke={primaryPct >= 80 ? "#14b8a6" : primaryPct >= 60 ? "#f59e0b" : "#ef4444"}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${(primaryPct / 100) * 327} 327`}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-bold text-gray-900">
                {primaryPct.toFixed(1)}%
              </span>
              <span className="text-xs text-gray-500">{primaryLabel}</span>
            </div>
          </div>
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Trees className="w-3.5 h-3.5" /> Trees
          </div>
          <div className="text-lg font-semibold text-gray-900">{result.n_estimators}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Database className="w-3.5 h-3.5" /> Samples
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {result.training_samples?.toLocaleString()}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Hash className="w-3.5 h-3.5" /> Features
          </div>
          <div className="text-lg font-semibold text-gray-900">{result.n_features}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
            <Timer className="w-3.5 h-3.5" /> Training Time
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {result.training_time_seconds?.toFixed(2)}s
          </div>
        </div>
      </div>

      {/* All Metrics */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-4">Training Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(metrics)
            .filter(([key, value]) => typeof value === "number" && key !== "confusion_matrix")
            .map(([key, value]) => (
              <div key={key} className="bg-gray-50 rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                  {key.replace(/_/g, " ")}
                </div>
                <div className="text-xl font-bold text-gray-900">
                  {(value as number).toFixed(4)}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Model Info */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Model Details</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Task Type:</span>
            <span className="font-semibold capitalize">{result.task_type}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Target Column:</span>
            <span className="font-semibold">{result.target_column || "N/A"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Estimators:</span>
            <span className="font-semibold">{result.n_estimators}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Model ID:</span>
            <code className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded">
              {result.model_id?.slice(0, 20)}...
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}

// ======================== Feature Importance Tab ========================

function FeatureImportanceTab({ data }: { data: any }) {
  const features = data?.features || [];
  const [sortBy, setSortBy] = useState<"importance" | "name">("importance");

  const sorted = [...features].sort((a: any, b: any) =>
    sortBy === "importance"
      ? b.importance - a.importance
      : a.feature.localeCompare(b.feature)
  );

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
        <BarChart3 className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />
        <p className="text-sm text-teal-800">
          <span className="font-semibold">Ensemble feature importance</span> is averaged
          across all trees in the forest. Features with higher importance are more useful
          for making predictions. This is one of Random Forest's key advantages — robust
          importance estimates.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-500">Sort by:</span>
        <button
          onClick={() => setSortBy("importance")}
          className={`text-xs px-3 py-1 rounded-full ${
            sortBy === "importance"
              ? "bg-teal-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          By Impact
        </button>
        <button
          onClick={() => setSortBy("name")}
          className={`text-xs px-3 py-1 rounded-full ${
            sortBy === "name"
              ? "bg-teal-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          By Name
        </button>
      </div>

      <div className="space-y-2">
        {sorted.map((feat: any, i: number) => (
          <div key={feat.feature} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-700 font-medium truncate max-w-[200px]" title={feat.feature}>
                {i + 1}. {feat.feature}
              </span>
              <span className="font-semibold text-gray-900">
                {(feat.importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-3 rounded-full bg-gray-100 overflow-hidden">
              <div
                className="h-full rounded-full bg-teal-500 transition-all duration-500"
                style={{ width: `${feat.bar_width_pct}%` }}
              />
            </div>
            <p className="text-xs text-gray-500">{feat.interpretation}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ======================== Tree Comparison Tab ========================

function TreeComparisonTab({ data }: { data: any }) {
  const trees = data?.trees || [];
  const summary = data?.summary || {};

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
        <GitCompare className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
        <p className="text-sm text-amber-800">
          <span className="font-semibold">Tree Comparison</span> — {data?.description || "Each tree in the forest is different because it trains on a random subset of data and considers random features at each split."}
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Total Trees</div>
          <div className="text-lg font-bold text-gray-900">{data?.n_estimators_total}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Avg Depth</div>
          <div className="text-lg font-bold text-gray-900">{summary.avg_depth}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Min Depth</div>
          <div className="text-lg font-bold text-gray-900">{summary.min_depth}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Max Depth</div>
          <div className="text-lg font-bold text-gray-900">{summary.max_depth}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Avg Leaves</div>
          <div className="text-lg font-bold text-gray-900">{summary.avg_leaves}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500">Shown</div>
          <div className="text-lg font-bold text-gray-900">{data?.n_trees_shown}</div>
        </div>
      </div>

      {/* Individual trees */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {trees.map((tree: any) => {
          const depthPct = summary.max_depth > 0
            ? (tree.depth / summary.max_depth) * 100
            : 50;
          return (
            <div
              key={tree.tree_index}
              className="bg-white border border-gray-200 rounded-lg p-4 space-y-2"
            >
              <div className="flex items-center gap-2">
                <TreePine className="w-4 h-4 text-teal-600" />
                <span className="text-sm font-semibold text-gray-900">
                  Tree {tree.tree_index + 1}
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Depth</span>
                  <span className="font-medium text-gray-700">{tree.depth}</span>
                </div>
                <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-teal-400"
                    style={{ width: `${depthPct}%` }}
                  />
                </div>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Leaves</span>
                <span className="font-medium text-gray-700">{tree.n_leaves}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ======================== Metric Explainer Tab ========================

function MetricExplainerTab({ data }: { data: any }) {
  const metrics = data?.metrics || [];
  const colorMap: Record<string, string> = {
    green: "bg-green-50 border-green-200",
    blue: "bg-blue-50 border-blue-200",
    orange: "bg-orange-50 border-orange-200",
    purple: "bg-purple-50 border-purple-200",
    yellow: "bg-yellow-50 border-yellow-200",
    red: "bg-red-50 border-red-200",
  };
  const barColorMap: Record<string, string> = {
    green: "bg-green-500",
    blue: "bg-blue-500",
    orange: "bg-orange-500",
    purple: "bg-purple-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-purple-200 bg-purple-50 px-4 py-3">
        <Gauge className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
        <p className="text-sm text-purple-800">
          <span className="font-semibold">Understanding your metrics</span> — Each metric
          tells a different part of the story about your ensemble model's performance.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {metrics.map((m: any) => (
          <div
            key={m.metric}
            className={`rounded-lg border p-4 space-y-2 ${colorMap[m.color] || "bg-gray-50 border-gray-200"}`}
          >
            <div className="flex justify-between items-center">
              <h4 className="font-semibold text-gray-900">{m.metric}</h4>
              {m.value_pct != null ? (
                <span className="text-lg font-bold text-gray-900">{m.value_pct}%</span>
              ) : (
                <span className="text-lg font-bold text-gray-900">{m.value}</span>
              )}
            </div>
            {m.value_pct != null && (
              <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
                <div
                  className={`h-full rounded-full ${barColorMap[m.color] || "bg-gray-500"}`}
                  style={{ width: `${Math.min(m.value_pct, 100)}%` }}
                />
              </div>
            )}
            <p className="text-sm text-gray-700">{m.analogy}</p>
            <p className="text-xs text-gray-500 italic">{m.when_useful}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ======================== Quiz Tab ========================

function QuizTab({ questions }: { questions: any[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  if (!questions.length) return null;

  const question = questions[currentQ];
  const isCorrect = selectedAnswer === question.correct_answer;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelectedAnswer(idx);
    setAnswered(true);
    const correct = idx === question.correct_answer;
    if (correct) setScore((s) => s + 1);
    setAnswers((a) => [...a, idx]);
  };

  const handleNext = () => {
    if (currentQ + 1 >= questions.length) {
      setShowResults(true);
    } else {
      setCurrentQ((q) => q + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    }
  };

  const handleRetry = () => {
    setCurrentQ(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setScore(0);
    setAnswers([]);
    setShowResults(false);
  };

  if (showResults) {
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          <Trophy
            className={`w-16 h-16 mx-auto mb-4 ${
              pct >= 80 ? "text-teal-500" : pct >= 50 ? "text-yellow-500" : "text-red-500"
            }`}
          />
          <h3 className="text-2xl font-bold text-gray-900">
            {score} / {questions.length}
          </h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80
              ? "Excellent! You understand random forests well!"
              : pct >= 50
              ? "Good job! Review the topics you missed."
              : "Keep learning! Ensemble methods have many concepts to master."}
          </p>
          <button
            onClick={handleRetry}
            className="mt-4 px-6 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors text-sm font-medium"
          >
            Try Again
          </button>
        </div>

        {/* Review */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700">Review</h4>
          {questions.map((q: any, i: number) => {
            const userAns = answers[i];
            const correct = userAns === q.correct_answer;
            return (
              <div key={i} className={`rounded-lg border p-3 ${correct ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}>
                <div className="flex items-start gap-2">
                  {correct ? (
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{q.question}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      Your answer: {q.options[userAns ?? 0]} {!correct && `| Correct: ${q.options[q.correct_answer]}`}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">{q.explanation}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />
        <p className="text-sm text-teal-800">
          <span className="font-semibold">Test your knowledge</span> about Random Forests!
          Answer {questions.length} questions to check your understanding.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex items-center justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-2.5 h-2.5 rounded-full transition-all ${
              i === currentQ
                ? "bg-teal-500 scale-125"
                : i < answers.length
                ? answers[i] === questions[i].correct_answer
                  ? "bg-green-400"
                  : "bg-red-400"
                : "bg-gray-300"
            }`}
          />
        ))}
      </div>

      {/* Question */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <p className="text-xs text-gray-400 mb-2">
          Question {currentQ + 1} of {questions.length}
        </p>
        <p className="text-base font-medium text-gray-900 mb-4">{question.question}</p>

        <div className="space-y-2">
          {question.options.map((opt: string, idx: number) => {
            let style = "border-gray-200 bg-white hover:bg-gray-50 text-gray-700";
            if (answered) {
              if (idx === question.correct_answer) {
                style = "border-green-300 bg-green-50 text-green-800";
              } else if (idx === selectedAnswer && !isCorrect) {
                style = "border-red-300 bg-red-50 text-red-800";
              } else {
                style = "border-gray-200 bg-gray-50 text-gray-400";
              }
            } else if (idx === selectedAnswer) {
              style = "border-teal-400 bg-teal-50 text-teal-700";
            }

            return (
              <button
                key={idx}
                onClick={() => handleSelect(idx)}
                disabled={answered}
                className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-colors ${style} disabled:cursor-default`}
              >
                <span className="font-medium mr-2">
                  {String.fromCharCode(65 + idx)}.
                </span>
                {opt}
              </button>
            );
          })}
        </div>

        {answered && (
          <div className={`mt-4 p-3 rounded-lg text-sm ${isCorrect ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"}`}>
            <p className="font-semibold mb-1">{isCorrect ? "Correct!" : "Not quite."}</p>
            <p>{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors text-sm font-medium"
          >
            {currentQ + 1 >= questions.length ? "See Results" : "Next Question"}
          </button>
        )}
      </div>
    </div>
  );
}

// ======================== How It Works Tab ========================

function HowItWorksTab({ nEstimators }: { nEstimators?: number }) {
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />
        <div className="text-sm text-teal-800 space-y-2">
          <p className="font-semibold">How Random Forest Works</p>
          <ol className="list-decimal list-inside space-y-1">
            <li><strong>Bootstrap sampling:</strong> Create {nEstimators || "N"} random subsets of the training data (with replacement)</li>
            <li><strong>Train trees:</strong> Train one decision tree on each bootstrap sample</li>
            <li><strong>Random feature selection:</strong> At each split, only consider a random subset of features</li>
            <li><strong>Independent predictions:</strong> Each tree makes its own prediction independently</li>
            <li><strong>Aggregate:</strong> Combine predictions via majority voting (classification) or averaging (regression)</li>
          </ol>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Trees className="w-4 h-4 text-teal-600" />
            Ensemble Power
          </h4>
          <p className="text-sm text-gray-600">
            By combining many diverse trees, individual errors cancel out.
            The ensemble is more accurate and stable than any single tree.
            This is called the "wisdom of the crowd" effect.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Layers className="w-4 h-4 text-blue-600" />
            Diversity
          </h4>
          <p className="text-sm text-gray-600">
            Two sources of randomness ensure diverse trees: bootstrap sampling
            (different data) and random feature selection at each split
            (different perspectives).
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Target className="w-4 h-4 text-orange-600" />
            Robustness
          </h4>
          <p className="text-sm text-gray-600">
            Random Forests resist overfitting better than single trees, handle
            missing values and outliers well, and provide reliable feature
            importance estimates.
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Key Hyperparameters</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">n_estimators:</span>
            <span className="text-gray-600">Number of trees in the forest. More trees = better but slower.</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">max_depth:</span>
            <span className="text-gray-600">Maximum depth of each tree. Limits individual tree complexity.</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">min_samples_split:</span>
            <span className="text-gray-600">Minimum samples required to split a node.</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold text-teal-700 shrink-0">max_features:</span>
            <span className="text-gray-600">Number of features considered at each split (source of randomness).</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default RandomForestExplorer;
