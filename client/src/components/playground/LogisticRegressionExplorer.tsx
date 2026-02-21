/**
 * LogisticRegressionExplorer - Interactive learning activities for Logistic Regression
 * Tabbed component: Results | Class Distribution | Metric Explainer | Sigmoid Explorer | Quiz
 */

import { useState, lazy, Suspense, type ReactNode } from "react";
import {
  ClipboardList,
  BarChart3,
  Gauge,
  TrendingUp,
  HelpCircle,
  CheckCircle,
  XCircle,
  Trophy,
  ChevronRight,
  Info,
  Cog,
} from "lucide-react";

const LogisticRegressionAnimation = lazy(() => import("./animations/LogisticRegressionAnimation"));

interface LogisticRegressionExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type ExplorerTab =
  | "results"
  | "distribution"
  | "metrics"
  | "sigmoid"
  | "quiz"
  | "how_it_works";

export const LogisticRegressionExplorer = ({
  result,
  renderResults,
}: LogisticRegressionExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "distribution",
      label: "Class Distribution",
      icon: BarChart3,
      available: !!result.class_distribution?.classes?.length,
    },
    {
      id: "metrics",
      label: "Metric Explainer",
      icon: Gauge,
      available: !!result.metric_explainer?.metrics?.length,
    },
    {
      id: "sigmoid",
      label: "Sigmoid Explorer",
      icon: TrendingUp,
      available: !!result.sigmoid_data,
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
                    ? "border-violet-600 text-violet-600"
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
      {activeTab === "results" && renderResults()}
      {activeTab === "distribution" && result.class_distribution && (
        <ClassDistributionTab data={result.class_distribution} />
      )}
      {activeTab === "metrics" && result.metric_explainer && (
        <MetricExplainerTab data={result.metric_explainer} />
      )}
      {activeTab === "sigmoid" && result.sigmoid_data && (
        <SigmoidTab data={result.sigmoid_data} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
      {activeTab === "how_it_works" && (
        <Suspense
          fallback={
            <div className="flex items-center justify-center py-12">
              <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            </div>
          }
        >
          <LogisticRegressionAnimation />
        </Suspense>
      )}
    </div>
  );
};

// --- Class Distribution Tab ---

function ClassDistributionTab({ data }: { data: any }) {
  const classes = data.classes || [];

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-violet-900">
          Training Data Class Distribution
        </h3>
        <p className="text-xs text-violet-700 mt-1">
          See how many samples belong to each class in your training data. Class
          balance affects how well the model learns each class.
        </p>
      </div>

      {/* Balance status */}
      <div
        className={`border rounded-lg p-4 ${
          data.is_balanced
            ? "bg-green-50 border-green-200"
            : "bg-yellow-50 border-yellow-200"
        }`}
      >
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4" />
          <span className="text-sm font-medium">{data.balance_message}</span>
        </div>
        {!data.is_balanced && (
          <p className="text-xs text-gray-600 mt-2">
            Imbalance ratio: {data.imbalance_ratio}:1
          </p>
        )}
      </div>

      {/* Bar chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-gray-900">
            {data.target_column} — {data.n_classes} classes
          </h4>
          <span className="text-xs text-gray-500">
            {data.total_samples?.toLocaleString()} total samples
          </span>
        </div>

        <div className="space-y-3">
          {classes.map((cls: any, idx: number) => {
            const colors = [
              "bg-violet-500",
              "bg-indigo-500",
              "bg-fuchsia-500",
              "bg-purple-500",
              "bg-pink-500",
              "bg-blue-500",
              "bg-cyan-500",
              "bg-teal-500",
            ];
            const bgColor = colors[idx % colors.length];

            return (
              <div key={cls.name}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-gray-800">
                    {cls.name}
                  </span>
                  <span className="text-xs text-gray-600">
                    {cls.count.toLocaleString()} ({cls.percentage}%)
                  </span>
                </div>
                <div className="bg-gray-100 rounded h-7 overflow-hidden">
                  <div
                    className={`${bgColor} h-full rounded flex items-center pl-2 transition-all`}
                    style={{ width: `${cls.bar_width_pct}%` }}
                  >
                    {cls.bar_width_pct > 15 && (
                      <span className="text-white text-xs font-medium">
                        {cls.percentage}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// --- Metric Explainer Tab ---

function MetricExplainerTab({ data }: { data: any }) {
  const metrics = data.metrics || [];

  const colorMap: Record<string, { bg: string; border: string; text: string }> = {
    blue: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-700" },
    green: { bg: "bg-green-50", border: "border-green-200", text: "text-green-700" },
    orange: { bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700" },
    purple: { bg: "bg-purple-50", border: "border-purple-200", text: "text-purple-700" },
  };

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-violet-900">
          Understanding Your Metrics
        </h3>
        <p className="text-xs text-violet-700 mt-1">
          Each metric measures a different aspect of model performance. Click on
          each card to learn what it means using your actual results.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {metrics.map((m: any) => {
          const colors = colorMap[m.color] || colorMap.blue;
          return (
            <div
              key={m.metric}
              className={`${colors.bg} ${colors.border} border rounded-lg p-4`}
            >
              {/* Metric value */}
              <div className="flex items-center justify-between mb-3">
                <h4 className={`text-sm font-bold ${colors.text}`}>
                  {m.metric}
                </h4>
                <div className="text-right">
                  <span className="text-2xl font-bold text-gray-900">
                    {m.value_pct}%
                  </span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="bg-white rounded-full h-3 mb-3 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    m.value_pct >= 80
                      ? "bg-green-500"
                      : m.value_pct >= 60
                        ? "bg-yellow-500"
                        : "bg-red-500"
                  }`}
                  style={{ width: `${m.value_pct}%` }}
                />
              </div>

              {/* Analogy */}
              <p className="text-xs text-gray-700 mb-2">{m.analogy}</p>

              {/* When useful */}
              <div className="bg-white bg-opacity-60 rounded p-2">
                <p className="text-xs text-gray-600">
                  <span className="font-semibold">When to use:</span>{" "}
                  {m.when_useful}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Sigmoid Explorer Tab ---

function SigmoidTab({ data }: { data: any }) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const xValues = data.x_values || [];
  const yValues = data.y_values || [];
  const annotations = data.annotations || [];

  // SVG dimensions
  const width = 600;
  const height = 300;
  const padding = { top: 20, right: 30, bottom: 40, left: 50 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const xMin = xValues[0] ?? -8;
  const xMax = xValues[xValues.length - 1] ?? 8;

  const scaleX = (x: number) =>
    padding.left + ((x - xMin) / (xMax - xMin)) * plotW;
  const scaleY = (y: number) =>
    padding.top + plotH - y * plotH;

  // Build SVG path
  const pathData = xValues
    .map((x: number, i: number) => {
      const sx = scaleX(x);
      const sy = scaleY(yValues[i]);
      return `${i === 0 ? "M" : "L"}${sx},${sy}`;
    })
    .join(" ");

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-violet-900">
          The Sigmoid Function
        </h3>
        <p className="text-xs text-violet-700 mt-1">{data.description}</p>
      </div>

      {/* Formula */}
      <div className="bg-gray-900 rounded-lg p-4 text-center">
        <code className="text-green-400 text-lg">{data.formula}</code>
      </div>

      {/* SVG Chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full"
          style={{ maxHeight: "300px" }}
        >
          {/* Grid lines */}
          <line
            x1={scaleX(0)}
            y1={padding.top}
            x2={scaleX(0)}
            y2={padding.top + plotH}
            stroke="#e5e7eb"
            strokeWidth="1"
          />
          <line
            x1={padding.left}
            y1={scaleY(0.5)}
            x2={padding.left + plotW}
            y2={scaleY(0.5)}
            stroke="#e5e7eb"
            strokeDasharray="4,4"
            strokeWidth="1"
          />
          <line
            x1={padding.left}
            y1={scaleY(0)}
            x2={padding.left + plotW}
            y2={scaleY(0)}
            stroke="#d1d5db"
            strokeWidth="1"
          />
          <line
            x1={padding.left}
            y1={scaleY(1)}
            x2={padding.left + plotW}
            y2={scaleY(1)}
            stroke="#d1d5db"
            strokeWidth="1"
          />

          {/* Sigmoid curve */}
          <path
            d={pathData}
            fill="none"
            stroke="#7c3aed"
            strokeWidth="3"
            strokeLinecap="round"
          />

          {/* Decision boundary region shading */}
          <rect
            x={padding.left}
            y={scaleY(1)}
            width={scaleX(0) - padding.left}
            height={plotH}
            fill="#ef4444"
            opacity="0.05"
          />
          <rect
            x={scaleX(0)}
            y={scaleY(1)}
            width={padding.left + plotW - scaleX(0)}
            height={plotH}
            fill="#22c55e"
            opacity="0.05"
          />

          {/* Threshold line at 0.5 */}
          <text
            x={padding.left - 5}
            y={scaleY(0.5) + 4}
            textAnchor="end"
            className="text-xs fill-gray-500"
            fontSize="10"
          >
            0.5
          </text>

          {/* Annotation dots */}
          {annotations.map((ann: any, i: number) => (
            <g key={i}>
              <circle
                cx={scaleX(ann.x)}
                cy={scaleY(ann.y)}
                r="5"
                fill="#7c3aed"
                stroke="white"
                strokeWidth="2"
              />
              <text
                x={scaleX(ann.x)}
                y={scaleY(ann.y) - 10}
                textAnchor="middle"
                className="text-xs fill-gray-700"
                fontSize="9"
              >
                {ann.label}
              </text>
            </g>
          ))}

          {/* Axis labels */}
          <text
            x={padding.left + plotW / 2}
            y={height - 5}
            textAnchor="middle"
            className="text-xs fill-gray-600"
            fontSize="11"
          >
            {data.x_label}
          </text>
          <text
            x={12}
            y={padding.top + plotH / 2}
            textAnchor="middle"
            transform={`rotate(-90, 12, ${padding.top + plotH / 2})`}
            className="text-xs fill-gray-600"
            fontSize="11"
          >
            {data.y_label}
          </text>

          {/* Y-axis labels */}
          <text
            x={padding.left - 5}
            y={scaleY(0) + 4}
            textAnchor="end"
            className="text-xs fill-gray-500"
            fontSize="10"
          >
            0
          </text>
          <text
            x={padding.left - 5}
            y={scaleY(1) + 4}
            textAnchor="end"
            className="text-xs fill-gray-500"
            fontSize="10"
          >
            1
          </text>

          {/* X-axis labels */}
          {[-8, -4, 0, 4, 8].map((x) => (
            <text
              key={x}
              x={scaleX(x)}
              y={padding.top + plotH + 15}
              textAnchor="middle"
              className="text-xs fill-gray-500"
              fontSize="10"
            >
              {x}
            </text>
          ))}

          {/* Interactive hover zone */}
          {xValues.map((x: number, i: number) => (
            <rect
              key={i}
              x={scaleX(x) - 3}
              y={padding.top}
              width={6}
              height={plotH}
              fill="transparent"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
              className="cursor-crosshair"
            />
          ))}

          {/* Hover tooltip */}
          {hoverIdx !== null && (
            <g>
              <circle
                cx={scaleX(xValues[hoverIdx])}
                cy={scaleY(yValues[hoverIdx])}
                r="4"
                fill="#7c3aed"
              />
              <rect
                x={scaleX(xValues[hoverIdx]) + 8}
                y={scaleY(yValues[hoverIdx]) - 20}
                width="110"
                height="30"
                rx="4"
                fill="white"
                stroke="#d1d5db"
              />
              <text
                x={scaleX(xValues[hoverIdx]) + 14}
                y={scaleY(yValues[hoverIdx]) - 4}
                className="text-xs fill-gray-800"
                fontSize="10"
              >
                z={xValues[hoverIdx].toFixed(1)}, P=
                {yValues[hoverIdx].toFixed(3)}
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Threshold explanation */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-2">
          How Predictions Work
        </h4>
        <p className="text-xs text-gray-600">{data.threshold_explanation}</p>
        <div className="flex gap-4 mt-3">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-100 rounded border border-red-200" />
            <span className="text-xs text-gray-600">
              P &lt; 0.5 → Class 0
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-100 rounded border border-green-200" />
            <span className="text-xs text-gray-600">
              P &ge; 0.5 → Class 1
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Quiz Tab ---

function QuizTab({ questions }: { questions: any[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>(
    new Array(questions.length).fill(null)
  );
  const [showResults, setShowResults] = useState(false);

  const question = questions[currentQ];

  const handleSelect = (optionIdx: number) => {
    if (answered) return;
    setSelectedAnswer(optionIdx);
    setAnswered(true);
    const newAnswers = [...answers];
    newAnswers[currentQ] = optionIdx;
    setAnswers(newAnswers);
    if (optionIdx === question.correct_answer) {
      setScore((s) => s + 1);
    }
  };

  const handleNext = () => {
    if (currentQ < questions.length - 1) {
      setCurrentQ((q) => q + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    } else {
      setShowResults(true);
    }
  };

  const handleRetry = () => {
    setCurrentQ(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setScore(0);
    setAnswers(new Array(questions.length).fill(null));
    setShowResults(false);
  };

  if (showResults) {
    const percentage = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-4">
        <div
          className={`text-center py-8 rounded-lg border-2 ${
            percentage >= 80
              ? "bg-green-50 border-green-300"
              : percentage >= 50
                ? "bg-yellow-50 border-yellow-300"
                : "bg-red-50 border-red-300"
          }`}
        >
          <Trophy
            className={`w-12 h-12 mx-auto mb-3 ${
              percentage >= 80
                ? "text-green-600"
                : percentage >= 50
                  ? "text-yellow-600"
                  : "text-red-600"
            }`}
          />
          <div className="text-3xl font-bold text-gray-900 mb-2">
            {score}/{questions.length}
          </div>
          <p className="text-sm text-gray-700">
            {percentage >= 80
              ? "Excellent understanding of logistic regression!"
              : percentage >= 50
                ? "Good effort! Review the explanations below."
                : "Keep learning! Try again after reviewing."}
          </p>
        </div>
        <button
          onClick={handleRetry}
          className="w-full py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 text-sm font-medium"
        >
          Try Again
        </button>
        <div className="space-y-3">
          {questions.map((q: any, qi: number) => (
            <div
              key={qi}
              className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm"
            >
              <p className="font-medium text-gray-800 mb-1">{q.question}</p>
              <p
                className={`text-xs ${
                  answers[qi] === q.correct_answer
                    ? "text-green-700"
                    : "text-red-700"
                }`}
              >
                Your answer: {q.options[answers[qi] ?? 0]}{" "}
                {answers[qi] === q.correct_answer
                  ? "(Correct)"
                  : "(Incorrect)"}
              </p>
              <p className="text-xs text-blue-700 mt-1">{q.explanation}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-violet-900">
          Knowledge Check
        </h3>
        <p className="text-xs text-violet-700 mt-1">
          Test your understanding of logistic regression using questions from
          YOUR model.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-3 h-3 rounded-full transition-all ${
              i === currentQ
                ? "bg-violet-600 scale-125"
                : i < currentQ
                  ? answers[i] === questions[i].correct_answer
                    ? "bg-green-500"
                    : "bg-red-400"
                  : "bg-gray-300"
            }`}
          />
        ))}
      </div>

      {/* Question card */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="text-xs text-gray-500 mb-2">
          Question {currentQ + 1} of {questions.length}
        </div>
        <p className="text-sm font-medium text-gray-900 mb-4">
          {question.question}
        </p>

        <div className="space-y-2">
          {question.options.map((option: string, oi: number) => {
            let optionClass =
              "bg-white border-gray-300 hover:border-violet-400 cursor-pointer";
            if (answered) {
              if (oi === question.correct_answer) {
                optionClass = "bg-green-50 border-green-500 text-green-900";
              } else if (
                oi === selectedAnswer &&
                oi !== question.correct_answer
              ) {
                optionClass = "bg-red-50 border-red-500 text-red-900";
              } else {
                optionClass =
                  "bg-gray-50 border-gray-200 text-gray-400 cursor-default";
              }
            }
            return (
              <button
                key={oi}
                onClick={() => handleSelect(oi)}
                disabled={answered}
                className={`w-full text-left p-3 rounded-lg border-2 text-sm transition-colors ${optionClass}`}
              >
                <span className="font-mono text-xs text-gray-400 mr-2">
                  {String.fromCharCode(65 + oi)}.
                </span>
                {option}
                {answered && oi === question.correct_answer && (
                  <CheckCircle className="w-4 h-4 text-green-600 inline ml-2" />
                )}
                {answered &&
                  oi === selectedAnswer &&
                  oi !== question.correct_answer && (
                    <XCircle className="w-4 h-4 text-red-600 inline ml-2" />
                  )}
              </button>
            );
          })}
        </div>

        {answered && (
          <div className="mt-4 p-3 bg-violet-50 border border-violet-200 rounded-lg">
            <p className="text-xs text-violet-800">{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 w-full py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 text-sm font-medium flex items-center justify-center gap-2"
          >
            {currentQ < questions.length - 1 ? (
              <>
                Next Question <ChevronRight className="w-4 h-4" />
              </>
            ) : (
              "See Results"
            )}
          </button>
        )}
      </div>
    </div>
  );
}
