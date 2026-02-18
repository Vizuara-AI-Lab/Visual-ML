/**
 * MissingValueExplorer - Interactive learning activities for Missing Value Handler
 * Tabbed component: Results | Health Score | Strategy Lab | Heatmap | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  Heart,
  FlaskConical,
  Grid3X3,
  HelpCircle,
  CheckCircle,
  XCircle,
  Trophy,
  Info,
  ChevronRight,
  ClipboardList,
} from "lucide-react";

interface MissingValueExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type ExplorerTab = "results" | "health" | "strategy" | "heatmap" | "quiz";

export const MissingValueExplorer = ({
  result,
  renderResults,
}: MissingValueExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    { id: "health", label: "Health Score", icon: Heart, available: true },
    {
      id: "strategy",
      label: "Strategy Lab",
      icon: FlaskConical,
      available:
        !!result.strategy_comparison &&
        Object.keys(result.strategy_comparison).length > 0,
    },
    {
      id: "heatmap",
      label: "Heatmap",
      icon: Grid3X3,
      available: !!result.missing_heatmap,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available:
        !!result.quiz_questions && result.quiz_questions.length > 0,
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
                    ? "border-blue-600 text-blue-600"
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
      {activeTab === "health" && (
        <HealthScoreTab
          beforeStats={result.before_stats}
          afterStats={result.after_stats}
          operationLog={result.operation_log}
        />
      )}
      {activeTab === "strategy" && result.strategy_comparison && (
        <StrategyLabTab data={result.strategy_comparison} />
      )}
      {activeTab === "heatmap" && result.missing_heatmap && (
        <HeatmapTab data={result.missing_heatmap} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Health Score Tab ---

function HealthScoreTab({
  beforeStats,
  afterStats,
  operationLog,
}: {
  beforeStats: any;
  afterStats: any;
  operationLog: any;
}) {
  // Compute health as percentage of complete cells, but floor so it never
  // rounds *up* to 100% when missing values still exist.
  const computeHealth = (stats: any): number => {
    const totalCells =
      (stats.total_rows || 1) * (stats.total_columns || 1);
    const totalMissing = stats.total_missing_values || 0;
    if (totalMissing === 0) return 100;
    // Floor to 1 decimal place so 99.98% → 99.9, never rounds to 100
    return Math.floor((1 - totalMissing / totalCells) * 1000) / 10;
  };

  const beforeHealth = computeHealth(beforeStats);
  const afterHealth = computeHealth(afterStats);
  const beforeMissingCount = beforeStats.total_missing_values || 0;
  const afterMissingCount = afterStats.total_missing_values || 0;

  const getColor = (score: number) => {
    if (score >= 80)
      return {
        bg: "bg-green-500",
        text: "text-green-600",
        stroke: "#22c55e",
      };
    if (score >= 50)
      return {
        bg: "bg-orange-500",
        text: "text-orange-600",
        stroke: "#f59e0b",
      };
    return { bg: "bg-red-500", text: "text-red-600", stroke: "#ef4444" };
  };

  // Determine badges — use actual missing count, not rounded health
  const badges: { label: string }[] = [];
  if (afterMissingCount === 0) {
    badges.push({ label: "Clean Dataset!" });
    badges.push({ label: "Zero Missing!" });
  }
  const strategiesUsed = new Set(
    Object.values(operationLog || {})
      .map((op: any) => op.strategy)
      .filter(Boolean)
  );
  if (strategiesUsed.size >= 3) badges.push({ label: "Strategy Master" });
  if (strategiesUsed.size >= 2) badges.push({ label: "Multi-Strategy" });

  const beforeMissing = beforeStats.missing_by_column || {};
  const afterMissing = afterStats.missing_by_column || {};
  const allCols = new Set([
    ...Object.keys(beforeMissing),
    ...Object.keys(afterMissing),
  ]);

  const renderRing = (score: number, label: string, missingCount: number) => {
    const radius = 55;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;
    const colors = getColor(score);
    // Show 1 decimal when < 100, integer when exactly 100
    const displayScore =
      score === 100 ? "100" : score.toFixed(1);
    return (
      <div className="flex flex-col items-center relative">
        <svg width="140" height="140" className="transform -rotate-90">
          <circle
            cx="70"
            cy="70"
            r={radius}
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="10"
          />
          <circle
            cx="70"
            cy="70"
            r={radius}
            fill="none"
            stroke={colors.stroke}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.8s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className={`text-2xl font-bold ${colors.text}`}>
            {displayScore}%
          </div>
          <div className="text-xs text-gray-500">{label}</div>
          <div className="text-[10px] text-gray-400 mt-0.5">
            {missingCount === 0
              ? "0 missing"
              : `${missingCount.toLocaleString()} missing`}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Score comparison */}
      <div className="flex items-center justify-center gap-8 py-4">
        {renderRing(beforeHealth, "Before", beforeMissingCount)}
        <div className="text-3xl text-gray-300 font-light">&rarr;</div>
        {renderRing(afterHealth, "After", afterMissingCount)}
      </div>

      {/* Summary stats */}
      {beforeMissingCount > 0 && (
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <div className="text-lg font-bold text-red-700">
              {beforeMissingCount.toLocaleString()}
            </div>
            <div className="text-[10px] text-red-600 uppercase tracking-wide font-medium">
              Missing Before
            </div>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="text-lg font-bold text-green-700">
              {afterMissingCount.toLocaleString()}
            </div>
            <div className="text-[10px] text-green-600 uppercase tracking-wide font-medium">
              Missing After
            </div>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="text-lg font-bold text-blue-700">
              {(beforeMissingCount - afterMissingCount).toLocaleString()}
            </div>
            <div className="text-[10px] text-blue-600 uppercase tracking-wide font-medium">
              Values Fixed
            </div>
          </div>
        </div>
      )}

      {/* Badges */}
      {badges.length > 0 && (
        <div className="flex justify-center gap-3 flex-wrap">
          {badges.map((badge, i) => (
            <div
              key={i}
              className="flex items-center gap-2 px-4 py-2 bg-yellow-50 border border-yellow-300 rounded-full"
            >
              <Trophy className="w-4 h-4 text-yellow-600" />
              <span className="text-sm font-semibold text-yellow-800">
                {badge.label}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Per-column completeness bars */}
      {allCols.size > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-4">
            Column Completeness
          </h4>
          <div className="space-y-3">
            {Array.from(allCols).map((col) => {
              const beforePct = beforeMissing[col]?.percentage || 0;
              const afterPct = afterMissing[col]?.percentage || 0;
              const beforeComplete = 100 - beforePct;
              const afterComplete = 100 - afterPct;
              const beforeCount = beforeMissing[col]?.count || 0;
              const afterCount = afterMissing[col]?.count || 0;
              return (
                <div key={col} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-medium text-gray-700">
                      {col}
                      <span className="text-gray-400 ml-1 font-normal">
                        ({beforeCount} &rarr; {afterCount} missing)
                      </span>
                    </span>
                    <span className="text-gray-500">
                      {beforeComplete.toFixed(1)}% &rarr;{" "}
                      {afterComplete.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex gap-1 h-3">
                    <div className="flex-1 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${getColor(beforeComplete).bg} opacity-50`}
                        style={{ width: `${beforeComplete}%` }}
                      />
                    </div>
                    <div className="flex-1 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${getColor(afterComplete).bg}`}
                        style={{ width: `${afterComplete}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-green-500 opacity-50 rounded-sm" />{" "}
              Before
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-green-500 rounded-sm" /> After
            </span>
          </div>
        </div>
      )}

      {/* Educational note */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <div>
            <h4 className="text-xs font-semibold text-blue-900 mb-1">
              What is Data Health?
            </h4>
            <p className="text-xs text-blue-700">
              Data health measures how complete your dataset is. A score of
              100% means every cell has a value. Missing values reduce the
              score. Handling missing values improves your dataset's health and
              makes your ML model more reliable.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Strategy Lab Tab ---

function StrategyLabTab({ data }: { data: Record<string, any> }) {
  const [selectedColumn, setSelectedColumn] = useState<string>(
    Object.keys(data)[0] || ""
  );

  const columns = Object.keys(data);
  const colData = data[selectedColumn];

  if (!colData) {
    return (
      <div className="text-center py-8 text-gray-500">
        No columns with missing values to compare
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Column selector */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900 mb-2">
          Strategy Comparison Lab
        </h3>
        <p className="text-xs text-indigo-700 mb-3">
          Select a column to see what each strategy would do with its{" "}
          {colData.missing_count} missing values.
        </p>
        <div className="flex flex-wrap gap-2">
          {columns.map((col) => (
            <button
              key={col}
              onClick={() => setSelectedColumn(col)}
              className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
                selectedColumn === col
                  ? "bg-indigo-600 text-white border-indigo-600"
                  : "bg-white text-gray-700 border-gray-300 hover:border-indigo-400"
              }`}
            >
              {col} ({data[col].missing_count} missing)
            </button>
          ))}
        </div>
      </div>

      {/* Sample values display */}
      <div className="bg-white border border-gray-200 rounded-lg p-3">
        <h4 className="text-xs font-semibold text-gray-700 mb-2">
          Sample Values
        </h4>
        <div className="flex flex-wrap gap-2">
          {colData.sample_values.map((val: any, i: number) => (
            <span
              key={i}
              className={`px-3 py-1 rounded text-sm font-mono ${
                val === null
                  ? "bg-red-100 text-red-700 border border-red-300"
                  : "bg-gray-100 text-gray-800 border border-gray-200"
              }`}
            >
              {val === null ? "NULL" : String(val)}
            </span>
          ))}
        </div>
      </div>

      {/* Strategy cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {Object.entries(colData.strategies).map(
          ([strategyName, strategyData]: [string, any]) => {
            const isApplied = colData.applied_strategy === strategyName;
            return (
              <div
                key={strategyName}
                className={`bg-white border-2 rounded-lg p-4 ${
                  isApplied
                    ? "border-green-500 shadow-md"
                    : "border-gray-200"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-sm font-semibold text-gray-900 capitalize">
                    {strategyName.replace("_", " ")}
                  </h5>
                  {isApplied && (
                    <span className="px-2 py-0.5 text-xs font-semibold bg-green-100 text-green-800 rounded-full border border-green-300">
                      Applied
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-600 mb-3">
                  {strategyData.description}
                </p>

                {strategyData.fill_value !== undefined &&
                  strategyData.fill_value !== null && (
                    <div className="bg-blue-50 rounded p-2 mb-2">
                      <span className="text-xs text-blue-700">
                        Would fill with:{" "}
                      </span>
                      <span className="text-sm font-semibold text-blue-900 font-mono">
                        {String(strategyData.fill_value)}
                      </span>
                    </div>
                  )}
                {strategyData.rows_removed !== undefined && (
                  <div className="bg-orange-50 rounded p-2 mb-2">
                    <span className="text-xs text-orange-700">
                      Would remove:{" "}
                    </span>
                    <span className="text-sm font-semibold text-orange-900">
                      {strategyData.rows_removed} rows
                    </span>
                  </div>
                )}

                {strategyData.when_to_use && (
                  <div className="mt-2 pt-2 border-t border-gray-100">
                    <p className="text-xs text-gray-500 italic">
                      Best when: {strategyData.when_to_use}
                    </p>
                  </div>
                )}
              </div>
            );
          }
        )}
      </div>

      {/* Educational note */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            There's no single "best" strategy — it depends on your data! For
            numbers with outliers, use <strong>Median</strong>. For categories,
            use <strong>Mode</strong>. If very few rows are affected,{" "}
            <strong>Drop</strong> is simplest. The strategy with the{" "}
            <span className="text-green-700 font-semibold">green border</span>{" "}
            is what was actually applied in your pipeline.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Heatmap Tab ---

function HeatmapTab({ data }: { data: any }) {
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);

  const {
    columns,
    grid,
    rows_sampled,
    total_rows,
    column_missing_pct,
    pattern,
  } = data;

  const cellSize = columns.length > 15 ? 16 : columns.length > 10 ? 20 : 28;

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-orange-900">
          Missing Data Heatmap
        </h3>
        <p className="text-xs text-orange-700 mt-1">
          Each cell shows whether a value is present (green) or missing (red).
          Showing {rows_sampled} sampled rows out of{" "}
          {total_rows.toLocaleString()} total.
        </p>
      </div>

      {/* Pattern detection */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 flex items-start gap-2">
        <Info className="w-4 h-4 text-purple-600 mt-0.5 shrink-0" />
        <p className="text-xs text-purple-800 font-medium">{pattern}</p>
      </div>

      {/* Heatmap grid */}
      <div className="overflow-auto bg-white border border-gray-200 rounded-lg p-4">
        {/* Column headers with missing % */}
        <div className="flex" style={{ paddingLeft: "40px" }}>
          {columns.map((col: string, ci: number) => (
            <div
              key={ci}
              className="text-center"
              style={{ width: cellSize, minWidth: cellSize }}
              title={`${col}: ${column_missing_pct[col]}% missing`}
            >
              <div
                className="text-[9px] text-gray-500 truncate origin-bottom-left whitespace-nowrap overflow-hidden"
                style={{ maxWidth: cellSize }}
              >
                {col.length > 4 ? col.substring(0, 3) + ".." : col}
              </div>
              <div
                className={`text-[8px] font-mono ${
                  column_missing_pct[col] > 5
                    ? "text-red-600 font-semibold"
                    : "text-gray-400"
                }`}
              >
                {column_missing_pct[col]}%
              </div>
            </div>
          ))}
        </div>

        {/* Grid rows */}
        <div className="mt-1">
          {grid.map((row: boolean[], ri: number) => (
            <div key={ri} className="flex items-center">
              <div className="text-[8px] text-gray-400 w-[40px] text-right pr-1 shrink-0">
                {data.row_indices?.[ri] ?? ri}
              </div>
              {row.map((isMissing: boolean, ci: number) => (
                <div
                  key={ci}
                  className={`transition-opacity ${
                    isMissing ? "bg-red-500" : "bg-green-400"
                  } ${
                    hoveredCell?.row === ri && hoveredCell?.col === ci
                      ? "opacity-70 ring-1 ring-gray-800"
                      : ""
                  }`}
                  style={{
                    width: cellSize,
                    height: Math.max(cellSize / 2.5, 6),
                    minWidth: cellSize,
                    border: "0.5px solid rgba(255,255,255,0.3)",
                  }}
                  onMouseEnter={() => setHoveredCell({ row: ri, col: ci })}
                  onMouseLeave={() => setHoveredCell(null)}
                  title={`Row ${data.row_indices?.[ri] ?? ri}, ${columns[ci]}: ${isMissing ? "MISSING" : "Present"}`}
                />
              ))}
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-3 pt-2 border-t border-gray-200 text-xs text-gray-600">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 bg-green-400 rounded-sm" /> Value
            Present
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 bg-red-500 rounded-sm" /> Missing Value
          </span>
        </div>
      </div>

      {/* Educational note */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Understanding <strong>where</strong> missing values appear helps
            you choose the right strategy. If values are missing randomly, any
            strategy works well. If they form a pattern (like all missing in
            one column), that column might need a specific approach.
          </p>
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
              ? "Excellent understanding!"
              : percentage >= 50
                ? "Good effort! Review the explanations below."
                : "Keep learning! Try again after reviewing."}
          </p>
        </div>
        <button
          onClick={handleRetry}
          className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
        >
          Try Again
        </button>
        {/* Review all questions */}
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
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">
          Knowledge Check
        </h3>
        <p className="text-xs text-indigo-700 mt-1">
          Test your understanding of missing value handling strategies using
          questions from YOUR dataset.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-3 h-3 rounded-full transition-all ${
              i === currentQ
                ? "bg-blue-600 scale-125"
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
              "bg-white border-gray-300 hover:border-blue-400 cursor-pointer";
            if (answered) {
              if (oi === question.correct_answer) {
                optionClass =
                  "bg-green-50 border-green-500 text-green-900";
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

        {/* Explanation */}
        {answered && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-800">{question.explanation}</p>
          </div>
        )}

        {/* Next button */}
        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium flex items-center justify-center gap-2"
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
