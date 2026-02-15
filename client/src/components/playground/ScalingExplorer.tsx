/**
 * ScalingExplorer - Interactive learning activities for Scaling node
 * Tabs: Results | Before & After | Method Comparison | Outlier Impact | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  ArrowLeftRight,
  BarChart3,
  AlertTriangle,
  HelpCircle,
  CheckCircle,
  XCircle,
  Trophy,
  Info,
  ChevronRight,
  ClipboardList,
} from "lucide-react";

interface ScalingExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type ExplorerTab = "results" | "before_after" | "comparison" | "outliers" | "quiz";

export const ScalingExplorer = ({ result, renderResults }: ScalingExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: { id: ExplorerTab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "before_after",
      label: "Before & After",
      icon: ArrowLeftRight,
      available: !!result.scaling_before_after?.columns && Object.keys(result.scaling_before_after.columns).length > 0,
    },
    {
      id: "comparison",
      label: "Method Comparison",
      icon: BarChart3,
      available: !!result.scaling_method_comparison && Object.keys(result.scaling_method_comparison).length > 0,
    },
    {
      id: "outliers",
      label: "Outlier Impact",
      icon: AlertTriangle,
      available: !!result.scaling_outlier_analysis && Object.keys(result.scaling_outlier_analysis).length > 0,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available: !!result.quiz_questions && result.quiz_questions.length > 0,
    },
  ];

  return (
    <div className="space-y-4">
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
                    ? "border-teal-600 text-teal-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
      </div>

      {activeTab === "results" && renderResults()}
      {activeTab === "before_after" && result.scaling_before_after && (
        <BeforeAfterTab data={result.scaling_before_after} />
      )}
      {activeTab === "comparison" && result.scaling_method_comparison && (
        <MethodComparisonTab data={result.scaling_method_comparison} appliedMethod={result.scaling_method} />
      )}
      {activeTab === "outliers" && result.scaling_outlier_analysis && (
        <OutlierImpactTab data={result.scaling_outlier_analysis} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Before & After Tab ---
function BeforeAfterTab({ data }: { data: any }) {
  const columns = Object.keys(data.columns || {});
  const [selectedColumn, setSelectedColumn] = useState(columns[0] || "");

  const colData = data.columns?.[selectedColumn];

  const StatCard = ({ label, stats, color }: { label: string; stats: any; color: "warm" | "cool" }) => {
    const bg = color === "warm" ? "bg-orange-50 border-orange-200" : "bg-teal-50 border-teal-200";
    const textColor = color === "warm" ? "text-orange-900" : "text-teal-900";
    const labelColor = color === "warm" ? "text-orange-600" : "text-teal-600";

    return (
      <div className={`border rounded-lg p-4 ${bg}`}>
        <h4 className={`text-sm font-semibold ${textColor} mb-3`}>{label}</h4>
        <div className="grid grid-cols-2 gap-2">
          {["mean", "std", "min", "max", "median"].map((stat) => (
            <div key={stat} className="bg-white/70 rounded p-2">
              <div className={`text-[10px] uppercase tracking-wide ${labelColor}`}>{stat}</div>
              <div className="text-sm font-semibold text-gray-900">{stats?.[stat] ?? "—"}</div>
            </div>
          ))}
        </div>
        {stats?.sample_values && stats.sample_values.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-200/50">
            <div className={`text-[10px] uppercase tracking-wide ${labelColor} mb-1`}>Sample Values</div>
            <div className="flex flex-wrap gap-1">
              {stats.sample_values.map((v: number, i: number) => (
                <span key={i} className="px-1.5 py-0.5 text-xs font-mono bg-white/70 rounded">{v}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-teal-900">Before & After Scaling</h3>
        <p className="text-xs text-teal-700 mt-1">
          Compare statistics before and after scaling to see how values change.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {columns.map((col) => (
          <button
            key={col}
            onClick={() => setSelectedColumn(col)}
            className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
              selectedColumn === col
                ? "bg-teal-600 text-white border-teal-600"
                : "bg-white text-gray-700 border-gray-300 hover:border-teal-400"
            }`}
          >
            {col}
          </button>
        ))}
      </div>

      {colData && (
        <>
          {/* Formula box */}
          <div className="bg-gray-900 text-green-400 rounded-lg p-3 text-center font-mono text-sm">
            <span className="text-gray-400">Formula: </span>{colData.formula}
          </div>

          {/* Before / After cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <StatCard label="Before Scaling" stats={colData.before} color="warm" />
            <StatCard label="After Scaling" stats={colData.after} color="cool" />
          </div>

          {/* Sample value arrows */}
          {colData.before?.sample_values && colData.after?.sample_values && (
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="text-xs font-semibold text-gray-700 mb-3">Value Transformation</h4>
              <div className="space-y-2">
                {colData.before.sample_values.map((bv: number, i: number) => {
                  const av = colData.after.sample_values[i];
                  if (av === undefined) return null;
                  return (
                    <div key={i} className="flex items-center gap-3">
                      <span className="px-3 py-1 bg-orange-100 text-orange-800 rounded font-mono text-sm min-w-[80px] text-right">{bv}</span>
                      <ChevronRight className="w-4 h-4 text-gray-400" />
                      <span className="px-3 py-1 bg-teal-100 text-teal-800 rounded font-mono text-sm min-w-[80px]">{av}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Scaling changes the range and distribution of values but preserves the relative ordering.
            After standard scaling, the mean becomes 0 and std becomes 1. After MinMax scaling, values
            fall between 0 and 1.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Method Comparison Tab ---
function MethodComparisonTab({ data, appliedMethod }: { data: any; appliedMethod: string }) {
  const columns = Object.keys(data);
  const [selectedColumn, setSelectedColumn] = useState(columns[0] || "");

  const colData = data[selectedColumn];
  const methods = ["standard", "minmax", "robust", "normalize"] as const;

  const methodColors: Record<string, { bg: string; border: string; text: string }> = {
    standard: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-900" },
    minmax: { bg: "bg-green-50", border: "border-green-200", text: "text-green-900" },
    robust: { bg: "bg-purple-50", border: "border-purple-200", text: "text-purple-900" },
    normalize: { bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-900" },
  };

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-teal-900">Method Comparison</h3>
        <p className="text-xs text-teal-700 mt-1">
          See what each scaling method would do to the same column.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {columns.map((col) => (
          <button
            key={col}
            onClick={() => setSelectedColumn(col)}
            className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
              selectedColumn === col
                ? "bg-teal-600 text-white border-teal-600"
                : "bg-white text-gray-700 border-gray-300 hover:border-teal-400"
            }`}
          >
            {col}
          </button>
        ))}
      </div>

      {colData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {methods.map((m) => {
            const mData = colData[m];
            if (!mData) return null;
            const isApplied = appliedMethod === m;
            const colors = methodColors[m];

            return (
              <div
                key={m}
                className={`border-2 rounded-lg p-4 ${isApplied ? "border-green-500 ring-1 ring-green-300" : colors.border} ${colors.bg}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className={`text-sm font-semibold capitalize ${colors.text}`}>{m}</h4>
                  {isApplied && (
                    <span className="px-2 py-0.5 text-xs font-semibold bg-green-100 text-green-800 rounded-full border border-green-300">Applied</span>
                  )}
                </div>
                <div className="bg-gray-900 text-green-400 rounded p-1.5 text-xs font-mono mb-2 text-center">
                  {mData.formula}
                </div>
                <p className="text-xs text-gray-600 mb-2">{mData.description}</p>

                {mData.sample_values && (
                  <div className="grid grid-cols-2 gap-1.5 mb-2">
                    {["mean", "std", "min", "max"].map((s) =>
                      mData[s] !== undefined ? (
                        <div key={s} className="bg-white/60 rounded p-1.5">
                          <div className="text-[9px] uppercase text-gray-500">{s}</div>
                          <div className="text-xs font-semibold">{mData[s]}</div>
                        </div>
                      ) : null
                    )}
                  </div>
                )}

                {mData.best_for && (
                  <div className="pt-2 border-t border-gray-200/50">
                    <p className="text-[10px] text-gray-500">Best for: {mData.best_for}</p>
                  </div>
                )}

                {mData.note && (
                  <div className="mt-2 px-2 py-1.5 bg-yellow-50 border border-yellow-200 rounded text-[10px] text-yellow-800">
                    {mData.note}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Each method handles data differently. <strong>Standard</strong> centers around 0.{" "}
            <strong>MinMax</strong> squeezes into [0,1]. <strong>Robust</strong> resists outliers.{" "}
            <strong>Normalizer</strong> makes each row have length 1. The method with the{" "}
            <span className="text-green-700 font-semibold">green border</span> is what was applied.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Outlier Impact Tab ---
function OutlierImpactTab({ data }: { data: any }) {
  const columns = Object.keys(data);
  const [selectedColumn, setSelectedColumn] = useState(columns[0] || "");

  const colData = data[selectedColumn];

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-orange-900">Outlier Impact Analysis</h3>
        <p className="text-xs text-orange-700 mt-1">
          See how outliers affect scaling and why Robust scaling can help.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {columns.map((col) => {
          const cd = data[col];
          return (
            <button
              key={col}
              onClick={() => setSelectedColumn(col)}
              className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
                selectedColumn === col
                  ? "bg-orange-600 text-white border-orange-600"
                  : "bg-white text-gray-700 border-gray-300 hover:border-orange-400"
              }`}
            >
              {col}
              {cd?.has_outliers && (
                <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-red-100 text-red-700 rounded-full">
                  {cd.outlier_count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {colData && (
        <>
          {/* Outlier badge */}
          <div className={`rounded-lg p-4 border ${colData.has_outliers ? "bg-red-50 border-red-200" : "bg-green-50 border-green-200"}`}>
            <div className="flex items-center gap-3">
              <AlertTriangle className={`w-5 h-5 ${colData.has_outliers ? "text-red-500" : "text-green-500"}`} />
              <div>
                <p className={`text-sm font-semibold ${colData.has_outliers ? "text-red-900" : "text-green-900"}`}>
                  {colData.has_outliers
                    ? `${colData.outlier_count} outlier${colData.outlier_count > 1 ? "s" : ""} detected out of ${colData.total_values} values`
                    : "No outliers detected"}
                </p>
                {colData.iqr_bounds && (
                  <p className="text-xs text-gray-600 mt-1">
                    IQR range: [{colData.iqr_bounds.lower}, {colData.iqr_bounds.upper}] (Q1={colData.iqr_bounds.q1}, Q3={colData.iqr_bounds.q3})
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Outlier values */}
          {colData.outlier_values && colData.outlier_values.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="text-xs font-semibold text-gray-700 mb-2">Outlier Values</h4>
              <div className="flex flex-wrap gap-2">
                {colData.outlier_values.map((v: number, i: number) => (
                  <span key={i} className="px-3 py-1 bg-red-100 text-red-800 rounded font-mono text-sm border border-red-200">{v}</span>
                ))}
              </div>
            </div>
          )}

          {/* Stats comparison */}
          {colData.stats_with_outliers && colData.stats_without_outliers && (
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="text-xs font-semibold text-gray-700 mb-3">Impact on Statistics</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-red-50 rounded-lg p-3 border border-red-200">
                  <h5 className="text-xs font-semibold text-red-800 mb-2">With Outliers</h5>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Mean:</span>
                      <span className="font-mono font-semibold">{colData.stats_with_outliers.mean}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Std:</span>
                      <span className="font-mono font-semibold">{colData.stats_with_outliers.std}</span>
                    </div>
                  </div>
                </div>
                <div className="bg-green-50 rounded-lg p-3 border border-green-200">
                  <h5 className="text-xs font-semibold text-green-800 mb-2">Without Outliers</h5>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Mean:</span>
                      <span className="font-mono font-semibold">{colData.stats_without_outliers.mean}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Std:</span>
                      <span className="font-mono font-semibold">{colData.stats_without_outliers.std}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Standard vs Robust */}
          {colData.standard_vs_robust && (
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="text-xs font-semibold text-gray-700 mb-3">Standard vs Robust Scaling</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                  <h5 className="text-xs font-semibold text-blue-800 mb-2">Standard Scaler</h5>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between"><span className="text-gray-600">Min:</span><span className="font-mono">{colData.standard_vs_robust.standard.min}</span></div>
                    <div className="flex justify-between"><span className="text-gray-600">Max:</span><span className="font-mono">{colData.standard_vs_robust.standard.max}</span></div>
                    <div className="flex justify-between"><span className="text-gray-600">Range:</span><span className="font-mono font-semibold">{colData.standard_vs_robust.standard.range}</span></div>
                  </div>
                </div>
                <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
                  <h5 className="text-xs font-semibold text-purple-800 mb-2">Robust Scaler</h5>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between"><span className="text-gray-600">Min:</span><span className="font-mono">{colData.standard_vs_robust.robust.min}</span></div>
                    <div className="flex justify-between"><span className="text-gray-600">Max:</span><span className="font-mono">{colData.standard_vs_robust.robust.max}</span></div>
                    <div className="flex justify-between"><span className="text-gray-600">Range:</span><span className="font-mono font-semibold">{colData.standard_vs_robust.robust.range}</span></div>
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-600 mt-2 italic">{colData.standard_vs_robust.message}</p>
            </div>
          )}

          {colData.message && !colData.has_outliers && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-3">
              <p className="text-xs text-green-700">{colData.message}</p>
            </div>
          )}
        </>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Outliers are extreme values that lie far from most data points. They distort the mean and
            standard deviation, which affects Standard Scaling. <strong>Robust Scaling</strong> uses
            median and IQR (Interquartile Range) instead, making it resistant to outlier influence.
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
  const [answers, setAnswers] = useState<(number | null)[]>(new Array(questions.length).fill(null));
  const [showResults, setShowResults] = useState(false);

  const question = questions[currentQ];

  const handleSelect = (optionIdx: number) => {
    if (answered) return;
    setSelectedAnswer(optionIdx);
    setAnswered(true);
    const newAnswers = [...answers];
    newAnswers[currentQ] = optionIdx;
    setAnswers(newAnswers);
    if (optionIdx === question.correct_answer) setScore((s) => s + 1);
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
        <div className={`text-center py-8 rounded-lg border-2 ${percentage >= 80 ? "bg-green-50 border-green-300" : percentage >= 50 ? "bg-yellow-50 border-yellow-300" : "bg-red-50 border-red-300"}`}>
          <Trophy className={`w-12 h-12 mx-auto mb-3 ${percentage >= 80 ? "text-green-600" : percentage >= 50 ? "text-yellow-600" : "text-red-600"}`} />
          <div className="text-3xl font-bold text-gray-900 mb-2">{score}/{questions.length}</div>
          <p className="text-sm text-gray-700">
            {percentage >= 80 ? "Excellent understanding!" : percentage >= 50 ? "Good effort! Review the explanations below." : "Keep learning! Try again after reviewing."}
          </p>
        </div>
        <button onClick={handleRetry} className="w-full py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 text-sm font-medium">Try Again</button>
        <div className="space-y-3">
          {questions.map((q: any, qi: number) => (
            <div key={qi} className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm">
              <p className="font-medium text-gray-800 mb-1">{q.question}</p>
              <p className={`text-xs ${answers[qi] === q.correct_answer ? "text-green-700" : "text-red-700"}`}>
                Your answer: {q.options[answers[qi] ?? 0]} {answers[qi] === q.correct_answer ? "(Correct)" : "(Incorrect)"}
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
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-teal-900">Knowledge Check</h3>
        <p className="text-xs text-teal-700 mt-1">Test your understanding of scaling using questions from YOUR dataset.</p>
      </div>

      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div key={i} className={`w-3 h-3 rounded-full transition-all ${i === currentQ ? "bg-teal-600 scale-125" : i < currentQ ? (answers[i] === questions[i].correct_answer ? "bg-green-500" : "bg-red-400") : "bg-gray-300"}`} />
        ))}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="text-xs text-gray-500 mb-2">Question {currentQ + 1} of {questions.length}</div>
        <p className="text-sm font-medium text-gray-900 mb-4">{question.question}</p>
        <div className="space-y-2">
          {question.options.map((option: string, oi: number) => {
            let optionClass = "bg-white border-gray-300 hover:border-teal-400 cursor-pointer";
            if (answered) {
              if (oi === question.correct_answer) optionClass = "bg-green-50 border-green-500 text-green-900";
              else if (oi === selectedAnswer && oi !== question.correct_answer) optionClass = "bg-red-50 border-red-500 text-red-900";
              else optionClass = "bg-gray-50 border-gray-200 text-gray-400 cursor-default";
            }
            return (
              <button key={oi} onClick={() => handleSelect(oi)} disabled={answered} className={`w-full text-left p-3 rounded-lg border-2 text-sm transition-colors ${optionClass}`}>
                <span className="font-mono text-xs text-gray-400 mr-2">{String.fromCharCode(65 + oi)}.</span>
                {option}
                {answered && oi === question.correct_answer && <CheckCircle className="w-4 h-4 text-green-600 inline ml-2" />}
                {answered && oi === selectedAnswer && oi !== question.correct_answer && <XCircle className="w-4 h-4 text-red-600 inline ml-2" />}
              </button>
            );
          })}
        </div>
        {answered && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-800">{question.explanation}</p>
          </div>
        )}
        {answered && (
          <button onClick={handleNext} className="mt-4 w-full py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 text-sm font-medium flex items-center justify-center gap-2">
            {currentQ < questions.length - 1 ? (<>Next Question <ChevronRight className="w-4 h-4" /></>) : "See Results"}
          </button>
        )}
      </div>
    </div>
  );
}
