/**
 * SplitExplorer - Interactive learning activities for Train/Test Split
 * Tabbed component: Results | Split Visualizer | Class Balance | Ratio Explorer | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  ClipboardList,
  PieChart,
  BarChart3,
  SlidersHorizontal,
  HelpCircle,
  CheckCircle,
  XCircle,
  Trophy,
  ChevronRight,
  Info,
} from "lucide-react";

interface SplitExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type ExplorerTab = "results" | "visualizer" | "balance" | "ratio" | "quiz";

export const SplitExplorer = ({
  result,
  renderResults,
}: SplitExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "visualizer",
      label: "Split Visualizer",
      icon: PieChart,
      available: !!result.split_visualization,
    },
    {
      id: "balance",
      label: "Class Balance",
      icon: BarChart3,
      available: !!result.class_balance,
    },
    {
      id: "ratio",
      label: "Ratio Explorer",
      icon: SlidersHorizontal,
      available: !!result.ratio_explorer,
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
                    ? "border-cyan-600 text-cyan-600"
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
      {activeTab === "visualizer" && result.split_visualization && (
        <SplitVisualizerTab data={result.split_visualization} />
      )}
      {activeTab === "balance" && result.class_balance && (
        <ClassBalanceTab data={result.class_balance} />
      )}
      {activeTab === "ratio" && result.ratio_explorer && (
        <RatioExplorerTab data={result.ratio_explorer} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Split Visualizer Tab ---

function SplitVisualizerTab({ data }: { data: any }) {
  const trainPct = data.train_percentage || 80;
  const testPct = data.test_percentage || 20;

  return (
    <div className="space-y-4">
      <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-cyan-900">
          How Your Data Was Split
        </h3>
        <p className="text-xs text-cyan-700 mt-1">
          The dataset was divided into training and test sets. The model learns
          from training data and is evaluated on test data it has never seen.
        </p>
      </div>

      {/* Visual split bar */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="mb-4 text-sm font-medium text-gray-700 text-center">
          {data.total_samples?.toLocaleString()} total samples
        </div>
        <div className="flex h-14 rounded-lg overflow-hidden border border-gray-300">
          <div
            className="bg-cyan-500 flex items-center justify-center text-white font-semibold text-sm transition-all"
            style={{ width: `${trainPct}%` }}
          >
            Train ({trainPct}%)
          </div>
          <div
            className="bg-orange-400 flex items-center justify-center text-white font-semibold text-sm transition-all"
            style={{ width: `${testPct}%` }}
          >
            Test ({testPct}%)
          </div>
        </div>

        {/* Sample counts */}
        <div className="flex justify-between mt-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-cyan-600">
              {data.train_samples?.toLocaleString()}
            </div>
            <div className="text-xs text-gray-500">Training samples</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-500">
              {data.test_samples?.toLocaleString()}
            </div>
            <div className="text-xs text-gray-500">Test samples</div>
          </div>
        </div>
      </div>

      {/* Feature/Column info */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xl font-bold text-gray-900">
            {data.n_features || 0}
          </div>
          <div className="text-xs text-gray-500">Features</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xl font-bold text-gray-900">
            {data.target_column || "N/A"}
          </div>
          <div className="text-xs text-gray-500">Target Column</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-xl font-bold text-gray-900 capitalize">
            {data.split_type || "random"}
          </div>
          <div className="text-xs text-gray-500">Split Method</div>
        </div>
      </div>

      {/* Sample rows preview */}
      {data.train_sample_rows && data.train_sample_rows.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Sample Training Rows (first 3)
          </h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs border-collapse">
              <thead className="bg-cyan-50">
                <tr>
                  {Object.keys(data.train_sample_rows[0]).map((col) => (
                    <th
                      key={col}
                      className="border border-gray-200 px-3 py-2 text-left font-semibold"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.train_sample_rows.map((row: any, idx: number) => (
                  <tr
                    key={idx}
                    className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                  >
                    {Object.values(row).map((val: any, ci: number) => (
                      <td key={ci} className="border border-gray-200 px-3 py-1">
                        {val === null ? (
                          <span className="text-gray-400 italic">null</span>
                        ) : (
                          String(val)
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// --- Class Balance Tab ---

function ClassBalanceTab({ data }: { data: any }) {
  const trainDist = data.train_distribution || {};
  const testDist = data.test_distribution || {};
  const classes = data.class_names || Object.keys(trainDist);

  return (
    <div className="space-y-4">
      <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-cyan-900">
          Class Distribution Comparison
        </h3>
        <p className="text-xs text-cyan-700 mt-1">
          Compare how each class is distributed between training and test sets.
          {data.is_stratified
            ? " Stratified splitting keeps proportions similar."
            : " Random splitting may cause slight differences."}
        </p>
      </div>

      {/* Balance status */}
      {data.balance_status && (
        <div
          className={`border rounded-lg p-4 ${
            data.balance_status === "balanced"
              ? "bg-green-50 border-green-200"
              : "bg-yellow-50 border-yellow-200"
          }`}
        >
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4" />
            <span className="text-sm font-medium">
              {data.balance_status === "balanced"
                ? "Classes are well-balanced across train and test sets!"
                : "There is some class imbalance - the model may need extra care."}
            </span>
          </div>
        </div>
      )}

      {/* Side-by-side bars */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-gray-900">
            Class Distribution
          </h4>
          <div className="flex gap-4 text-xs">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-cyan-500 rounded" /> Train
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-orange-400 rounded" /> Test
            </span>
          </div>
        </div>

        <div className="space-y-4">
          {classes.map((cls: string) => {
            const trainData = trainDist[cls] || {};
            const testData = testDist[cls] || {};
            const trainPct = trainData.percentage || 0;
            const testPct = testData.percentage || 0;

            return (
              <div key={cls}>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-gray-800">
                    {cls}
                  </span>
                  <span className="text-xs text-gray-500">
                    Train: {trainData.count || 0} ({trainPct}%) | Test:{" "}
                    {testData.count || 0} ({testPct}%)
                  </span>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="w-12 text-xs text-gray-400">Train</div>
                    <div className="flex-1 bg-gray-100 rounded h-5 overflow-hidden">
                      <div
                        className="bg-cyan-500 h-full rounded transition-all"
                        style={{ width: `${trainPct}%` }}
                      />
                    </div>
                    <div className="w-12 text-xs text-right text-gray-600">
                      {trainPct}%
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-12 text-xs text-gray-400">Test</div>
                    <div className="flex-1 bg-gray-100 rounded h-5 overflow-hidden">
                      <div
                        className="bg-orange-400 h-full rounded transition-all"
                        style={{ width: `${testPct}%` }}
                      />
                    </div>
                    <div className="w-12 text-xs text-right text-gray-600">
                      {testPct}%
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Max deviation */}
      {data.max_deviation !== undefined && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-600">
            Maximum class proportion difference between train and test:{" "}
            <span
              className={`font-semibold ${
                data.max_deviation < 5 ? "text-green-600" : "text-orange-600"
              }`}
            >
              {data.max_deviation}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// --- Ratio Explorer Tab ---

function RatioExplorerTab({ data }: { data: any }) {
  const ratios = data.ratios || [];
  const appliedRatio = data.applied_ratio || "80/20";

  return (
    <div className="space-y-4">
      <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-cyan-900">
          Explore Different Split Ratios
        </h3>
        <p className="text-xs text-cyan-700 mt-1">
          See what happens with different train/test ratios. Each ratio has
          trade-offs between having enough data to learn from and enough to test
          on.
        </p>
      </div>

      <div className="space-y-3">
        {ratios.map((ratio: any) => {
          const isApplied = ratio.label === appliedRatio;
          return (
            <div
              key={ratio.label}
              className={`border rounded-lg p-4 transition-all ${
                isApplied
                  ? "border-cyan-500 bg-cyan-50 ring-2 ring-cyan-200"
                  : "border-gray-200 bg-white"
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold text-gray-900">
                    {ratio.label}
                  </span>
                  {isApplied && (
                    <span className="px-2 py-0.5 bg-cyan-600 text-white rounded-full text-xs font-medium">
                      Applied
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-500">
                  Train: {ratio.train_count?.toLocaleString()} | Test:{" "}
                  {ratio.test_count?.toLocaleString()}
                </div>
              </div>

              {/* Visual bar */}
              <div className="flex h-6 rounded overflow-hidden border border-gray-200 mb-3">
                <div
                  className="bg-cyan-400 flex items-center justify-center text-white text-xs font-medium"
                  style={{ width: `${ratio.train_pct}%` }}
                >
                  {ratio.train_pct}%
                </div>
                <div
                  className="bg-orange-300 flex items-center justify-center text-white text-xs font-medium"
                  style={{ width: `${ratio.test_pct}%` }}
                >
                  {ratio.test_pct}%
                </div>
              </div>

              {/* Pros/Cons */}
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div className="font-semibold text-green-700 mb-1">Pros</div>
                  <div className="text-gray-600">
                    {Array.isArray(ratio.pros)
                      ? ratio.pros.map((pro: string, i: number) => (
                          <p key={i}>+ {pro}</p>
                        ))
                      : ratio.pros && <p>+ {ratio.pros}</p>}
                  </div>
                </div>
                <div>
                  <div className="font-semibold text-red-700 mb-1">Cons</div>
                  <div className="text-gray-600">
                    {Array.isArray(ratio.cons)
                      ? ratio.cons.map((con: string, i: number) => (
                          <p key={i}>- {con}</p>
                        ))
                      : ratio.cons && <p>- {ratio.cons}</p>}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
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
              ? "Excellent understanding of data splitting!"
              : percentage >= 50
                ? "Good effort! Review the explanations below."
                : "Keep learning! Try again after reviewing."}
          </p>
        </div>
        <button
          onClick={handleRetry}
          className="w-full py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 text-sm font-medium"
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
      <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-cyan-900">
          Knowledge Check
        </h3>
        <p className="text-xs text-cyan-700 mt-1">
          Test your understanding of train/test splitting using questions from
          YOUR dataset.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-3 h-3 rounded-full transition-all ${
              i === currentQ
                ? "bg-cyan-600 scale-125"
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
              "bg-white border-gray-300 hover:border-cyan-400 cursor-pointer";
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
          <div className="mt-4 p-3 bg-cyan-50 border border-cyan-200 rounded-lg">
            <p className="text-xs text-cyan-800">{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 w-full py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 text-sm font-medium flex items-center justify-center gap-2"
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
