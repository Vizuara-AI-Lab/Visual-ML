/**
 * LinearRegressionExplorer - Interactive learning activities for Linear Regression
 * Tabbed component: Results | Coefficient Explorer | Equation Builder | Prediction Playground | Quiz
 */

import { useState, useMemo, type ReactNode } from "react";
import {
  ClipboardList,
  BarChart3,
  Calculator,
  SlidersHorizontal,
  HelpCircle,
  CheckCircle,
  XCircle,
  Trophy,
  ChevronRight,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

interface LinearRegressionExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type ExplorerTab =
  | "results"
  | "coefficients"
  | "equation"
  | "playground"
  | "quiz";

export const LinearRegressionExplorer = ({
  result,
  renderResults,
}: LinearRegressionExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: {
    id: ExplorerTab;
    label: string;
    icon: any;
    available: boolean;
  }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "coefficients",
      label: "Coefficient Explorer",
      icon: BarChart3,
      available: !!result.coefficient_analysis?.features?.length,
    },
    {
      id: "equation",
      label: "Equation Builder",
      icon: Calculator,
      available: !!result.equation_data,
    },
    {
      id: "playground",
      label: "Prediction Playground",
      icon: SlidersHorizontal,
      available: !!result.prediction_playground?.features?.length,
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
                    ? "border-emerald-600 text-emerald-600"
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
      {activeTab === "coefficients" && result.coefficient_analysis && (
        <CoefficientTab data={result.coefficient_analysis} />
      )}
      {activeTab === "equation" && result.equation_data && (
        <EquationTab data={result.equation_data} />
      )}
      {activeTab === "playground" && result.prediction_playground && (
        <PredictionPlaygroundTab data={result.prediction_playground} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Coefficient Explorer Tab ---

function CoefficientTab({ data }: { data: any }) {
  const features = data.features || [];
  const [sortBy, setSortBy] = useState<"rank" | "name">("rank");

  const sorted = useMemo(() => {
    const copy = [...features];
    if (sortBy === "name") copy.sort((a: any, b: any) => a.name.localeCompare(b.name));
    return copy;
  }, [features, sortBy]);

  return (
    <div className="space-y-4">
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-emerald-900">
          Feature Coefficients
        </h3>
        <p className="text-xs text-emerald-700 mt-1">
          Each feature has a coefficient showing its impact on the prediction.
          Positive = increases prediction, Negative = decreases prediction.
          Larger bars = stronger influence.
        </p>
      </div>

      {/* Sort toggle */}
      <div className="flex gap-2 text-xs">
        <button
          onClick={() => setSortBy("rank")}
          className={`px-3 py-1 rounded-full ${
            sortBy === "rank"
              ? "bg-emerald-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          By Impact
        </button>
        <button
          onClick={() => setSortBy("name")}
          className={`px-3 py-1 rounded-full ${
            sortBy === "name"
              ? "bg-emerald-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          By Name
        </button>
      </div>

      {/* Coefficient bars */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-3">
        {sorted.map((feat: any) => (
          <div key={feat.name}>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                {feat.direction === "positive" ? (
                  <ArrowUpRight className="w-3 h-3 text-green-600" />
                ) : (
                  <ArrowDownRight className="w-3 h-3 text-red-500" />
                )}
                <span className="text-sm font-medium text-gray-800">
                  {feat.name}
                </span>
              </div>
              <span className="text-xs font-mono text-gray-600">
                {feat.coefficient > 0 ? "+" : ""}
                {feat.coefficient}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-gray-100 rounded h-5 overflow-hidden">
                <div
                  className={`h-full rounded transition-all ${
                    feat.direction === "positive"
                      ? "bg-green-400"
                      : "bg-red-400"
                  }`}
                  style={{ width: `${feat.bar_width_pct}%` }}
                />
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-0.5">
              {feat.interpretation}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Equation Builder Tab ---

function EquationTab({ data }: { data: any }) {
  const [showFull, setShowFull] = useState(false);
  const terms = data.terms || [];
  const displayTerms = showFull ? terms : terms.slice(0, 6);

  return (
    <div className="space-y-4">
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-emerald-900">
          Regression Equation
        </h3>
        <p className="text-xs text-emerald-700 mt-1">
          Linear regression learns an equation: y = b + w1*x1 + w2*x2 + ... Each
          term shows how a feature contributes to the prediction.
        </p>
      </div>

      {/* Full equation string */}
      <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
        <code className="text-green-400 text-sm whitespace-nowrap">
          {data.equation_string || "N/A"}
        </code>
      </div>

      {/* Intercept */}
      <div className="bg-white border border-emerald-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm font-medium text-gray-700">
              Intercept (baseline)
            </span>
            <p className="text-xs text-gray-500 mt-1">
              The prediction when all features are zero
            </p>
          </div>
          <span className="text-lg font-bold text-emerald-600 font-mono">
            {data.intercept}
          </span>
        </div>
      </div>

      {/* Term cards */}
      <div className="space-y-2">
        {displayTerms.map((term: any, idx: number) => (
          <div
            key={idx}
            className="bg-white border border-gray-200 rounded-lg p-3 flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <span
                className={`text-lg font-bold ${
                  term.coefficient >= 0 ? "text-green-600" : "text-red-500"
                }`}
              >
                {term.sign}
              </span>
              <div>
                <span className="text-sm font-medium text-gray-800">
                  {term.feature}
                </span>
              </div>
            </div>
            <span className="font-mono text-sm text-gray-700">
              {Math.abs(term.coefficient)}
            </span>
          </div>
        ))}
      </div>

      {terms.length > 6 && (
        <button
          onClick={() => setShowFull(!showFull)}
          className="text-sm text-emerald-600 hover:text-emerald-700 font-medium"
        >
          {showFull ? "Show less" : `Show all ${terms.length} terms`}
        </button>
      )}
    </div>
  );
}

// --- Prediction Playground Tab ---

function PredictionPlaygroundTab({ data }: { data: any }) {
  const features = data.features || [];
  const intercept = data.intercept || 0;

  const [sliderValues, setSliderValues] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    features.forEach((f: any) => {
      initial[f.name] = f.mean;
    });
    return initial;
  });

  const prediction = useMemo(() => {
    let pred = intercept;
    features.forEach((f: any) => {
      pred += f.coefficient * (sliderValues[f.name] ?? f.mean);
    });
    return pred;
  }, [sliderValues, features, intercept]);

  const handleReset = () => {
    const reset: Record<string, number> = {};
    features.forEach((f: any) => {
      reset[f.name] = f.mean;
    });
    setSliderValues(reset);
  };

  return (
    <div className="space-y-4">
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-emerald-900">
          Prediction Playground
        </h3>
        <p className="text-xs text-emerald-700 mt-1">
          Drag the sliders to change feature values and watch the prediction
          update in real time. This is how the model uses the equation!
        </p>
      </div>

      {/* Live prediction */}
      <div className="bg-emerald-600 text-white rounded-lg p-6 text-center">
        <div className="text-xs uppercase tracking-wide opacity-80 mb-1">
          Predicted Value
        </div>
        <div className="text-4xl font-bold font-mono">{prediction.toFixed(4)}</div>
        <div className="text-xs mt-2 opacity-70">
          Intercept ({intercept}) + weighted feature sums
        </div>
      </div>

      {/* Reset button */}
      <button
        onClick={handleReset}
        className="text-xs text-emerald-600 hover:text-emerald-700 font-medium"
      >
        Reset all to mean values
      </button>

      {/* Feature sliders */}
      <div className="space-y-4">
        {features.map((feat: any) => {
          const val = sliderValues[feat.name] ?? feat.mean;
          const contribution = feat.coefficient * val;
          return (
            <div
              key={feat.name}
              className="bg-white border border-gray-200 rounded-lg p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-800">
                  {feat.name}
                </span>
                <div className="text-right">
                  <span className="text-sm font-mono text-gray-900">
                    {val.toFixed(2)}
                  </span>
                  <span
                    className={`ml-2 text-xs font-mono ${
                      contribution >= 0 ? "text-green-600" : "text-red-500"
                    }`}
                  >
                    ({contribution >= 0 ? "+" : ""}
                    {contribution.toFixed(4)})
                  </span>
                </div>
              </div>
              <input
                type="range"
                min={feat.min}
                max={feat.max}
                step={feat.step}
                value={val}
                onChange={(e) =>
                  setSliderValues((prev) => ({
                    ...prev,
                    [feat.name]: parseFloat(e.target.value),
                  }))
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>{feat.min}</span>
                <span className="text-gray-500">coef: {feat.coefficient}</span>
                <span>{feat.max}</span>
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
              ? "Excellent understanding of linear regression!"
              : percentage >= 50
                ? "Good effort! Review the explanations below."
                : "Keep learning! Try again after reviewing."}
          </p>
        </div>
        <button
          onClick={handleRetry}
          className="w-full py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 text-sm font-medium"
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
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-emerald-900">
          Knowledge Check
        </h3>
        <p className="text-xs text-emerald-700 mt-1">
          Test your understanding of linear regression using questions from YOUR
          model.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-3 h-3 rounded-full transition-all ${
              i === currentQ
                ? "bg-emerald-600 scale-125"
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
              "bg-white border-gray-300 hover:border-emerald-400 cursor-pointer";
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
          <div className="mt-4 p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
            <p className="text-xs text-emerald-800">{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 w-full py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 text-sm font-medium flex items-center justify-center gap-2"
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
