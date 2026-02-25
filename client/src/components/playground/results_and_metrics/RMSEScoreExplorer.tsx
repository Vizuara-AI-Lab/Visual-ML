/**
 * RMSE Score Explorer — tabbed interactive explorer for Root Mean Squared Error
 * Tabs: Results | MSE vs RMSE | Quiz | How It Works
 */

import { useState, useMemo } from "react";
import {
  ClipboardList,
  Sparkles,
  HelpCircle,
  Cog,
  Trophy,
  CheckCircle,
  XCircle,
  ArrowDown,
  Info,
  TrendingUp,
  BarChart3,
} from "lucide-react";
import { RMSEScoreResult } from "./RMSEScoreResult";

// =========================================================================
// Types
// =========================================================================

type RMSETab = "results" | "activity" | "quiz" | "how_it_works";

interface RMSEScoreExplorerProps {
  result: any;
}

interface PredPoint {
  x: number;
  actual: number;
  predicted: number;
}

// =========================================================================
// Utilities
// =========================================================================

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generatePredPoints(seed: number, n = 8): PredPoint[] {
  const rng = mulberry32(seed);
  const pts: PredPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = i + 1;
    const actual = 50 + rng() * 50; // e.g. house prices in $K
    const predicted = actual + (rng() - 0.5) * 30;
    pts.push({ x, actual, predicted });
  }
  return pts;
}

function calcMSE(pts: PredPoint[]): number {
  if (pts.length === 0) return 0;
  return pts.reduce((s, p) => s + (p.predicted - p.actual) ** 2, 0) / pts.length;
}

function calcMAE(pts: PredPoint[]): number {
  if (pts.length === 0) return 0;
  return pts.reduce((s, p) => s + Math.abs(p.predicted - p.actual), 0) / pts.length;
}

// =========================================================================
// Main Explorer
// =========================================================================

export function RMSEScoreExplorer({ result }: RMSEScoreExplorerProps) {
  const [activeTab, setActiveTab] = useState<RMSETab>("results");

  const tabs: { id: RMSETab; label: string; icon: any }[] = [
    { id: "results", label: "Results", icon: ClipboardList },
    { id: "activity", label: "MSE vs RMSE", icon: Sparkles },
    { id: "quiz", label: "Quiz", icon: HelpCircle },
    { id: "how_it_works", label: "How It Works", icon: Cog },
  ];

  return (
    <div className="space-y-4">
      <div className="flex border-b border-gray-200 overflow-x-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? "border-purple-500 text-purple-700"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="min-h-[400px]">
        {activeTab === "results" && <RMSEScoreResult result={result} />}
        {activeTab === "activity" && <MSEvsRMSETab />}
        {activeTab === "quiz" && <QuizTab />}
        {activeTab === "how_it_works" && <HowItWorksTab />}
      </div>
    </div>
  );
}

// =========================================================================
// MSE vs RMSE Tab
// =========================================================================

function MSEvsRMSETab() {
  const [unitLabel, setUnitLabel] = useState("$K");
  const [points] = useState<PredPoint[]>(() => generatePredPoints(77));

  const mse = useMemo(() => calcMSE(points), [points]);
  const rmse = Math.sqrt(mse);
  const mae = useMemo(() => calcMAE(points), [points]);

  const errors = points.map((p) => ({
    error: p.predicted - p.actual,
    absError: Math.abs(p.predicted - p.actual),
    sqError: (p.predicted - p.actual) ** 2,
  }));

  const maxSqError = Math.max(...errors.map((e) => e.sqError));
  const maxAbsError = Math.max(...errors.map((e) => e.absError));

  const unitOptions = [
    { label: "$K (house prices)", value: "$K" },
    { label: "°C (temperature)", value: "°C" },
    { label: "kg (weight)", value: "kg" },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-purple-200 bg-purple-50 px-4 py-3">
        <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
        <p className="text-sm text-purple-800">
          <span className="font-semibold">MSE is in squared units</span> — hard to interpret.
          RMSE takes the square root to give you error in the <strong>same units</strong> as your data.
        </p>
      </div>

      {/* Unit selector */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-500">Imagine predicting in:</span>
        {unitOptions.map((opt) => (
          <button
            key={opt.value}
            onClick={() => setUnitLabel(opt.value)}
            className={`text-xs px-3 py-1 rounded-full transition-colors ${
              unitLabel === opt.value
                ? "bg-purple-600 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Conversion visual */}
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="grid grid-cols-3 gap-4 items-center">
          {/* MSE */}
          <div className="text-center">
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <div className="text-[10px] text-orange-600 font-semibold uppercase mb-1">MSE</div>
              <div className="text-2xl font-bold text-orange-700">{mse.toFixed(1)}</div>
              <div className="text-xs text-orange-500 mt-1">{unitLabel}²</div>
              <p className="text-[10px] text-gray-400 mt-2">Squared units — hard to interpret</p>
            </div>
          </div>

          {/* Arrow */}
          <div className="text-center">
            <div className="flex flex-col items-center gap-1">
              <ArrowDown className="w-8 h-8 text-purple-400 rotate-[-90deg]" />
              <div className="bg-purple-100 rounded-lg px-3 py-1">
                <span className="text-xs font-mono font-bold text-purple-700">sqrt()</span>
              </div>
            </div>
          </div>

          {/* RMSE */}
          <div className="text-center">
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="text-[10px] text-purple-600 font-semibold uppercase mb-1">RMSE</div>
              <div className="text-2xl font-bold text-purple-700">{rmse.toFixed(1)}</div>
              <div className="text-xs text-purple-500 mt-1">{unitLabel}</div>
              <p className="text-[10px] text-gray-400 mt-2">Original units — easy to interpret!</p>
            </div>
          </div>
        </div>

        <div className="mt-4 bg-purple-50 border border-purple-100 rounded-lg p-3 text-center">
          <p className="text-sm text-purple-800">
            "On average, predictions are about <strong>{rmse.toFixed(1)} {unitLabel}</strong> off."
          </p>
          <p className="text-xs text-gray-500 mt-1">
            (Compare: MSE = {mse.toFixed(1)} {unitLabel}² means nothing intuitively)
          </p>
        </div>
      </div>

      {/* Error breakdown table */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Error Breakdown Per Point</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-left text-[10px] font-bold text-gray-500 uppercase pb-2 pr-3">#</th>
                <th className="text-right text-[10px] font-bold text-gray-500 uppercase pb-2 px-2">Actual</th>
                <th className="text-right text-[10px] font-bold text-gray-500 uppercase pb-2 px-2">Predicted</th>
                <th className="text-right text-[10px] font-bold text-gray-500 uppercase pb-2 px-2">Error</th>
                <th className="text-right text-[10px] font-bold text-orange-500 uppercase pb-2 px-2">Error² (MSE)</th>
                <th className="text-right text-[10px] font-bold text-cyan-500 uppercase pb-2 px-2">|Error| (MAE)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {points.map((p, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="py-2 pr-3 text-gray-400">{i + 1}</td>
                  <td className="py-2 px-2 text-right font-medium">{p.actual.toFixed(1)}</td>
                  <td className="py-2 px-2 text-right font-medium">{p.predicted.toFixed(1)}</td>
                  <td className="py-2 px-2 text-right text-gray-600">{errors[i].error.toFixed(1)}</td>
                  <td className="py-2 px-2 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-16 h-2 rounded-full bg-gray-100 overflow-hidden">
                        <div className="h-full bg-orange-400 rounded-full" style={{ width: `${(errors[i].sqError / maxSqError) * 100}%` }} />
                      </div>
                      <span className="text-orange-600 font-medium w-14 text-right">{errors[i].sqError.toFixed(0)}</span>
                    </div>
                  </td>
                  <td className="py-2 px-2 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-16 h-2 rounded-full bg-gray-100 overflow-hidden">
                        <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${(errors[i].absError / maxAbsError) * 100}%` }} />
                      </div>
                      <span className="text-cyan-600 font-medium w-14 text-right">{errors[i].absError.toFixed(1)}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t-2 border-gray-200 font-bold">
                <td colSpan={4} className="py-2 text-right text-gray-700 pr-2">Average:</td>
                <td className="py-2 px-2 text-right text-orange-700">
                  {mse.toFixed(1)} <span className="text-[10px] font-normal text-gray-400">= MSE</span>
                </td>
                <td className="py-2 px-2 text-right text-cyan-700">
                  {mae.toFixed(1)} <span className="text-[10px] font-normal text-gray-400">= MAE</span>
                </td>
              </tr>
              <tr className="font-bold">
                <td colSpan={4} className="py-1 text-right text-gray-700 pr-2">sqrt(MSE) = RMSE:</td>
                <td className="py-1 px-2 text-right text-purple-700" colSpan={2}>
                  {rmse.toFixed(1)} {unitLabel}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* RMSE >= MAE insight */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-purple-900 mb-2">RMSE is always &ge; MAE</h4>
        <div className="flex items-center gap-4">
          <div className="text-center">
            <span className="text-lg font-bold text-purple-700">RMSE = {rmse.toFixed(2)}</span>
          </div>
          <span className="text-gray-400">&ge;</span>
          <div className="text-center">
            <span className="text-lg font-bold text-cyan-700">MAE = {mae.toFixed(2)}</span>
          </div>
          <span className="text-gray-400">|</span>
          <div className="text-center">
            <span className="text-sm text-gray-600">Gap: {(rmse - mae).toFixed(2)}</span>
          </div>
        </div>
        <p className="text-xs text-purple-700 mt-2">
          A large gap between RMSE and MAE means some predictions have much bigger errors than others.
          If RMSE ≈ MAE, errors are uniform in size.
        </p>
      </div>
    </div>
  );
}

// =========================================================================
// Quiz Tab
// =========================================================================

const RMSE_QUIZ = [
  {
    question: "What is the relationship between RMSE and MSE?",
    options: ["RMSE = MSE / n", "RMSE = MSE²", "RMSE = sqrt(MSE)", "RMSE = MSE × 2"],
    correct_answer: 2,
    explanation: "RMSE is the square root of MSE. This converts the metric from squared units back to the original units.",
  },
  {
    question: "Why is RMSE often preferred over MSE for reporting?",
    options: [
      "RMSE is always smaller than MSE",
      "RMSE is in the same units as the target variable",
      "RMSE is less sensitive to outliers",
      "RMSE is faster to compute",
    ],
    correct_answer: 1,
    explanation: "RMSE is in the same units as the target (dollars, degrees, meters), while MSE is in squared units. This makes RMSE directly interpretable.",
  },
  {
    question: "If RMSE = 3°C for a temperature model, what does this mean?",
    options: [
      "The model is 3% accurate",
      "On average, predictions are about 3 degrees off",
      "The model explains 3% of variance",
      "There are 3 outliers",
    ],
    correct_answer: 1,
    explanation: "RMSE = 3°C means the typical prediction error is about 3 degrees Celsius. It gives you an intuitive sense of 'how wrong' the model usually is.",
  },
  {
    question: "Is RMSE always greater than or equal to MAE?",
    options: [
      "Yes, always",
      "No, MAE is always greater",
      "They are always equal",
      "It depends on the dataset",
    ],
    correct_answer: 0,
    explanation: "RMSE >= MAE always holds. Squaring amplifies larger errors, and sqrt doesn't fully undo this. The gap tells you about error variance.",
  },
  {
    question: "Model A: RMSE=5, Model B: RMSE=3. Which is better?",
    options: [
      "Model A (higher is better)",
      "Model B (lower RMSE = smaller errors)",
      "Cannot compare without the dataset",
      "They are equal",
    ],
    correct_answer: 1,
    explanation: "Lower RMSE means smaller prediction errors. Model B's predictions are typically about 3 units off vs Model A's 5 units. (Assuming same test set.)",
  },
];

function QuizTab() {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = RMSE_QUIZ;
  const question = questions[currentQ];
  const isCorrect = selectedAnswer === question.correct_answer;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelectedAnswer(idx);
    setAnswered(true);
    if (idx === question.correct_answer) setScore((s) => s + 1);
    setAnswers((a) => [...a, idx]);
  };

  const handleNext = () => {
    if (currentQ + 1 >= questions.length) setShowResults(true);
    else { setCurrentQ((q) => q + 1); setSelectedAnswer(null); setAnswered(false); }
  };

  const handleRetry = () => {
    setCurrentQ(0); setSelectedAnswer(null); setAnswered(false);
    setScore(0); setAnswers([]); setShowResults(false);
  };

  if (showResults) {
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          <Trophy className={`w-16 h-16 mx-auto mb-4 ${pct >= 80 ? "text-green-500" : pct >= 50 ? "text-yellow-500" : "text-red-500"}`} />
          <h3 className="text-2xl font-bold text-gray-900">{score} / {questions.length}</h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80 ? "Excellent! You understand RMSE well!" : pct >= 50 ? "Good job! Review what you missed." : "Keep learning about error metrics!"}
          </p>
          <button onClick={handleRetry} className="mt-4 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium">Try Again</button>
        </div>
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700">Review</h4>
          {questions.map((q, i) => {
            const userAns = answers[i];
            const correct = userAns === q.correct_answer;
            return (
              <div key={i} className={`rounded-lg border p-3 ${correct ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}>
                <div className="flex items-start gap-2">
                  {correct ? <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 shrink-0" /> : <XCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{q.question}</p>
                    <p className="text-xs text-gray-600 mt-1">Your answer: {q.options[userAns ?? 0]}{!correct && ` | Correct: ${q.options[q.correct_answer]}`}</p>
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
      <div className="flex items-start gap-3 rounded-lg border border-purple-200 bg-purple-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
        <p className="text-sm text-purple-800"><span className="font-semibold">Test your knowledge</span> about RMSE!</p>
      </div>
      <div className="flex items-center justify-center gap-2">
        {questions.map((_, i) => (
          <div key={i} className={`w-2.5 h-2.5 rounded-full transition-all ${i === currentQ ? "bg-purple-500 scale-125" : i < answers.length ? (answers[i] === questions[i].correct_answer ? "bg-green-400" : "bg-red-400") : "bg-gray-300"}`} />
        ))}
      </div>
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <p className="text-xs text-gray-400 mb-2">Question {currentQ + 1} of {questions.length}</p>
        <p className="text-base font-medium text-gray-900 mb-4">{question.question}</p>
        <div className="space-y-2">
          {question.options.map((opt, idx) => {
            let style = "border-gray-200 bg-white hover:bg-gray-50 text-gray-700";
            if (answered) {
              if (idx === question.correct_answer) style = "border-green-300 bg-green-50 text-green-800";
              else if (idx === selectedAnswer && !isCorrect) style = "border-red-300 bg-red-50 text-red-800";
              else style = "border-gray-200 bg-gray-50 text-gray-400";
            }
            return (
              <button key={idx} onClick={() => handleSelect(idx)} disabled={answered} className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-colors ${style} disabled:cursor-default`}>
                <span className="font-medium mr-2">{String.fromCharCode(65 + idx)}.</span>{opt}
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
          <button onClick={handleNext} className="mt-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium">
            {currentQ + 1 >= questions.length ? "See Results" : "Next Question"}
          </button>
        )}
      </div>
    </div>
  );
}

// =========================================================================
// How It Works Tab
// =========================================================================

function HowItWorksTab() {
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-purple-200 bg-purple-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
        <div className="text-sm text-purple-800 space-y-2">
          <p className="font-semibold">How RMSE Works</p>
          <ol className="list-decimal list-inside space-y-1">
            <li>Compute each error: predicted - actual</li>
            <li>Square each error (same as MSE)</li>
            <li>Average the squared errors (= MSE)</li>
            <li>Take the <strong>square root</strong> of MSE to get RMSE</li>
          </ol>
          <p className="mt-2">The sqrt step converts squared units back to original units, making RMSE interpretable.</p>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">The Formula</h4>
        <div className="bg-gray-50 rounded-lg p-4 inline-block">
          <p className="text-lg font-mono text-gray-800">
            RMSE = &radic;( (1/n) &times; &Sigma;(y<sub>i</sub> - y&#770;<sub>i</sub>)² )
          </p>
          <p className="mt-2 text-xs text-gray-500">= sqrt(MSE)</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-purple-600" />
            Interpretable Units
          </h4>
          <p className="text-sm text-gray-600">
            If your target is in dollars, RMSE is in dollars too. "Average error of $5" is
            much clearer than "MSE of 25 dollars-squared".
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-cyan-600" />
            RMSE vs MAE
          </h4>
          <p className="text-sm text-gray-600">
            RMSE &ge; MAE always. The gap between them reveals error variance: large gap = some predictions
            are much worse than others. Small gap = errors are uniform.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Info className="w-4 h-4 text-green-600" />
            When to Use
          </h4>
          <p className="text-sm text-gray-600">
            Use RMSE as the standard reporting metric for regression. It's more interpretable than MSE and
            still penalizes large errors (unlike MAE). Ideal for comparing models on the same dataset.
          </p>
        </div>
      </div>
    </div>
  );
}
