/**
 * MAE Score Explorer — tabbed interactive explorer for Mean Absolute Error
 * Tabs: Results | Outlier Robustness | Quiz | How It Works
 */

import { useState, useMemo, useCallback, useRef } from "react";
import {
  ClipboardList,
  Sparkles,
  HelpCircle,
  Cog,
  Trophy,
  CheckCircle,
  XCircle,
  RotateCcw,
  Shuffle,
  Shield,
  AlertTriangle,
  Info,
  TrendingUp,
} from "lucide-react";
import { MAEScoreResult } from "./MAEScoreResult";

// =========================================================================
// Types
// =========================================================================

type MAETab = "results" | "activity" | "quiz" | "how_it_works";

interface MAEScoreExplorerProps {
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

function generatePredPoints(seed: number, n = 10): PredPoint[] {
  const rng = mulberry32(seed);
  const pts: PredPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = i + 1;
    const actual = 3 + rng() * 6;
    const predicted = actual + (rng() - 0.5) * 2;
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

export function MAEScoreExplorer({ result }: MAEScoreExplorerProps) {
  const [activeTab, setActiveTab] = useState<MAETab>("results");

  const tabs: { id: MAETab; label: string; icon: any }[] = [
    { id: "results", label: "Results", icon: ClipboardList },
    { id: "activity", label: "Outlier Robustness", icon: Shield },
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
                  ? "border-cyan-500 text-cyan-700"
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
        {activeTab === "results" && <MAEScoreResult result={result} />}
        {activeTab === "activity" && <OutlierRobustnessTab />}
        {activeTab === "quiz" && <QuizTab />}
        {activeTab === "how_it_works" && <HowItWorksTab />}
      </div>
    </div>
  );
}

// =========================================================================
// Outlier Robustness Tab
// =========================================================================

const W = 520, H = 340;
const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;

function OutlierRobustnessTab() {
  const [seed, setSeed] = useState(55);
  const [basePoints] = useState<PredPoint[]>(() => generatePredPoints(55, 8));
  const [outlierDistance, setOutlierDistance] = useState(0);

  // Add outlier as an extra point
  const points = useMemo(() => {
    const pts = [...basePoints];
    if (outlierDistance > 0) {
      const lastActual = pts[pts.length - 1].actual;
      pts.push({
        x: pts.length + 1,
        actual: lastActual,
        predicted: lastActual + outlierDistance,
      });
    }
    return pts;
  }, [basePoints, outlierDistance]);

  const mse = useMemo(() => calcMSE(points), [points]);
  const mae = useMemo(() => calcMAE(points), [points]);
  const rmse = Math.sqrt(mse);

  // Baseline metrics (no outlier)
  const baseMSE = useMemo(() => calcMSE(basePoints), [basePoints]);
  const baseMAE = useMemo(() => calcMAE(basePoints), [basePoints]);

  const yRange = useMemo(() => {
    const allY = points.flatMap((p) => [p.actual, p.predicted]);
    const mn = Math.min(...allY);
    const mx = Math.max(...allY);
    const pad = Math.max((mx - mn) * 0.2, 2);
    return { min: mn - pad, max: mx + pad };
  }, [points]);

  const xRange = useMemo(() => ({ min: 0, max: points.length + 1 }), [points]);

  const toSX = useCallback((x: number) => PAD.left + ((x - xRange.min) / (xRange.max - xRange.min)) * CW, [xRange]);
  const toSY = useCallback((y: number) => PAD.top + (1 - (y - yRange.min) / (yRange.max - yRange.min)) * CH, [yRange]);

  // Data for growth comparison chart
  const growthData = useMemo(() => {
    const steps: { dist: number; mse: number; mae: number }[] = [];
    for (let d = 0; d <= 20; d += 1) {
      const pts = [...basePoints];
      if (d > 0) {
        const lastActual = pts[pts.length - 1].actual;
        pts.push({ x: pts.length + 1, actual: lastActual, predicted: lastActual + d });
      }
      steps.push({ dist: d, mse: calcMSE(pts), mae: calcMAE(pts) });
    }
    return steps;
  }, [basePoints]);

  const maxMetric = Math.max(...growthData.map((d) => d.mse));

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <Shield className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <p className="text-sm text-cyan-800">
          <span className="font-semibold">Move the outlier slider</span> to see how MAE stays stable while
          MSE explodes. This is why MAE is called "robust to outliers".
        </p>
      </div>

      {/* Outlier Slider */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-2">
          Outlier Distance: <span className="text-cyan-600">{outlierDistance.toFixed(1)}</span>
        </h4>
        <input
          type="range"
          min={0}
          max={20}
          step={0.5}
          value={outlierDistance}
          onChange={(e) => setOutlierDistance(parseFloat(e.target.value))}
          className="w-full accent-cyan-500"
        />
        <div className="flex justify-between mt-1 text-xs text-gray-400">
          <span>No outlier</span>
          <span>Extreme</span>
        </div>
      </div>

      {/* Side-by-side metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-4 text-center">
          <div className="text-[10px] text-cyan-600 font-semibold uppercase mb-1">MAE</div>
          <div className="text-3xl font-bold text-cyan-700">{mae.toFixed(3)}</div>
          {outlierDistance > 0 && (
            <div className="text-xs text-cyan-500 mt-1">
              +{((mae - baseMAE) / baseMAE * 100).toFixed(0)}% from baseline
            </div>
          )}
        </div>
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-center">
          <div className="text-[10px] text-orange-600 font-semibold uppercase mb-1">MSE</div>
          <div className="text-3xl font-bold text-orange-700">{mse.toFixed(3)}</div>
          {outlierDistance > 0 && (
            <div className="text-xs text-orange-500 mt-1">
              +{((mse - baseMSE) / baseMSE * 100).toFixed(0)}% from baseline
            </div>
          )}
        </div>
      </div>

      {/* Scatter plot */}
      <div className="bg-white border border-gray-200 rounded-lg p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <line x1={PAD.left} y1={PAD.top + CH} x2={PAD.left + CW} y2={PAD.top + CH} stroke="#94a3b8" strokeWidth={1} />
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + CH} stroke="#94a3b8" strokeWidth={1} />

          {points.map((p, i) => {
            const sx = toSX(p.x);
            const syActual = toSY(p.actual);
            const syPred = toSY(p.predicted);
            const isOutlier = i === points.length - 1 && outlierDistance > 0;

            return (
              <g key={i}>
                {/* Absolute error bar */}
                <line x1={sx} y1={syActual} x2={sx} y2={syPred} stroke={isOutlier ? "#ef4444" : "#06b6d4"} strokeWidth={isOutlier ? 3 : 2} opacity={0.6} />
                <circle cx={sx} cy={syActual} r={4} fill="#3b82f6" stroke="white" strokeWidth={1.5} />
                <circle cx={sx} cy={syPred} r={4} fill={isOutlier ? "#ef4444" : "#06b6d4"} stroke="white" strokeWidth={1.5} />
                {isOutlier && (
                  <text x={sx + 8} y={(syActual + syPred) / 2 + 3} className="text-[9px] fill-red-500 font-bold">
                    Outlier!
                  </text>
                )}
              </g>
            );
          })}

          <circle cx={PAD.left + 10} cy={PAD.top + 10} r={3} fill="#3b82f6" />
          <text x={PAD.left + 18} y={PAD.top + 13} className="text-[10px] fill-gray-600">Actual</text>
          <circle cx={PAD.left + 70} cy={PAD.top + 10} r={3} fill="#06b6d4" />
          <text x={PAD.left + 78} y={PAD.top + 13} className="text-[10px] fill-gray-600">Predicted</text>
        </svg>
      </div>

      {/* Growth comparison chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">
          <AlertTriangle className="w-4 h-4 inline mr-1 text-orange-500" />
          How MAE vs MSE Grow With Outlier Distance
        </h4>
        <svg viewBox="0 0 480 160" className="w-full">
          <line x1={40} y1={140} x2={460} y2={140} stroke="#94a3b8" strokeWidth={1} />
          <line x1={40} y1={10} x2={40} y2={140} stroke="#94a3b8" strokeWidth={1} />

          {/* MSE line (orange) */}
          <polyline
            points={growthData.map((d, i) => `${40 + (i / (growthData.length - 1)) * 420},${140 - (d.mse / maxMetric) * 120}`).join(" ")}
            fill="none"
            stroke="#f97316"
            strokeWidth={2}
          />

          {/* MAE line (cyan) */}
          <polyline
            points={growthData.map((d, i) => `${40 + (i / (growthData.length - 1)) * 420},${140 - (d.mae / maxMetric) * 120}`).join(" ")}
            fill="none"
            stroke="#06b6d4"
            strokeWidth={2}
          />

          {/* Current position marker */}
          {outlierDistance > 0 && (
            <line
              x1={40 + (outlierDistance / 20) * 420}
              y1={10}
              x2={40 + (outlierDistance / 20) * 420}
              y2={140}
              stroke="#94a3b8"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
          )}

          {/* Legend */}
          <line x1={60} y1={15} x2={80} y2={15} stroke="#f97316" strokeWidth={2} />
          <text x={84} y={18} className="text-[9px] fill-gray-600">MSE (quadratic growth)</text>
          <line x1={240} y1={15} x2={260} y2={15} stroke="#06b6d4" strokeWidth={2} />
          <text x={264} y={18} className="text-[9px] fill-gray-600">MAE (linear growth)</text>

          {/* Axis labels */}
          <text x={250} y={156} textAnchor="middle" className="text-[9px] fill-gray-500">Outlier Distance</text>
          <text x={12} y={80} textAnchor="middle" className="text-[9px] fill-gray-500" transform="rotate(-90, 12, 80)">Metric Value</text>
        </svg>
      </div>

      {/* Key Insight */}
      <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-4">
        <p className="text-sm text-cyan-800">
          <strong>Key Insight:</strong> MAE grows linearly with outlier distance (error of 10 adds 10 to MAE).
          MSE grows quadratically (error of 10 adds 100). This is why MAE is preferred when your data has outliers
          and you don't want a single bad prediction to dominate the metric.
        </p>
      </div>
    </div>
  );
}

// =========================================================================
// Quiz Tab
// =========================================================================

const MAE_QUIZ = [
  {
    question: "What does MAE measure?",
    options: [
      "The average of the squared prediction errors",
      "The average of the absolute prediction errors",
      "The maximum prediction error",
      "The median prediction error",
    ],
    correct_answer: 1,
    explanation:
      "MAE calculates the average of the absolute differences between predicted and actual values. Absolute values ensure errors don't cancel out, but unlike MSE, errors are not squared.",
  },
  {
    question: "Predictions: [3, 5, 7], Actual: [2, 5, 10]. What is the MAE?",
    options: ["1.0", "1.33", "2.0", "4.33"],
    correct_answer: 1,
    explanation: "Errors: |3-2|=1, |5-5|=0, |7-10|=3. MAE = (1+0+3)/3 = 4/3 ≈ 1.33.",
  },
  {
    question: "Why is MAE more robust to outliers than MSE?",
    options: [
      "MAE ignores outliers completely",
      "MAE uses absolute values so large errors aren't amplified",
      "MAE only considers the median error",
      "MAE removes the top 10% of errors",
    ],
    correct_answer: 1,
    explanation:
      "With MSE, an error of 100 contributes 10,000. With MAE, it contributes only 100. MAE treats all errors proportionally to their magnitude without amplification.",
  },
  {
    question: "When would you prefer MSE over MAE?",
    options: [
      "When outliers should be ignored",
      "When large errors are especially costly and should be penalized more",
      "When you want a simpler calculation",
      "When dealing with classification",
    ],
    correct_answer: 1,
    explanation:
      "If large prediction errors are much worse than small ones (e.g., safety-critical), MSE's quadratic penalty makes it a better loss function.",
  },
  {
    question: "If MAE = RMSE for a model, what can you conclude?",
    options: [
      "The model is perfect",
      "All prediction errors have the same magnitude",
      "The model is overfitting",
      "The dataset has no outliers",
    ],
    correct_answer: 1,
    explanation:
      "MAE = RMSE only when all errors have exactly the same absolute value. Any variance in error sizes makes RMSE > MAE.",
  },
];

function QuizTab() {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = MAE_QUIZ;
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
            {pct >= 80 ? "Excellent! You understand MAE well!" : pct >= 50 ? "Good job! Review the topics you missed." : "Keep learning! Understanding error metrics is key."}
          </p>
          <button onClick={handleRetry} className="mt-4 px-6 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors text-sm font-medium">Try Again</button>
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
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <p className="text-sm text-cyan-800"><span className="font-semibold">Test your knowledge</span> about Mean Absolute Error!</p>
      </div>
      <div className="flex items-center justify-center gap-2">
        {questions.map((_, i) => (
          <div key={i} className={`w-2.5 h-2.5 rounded-full transition-all ${i === currentQ ? "bg-cyan-500 scale-125" : i < answers.length ? (answers[i] === questions[i].correct_answer ? "bg-green-400" : "bg-red-400") : "bg-gray-300"}`} />
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
          <button onClick={handleNext} className="mt-4 px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors text-sm font-medium">
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
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <div className="text-sm text-cyan-800 space-y-2">
          <p className="font-semibold">How MAE Works</p>
          <ol className="list-decimal list-inside space-y-1">
            <li>For each data point, compute the <strong>error</strong>: predicted - actual</li>
            <li>Take the <strong>absolute value</strong> of each error (no squaring!)</li>
            <li><strong>Average</strong> all absolute errors to get MAE</li>
          </ol>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">The Formula</h4>
        <div className="bg-gray-50 rounded-lg p-4 inline-block">
          <p className="text-lg font-mono text-gray-800">
            MAE = (1/n) &times; &Sigma;|y<sub>i</sub> - y&#770;<sub>i</sub>|
          </p>
          <p className="mt-2 text-xs text-gray-500">
            where y<sub>i</sub> = actual, y&#770;<sub>i</sub> = predicted, n = samples
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-600" />
            Linear Penalty
          </h4>
          <p className="text-sm text-gray-600">
            MAE treats errors proportionally. An error of 10 is exactly 10x worse than an error of 1.
            No amplification like MSE's quadratic penalty.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Shield className="w-4 h-4 text-green-600" />
            Outlier Robustness
          </h4>
          <p className="text-sm text-gray-600">
            One extreme outlier won't dominate MAE the way it does MSE. This makes MAE a better choice
            when your data has noisy outliers you can't remove.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Info className="w-4 h-4 text-orange-600" />
            When to Use
          </h4>
          <p className="text-sm text-gray-600">
            Use MAE when all errors should be weighted equally, when your data has outliers,
            or when you want an intuitive "average error" in the target's original units.
          </p>
        </div>
      </div>
    </div>
  );
}
