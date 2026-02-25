/**
 * MSE Score Explorer — tabbed interactive explorer for Mean Squared Error
 * Tabs: Results | Error Visualizer | Quiz | How It Works
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
  Eye,
  Square,
  AlertTriangle,
  Info,
  TrendingUp,
} from "lucide-react";
import { MSEScoreResult } from "./MSEScoreResult";

// =========================================================================
// Types
// =========================================================================

type MSETab = "results" | "activity" | "quiz" | "how_it_works";

interface MSEScoreExplorerProps {
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
    const actual = 2 + rng() * 8;
    const predicted = actual + (rng() - 0.5) * 3;
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

export function MSEScoreExplorer({ result }: MSEScoreExplorerProps) {
  const [activeTab, setActiveTab] = useState<MSETab>("results");

  const tabs: { id: MSETab; label: string; icon: any }[] = [
    { id: "results", label: "Results", icon: ClipboardList },
    { id: "activity", label: "Error Visualizer", icon: Sparkles },
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
                  ? "border-orange-500 text-orange-700"
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
        {activeTab === "results" && <MSEScoreResult result={result} />}
        {activeTab === "activity" && <ErrorVisualizerTab />}
        {activeTab === "quiz" && <QuizTab />}
        {activeTab === "how_it_works" && <HowItWorksTab />}
      </div>
    </div>
  );
}

// =========================================================================
// Error Visualizer Tab
// =========================================================================

const W = 520, H = 380;
const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;

function ErrorVisualizerTab() {
  const [seed, setSeed] = useState(42);
  const [points, setPoints] = useState<PredPoint[]>(() => generatePredPoints(42));
  const [showSquares, setShowSquares] = useState(true);
  const [outlierDistance, setOutlierDistance] = useState(0);
  const dragging = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Add outlier effect to last point
  const effectivePoints = useMemo(() => {
    const pts = [...points];
    if (pts.length > 0 && outlierDistance > 0) {
      const last = { ...pts[pts.length - 1] };
      last.predicted = last.actual + outlierDistance;
      pts[pts.length - 1] = last;
    }
    return pts;
  }, [points, outlierDistance]);

  const yRange = useMemo(() => {
    const allY = effectivePoints.flatMap((p) => [p.actual, p.predicted]);
    const mn = Math.min(...allY);
    const mx = Math.max(...allY);
    const pad = Math.max((mx - mn) * 0.2, 1);
    return { min: mn - pad, max: mx + pad };
  }, [effectivePoints]);

  const xRange = useMemo(() => {
    return { min: 0, max: effectivePoints.length + 1 };
  }, [effectivePoints]);

  const toSX = useCallback((x: number) => PAD.left + ((x - xRange.min) / (xRange.max - xRange.min)) * CW, [xRange]);
  const toSY = useCallback((y: number) => PAD.top + (1 - (y - yRange.min) / (yRange.max - yRange.min)) * CH, [yRange]);
  const fromSY = useCallback((sy: number) => yRange.min + (1 - (sy - PAD.top) / CH) * (yRange.max - yRange.min), [yRange]);

  const mse = useMemo(() => calcMSE(effectivePoints), [effectivePoints]);
  const mae = useMemo(() => calcMAE(effectivePoints), [effectivePoints]);

  const getSVGCoords = useCallback(
    (e: React.MouseEvent) => {
      const svg = svgRef.current;
      if (!svg) return { x: 0, y: 0 };
      const rect = svg.getBoundingClientRect();
      return {
        x: ((e.clientX - rect.left) / rect.width) * W,
        y: ((e.clientY - rect.top) / rect.height) * H,
      };
    },
    []
  );

  const handleMouseDown = useCallback((idx: number) => (e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = idx;
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (dragging.current === null) return;
      const { y: sy } = getSVGCoords(e);
      const dataY = fromSY(sy);
      setPoints((prev) => {
        const next = [...prev];
        next[dragging.current!] = { ...next[dragging.current!], predicted: dataY };
        return next;
      });
    },
    [getSVGCoords, fromSY]
  );

  const handleMouseUp = useCallback(() => {
    dragging.current = null;
  }, []);

  const handleNewData = () => {
    const newSeed = seed + 7;
    setSeed(newSeed);
    setPoints(generatePredPoints(newSeed));
    setOutlierDistance(0);
  };

  const handleReset = () => {
    setPoints(generatePredPoints(seed));
    setOutlierDistance(0);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-orange-200 bg-orange-50 px-4 py-3">
        <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-orange-600" />
        <p className="text-sm text-orange-800">
          <span className="font-semibold">Drag the orange prediction dots</span> up or down to see how
          MSE changes. Notice the squared error rectangles — large errors get amplified!
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => setShowSquares(!showSquares)}
          className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
            showSquares
              ? "bg-orange-600 text-white border-orange-600"
              : "bg-white text-orange-700 border-orange-300 hover:bg-orange-50"
          }`}
        >
          <Square className="w-3 h-3 inline mr-1" />
          {showSquares ? "Hide Squares" : "Show Squares"}
        </button>
        <button
          onClick={handleNewData}
          className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-colors font-medium"
        >
          <Shuffle className="w-3 h-3 inline mr-1" />
          New Data
        </button>
        <button
          onClick={handleReset}
          className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-colors font-medium"
        >
          <RotateCcw className="w-3 h-3 inline mr-1" />
          Reset
        </button>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">MSE</div>
          <div className="text-2xl font-bold text-orange-600">{mse.toFixed(3)}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">MAE</div>
          <div className="text-2xl font-bold text-cyan-600">{mae.toFixed(3)}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">RMSE</div>
          <div className="text-2xl font-bold text-purple-600">{Math.sqrt(mse).toFixed(3)}</div>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-2">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Axes */}
          <line x1={PAD.left} y1={PAD.top + CH} x2={PAD.left + CW} y2={PAD.top + CH} stroke="#94a3b8" strokeWidth={1} />
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + CH} stroke="#94a3b8" strokeWidth={1} />

          {effectivePoints.map((p, i) => {
            const sx = toSX(p.x);
            const syActual = toSY(p.actual);
            const syPred = toSY(p.predicted);
            const error = Math.abs(p.predicted - p.actual);
            const sqError = error * error;
            const maxErr = (yRange.max - yRange.min) * 0.3;
            const squareSize = Math.min(error / maxErr, 1) * 40;

            return (
              <g key={i}>
                {/* Error line */}
                <line
                  x1={sx}
                  y1={syActual}
                  x2={sx}
                  y2={syPred}
                  stroke="#ef4444"
                  strokeWidth={1.5}
                  opacity={0.5}
                />

                {/* Squared error rectangle */}
                {showSquares && error > 0.01 && (
                  <rect
                    x={sx + 2}
                    y={Math.min(syActual, syPred)}
                    width={squareSize}
                    height={Math.abs(syPred - syActual)}
                    fill="#ef4444"
                    opacity={0.15}
                    stroke="#ef4444"
                    strokeWidth={0.5}
                    strokeDasharray="3 2"
                  />
                )}

                {/* Error label */}
                {error > 0.3 && (
                  <text
                    x={sx + squareSize + 6}
                    y={(syActual + syPred) / 2 + 3}
                    className="text-[9px] fill-red-500 font-medium"
                  >
                    {sqError.toFixed(1)}
                  </text>
                )}

                {/* Actual point (blue) */}
                <circle cx={sx} cy={syActual} r={5} fill="#3b82f6" stroke="white" strokeWidth={1.5} />

                {/* Predicted point (orange, draggable) */}
                <circle
                  cx={sx}
                  cy={syPred}
                  r={6}
                  fill="#f97316"
                  stroke="white"
                  strokeWidth={2}
                  className="cursor-grab active:cursor-grabbing"
                  onMouseDown={handleMouseDown(i)}
                />
              </g>
            );
          })}

          {/* Legend */}
          <circle cx={PAD.left + 10} cy={PAD.top + 10} r={4} fill="#3b82f6" />
          <text x={PAD.left + 20} y={PAD.top + 14} className="text-[10px] fill-gray-600">Actual</text>
          <circle cx={PAD.left + 70} cy={PAD.top + 10} r={4} fill="#f97316" />
          <text x={PAD.left + 80} y={PAD.top + 14} className="text-[10px] fill-gray-600">Predicted (drag me!)</text>
        </svg>
      </div>

      {/* Outlier Impact Slider */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-2 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-orange-500" />
          Outlier Impact: Move one prediction far away
        </h4>
        <p className="text-xs text-gray-500 mb-3">
          Watch how MSE explodes compared to MAE when one prediction has a large error.
        </p>
        <input
          type="range"
          min={0}
          max={20}
          step={0.5}
          value={outlierDistance}
          onChange={(e) => setOutlierDistance(parseFloat(e.target.value))}
          className="w-full accent-orange-500"
        />
        <div className="flex justify-between mt-1 text-xs text-gray-500">
          <span>No outlier</span>
          <span>Distance: {outlierDistance.toFixed(1)}</span>
          <span>Extreme outlier</span>
        </div>
        {outlierDistance > 0 && (
          <div className="grid grid-cols-2 gap-3 mt-3">
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 text-center">
              <div className="text-[10px] text-orange-600 font-semibold uppercase">MSE Impact</div>
              <div className="text-lg font-bold text-orange-700">{mse.toFixed(2)}</div>
              <div className="text-[10px] text-orange-500">Grows quadratically</div>
            </div>
            <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3 text-center">
              <div className="text-[10px] text-cyan-600 font-semibold uppercase">MAE Impact</div>
              <div className="text-lg font-bold text-cyan-700">{mae.toFixed(2)}</div>
              <div className="text-[10px] text-cyan-500">Grows linearly</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// =========================================================================
// Quiz Tab
// =========================================================================

const MSE_QUIZ = [
  {
    question: "Why does MSE square the errors instead of just averaging them?",
    options: [
      "To make the math easier",
      "To ensure errors are positive and penalize large errors more",
      "Because negative errors don't matter",
      "It's just a convention",
    ],
    correct_answer: 1,
    explanation:
      "Squaring serves two purposes: it makes all errors positive (no cancellation), and it disproportionately penalizes large errors. An error of 10 contributes 100, while an error of 1 contributes only 1.",
  },
  {
    question: "If your model has MSE = 25, what is the RMSE?",
    options: ["625", "5", "12.5", "Cannot be determined"],
    correct_answer: 1,
    explanation: "RMSE = sqrt(MSE) = sqrt(25) = 5. RMSE converts MSE back to the original units.",
  },
  {
    question: "Predictions: [2, 4, 6], Actual: [1, 4, 8]. What is the MSE?",
    options: ["1.0", "1.67", "3.0", "5.0"],
    correct_answer: 1,
    explanation:
      "Errors: (2-1)²=1, (4-4)²=0, (6-8)²=4. MSE = (1+0+4)/3 = 5/3 ≈ 1.67.",
  },
  {
    question: "A model has MSE = 0.001. Is it a good model?",
    options: [
      "Yes, low MSE always means good",
      "Not necessarily — it depends on the scale of the target",
      "No, MSE should be exactly 0",
      "Only if R² is also high",
    ],
    correct_answer: 1,
    explanation:
      "MSE depends on the scale of the target variable. If predicting house prices in millions, MSE=0.001 is great. Always compare MSE relative to the target variable's range.",
  },
  {
    question: "What is the main disadvantage of MSE compared to MAE?",
    options: [
      "MSE is harder to compute",
      "MSE is more sensitive to outliers because errors are squared",
      "MSE can be negative",
      "MSE doesn't work for regression",
    ],
    correct_answer: 1,
    explanation:
      "Because MSE squares errors, a single large outlier can dominate the entire metric. MAE is more robust to outliers since it doesn't amplify large errors.",
  },
];

function QuizTab() {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = MSE_QUIZ;
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
          <Trophy className={`w-16 h-16 mx-auto mb-4 ${pct >= 80 ? "text-green-500" : pct >= 50 ? "text-yellow-500" : "text-red-500"}`} />
          <h3 className="text-2xl font-bold text-gray-900">{score} / {questions.length}</h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80 ? "Excellent! You understand MSE well!" : pct >= 50 ? "Good job! Review the topics you missed." : "Keep learning! MSE concepts are important for regression."}
          </p>
          <button onClick={handleRetry} className="mt-4 px-6 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors text-sm font-medium">
            Try Again
          </button>
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
      <div className="flex items-start gap-3 rounded-lg border border-orange-200 bg-orange-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-orange-600" />
        <p className="text-sm text-orange-800">
          <span className="font-semibold">Test your knowledge</span> about Mean Squared Error!
        </p>
      </div>

      <div className="flex items-center justify-center gap-2">
        {questions.map((_, i) => (
          <div key={i} className={`w-2.5 h-2.5 rounded-full transition-all ${i === currentQ ? "bg-orange-500 scale-125" : i < answers.length ? (answers[i] === questions[i].correct_answer ? "bg-green-400" : "bg-red-400") : "bg-gray-300"}`} />
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
          <button onClick={handleNext} className="mt-4 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors text-sm font-medium">
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
      <div className="flex items-start gap-3 rounded-lg border border-orange-200 bg-orange-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-orange-600" />
        <div className="text-sm text-orange-800 space-y-2">
          <p className="font-semibold">How MSE Works</p>
          <ol className="list-decimal list-inside space-y-1">
            <li>For each data point, compute the <strong>error</strong>: predicted - actual</li>
            <li><strong>Square</strong> each error (removes negatives, amplifies large errors)</li>
            <li><strong>Average</strong> all squared errors to get MSE</li>
          </ol>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">The Formula</h4>
        <div className="bg-gray-50 rounded-lg p-4 inline-block">
          <p className="text-lg font-mono text-gray-800">
            MSE = (1/n) &times; &Sigma;(y<sub>i</sub> - y&#770;<sub>i</sub>)²
          </p>
          <p className="mt-2 text-xs text-gray-500">
            where y<sub>i</sub> = actual value, y&#770;<sub>i</sub> = predicted value, n = number of samples
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Square className="w-4 h-4 text-orange-600" />
            Squaring Effect
          </h4>
          <p className="text-sm text-gray-600">
            An error of 1 contributes 1 to MSE. An error of 10 contributes 100. This means MSE heavily
            penalizes large mistakes — one bad prediction can dominate the score.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            Scale Dependence
          </h4>
          <p className="text-sm text-gray-600">
            MSE is in <strong>squared units</strong> of the target variable. If predicting prices in
            dollars, MSE is in dollars². This makes it hard to interpret directly — use RMSE instead.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Info className="w-4 h-4 text-green-600" />
            When to Use
          </h4>
          <p className="text-sm text-gray-600">
            MSE is the standard loss function for regression. Use it when large errors are especially
            bad (safety-critical applications). For reporting, prefer RMSE (same units as target).
          </p>
        </div>
      </div>
    </div>
  );
}
