/**
 * R2 Score Explorer — tabbed interactive explorer for R² metric
 * Tabs: Results | Interactive Fit | Quiz | How It Works
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
  Target,
  TrendingUp,
  BarChart3,
  Info,
} from "lucide-react";
import { R2ScoreResult } from "./R2ScoreResult";

// =========================================================================
// Types
// =========================================================================

type R2Tab = "results" | "activity" | "quiz" | "how_it_works";

interface R2ScoreExplorerProps {
  result: any;
}

interface DataPoint {
  x: number;
  y: number;
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

function generatePoints(seed: number, n = 20, noise = 0.5): DataPoint[] {
  const rng = mulberry32(seed);
  const slope = 0.4 + rng() * 1.2;
  const intercept = 1 + rng() * 2;
  const pts: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = 0.5 + rng() * 9;
    const y = slope * x + intercept + (rng() - 0.5) * 2 * noise * 3;
    pts.push({ x, y });
  }
  return pts;
}

function computeOLS(points: DataPoint[]): { m: number; b: number } {
  const n = points.length;
  if (n === 0) return { m: 0, b: 0 };
  let sumX = 0, sumY = 0;
  for (const p of points) { sumX += p.x; sumY += p.y; }
  const meanX = sumX / n;
  const meanY = sumY / n;
  let num = 0, den = 0;
  for (const p of points) {
    const dx = p.x - meanX;
    num += dx * (p.y - meanY);
    den += dx * dx;
  }
  const m = den !== 0 ? num / den : 0;
  return { m, b: meanY - m * meanX };
}

function computeR2(points: DataPoint[], m: number, b: number): number {
  const n = points.length;
  if (n === 0) return 0;
  const meanY = points.reduce((s, p) => s + p.y, 0) / n;
  let ssTot = 0, ssRes = 0;
  for (const p of points) {
    ssTot += (p.y - meanY) ** 2;
    ssRes += (p.y - (m * p.x + b)) ** 2;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

function computeMSE(points: DataPoint[], m: number, b: number): number {
  if (points.length === 0) return 0;
  let sum = 0;
  for (const p of points) sum += (p.y - (m * p.x + b)) ** 2;
  return sum / points.length;
}

// =========================================================================
// Main Explorer
// =========================================================================

export function R2ScoreExplorer({ result }: R2ScoreExplorerProps) {
  const [activeTab, setActiveTab] = useState<R2Tab>("results");

  const tabs: { id: R2Tab; label: string; icon: any }[] = [
    { id: "results", label: "Results", icon: ClipboardList },
    { id: "activity", label: "Interactive Fit", icon: Sparkles },
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
                  ? "border-green-500 text-green-700"
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
        {activeTab === "results" && <R2ScoreResult result={result} />}
        {activeTab === "activity" && <InteractiveFitTab />}
        {activeTab === "quiz" && <QuizTab />}
        {activeTab === "how_it_works" && <HowItWorksTab />}
      </div>
    </div>
  );
}

// =========================================================================
// Interactive Fit Tab
// =========================================================================

const W = 520, H = 360;
const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;

function InteractiveFitTab() {
  const [seed, setSeed] = useState(42);
  const [points, setPoints] = useState<DataPoint[]>(() => generatePoints(42));
  const [slope, setSlope] = useState(1);
  const [intercept, setIntercept] = useState(2);
  const [showBestFit, setShowBestFit] = useState(false);
  const [addMode, setAddMode] = useState(false);
  const dragging = useRef<"left" | "right" | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const xRange = useMemo(() => {
    const xs = points.map((p) => p.x);
    return { min: Math.min(...xs) - 0.5, max: Math.max(...xs) + 0.5 };
  }, [points]);

  const yRange = useMemo(() => {
    const ys = points.map((p) => p.y);
    const mn = Math.min(...ys);
    const mx = Math.max(...ys);
    const pad = (mx - mn) * 0.15 || 1;
    return { min: mn - pad, max: mx + pad };
  }, [points]);

  const toSX = useCallback((x: number) => PAD.left + ((x - xRange.min) / (xRange.max - xRange.min)) * CW, [xRange]);
  const toSY = useCallback((y: number) => PAD.top + (1 - (y - yRange.min) / (yRange.max - yRange.min)) * CH, [yRange]);
  const fromSX = useCallback((sx: number) => xRange.min + ((sx - PAD.left) / CW) * (xRange.max - xRange.min), [xRange]);
  const fromSY = useCallback((sy: number) => yRange.min + (1 - (sy - PAD.top) / CH) * (yRange.max - yRange.min), [yRange]);

  const bestFit = useMemo(() => computeOLS(points), [points]);
  const activeM = showBestFit ? bestFit.m : slope;
  const activeB = showBestFit ? bestFit.b : intercept;
  const r2 = useMemo(() => computeR2(points, activeM, activeB), [points, activeM, activeB]);
  const mse = useMemo(() => computeMSE(points, activeM, activeB), [points, activeM, activeB]);
  const bestR2 = useMemo(() => computeR2(points, bestFit.m, bestFit.b), [points, bestFit]);

  const handleX1 = xRange.min + (xRange.max - xRange.min) * 0.2;
  const handleX2 = xRange.min + (xRange.max - xRange.min) * 0.8;
  const handleY1 = activeM * handleX1 + activeB;
  const handleY2 = activeM * handleX2 + activeB;

  const lineX1 = xRange.min;
  const lineX2 = xRange.max;
  const lineY1 = activeM * lineX1 + activeB;
  const lineY2 = activeM * lineX2 + activeB;

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

  const handleMouseDown = useCallback(
    (handle: "left" | "right") => (e: React.MouseEvent) => {
      e.preventDefault();
      if (!showBestFit) dragging.current = handle;
    },
    [showBestFit]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging.current) return;
      const { x: sx, y: sy } = getSVGCoords(e);
      const dataY = fromSY(sy);
      if (dragging.current === "left") {
        const newM = (activeM * handleX2 + activeB - dataY) / (handleX2 - handleX1);
        const newB = dataY - newM * handleX1;
        setSlope(newM);
        setIntercept(newB);
      } else {
        const newM = (dataY - (activeM * handleX1 + activeB)) / (handleX2 - handleX1);
        const newB = activeM * handleX1 + activeB - newM * handleX1;
        setSlope(newM);
        setIntercept(newB);
      }
    },
    [activeM, activeB, handleX1, handleX2, getSVGCoords, fromSY]
  );

  const handleMouseUp = useCallback(() => {
    dragging.current = null;
  }, []);

  const handleSVGClick = useCallback(
    (e: React.MouseEvent) => {
      if (!addMode || dragging.current) return;
      const { x: sx, y: sy } = getSVGCoords(e);
      const dataX = fromSX(sx);
      const dataY = fromSY(sy);
      if (dataX >= xRange.min && dataX <= xRange.max && dataY >= yRange.min && dataY <= yRange.max) {
        setPoints((prev) => [...prev, { x: dataX, y: dataY }]);
      }
    },
    [addMode, getSVGCoords, fromSX, fromSY, xRange, yRange]
  );

  const handleNewData = () => {
    const newSeed = seed + 7;
    setSeed(newSeed);
    setPoints(generatePoints(newSeed));
    setShowBestFit(false);
    setSlope(1);
    setIntercept(2);
  };

  const r2Color = r2 >= 0.8 ? "text-green-600" : r2 >= 0.5 ? "text-yellow-600" : "text-red-600";
  const r2Badge = r2 >= 0.8 ? "bg-green-100 text-green-800" : r2 >= 0.5 ? "bg-yellow-100 text-yellow-800" : "bg-red-100 text-red-800";

  // Tick generators
  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    const step = Math.ceil((xRange.max - xRange.min) / 5);
    for (let t = Math.ceil(xRange.min); t <= xRange.max; t += step) ticks.push(t);
    return ticks;
  }, [xRange]);

  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const step = Math.ceil((yRange.max - yRange.min) / 5);
    for (let t = Math.ceil(yRange.min); t <= yRange.max; t += step) ticks.push(t);
    return ticks;
  }, [yRange]);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-green-200 bg-green-50 px-4 py-3">
        <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
        <p className="text-sm text-green-800">
          <span className="font-semibold">Drag the handles</span> on the regression line to see how
          R² changes in real time. Try to maximize R² by fitting the line to the data points!
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => { setShowBestFit(!showBestFit); }}
          className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
            showBestFit
              ? "bg-green-600 text-white border-green-600"
              : "bg-white text-green-700 border-green-300 hover:bg-green-50"
          }`}
        >
          <Eye className="w-3 h-3 inline mr-1" />
          {showBestFit ? "Hide Best Fit" : "Show Best Fit"}
        </button>
        <button
          onClick={handleNewData}
          className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-colors font-medium"
        >
          <Shuffle className="w-3 h-3 inline mr-1" />
          New Data
        </button>
        <button
          onClick={() => setAddMode(!addMode)}
          className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
            addMode
              ? "bg-blue-600 text-white border-blue-600"
              : "bg-white text-blue-700 border-blue-300 hover:bg-blue-50"
          }`}
        >
          <Target className="w-3 h-3 inline mr-1" />
          {addMode ? "Adding Points..." : "Add Points"}
        </button>
        <button
          onClick={() => { setPoints(generatePoints(seed)); setSlope(1); setIntercept(2); setShowBestFit(false); }}
          className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 transition-colors font-medium"
        >
          <RotateCcw className="w-3 h-3 inline mr-1" />
          Reset
        </button>
      </div>

      {/* Metrics panel */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">Your R²</div>
          <div className={`text-2xl font-bold ${r2Color}`}>{r2.toFixed(4)}</div>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${r2Badge}`}>
            {r2 >= 0.8 ? "Great" : r2 >= 0.5 ? "Moderate" : "Poor"}
          </span>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">Best R²</div>
          <div className="text-2xl font-bold text-green-600">{bestR2.toFixed(4)}</div>
          <span className="text-[10px] text-gray-500">OLS fit</span>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">MSE</div>
          <div className="text-2xl font-bold text-orange-600">{mse.toFixed(3)}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">Points</div>
          <div className="text-2xl font-bold text-gray-800">{points.length}</div>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-2">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className={`w-full ${addMode ? "cursor-crosshair" : "cursor-default"}`}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleSVGClick}
        >
          {/* Grid lines */}
          {xTicks.map((t) => (
            <line key={`gx-${t}`} x1={toSX(t)} y1={PAD.top} x2={toSX(t)} y2={PAD.top + CH} stroke="#f1f5f9" strokeWidth={1} />
          ))}
          {yTicks.map((t) => (
            <line key={`gy-${t}`} x1={PAD.left} y1={toSY(t)} x2={PAD.left + CW} y2={toSY(t)} stroke="#f1f5f9" strokeWidth={1} />
          ))}

          {/* Axes */}
          <line x1={PAD.left} y1={PAD.top + CH} x2={PAD.left + CW} y2={PAD.top + CH} stroke="#94a3b8" strokeWidth={1} />
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + CH} stroke="#94a3b8" strokeWidth={1} />

          {/* Axis labels */}
          {xTicks.map((t) => (
            <text key={`xl-${t}`} x={toSX(t)} y={PAD.top + CH + 16} textAnchor="middle" className="text-[10px] fill-gray-500">
              {t}
            </text>
          ))}
          {yTicks.map((t) => (
            <text key={`yl-${t}`} x={PAD.left - 8} y={toSY(t) + 3} textAnchor="end" className="text-[10px] fill-gray-500">
              {t.toFixed(0)}
            </text>
          ))}

          {/* Mean line (baseline) */}
          {(() => {
            const meanY = points.reduce((s, p) => s + p.y, 0) / points.length;
            return (
              <line
                x1={PAD.left}
                y1={toSY(meanY)}
                x2={PAD.left + CW}
                y2={toSY(meanY)}
                stroke="#cbd5e1"
                strokeWidth={1}
                strokeDasharray="6 4"
              />
            );
          })()}

          {/* Residual lines */}
          {points.map((p, i) => {
            const pred = activeM * p.x + activeB;
            return (
              <line
                key={`res-${i}`}
                x1={toSX(p.x)}
                y1={toSY(p.y)}
                x2={toSX(p.x)}
                y2={toSY(pred)}
                stroke={p.y > pred ? "#ef4444" : "#3b82f6"}
                strokeWidth={1.5}
                opacity={0.4}
              />
            );
          })}

          {/* Regression line */}
          <line
            x1={toSX(lineX1)}
            y1={toSY(lineY1)}
            x2={toSX(lineX2)}
            y2={toSY(lineY2)}
            stroke="#16a34a"
            strokeWidth={2.5}
          />

          {/* Data points */}
          {points.map((p, i) => (
            <circle
              key={`pt-${i}`}
              cx={toSX(p.x)}
              cy={toSY(p.y)}
              r={4}
              fill="#3b82f6"
              stroke="white"
              strokeWidth={1.5}
            />
          ))}

          {/* Drag handles */}
          {!showBestFit && (
            <>
              <circle
                cx={toSX(handleX1)}
                cy={toSY(handleY1)}
                r={8}
                fill="#16a34a"
                stroke="white"
                strokeWidth={2}
                className="cursor-grab active:cursor-grabbing"
                onMouseDown={handleMouseDown("left")}
              />
              <circle
                cx={toSX(handleX2)}
                cy={toSY(handleY2)}
                r={8}
                fill="#16a34a"
                stroke="white"
                strokeWidth={2}
                className="cursor-grab active:cursor-grabbing"
                onMouseDown={handleMouseDown("right")}
              />
            </>
          )}
        </svg>
      </div>

      {/* R² Comparison Reference */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">What different R² values look like</h4>
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: "R² = 0.30 (Poor)", seed: 100, noise: 2.5, color: "red" },
            { label: "R² = 0.70 (Good)", seed: 200, noise: 1.0, color: "yellow" },
            { label: "R² = 0.95 (Excellent)", seed: 300, noise: 0.3, color: "green" },
          ].map((ref) => {
            const refPts = generatePoints(ref.seed, 15, ref.noise);
            const fit = computeOLS(refPts);
            const refR2 = computeR2(refPts, fit.m, fit.b);
            const rxs = refPts.map((p) => p.x);
            const rys = refPts.map((p) => p.y);
            const rxMin = Math.min(...rxs) - 0.5;
            const rxMax = Math.max(...rxs) + 0.5;
            const ryMin = Math.min(...rys) - 1;
            const ryMax = Math.max(...rys) + 1;
            const rToSX = (x: number) => 10 + ((x - rxMin) / (rxMax - rxMin)) * 120;
            const rToSY = (y: number) => 10 + (1 - (y - ryMin) / (ryMax - ryMin)) * 80;

            return (
              <div key={ref.seed} className="text-center">
                <svg viewBox="0 0 140 100" className="w-full bg-gray-50 rounded border border-gray-100">
                  <line
                    x1={rToSX(rxMin)}
                    y1={rToSY(fit.m * rxMin + fit.b)}
                    x2={rToSX(rxMax)}
                    y2={rToSY(fit.m * rxMax + fit.b)}
                    stroke="#16a34a"
                    strokeWidth={1.5}
                  />
                  {refPts.map((p, i) => (
                    <circle key={i} cx={rToSX(p.x)} cy={rToSY(p.y)} r={2.5} fill="#3b82f6" />
                  ))}
                </svg>
                <p className="text-xs text-gray-600 mt-1 font-medium">{ref.label}</p>
                <p className="text-[10px] text-gray-400">Actual: {refR2.toFixed(2)}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// Quiz Tab
// =========================================================================

const R2_QUIZ = [
  {
    question: "What does an R² score of 0.85 mean?",
    options: [
      "The model is 85% accurate",
      "The model explains 85% of the variance in the target variable",
      "85% of the predictions are correct",
      "The model has an 85% chance of being right",
    ],
    correct_answer: 1,
    explanation:
      "R² measures the proportion of variance in the dependent variable that is predictable from the independent variables. 0.85 means 85% of the variance is explained by the model.",
  },
  {
    question: "What is the range of R² score?",
    options: [
      "0 to 100",
      "-1 to 1",
      "0 to 1 (can be negative in practice)",
      "Depends on the dataset",
    ],
    correct_answer: 2,
    explanation:
      "R² theoretically ranges from 0 to 1, but can be negative when the model performs worse than simply predicting the mean. Negative R² means the model is worse than useless.",
  },
  {
    question: "If R² = 0, what does it mean?",
    options: [
      "The model is perfectly wrong",
      "The model predicts the mean value for all inputs",
      "There is no data",
      "The model has zero features",
    ],
    correct_answer: 1,
    explanation:
      "R² = 0 means the model's predictions are no better than always predicting the mean of the target variable. The model captures none of the variance.",
  },
  {
    question: "Adding more features to a model will always increase R² on training data. True?",
    options: [
      "Yes, more features always help",
      "No, it can decrease R²",
      "Yes for training data, but not necessarily for test data",
      "Only if the features are correlated",
    ],
    correct_answer: 2,
    explanation:
      "On training data, R² never decreases when adding features. But on test data, irrelevant features cause overfitting and can decrease R². This is why Adjusted R² exists — it penalizes extra features.",
  },
  {
    question: "Which metric should you use alongside R² to better understand model performance?",
    options: [
      "Accuracy",
      "RMSE or MAE (error in original units)",
      "F1 Score",
      "AUC-ROC",
    ],
    correct_answer: 1,
    explanation:
      "R² tells you the proportion of variance explained, but not the actual magnitude of errors. RMSE or MAE give you error in the same units as the target variable, which is more interpretable.",
  },
];

function QuizTab() {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = R2_QUIZ;
  if (!questions.length) return null;

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
          <Trophy
            className={`w-16 h-16 mx-auto mb-4 ${
              pct >= 80 ? "text-green-500" : pct >= 50 ? "text-yellow-500" : "text-red-500"
            }`}
          />
          <h3 className="text-2xl font-bold text-gray-900">
            {score} / {questions.length}
          </h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80
              ? "Excellent! You understand R² well!"
              : pct >= 50
                ? "Good job! Review the topics you missed."
                : "Keep learning! R² has subtleties worth mastering."}
          </p>
          <button
            onClick={handleRetry}
            className="mt-4 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
          >
            Try Again
          </button>
        </div>
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700">Review</h4>
          {questions.map((q, i) => {
            const userAns = answers[i];
            const correct = userAns === q.correct_answer;
            return (
              <div
                key={i}
                className={`rounded-lg border p-3 ${
                  correct ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"
                }`}
              >
                <div className="flex items-start gap-2">
                  {correct ? (
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{q.question}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      Your answer: {q.options[userAns ?? 0]}
                      {!correct && ` | Correct: ${q.options[q.correct_answer]}`}
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
      <div className="flex items-start gap-3 rounded-lg border border-green-200 bg-green-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
        <p className="text-sm text-green-800">
          <span className="font-semibold">Test your knowledge</span> about R² Score! Answer{" "}
          {questions.length} questions to check your understanding.
        </p>
      </div>

      <div className="flex items-center justify-center gap-2">
        {questions.map((_, i) => (
          <div
            key={i}
            className={`w-2.5 h-2.5 rounded-full transition-all ${
              i === currentQ
                ? "bg-green-500 scale-125"
                : i < answers.length
                  ? answers[i] === questions[i].correct_answer
                    ? "bg-green-400"
                    : "bg-red-400"
                  : "bg-gray-300"
            }`}
          />
        ))}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <p className="text-xs text-gray-400 mb-2">
          Question {currentQ + 1} of {questions.length}
        </p>
        <p className="text-base font-medium text-gray-900 mb-4">{question.question}</p>

        <div className="space-y-2">
          {question.options.map((opt, idx) => {
            let style = "border-gray-200 bg-white hover:bg-gray-50 text-gray-700";
            if (answered) {
              if (idx === question.correct_answer) {
                style = "border-green-300 bg-green-50 text-green-800";
              } else if (idx === selectedAnswer && !isCorrect) {
                style = "border-red-300 bg-red-50 text-red-800";
              } else {
                style = "border-gray-200 bg-gray-50 text-gray-400";
              }
            }

            return (
              <button
                key={idx}
                onClick={() => handleSelect(idx)}
                disabled={answered}
                className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-colors ${style} disabled:cursor-default`}
              >
                <span className="font-medium mr-2">{String.fromCharCode(65 + idx)}.</span>
                {opt}
              </button>
            );
          })}
        </div>

        {answered && (
          <div
            className={`mt-4 p-3 rounded-lg text-sm ${
              isCorrect ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
            }`}
          >
            <p className="font-semibold mb-1">{isCorrect ? "Correct!" : "Not quite."}</p>
            <p>{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
          >
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
      <div className="flex items-start gap-3 rounded-lg border border-green-200 bg-green-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
        <div className="text-sm text-green-800 space-y-2">
          <p className="font-semibold">How R² Score Works</p>
          <ol className="list-decimal list-inside space-y-1">
            <li>Calculate <strong>SS_tot</strong> — total variance of the actual values around their mean</li>
            <li>Calculate <strong>SS_res</strong> — sum of squared differences between actual and predicted values</li>
            <li>Compute <strong>R² = 1 - (SS_res / SS_tot)</strong></li>
            <li>If SS_res = 0 (perfect predictions), R² = 1</li>
            <li>If SS_res = SS_tot (predictions equal the mean), R² = 0</li>
          </ol>
        </div>
      </div>

      {/* Formula */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">The Formula</h4>
        <div className="bg-gray-50 rounded-lg p-4 inline-block">
          <p className="text-lg font-mono text-gray-800">
            R² = 1 - (SS<sub>res</sub> / SS<sub>tot</sub>)
          </p>
          <div className="mt-3 grid grid-cols-2 gap-4 text-xs text-gray-600">
            <div>
              <strong className="text-gray-800">SS<sub>res</sub></strong> = &Sigma;(y<sub>i</sub> - y&#770;<sub>i</sub>)²
              <p className="mt-1">Sum of squared residuals (prediction errors)</p>
            </div>
            <div>
              <strong className="text-gray-800">SS<sub>tot</sub></strong> = &Sigma;(y<sub>i</sub> - y&#772;)²
              <p className="mt-1">Total sum of squares (variance around mean)</p>
            </div>
          </div>
        </div>
      </div>

      {/* Concept cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-green-600" />
            Variance Explained
          </h4>
          <p className="text-sm text-gray-600">
            R² tells you what fraction of the total variability in your target variable your model can
            account for. Think of it as a "percentage of pattern captured".
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-blue-600" />
            Baseline Comparison
          </h4>
          <p className="text-sm text-gray-600">
            R² compares your model against a "dumb" baseline that always predicts the mean. R²=0 means
            your model is no better than this baseline. R²=1 means perfect predictions.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Info className="w-4 h-4 text-orange-600" />
            Limitations
          </h4>
          <p className="text-sm text-gray-600">
            R² always increases when you add more features (on training data), even irrelevant ones.
            Use <strong>Adjusted R²</strong> to account for this. Also, R² doesn't tell you the
            magnitude of errors — pair it with RMSE or MAE.
          </p>
        </div>
      </div>
    </div>
  );
}
