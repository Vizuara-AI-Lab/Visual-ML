/**
 * Confusion Matrix Explorer — tabbed interactive explorer
 * Tabs: Results | Threshold Explorer | Real-World Scenarios | Quiz | How It Works
 */

import { useState, useMemo } from "react";
import {
  ClipboardList,
  SlidersHorizontal,
  Globe,
  HelpCircle,
  Cog,
  Trophy,
  CheckCircle,
  XCircle,
  Heart,
  Mail,
  Shield,
  Target,
  AlertTriangle,
  Info,
  Scale,
} from "lucide-react";
import { ConfusionMatrixResult } from "./ConfusionMatrixResult";

// =========================================================================
// Types
// =========================================================================

type CMTab = "results" | "threshold" | "scenarios" | "quiz" | "how_it_works";

interface ConfusionMatrixExplorerProps {
  result: Record<string, unknown>;
}

interface Sample {
  trueLabel: 0 | 1;
  predictedProb: number;
}

// =========================================================================
// Utilities
// =========================================================================

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function gaussianSample(rand: () => number, mean: number, std: number): number {
  const u1 = rand();
  const u2 = rand();
  const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

function generateSamples(seed: number, n = 200): Sample[] {
  const rand = seededRandom(seed);
  const samples: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const trueLabel = rand() < 0.4 ? 1 : 0;
    const prob = trueLabel === 1
      ? Math.min(1, Math.max(0, gaussianSample(rand, 0.65, 0.18)))
      : Math.min(1, Math.max(0, gaussianSample(rand, 0.35, 0.18)));
    samples.push({ trueLabel: trueLabel as 0 | 1, predictedProb: prob });
  }
  return samples;
}

function computeMetrics(samples: Sample[], threshold: number) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const s of samples) {
    const pred = s.predictedProb >= threshold ? 1 : 0;
    if (pred === 1 && s.trueLabel === 1) tp++;
    else if (pred === 1 && s.trueLabel === 0) fp++;
    else if (pred === 0 && s.trueLabel === 0) tn++;
    else fn++;
  }
  const accuracy = (tp + tn) / (tp + fp + tn + fn) || 0;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  const specificity = tn / (tn + fp) || 0;
  return { tp, fp, tn, fn, accuracy, precision, recall, f1, specificity };
}

// =========================================================================
// Main Explorer
// =========================================================================

export function ConfusionMatrixExplorer({ result }: ConfusionMatrixExplorerProps) {
  const [activeTab, setActiveTab] = useState<CMTab>("results");

  const tabs: { id: CMTab; label: string; icon: any }[] = [
    { id: "results", label: "Results", icon: ClipboardList },
    { id: "threshold", label: "Threshold Explorer", icon: SlidersHorizontal },
    { id: "scenarios", label: "Real-World Scenarios", icon: Globe },
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
                  ? "border-violet-500 text-violet-700"
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
        {activeTab === "results" && <ConfusionMatrixResult result={result} />}
        {activeTab === "threshold" && <ThresholdExplorerTab />}
        {activeTab === "scenarios" && <ScenariosTab />}
        {activeTab === "quiz" && <QuizTab />}
        {activeTab === "how_it_works" && <HowItWorksTab />}
      </div>
    </div>
  );
}

// =========================================================================
// Threshold Explorer Tab
// =========================================================================

function ThresholdExplorerTab() {
  const [threshold, setThreshold] = useState(0.5);
  const samples = useMemo(() => generateSamples(42), []);
  const metrics = useMemo(() => computeMetrics(samples, threshold), [samples, threshold]);

  // PR curve data
  const prCurve = useMemo(() => {
    const points: { threshold: number; precision: number; recall: number }[] = [];
    for (let t = 0.05; t <= 0.95; t += 0.05) {
      const m = computeMetrics(samples, t);
      points.push({ threshold: t, precision: m.precision, recall: m.recall });
    }
    return points;
  }, [samples]);

  const getCellColor = (value: number, total: number, isCorrect: boolean) => {
    const ratio = total > 0 ? value / total : 0;
    if (isCorrect) return `rgba(16, 185, 129, ${0.15 + ratio * 0.55})`;
    if (value === 0) return "rgba(241, 245, 249, 0.5)";
    return `rgba(239, 68, 68, ${0.1 + ratio * 0.4})`;
  };

  const total = metrics.tp + metrics.fp + metrics.tn + metrics.fn;

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <SlidersHorizontal className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <p className="text-sm text-violet-800">
          <span className="font-semibold">Adjust the threshold</span> to see how it affects the confusion
          matrix and all derived metrics. A higher threshold means fewer positives (higher precision, lower recall).
        </p>
      </div>

      {/* Threshold Slider */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-semibold text-gray-900">
            Classification Threshold: <span className="text-violet-600">{threshold.toFixed(2)}</span>
          </h4>
        </div>
        <input
          type="range"
          min={0.05}
          max={0.95}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full accent-violet-500"
        />
        <div className="flex justify-between text-[10px] text-gray-400 mt-1">
          <span>More Positives (High Recall)</span>
          <span>Fewer Positives (High Precision)</span>
        </div>
      </div>

      {/* Live Confusion Matrix + Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Mini matrix */}
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Confusion Matrix</h4>
          <div className="overflow-x-auto">
            <table className="mx-auto border-separate" style={{ borderSpacing: "4px" }}>
              <thead>
                <tr>
                  <th className="w-16" />
                  <th className="text-center text-[10px] font-bold text-indigo-600 uppercase px-2">Pred +</th>
                  <th className="text-center text-[10px] font-bold text-indigo-600 uppercase px-2">Pred -</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="text-right text-[10px] font-bold text-purple-600 uppercase pr-2">Actual +</td>
                  <td
                    className="text-center rounded-lg p-3 min-w-[72px]"
                    style={{ backgroundColor: getCellColor(metrics.tp, total, true) }}
                  >
                    <div className="text-lg font-bold text-emerald-800">{metrics.tp}</div>
                    <div className="text-[9px] text-emerald-600 font-medium">TP</div>
                  </td>
                  <td
                    className="text-center rounded-lg p-3 min-w-[72px]"
                    style={{ backgroundColor: getCellColor(metrics.fn, total, false) }}
                  >
                    <div className="text-lg font-bold text-red-800">{metrics.fn}</div>
                    <div className="text-[9px] text-red-600 font-medium">FN</div>
                  </td>
                </tr>
                <tr>
                  <td className="text-right text-[10px] font-bold text-purple-600 uppercase pr-2">Actual -</td>
                  <td
                    className="text-center rounded-lg p-3 min-w-[72px]"
                    style={{ backgroundColor: getCellColor(metrics.fp, total, false) }}
                  >
                    <div className="text-lg font-bold text-red-800">{metrics.fp}</div>
                    <div className="text-[9px] text-red-600 font-medium">FP</div>
                  </td>
                  <td
                    className="text-center rounded-lg p-3 min-w-[72px]"
                    style={{ backgroundColor: getCellColor(metrics.tn, total, true) }}
                  >
                    <div className="text-lg font-bold text-emerald-800">{metrics.tn}</div>
                    <div className="text-[9px] text-emerald-600 font-medium">TN</div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Metrics bars */}
        <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-3">
          <h4 className="text-sm font-semibold text-gray-900 mb-1">Metrics</h4>
          {[
            { label: "Accuracy", value: metrics.accuracy, color: "bg-violet-500" },
            { label: "Precision", value: metrics.precision, color: "bg-blue-500" },
            { label: "Recall", value: metrics.recall, color: "bg-emerald-500" },
            { label: "F1 Score", value: metrics.f1, color: "bg-amber-500" },
            { label: "Specificity", value: metrics.specificity, color: "bg-sky-500" },
          ].map((m) => (
            <div key={m.label}>
              <div className="flex justify-between text-xs mb-0.5">
                <span className="font-medium text-gray-700">{m.label}</span>
                <span className="font-bold text-gray-900">{(m.value * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                <div
                  className={`h-full rounded-full ${m.color} transition-all duration-300`}
                  style={{ width: `${m.value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Precision-Recall Curve */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Precision-Recall Tradeoff</h4>
        <svg viewBox="0 0 460 200" className="w-full">
          {/* Grid */}
          <line x1={40} y1={180} x2={440} y2={180} stroke="#e2e8f0" strokeWidth={1} />
          <line x1={40} y1={10} x2={40} y2={180} stroke="#e2e8f0" strokeWidth={1} />
          {[0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={40} y1={180 - v * 170} x2={440} y2={180 - v * 170} stroke="#f1f5f9" strokeWidth={1} />
              <text x={35} y={180 - v * 170 + 3} textAnchor="end" className="text-[8px] fill-gray-400">{(v * 100).toFixed(0)}%</text>
            </g>
          ))}

          {/* Precision line (blue) */}
          <polyline
            points={prCurve.map((p) => `${40 + p.threshold * 400},${180 - p.precision * 170}`).join(" ")}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={2}
          />
          {/* Recall line (green) */}
          <polyline
            points={prCurve.map((p) => `${40 + p.threshold * 400},${180 - p.recall * 170}`).join(" ")}
            fill="none"
            stroke="#10b981"
            strokeWidth={2}
          />

          {/* Current threshold marker */}
          <line
            x1={40 + threshold * 400}
            y1={10}
            x2={40 + threshold * 400}
            y2={180}
            stroke="#8b5cf6"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          <circle
            cx={40 + threshold * 400}
            cy={180 - metrics.precision * 170}
            r={4}
            fill="#3b82f6"
            stroke="white"
            strokeWidth={1.5}
          />
          <circle
            cx={40 + threshold * 400}
            cy={180 - metrics.recall * 170}
            r={4}
            fill="#10b981"
            stroke="white"
            strokeWidth={1.5}
          />

          {/* Legend */}
          <line x1={60} y1={15} x2={80} y2={15} stroke="#3b82f6" strokeWidth={2} />
          <text x={84} y={18} className="text-[9px] fill-gray-600">Precision</text>
          <line x1={160} y1={15} x2={180} y2={15} stroke="#10b981" strokeWidth={2} />
          <text x={184} y={18} className="text-[9px] fill-gray-600">Recall</text>

          {/* Axis labels */}
          <text x={240} y={198} textAnchor="middle" className="text-[9px] fill-gray-500">Threshold</text>
        </svg>
      </div>
    </div>
  );
}

// =========================================================================
// Real-World Scenarios Tab
// =========================================================================

function ScenariosTab() {
  const [activeScenario, setActiveScenario] = useState<"medical" | "spam" | "fraud">("medical");

  const scenarios = {
    medical: {
      icon: Heart,
      color: "red",
      title: "Medical Diagnosis",
      subtitle: "Detecting a rare disease affecting 1% of patients",
      description: "A hospital uses an ML model to screen patients for a rare disease.",
      stats: {
        population: 10000,
        positive: 100,
        negative: 9900,
        // "Always predict negative" baseline
        baselineAccuracy: 99,
        baselineRecall: 0,
        // Model predictions
        modelTP: 90, modelFP: 495, modelFN: 10, modelTN: 9405,
      },
      keyInsight: "99% accuracy sounds great, but a model that always says 'healthy' also gets 99%! What matters is Recall — catching the 100 sick patients. Missing even 10 (FN=10) could be life-threatening.",
      metricFocus: "Recall (Sensitivity) is critical. Missing a disease (FN) is much worse than a false alarm (FP).",
    },
    spam: {
      icon: Mail,
      color: "blue",
      title: "Spam Filter",
      subtitle: "Classifying emails as spam or not-spam",
      description: "An email service uses ML to filter spam. 30% of emails are spam.",
      stats: {
        population: 1000,
        positive: 300,
        negative: 700,
        baselineAccuracy: 70,
        baselineRecall: 0,
        modelTP: 270, modelFP: 35, modelFN: 30, modelTN: 665,
      },
      keyInsight: "Here, False Positives matter too! If a real email gets flagged as spam (FP=35), the user misses important messages. You need a balance between Precision and Recall.",
      metricFocus: "Precision is important. Sending legitimate emails to spam (FP) frustrates users.",
    },
    fraud: {
      icon: Shield,
      color: "violet",
      title: "Fraud Detection",
      subtitle: "Detecting fraudulent transactions (0.1% fraud rate)",
      description: "A bank flags potentially fraudulent credit card transactions.",
      stats: {
        population: 100000,
        positive: 100,
        negative: 99900,
        baselineAccuracy: 99.9,
        baselineRecall: 0,
        modelTP: 85, modelFP: 999, modelFN: 15, modelTN: 98901,
      },
      keyInsight: "With 0.1% fraud rate, accuracy is meaningless (99.9% by always predicting legit). The model catches 85 of 100 frauds but flags 999 legitimate transactions for review. The tradeoff is cost of review vs cost of missed fraud.",
      metricFocus: "Recall + Precision tradeoff. Missing fraud (FN) loses money; too many false alarms (FP) waste investigation resources.",
    },
  };

  const s = scenarios[activeScenario];
  const Icon = s.icon;
  const colorMap: Record<string, { bg: string; border: string; text: string; badge: string }> = {
    red: { bg: "bg-red-50", border: "border-red-200", text: "text-red-800", badge: "bg-red-100 text-red-700" },
    blue: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-800", badge: "bg-blue-100 text-blue-700" },
    violet: { bg: "bg-violet-50", border: "border-violet-200", text: "text-violet-800", badge: "bg-violet-100 text-violet-700" },
  };
  const c = colorMap[s.color];

  const modelAccuracy = ((s.stats.modelTP + s.stats.modelTN) / s.stats.population * 100).toFixed(1);
  const modelPrecision = (s.stats.modelTP / (s.stats.modelTP + s.stats.modelFP) * 100).toFixed(1);
  const modelRecall = (s.stats.modelTP / (s.stats.modelTP + s.stats.modelFN) * 100).toFixed(1);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <Globe className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <p className="text-sm text-violet-800">
          <span className="font-semibold">Explore real-world scenarios</span> where the confusion matrix
          reveals things that accuracy alone cannot. Click each scenario to learn the tradeoffs.
        </p>
      </div>

      {/* Scenario tabs */}
      <div className="flex gap-2">
        {(Object.entries(scenarios) as [typeof activeScenario, typeof s][]).map(([key, scenario]) => {
          const SIcon = scenario.icon;
          return (
            <button
              key={key}
              onClick={() => setActiveScenario(key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${
                activeScenario === key
                  ? "bg-violet-600 text-white border-violet-600"
                  : "bg-white text-gray-700 border-gray-200 hover:bg-gray-50"
              }`}
            >
              <SIcon className="w-4 h-4" />
              {scenario.title}
            </button>
          );
        })}
      </div>

      {/* Active scenario */}
      <div className={`${c.bg} ${c.border} border rounded-xl p-5`}>
        <div className="flex items-center gap-3 mb-3">
          <div className={`w-10 h-10 rounded-lg ${c.badge} flex items-center justify-center`}>
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <h3 className={`text-lg font-bold ${c.text}`}>{s.title}</h3>
            <p className="text-xs text-gray-600">{s.subtitle}</p>
          </div>
        </div>
        <p className="text-sm text-gray-700 mb-4">{s.description}</p>

        {/* Baseline vs Model comparison */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-white rounded-lg border border-gray-200 p-3">
            <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">
              "Always predict negative" baseline
            </div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Accuracy:</span>
                <span className="font-bold text-green-600">{s.stats.baselineAccuracy}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Recall:</span>
                <span className="font-bold text-red-600">{s.stats.baselineRecall}%</span>
              </div>
              <p className="text-[10px] text-gray-400 mt-1">Misses ALL positive cases!</p>
            </div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-3">
            <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">ML Model</div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Accuracy:</span>
                <span className="font-bold">{modelAccuracy}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Precision:</span>
                <span className="font-bold">{modelPrecision}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Recall:</span>
                <span className="font-bold text-green-600">{modelRecall}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Model confusion matrix */}
        <div className="bg-white rounded-lg border border-gray-200 p-3 mb-4">
          <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">Model Confusion Matrix</div>
          <div className="grid grid-cols-4 gap-2 text-center text-sm">
            <div className="bg-emerald-50 border border-emerald-200 rounded p-2">
              <div className="text-lg font-bold text-emerald-700">{s.stats.modelTP}</div>
              <div className="text-[9px] text-emerald-600">TP</div>
            </div>
            <div className="bg-red-50 border border-red-200 rounded p-2">
              <div className="text-lg font-bold text-red-700">{s.stats.modelFP}</div>
              <div className="text-[9px] text-red-600">FP</div>
            </div>
            <div className="bg-red-50 border border-red-200 rounded p-2">
              <div className="text-lg font-bold text-red-700">{s.stats.modelFN}</div>
              <div className="text-[9px] text-red-600">FN</div>
            </div>
            <div className="bg-emerald-50 border border-emerald-200 rounded p-2">
              <div className="text-lg font-bold text-emerald-700">{s.stats.modelTN}</div>
              <div className="text-[9px] text-emerald-600">TN</div>
            </div>
          </div>
        </div>

        {/* Key insight */}
        <div className="bg-white rounded-lg border border-yellow-200 p-3">
          <h4 className="text-sm font-bold text-yellow-800 flex items-center gap-1 mb-1">
            <AlertTriangle className="w-3.5 h-3.5" /> Key Insight
          </h4>
          <p className="text-sm text-gray-700">{s.keyInsight}</p>
          <p className="text-xs text-gray-500 mt-2 italic">{s.metricFocus}</p>
        </div>
      </div>
    </div>
  );
}

// =========================================================================
// Quiz Tab
// =========================================================================

const CM_QUIZ = [
  {
    question: "What does a False Positive (FP) represent?",
    options: [
      "Correctly predicted the positive class",
      "Incorrectly predicted positive when actual was negative",
      "Incorrectly predicted negative when actual was positive",
      "Correctly predicted the negative class",
    ],
    correct_answer: 1,
    explanation: "A False Positive (Type I error) is a false alarm — the model says 'positive' but the truth is 'negative'.",
  },
  {
    question: "A medical test has 95% accuracy on a disease affecting 1% of people. If the test says 'positive', what is the approximate precision?",
    options: ["About 95%", "About 50%", "About 16%", "About 1%"],
    correct_answer: 2,
    explanation: "With 1000 people: 10 sick (9.5 detected), 990 healthy (49.5 false alarms). Precision = 9.5/(9.5+49.5) ≈ 16%. This is the base rate fallacy.",
  },
  {
    question: "When should you prioritize Recall over Precision?",
    options: [
      "When false positives are very costly",
      "When false negatives are very costly (can't miss positives)",
      "When the dataset is balanced",
      "When you want the simplest model",
    ],
    correct_answer: 1,
    explanation: "High recall means catching all positives, even at the cost of false alarms. Critical in medical diagnosis, fraud detection, and security.",
  },
  {
    question: "What is the F1 Score?",
    options: [
      "Average of precision and recall",
      "Harmonic mean of precision and recall",
      "Product of precision and recall",
      "Maximum of precision and recall",
    ],
    correct_answer: 1,
    explanation: "F1 = 2 × (P × R) / (P + R). The harmonic mean penalizes extreme imbalances — if either P or R is very low, F1 is low too.",
  },
  {
    question: "Raising the threshold from 0.5 to 0.8 will generally...",
    options: [
      "Increase both precision and recall",
      "Increase precision but decrease recall",
      "Decrease precision but increase recall",
      "Have no effect",
    ],
    correct_answer: 1,
    explanation: "A higher threshold requires more confidence to predict 'positive'. This reduces false positives (higher precision) but misses some true positives (lower recall).",
  },
];

function QuizTab() {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = CM_QUIZ;
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
            {pct >= 80 ? "Excellent! You understand confusion matrices!" : pct >= 50 ? "Good job! Review what you missed." : "Keep learning! Confusion matrices are fundamental."}
          </p>
          <button onClick={handleRetry} className="mt-4 px-6 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm font-medium">Try Again</button>
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
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <p className="text-sm text-violet-800"><span className="font-semibold">Test your knowledge</span> about Confusion Matrices!</p>
      </div>
      <div className="flex items-center justify-center gap-2">
        {questions.map((_, i) => (
          <div key={i} className={`w-2.5 h-2.5 rounded-full transition-all ${i === currentQ ? "bg-violet-500 scale-125" : i < answers.length ? (answers[i] === questions[i].correct_answer ? "bg-green-400" : "bg-red-400") : "bg-gray-300"}`} />
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
          <button onClick={handleNext} className="mt-4 px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm font-medium">
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
      <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3">
        <Cog className="mt-0.5 h-5 w-5 shrink-0 text-violet-600" />
        <div className="text-sm text-violet-800 space-y-2">
          <p className="font-semibold">How Confusion Matrices Work</p>
          <ol className="list-decimal list-inside space-y-1">
            <li>Compare each prediction to the actual label</li>
            <li>Count into 4 buckets: TP, FP, TN, FN</li>
            <li>Derive metrics: Accuracy, Precision, Recall, F1</li>
          </ol>
        </div>
      </div>

      {/* Visual matrix explanation */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">The 2x2 Matrix</h4>
        <div className="grid grid-cols-2 gap-3 max-w-md mx-auto">
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-center">
            <div className="text-sm font-bold text-emerald-800">True Positive (TP)</div>
            <p className="text-[10px] text-emerald-600 mt-1">Predicted + | Actual +</p>
            <p className="text-[10px] text-gray-500 mt-1">Correct detection</p>
          </div>
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-center">
            <div className="text-sm font-bold text-red-800">False Negative (FN)</div>
            <p className="text-[10px] text-red-600 mt-1">Predicted - | Actual +</p>
            <p className="text-[10px] text-gray-500 mt-1">Missed case (Type II)</p>
          </div>
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-center">
            <div className="text-sm font-bold text-red-800">False Positive (FP)</div>
            <p className="text-[10px] text-red-600 mt-1">Predicted + | Actual -</p>
            <p className="text-[10px] text-gray-500 mt-1">False alarm (Type I)</p>
          </div>
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-center">
            <div className="text-sm font-bold text-emerald-800">True Negative (TN)</div>
            <p className="text-[10px] text-emerald-600 mt-1">Predicted - | Actual -</p>
            <p className="text-[10px] text-gray-500 mt-1">Correct rejection</p>
          </div>
        </div>
      </div>

      {/* Formula cards */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Derived Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <div className="text-xs font-bold text-gray-700 mb-1">Accuracy</div>
            <div className="text-[10px] font-mono text-gray-600">(TP+TN) / Total</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <div className="text-xs font-bold text-gray-700 mb-1">Precision</div>
            <div className="text-[10px] font-mono text-gray-600">TP / (TP+FP)</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <div className="text-xs font-bold text-gray-700 mb-1">Recall</div>
            <div className="text-[10px] font-mono text-gray-600">TP / (TP+FN)</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <div className="text-xs font-bold text-gray-700 mb-1">F1 Score</div>
            <div className="text-[10px] font-mono text-gray-600">2PR / (P+R)</div>
          </div>
        </div>
      </div>

      {/* Concept cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Target className="w-4 h-4 text-red-600" />
            Type I vs Type II Errors
          </h4>
          <p className="text-sm text-gray-600">
            <strong>Type I (FP):</strong> False alarm — you said "yes" when it was "no".
            <br />
            <strong>Type II (FN):</strong> Missed detection — you said "no" when it was "yes".
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Scale className="w-4 h-4 text-violet-600" />
            The Threshold Tradeoff
          </h4>
          <p className="text-sm text-gray-600">
            Lowering the threshold catches more positives (higher recall) but also creates more false alarms
            (lower precision). The right balance depends on your use case.
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Info className="w-4 h-4 text-blue-600" />
            When to Use
          </h4>
          <p className="text-sm text-gray-600">
            Use confusion matrices for any classification problem. Essential when classes are imbalanced
            (accuracy is misleading) or when different types of errors have different costs.
          </p>
        </div>
      </div>
    </div>
  );
}
