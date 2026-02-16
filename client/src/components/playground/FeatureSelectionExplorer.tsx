/**
 * FeatureSelectionExplorer - Interactive learning activities for Feature Selection node
 * Tabs: Results | Score Chart | Correlation Map | Threshold Simulator | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  BarChart3,
  Grid3X3,
  SlidersHorizontal,
  HelpCircle,
  CheckCircle,
  XCircle,
  Trophy,
  Info,
  ChevronRight,
  ClipboardList,
} from "lucide-react";

interface FeatureSelectionExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type ExplorerTab = "results" | "scores" | "correlation" | "threshold" | "quiz";

export const FeatureSelectionExplorer = ({ result, renderResults }: FeatureSelectionExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: { id: ExplorerTab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "scores",
      label: "Score Chart",
      icon: BarChart3,
      available: !!result.feature_score_details?.features && result.feature_score_details.features.length > 0,
    },
    {
      id: "correlation",
      label: "Correlation Map",
      icon: Grid3X3,
      available: !!result.feature_correlation_matrix?.columns && result.feature_correlation_matrix.columns.length >= 2,
    },
    {
      id: "threshold",
      label: "Threshold Simulator",
      icon: SlidersHorizontal,
      available: !!result.threshold_simulation_data?.features_sorted && result.threshold_simulation_data.features_sorted.length > 0,
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
                    ? "border-purple-600 text-purple-600"
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
      {activeTab === "scores" && result.feature_score_details && (
        <ScoreChartTab data={result.feature_score_details} />
      )}
      {activeTab === "correlation" && result.feature_correlation_matrix && (
        <CorrelationMapTab data={result.feature_correlation_matrix} />
      )}
      {activeTab === "threshold" && result.threshold_simulation_data && (
        <ThresholdSimulatorTab data={result.threshold_simulation_data} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Score Chart Tab ---
function ScoreChartTab({ data }: { data: any }) {
  const features = data.features || [];
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">Feature Scores</h3>
        <p className="text-xs text-purple-700 mt-1">
          Each feature scored by {data.method}. Green = selected, Red = removed.
          {data.selected_count !== undefined && (
            <span className="ml-1">({data.selected_count} selected, {data.removed_count} removed)</span>
          )}
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="space-y-1.5">
          {features.map((feat: any, idx: number) => (
            <div
              key={feat.name}
              className="flex items-center gap-3 group"
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              <div className="w-32 text-xs font-medium text-gray-700 truncate text-right flex-shrink-0" title={feat.name}>
                {feat.name}
              </div>
              <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden relative">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${feat.selected ? "bg-green-500" : "bg-red-400"}`}
                  style={{ width: `${Math.max(feat.bar_width_pct, 2)}%` }}
                />
              </div>
              <div className="w-20 text-xs font-mono text-gray-600 text-right flex-shrink-0">
                {feat.score}
              </div>
              <div className="w-16 flex-shrink-0">
                {feat.selected ? (
                  <span className="text-[10px] font-semibold text-green-700 bg-green-100 px-1.5 py-0.5 rounded">KEPT</span>
                ) : (
                  <span className="text-[10px] font-semibold text-red-700 bg-red-100 px-1.5 py-0.5 rounded">CUT</span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Tooltip for hovered feature */}
        {hoveredIdx !== null && features[hoveredIdx]?.stats && (
          <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="text-xs font-semibold text-gray-800 mb-1">{features[hoveredIdx].name} — Stats</div>
            <div className="grid grid-cols-4 gap-2 text-xs">
              {Object.entries(features[hoveredIdx].stats).map(([k, v]: [string, any]) => (
                <div key={k}>
                  <span className="text-gray-500 uppercase text-[9px]">{k}: </span>
                  <span className="font-mono font-semibold">{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Features with higher {data.method === "variance" ? "variance" : "correlation"} scores
            contain more useful information for the model. Features below the threshold are removed
            to simplify the model without losing much predictive power.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Correlation Map Tab ---
function CorrelationMapTab({ data }: { data: any }) {
  const { columns, matrix, highly_correlated_pairs, feature_status } = data;
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number } | null>(null);

  const getColor = (val: number) => {
    const abs = Math.abs(val);
    if (abs > 0.8) return val > 0 ? "bg-red-600 text-white" : "bg-blue-600 text-white";
    if (abs > 0.6) return val > 0 ? "bg-red-400 text-white" : "bg-blue-400 text-white";
    if (abs > 0.4) return val > 0 ? "bg-red-200 text-red-900" : "bg-blue-200 text-blue-900";
    if (abs > 0.2) return val > 0 ? "bg-red-100 text-red-800" : "bg-blue-100 text-blue-800";
    return "bg-gray-100 text-gray-600";
  };

  const cellSize = columns.length > 12 ? 28 : columns.length > 8 ? 36 : 44;

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">Correlation Map</h3>
        <p className="text-xs text-purple-700 mt-1">
          Shows how strongly each pair of features is related. High correlation = redundant information.
        </p>
      </div>

      {/* Matrix */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-auto">
        <div className="inline-block">
          {/* Header row */}
          <div className="flex" style={{ paddingLeft: cellSize + 4 }}>
            {columns.map((col: string, ci: number) => (
              <div
                key={ci}
                className="text-center overflow-hidden"
                style={{ width: cellSize, minWidth: cellSize }}
                title={col}
              >
                <div
                  className={`text-[8px] truncate font-medium ${
                    feature_status?.[col] === "removed" ? "text-red-600" : feature_status?.[col] === "selected" ? "text-green-700" : "text-gray-600"
                  }`}
                  style={{ maxWidth: cellSize }}
                >
                  {col.length > 5 ? col.substring(0, 4) + ".." : col}
                </div>
              </div>
            ))}
          </div>

          {/* Matrix rows */}
          {matrix.map((row: number[], ri: number) => (
            <div key={ri} className="flex items-center">
              <div
                className={`text-[8px] font-medium truncate text-right pr-1 ${
                  feature_status?.[columns[ri]] === "removed" ? "text-red-600" : feature_status?.[columns[ri]] === "selected" ? "text-green-700" : "text-gray-600"
                }`}
                style={{ width: cellSize, minWidth: cellSize }}
                title={columns[ri]}
              >
                {columns[ri]?.length > 6 ? columns[ri].substring(0, 5) + ".." : columns[ri]}
              </div>
              {row.map((val: number, ci: number) => (
                <div
                  key={ci}
                  className={`flex items-center justify-center text-[8px] font-mono ${getColor(val)} ${
                    hoveredCell?.r === ri && hoveredCell?.c === ci ? "ring-2 ring-gray-800" : ""
                  }`}
                  style={{ width: cellSize, height: cellSize, minWidth: cellSize }}
                  onMouseEnter={() => setHoveredCell({ r: ri, c: ci })}
                  onMouseLeave={() => setHoveredCell(null)}
                  title={`${columns[ri]} vs ${columns[ci]}: ${val}`}
                >
                  {cellSize >= 36 ? val.toFixed(2) : ""}
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Hovered cell info */}
        {hoveredCell && (
          <div className="mt-2 text-xs text-gray-700">
            <span className="font-semibold">{columns[hoveredCell.r]}</span> vs{" "}
            <span className="font-semibold">{columns[hoveredCell.c]}</span>:{" "}
            <span className="font-mono font-bold">{matrix[hoveredCell.r]?.[hoveredCell.c]?.toFixed(4)}</span>
          </div>
        )}

        {/* Legend */}
        <div className="flex items-center gap-3 mt-3 pt-2 border-t border-gray-200">
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-blue-600 rounded-sm" />
            <span className="text-[9px] text-gray-500">-1 (negative)</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-gray-100 rounded-sm" />
            <span className="text-[9px] text-gray-500">0 (none)</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-red-600 rounded-sm" />
            <span className="text-[9px] text-gray-500">+1 (positive)</span>
          </div>
          <span className="mx-2 text-gray-300">|</span>
          <span className="text-[9px] text-green-700 font-medium">Green = Selected</span>
          <span className="text-[9px] text-red-600 font-medium">Red = Removed</span>
        </div>
      </div>

      {/* Highly correlated pairs */}
      {highly_correlated_pairs && highly_correlated_pairs.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-gray-700 mb-3">Highly Correlated Pairs</h4>
          <div className="space-y-2">
            {highly_correlated_pairs.map((pair: any, idx: number) => (
              <div key={idx} className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg text-sm">
                <span className="font-medium text-gray-800">{pair.feature_a}</span>
                <span className="text-gray-400">&harr;</span>
                <span className="font-medium text-gray-800">{pair.feature_b}</span>
                <span className="ml-auto font-mono text-xs text-orange-700 bg-orange-100 px-2 py-0.5 rounded">
                  r = {pair.correlation}
                </span>
                {pair.which_removed && (
                  <span className="text-[10px] text-red-600 bg-red-50 px-1.5 py-0.5 rounded border border-red-200">
                    {pair.which_removed} removed
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            Correlation measures how two features move together (-1 to +1). Values near +1 or -1
            mean the features carry nearly identical information. Removing one of a highly correlated
            pair reduces redundancy without losing predictive power.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Threshold Simulator Tab ---
function ThresholdSimulatorTab({ data }: { data: any }) {
  const { features_sorted, applied_threshold, method, score_range } = data;
  const [threshold, setThreshold] = useState(applied_threshold);

  // For variance: features below threshold are removed
  // For correlation: features above threshold are removed
  const isVariance = method === "variance";
  const selected = features_sorted.filter((f: any) =>
    isVariance ? f.score >= threshold : f.score <= threshold
  );
  const removed = features_sorted.filter((f: any) =>
    isVariance ? f.score < threshold : f.score > threshold
  );

  const sliderMin = score_range?.min ?? 0;
  const sliderMax = score_range?.max ?? 1;
  const step = (sliderMax - sliderMin) / 100 || 0.01;

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">Threshold Simulator</h3>
        <p className="text-xs text-purple-700 mt-1">
          Drag the slider to see how changing the threshold affects which features are kept or removed.
        </p>
      </div>

      {/* Slider */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-600">Threshold</span>
          <span className="text-sm font-mono font-semibold text-purple-700">{threshold.toFixed(4)}</span>
        </div>
        <input
          type="range"
          min={sliderMin}
          max={sliderMax}
          step={step}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full accent-purple-600"
        />
        <div className="flex justify-between text-[10px] text-gray-400 mt-1">
          <span>{sliderMin.toFixed(4)}</span>
          <span className="text-purple-600 font-medium">Applied: {applied_threshold}</span>
          <span>{sliderMax.toFixed(4)}</span>
        </div>
      </div>

      {/* Counter */}
      <div className="flex gap-4">
        <div className="flex-1 bg-green-50 border border-green-200 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-green-700">{selected.length}</div>
          <div className="text-xs text-green-600">Selected</div>
        </div>
        <div className="flex-1 bg-red-50 border border-red-200 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-red-700">{removed.length}</div>
          <div className="text-xs text-red-600">Removed</div>
        </div>
      </div>

      {/* Two-column list */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-green-200 rounded-lg p-3">
          <h4 className="text-xs font-semibold text-green-800 mb-2">Selected Features</h4>
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {selected.length === 0 && <p className="text-xs text-gray-400 italic">None</p>}
            {selected.map((f: any) => (
              <div key={f.name} className="flex justify-between items-center px-2 py-1 bg-green-50 rounded text-xs">
                <span className="font-medium text-gray-800 truncate">{f.name}</span>
                <span className="font-mono text-green-700 ml-2">{f.score.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white border border-red-200 rounded-lg p-3">
          <h4 className="text-xs font-semibold text-red-800 mb-2">Removed Features</h4>
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {removed.length === 0 && <p className="text-xs text-gray-400 italic">None</p>}
            {removed.map((f: any) => (
              <div key={f.name} className="flex justify-between items-center px-2 py-1 bg-red-50 rounded text-xs">
                <span className="font-medium text-gray-800 truncate">{f.name}</span>
                <span className="font-mono text-red-700 ml-2">{f.score.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            {isVariance
              ? "For variance threshold: features with variance BELOW the threshold are removed (too little variation to be useful)."
              : "For correlation threshold: features with max correlation ABOVE the threshold are removed (too redundant)."}
            {" "}Try moving the slider to see how the threshold affects your feature set!
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
        <button onClick={handleRetry} className="w-full py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-sm font-medium">Try Again</button>
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
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">Knowledge Check</h3>
        <p className="text-xs text-purple-700 mt-1">Test your understanding of feature selection using questions from YOUR dataset.</p>
      </div>

      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div key={i} className={`w-3 h-3 rounded-full transition-all ${i === currentQ ? "bg-purple-600 scale-125" : i < currentQ ? (answers[i] === questions[i].correct_answer ? "bg-green-500" : "bg-red-400") : "bg-gray-300"}`} />
        ))}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="text-xs text-gray-500 mb-2">Question {currentQ + 1} of {questions.length}</div>
        <p className="text-sm font-medium text-gray-900 mb-4">{question.question}</p>
        <div className="space-y-2">
          {question.options.map((option: string, oi: number) => {
            let optionClass = "bg-white border-gray-300 hover:border-purple-400 cursor-pointer";
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
          <button onClick={handleNext} className="mt-4 w-full py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-sm font-medium flex items-center justify-center gap-2">
            {currentQ < questions.length - 1 ? (<>Next Question <ChevronRight className="w-4 h-4" /></>) : "See Results"}
          </button>
        )}
      </div>
    </div>
  );
}
