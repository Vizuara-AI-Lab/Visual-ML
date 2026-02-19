/**
 * Confusion Matrix Result Visualization Component
 */

import { useState } from "react";

interface ConfusionMatrixProps {
  result: Record<string, unknown>;
}

export function ConfusionMatrixResult({ result }: ConfusionMatrixProps) {
  const confusionMatrix = (result.confusion_matrix as number[][]) ?? [];
  const classLabels = (result.class_labels as string[]) ?? [];
  const totalSamples = (result.total_samples as number) ?? 0;
  const accuracy = (result.accuracy as number) ?? 0;
  const truePositives =
    (result.true_positives as Record<string, number>) ?? {};
  const falsePositives =
    (result.false_positives as Record<string, number>) ?? {};
  const trueNegatives =
    (result.true_negatives as Record<string, number>) ?? {};
  const falseNegatives =
    (result.false_negatives as Record<string, number>) ?? {};

  const [showPercentages, setShowPercentages] = useState(true);
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);

  // Calculate maximum value for color scaling
  const allValues = confusionMatrix.flat().filter((v) => !isNaN(v));
  const maxValue = Math.max(...allValues, 1);

  // Total correct predictions
  const correctPredictions = confusionMatrix.reduce(
    (sum, row, i) => sum + (row[i] ?? 0),
    0,
  );
  const incorrectPredictions = totalSamples - correctPredictions;

  // Get color for matrix cell
  const getCellStyle = (
    value: number,
    isCorrect: boolean,
  ): React.CSSProperties => {
    const ratio = maxValue > 0 ? value / maxValue : 0;
    if (isCorrect) {
      // Diagonal cells: green scale
      const alpha = 0.15 + ratio * 0.55;
      return { backgroundColor: `rgba(16, 185, 129, ${alpha})` };
    }
    // Off-diagonal cells: red scale (only if value > 0)
    if (value === 0) return { backgroundColor: "rgba(241, 245, 249, 0.5)" };
    const alpha = 0.1 + ratio * 0.4;
    return { backgroundColor: `rgba(239, 68, 68, ${alpha})` };
  };

  if (
    !confusionMatrix ||
    confusionMatrix.length === 0 ||
    !classLabels ||
    classLabels.length === 0
  ) {
    return (
      <div className="p-10 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
          <svg
            className="w-8 h-8 text-slate-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M3 10h18M3 14h18M10 3v18M14 3v18"
            />
          </svg>
        </div>
        <p className="text-slate-600 font-medium mb-1">
          No Confusion Matrix Data
        </p>
        <p className="text-slate-400 text-sm">
          Run the pipeline to generate confusion matrix results
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-linear-to-br from-violet-50 to-indigo-50 rounded-xl p-5 border border-violet-200">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-violet-500 flex items-center justify-center shadow-md">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 10h18M3 14h18M10 3v18M14 3v18"
              />
            </svg>
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-bold text-violet-900">
              Confusion Matrix
            </h3>
            <p className="text-sm text-violet-600">
              {classLabels.length} classes &middot;{" "}
              {totalSamples.toLocaleString()} samples evaluated
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-1">
            Accuracy
          </div>
          <div className="text-2xl font-bold text-violet-600">
            {(accuracy * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-1">
            Total Samples
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {totalSamples.toLocaleString()}
          </div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-1">
            Correct
          </div>
          <div className="text-2xl font-bold text-emerald-600">
            {correctPredictions.toLocaleString()}
          </div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4 text-center">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-1">
            Incorrect
          </div>
          <div className="text-2xl font-bold text-red-500">
            {incorrectPredictions.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Accuracy Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-600">
            Prediction Accuracy
          </span>
          <span className="text-xs font-bold text-violet-700">
            {(accuracy * 100).toFixed(2)}%
          </span>
        </div>
        <div className="h-3 rounded-full overflow-hidden flex bg-slate-100">
          <div
            className="bg-emerald-500 transition-all duration-500 rounded-l-full"
            style={{ width: `${accuracy * 100}%` }}
          />
          <div
            className="bg-red-400 transition-all duration-500"
            style={{ width: `${(1 - accuracy) * 100}%` }}
          />
        </div>
        <div className="flex justify-between mt-1.5">
          <span className="text-[10px] text-emerald-600 font-medium">
            Correct: {correctPredictions}
          </span>
          <span className="text-[10px] text-red-500 font-medium">
            Incorrect: {incorrectPredictions}
          </span>
        </div>
      </div>

      {/* Matrix Visualization */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-bold text-slate-800">
            Prediction Matrix
          </h4>
          <button
            onClick={() => setShowPercentages(!showPercentages)}
            className={`text-[11px] font-medium px-3 py-1.5 rounded-lg border transition-colors ${
              showPercentages
                ? "bg-violet-50 border-violet-200 text-violet-700"
                : "bg-slate-50 border-slate-200 text-slate-600"
            }`}
          >
            {showPercentages ? "Showing %" : "Showing counts"}
          </button>
        </div>

        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Predicted label */}
            <div className="flex items-center justify-center mb-2 ml-20">
              <span className="text-[10px] font-bold uppercase tracking-widest text-indigo-500">
                Predicted Class
              </span>
            </div>

            <div className="flex">
              {/* Actual label (vertical) */}
              <div className="flex items-center justify-center mr-2 w-5">
                <span
                  className="text-[10px] font-bold uppercase tracking-widest text-purple-500 whitespace-nowrap"
                  style={{
                    writingMode: "vertical-rl",
                    transform: "rotate(180deg)",
                  }}
                >
                  Actual Class
                </span>
              </div>

              <table className="border-separate" style={{ borderSpacing: "3px" }}>
                <thead>
                  <tr>
                    <th className="w-14" />
                    {classLabels.map((label) => (
                      <th
                        key={`pred-${label}`}
                        className="px-2 py-2 text-center"
                      >
                        <span className="text-[11px] font-semibold text-indigo-700 bg-indigo-50 px-2.5 py-1 rounded-md inline-block">
                          {String(label)}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {classLabels.map((actualLabel, i) => (
                    <tr key={`row-${actualLabel}`}>
                      <td className="px-2 py-2 text-right">
                        <span className="text-[11px] font-semibold text-purple-700 bg-purple-50 px-2.5 py-1 rounded-md inline-block">
                          {String(actualLabel)}
                        </span>
                      </td>
                      {classLabels.map((_, j) => {
                        const count = confusionMatrix[i]?.[j] ?? 0;
                        const percentage =
                          totalSamples > 0 ? (count / totalSamples) * 100 : 0;
                        const isCorrect = i === j;
                        const isHovered =
                          hoveredCell?.row === i && hoveredCell?.col === j;

                        return (
                          <td
                            key={`cell-${i}-${j}`}
                            className={`relative text-center rounded-lg transition-all duration-150 cursor-default ${
                              isHovered ? "ring-2 ring-violet-400 ring-offset-1" : ""
                            }`}
                            style={{
                              ...getCellStyle(count, isCorrect),
                              minWidth: "72px",
                              minHeight: "56px",
                            }}
                            onMouseEnter={() =>
                              setHoveredCell({ row: i, col: j })
                            }
                            onMouseLeave={() => setHoveredCell(null)}
                          >
                            <div className="flex flex-col items-center py-2.5 px-3">
                              <span
                                className={`text-lg font-bold ${
                                  isCorrect
                                    ? "text-emerald-800"
                                    : count > 0
                                      ? "text-red-800"
                                      : "text-slate-400"
                                }`}
                              >
                                {showPercentages
                                  ? `${percentage.toFixed(1)}%`
                                  : count.toLocaleString()}
                              </span>
                              {showPercentages && (
                                <span className="text-[10px] text-slate-500 mt-0.5">
                                  n={count}
                                </span>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Matrix legend */}
            <div className="flex items-center justify-center gap-6 mt-4 pt-3 border-t border-slate-100">
              <div className="flex items-center gap-1.5">
                <div className="w-3.5 h-3.5 rounded bg-emerald-300" />
                <span className="text-[10px] text-slate-500 font-medium">
                  Correct (diagonal)
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3.5 h-3.5 rounded bg-red-200" />
                <span className="text-[10px] text-slate-500 font-medium">
                  Misclassified
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3.5 h-3.5 rounded bg-slate-100 border border-slate-200" />
                <span className="text-[10px] text-slate-500 font-medium">
                  Zero
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Per-Class Performance Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <h4 className="text-sm font-bold text-slate-800 mb-4">
          Per-Class Performance
        </h4>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="border-b-2 border-slate-200">
                <th className="text-left text-[10px] font-bold text-slate-500 uppercase tracking-wider pb-3 pr-4">
                  Class
                </th>
                <th className="text-center text-[10px] font-bold text-slate-500 uppercase tracking-wider pb-3 px-3">
                  Precision
                </th>
                <th className="text-center text-[10px] font-bold text-slate-500 uppercase tracking-wider pb-3 px-3">
                  Recall
                </th>
                <th className="text-center text-[10px] font-bold text-slate-500 uppercase tracking-wider pb-3 px-3">
                  F1 Score
                </th>
                <th className="text-center text-[10px] font-bold text-emerald-600 uppercase tracking-wider pb-3 px-2">
                  TP
                </th>
                <th className="text-center text-[10px] font-bold text-red-500 uppercase tracking-wider pb-3 px-2">
                  FP
                </th>
                <th className="text-center text-[10px] font-bold text-sky-600 uppercase tracking-wider pb-3 px-2">
                  TN
                </th>
                <th className="text-center text-[10px] font-bold text-amber-600 uppercase tracking-wider pb-3 px-2">
                  FN
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {classLabels.map((label) => {
                const tp = truePositives[label] ?? 0;
                const fp = falsePositives[label] ?? 0;
                const tn = trueNegatives[label] ?? 0;
                const fn = falseNegatives[label] ?? 0;

                const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
                const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
                const f1 =
                  precision + recall > 0
                    ? (2 * precision * recall) / (precision + recall)
                    : 0;

                return (
                  <tr key={label} className="group hover:bg-slate-50/80">
                    <td className="py-3 pr-4">
                      <span className="text-sm font-semibold text-slate-800">
                        {String(label)}
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      <div className="flex flex-col items-center">
                        <span className="text-sm font-bold text-slate-800">
                          {(precision * 100).toFixed(1)}%
                        </span>
                        <div className="w-full mt-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-violet-400"
                            style={{ width: `${precision * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      <div className="flex flex-col items-center">
                        <span className="text-sm font-bold text-slate-800">
                          {(recall * 100).toFixed(1)}%
                        </span>
                        <div className="w-full mt-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-indigo-400"
                            style={{ width: `${recall * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      <div className="flex flex-col items-center">
                        <span className="text-sm font-bold text-slate-800">
                          {(f1 * 100).toFixed(1)}%
                        </span>
                        <div className="w-full mt-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-emerald-400"
                            style={{ width: `${f1 * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-2 text-center">
                      <span className="text-xs font-semibold text-emerald-700 bg-emerald-50 px-2 py-1 rounded">
                        {tp}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-center">
                      <span className="text-xs font-semibold text-red-700 bg-red-50 px-2 py-1 rounded">
                        {fp}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-center">
                      <span className="text-xs font-semibold text-sky-700 bg-sky-50 px-2 py-1 rounded">
                        {tn}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-center">
                      <span className="text-xs font-semibold text-amber-700 bg-amber-50 px-2 py-1 rounded">
                        {fn}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Understanding Guide */}
      <div className="bg-slate-50 rounded-xl border border-slate-200 p-4">
        <h4 className="text-xs font-bold text-slate-600 uppercase tracking-wider mb-3">
          Understanding the Metrics
        </h4>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-[11px]">
          <div className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400 mt-1 shrink-0" />
            <span className="text-slate-600">
              <strong className="text-slate-700">Precision</strong> &mdash; Of
              all predicted positives, how many are correct?
            </span>
          </div>
          <div className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1 shrink-0" />
            <span className="text-slate-600">
              <strong className="text-slate-700">Recall</strong> &mdash; Of all
              actual positives, how many were found?
            </span>
          </div>
          <div className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1 shrink-0" />
            <span className="text-slate-600">
              <strong className="text-slate-700">F1 Score</strong> &mdash;
              Harmonic mean of precision and recall
            </span>
          </div>
          <div className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-400 mt-1 shrink-0" />
            <span className="text-slate-600">
              <strong className="text-slate-700">Diagonal</strong> &mdash;
              Correct predictions (higher is better)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
