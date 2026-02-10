/**
 * Confusion Matrix Result Visualization Component
 */

interface ConfusionMatrixProps {
  result: any;
}

export function ConfusionMatrixResult({ result }: ConfusionMatrixProps) {
  // Debug logging
  console.log("ConfusionMatrixResult - Full result:", result);

  const confusionMatrix = result.confusion_matrix ?? [];
  const classLabels = result.class_labels ?? [];
  const totalSamples = result.total_samples ?? 0;
  const accuracy = result.accuracy ?? 0;
  const truePositives = result.true_positives ?? {};
  const falsePositives = result.false_positives ?? {};
  const trueNegatives = result.true_negatives ?? {};
  const falseNegatives = result.false_negatives ?? {};

  console.log("ConfusionMatrixResult - Parsed data:", {
    confusionMatrix,
    classLabels,
    totalSamples,
    accuracy,
    truePositives,
    falsePositives,
    trueNegatives,
    falseNegatives,
  });

  // Calculate maximum value for color scaling
  const maxValue = Math.max(
    ...confusionMatrix.flat().filter((v: number) => !isNaN(v)),
  );

  // Calculate percentage for each cell
  const getPercentage = (value: number): number => {
    return totalSamples > 0 ? (value / totalSamples) * 100 : 0;
  };

  // Get color intensity based on value
  const getColorIntensity = (value: number): string => {
    if (maxValue === 0) return "bg-gray-50";
    const intensity = Math.floor((value / maxValue) * 4);

    const colorMap: { [key: number]: string } = {
      0: "bg-blue-50",
      1: "bg-blue-100",
      2: "bg-blue-200",
      3: "bg-blue-300",
      4: "bg-blue-400 text-white",
    };

    return colorMap[intensity] || "bg-blue-50";
  };

  if (
    !confusionMatrix ||
    confusionMatrix.length === 0 ||
    !classLabels ||
    classLabels.length === 0
  ) {
    return (
      <div className="p-8 text-center">
        <div className="text-gray-400 mb-2">
          <svg
            className="w-16 h-16 mx-auto"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        </div>
        <p className="text-gray-600 font-medium mb-1">
          No Confusion Matrix Data
        </p>
        <p className="text-gray-500 text-sm">
          The confusion matrix computation completed, but no data was received.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overall Metrics Card */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-1">
              Overall Accuracy
            </h3>
            <p className="text-3xl font-bold text-blue-600">
              {(accuracy * 100).toFixed(2)}%
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Total Samples</p>
            <p className="text-2xl font-semibold text-gray-800">
              {totalSamples.toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          Confusion Matrix
        </h3>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            <table className="border-collapse border border-gray-300">
              <thead>
                <tr>
                  <th
                    className="border border-gray-300 bg-gray-100 px-4 py-2 text-center font-semibold"
                    rowSpan={2}
                    colSpan={2}
                  >
                    <div className="flex flex-col items-center justify-center h-full">
                      <span className="text-xs text-gray-500 mb-1">
                        Actual ↓
                      </span>
                      <span className="text-xs text-gray-500">Predicted →</span>
                    </div>
                  </th>
                  <th
                    className="border border-gray-300 bg-indigo-100 px-4 py-2 text-center font-semibold text-indigo-900"
                    colSpan={classLabels.length}
                  >
                    Predicted Class
                  </th>
                </tr>
                <tr>
                  {classLabels.map((label) => (
                    <th
                      key={`pred-${label}`}
                      className="border border-gray-300 bg-indigo-50 px-4 py-2 text-center font-medium text-indigo-800"
                    >
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {classLabels.map((actualLabel, i) => (
                  <tr key={`row-${actualLabel}`}>
                    {i === 0 && (
                      <th
                        className="border border-gray-300 bg-purple-100 px-4 py-2 text-center font-semibold text-purple-900"
                        rowSpan={classLabels.length}
                      >
                        <div className="flex items-center justify-center h-full">
                          <span className="transform -rotate-180 writing-mode-vertical">
                            Actual Class
                          </span>
                        </div>
                      </th>
                    )}
                    <th className="border border-gray-300 bg-purple-50 px-4 py-2 text-center font-medium text-purple-800">
                      {actualLabel}
                    </th>
                    {classLabels.map((_, j) => {
                      const count = confusionMatrix[i]?.[j] ?? 0;
                      const percentage = getPercentage(count);
                      const isCorrect = i === j;

                      return (
                        <td
                          key={`cell-${i}-${j}`}
                          className={`border border-gray-300 px-4 py-3 text-center ${
                            isCorrect
                              ? "bg-green-100 font-bold text-green-900"
                              : getColorIntensity(count)
                          }`}
                        >
                          <div className="flex flex-col items-center">
                            <span className="text-lg font-semibold">
                              {count.toLocaleString()}
                            </span>
                            <span
                              className={`text-xs ${
                                isCorrect ? "text-green-700" : "text-gray-600"
                              }`}
                            >
                              ({percentage.toFixed(1)}%)
                            </span>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Per-Class Metrics */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          Per-Class Metrics
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              <div
                key={label}
                className="border border-gray-200 rounded-lg p-4 bg-white shadow-sm hover:shadow-md transition-shadow"
              >
                <h4 className="font-semibold text-gray-800 mb-3 pb-2 border-b border-gray-200">
                  Class: {label}
                </h4>
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-green-50 rounded p-2">
                      <p className="text-xs text-green-700 font-medium">
                        True Positives
                      </p>
                      <p className="text-xl font-bold text-green-800">{tp}</p>
                    </div>
                    <div className="bg-red-50 rounded p-2">
                      <p className="text-xs text-red-700 font-medium">
                        False Positives
                      </p>
                      <p className="text-xl font-bold text-red-800">{fp}</p>
                    </div>
                    <div className="bg-blue-50 rounded p-2">
                      <p className="text-xs text-blue-700 font-medium">
                        True Negatives
                      </p>
                      <p className="text-xl font-bold text-blue-800">{tn}</p>
                    </div>
                    <div className="bg-orange-50 rounded p-2">
                      <p className="text-xs text-orange-700 font-medium">
                        False Negatives
                      </p>
                      <p className="text-xl font-bold text-orange-800">{fn}</p>
                    </div>
                  </div>
                  <div className="pt-2 border-t border-gray-200 space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Precision:</span>
                      <span className="font-semibold text-gray-800">
                        {(precision * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Recall:</span>
                      <span className="font-semibold text-gray-800">
                        {(recall * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">F1-Score:</span>
                      <span className="font-semibold text-gray-800">
                        {(f1 * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">
          📚 Understanding the Confusion Matrix
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div>
            <p className="font-medium text-green-700">✓ True Positive (TP)</p>
            <p className="text-gray-600 text-xs">
              Correctly predicted positive class
            </p>
          </div>
          <div>
            <p className="font-medium text-red-700">✗ False Positive (FP)</p>
            <p className="text-gray-600 text-xs">
              Incorrectly predicted positive (Type I error)
            </p>
          </div>
          <div>
            <p className="font-medium text-blue-700">✓ True Negative (TN)</p>
            <p className="text-gray-600 text-xs">
              Correctly predicted negative class
            </p>
          </div>
          <div>
            <p className="font-medium text-orange-700">✗ False Negative (FN)</p>
            <p className="text-gray-600 text-xs">
              Incorrectly predicted negative (Type II error)
            </p>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-gray-300 space-y-1 text-xs text-gray-600">
          <p>
            <strong>Precision:</strong> TP / (TP + FP) - How many predicted
            positives are correct?
          </p>
          <p>
            <strong>Recall:</strong> TP / (TP + FN) - How many actual positives
            were found?
          </p>
          <p>
            <strong>F1-Score:</strong> Harmonic mean of Precision and Recall
          </p>
        </div>
      </div>
    </div>
  );
}
