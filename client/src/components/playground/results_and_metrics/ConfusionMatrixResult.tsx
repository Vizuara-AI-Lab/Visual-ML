/**
 * Confusion Matrix Result Visualization Component
 */

interface ConfusionMatrixProps {
  result: any;
}

export function ConfusionMatrixResult({ result }: ConfusionMatrixProps) {
  // Debug: Log the result to see what we're receiving
  console.log("ConfusionMatrixResult - Full result:", result);
  
  const confusionMatrix = result.confusion_matrix ?? [];
  const classLabels = result.class_labels ?? [];
  const accuracy = result.accuracy ?? 0;
  const totalSamples = result.total_samples ?? 0;
  const truePositives = result.true_positives ?? {};
  const trueNegatives = result.true_negatives ?? {};
  const falsePositives = result.false_positives ?? {};
  const falseNegatives = result.false_negatives ?? {};

  console.log("ConfusionMatrixResult - Parsed data:", {
    confusionMatrix,
    classLabels,
    accuracy,
    totalSamples,
    truePositives,
    falsePositives,
    trueNegatives,
    falseNegatives,
  });

  // Calculate per-class metrics
  const getMetrics = (classLabel: string) => {
    const tp = truePositives[classLabel] ?? 0;
    const tn = trueNegatives[classLabel] ?? 0;
    const fp = falsePositives[classLabel] ?? 0;
    const fn = falseNegatives[classLabel] ?? 0;

    const precision = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
    const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
    const f1Score =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;

    return { tp, tn, fp, fn, precision, recall, f1Score };
  };

  // Check if we have data
  if (!confusionMatrix || confusionMatrix.length === 0 || !classLabels || classLabels.length === 0) {
    return (
      <div className="space-y-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <h3 className="text-lg font-semibold text-yellow-900 mb-2">
            No Confusion Matrix Data
          </h3>
          <p className="text-sm text-yellow-700">
            The confusion matrix computation completed, but no data was received.
          </p>
          <p className="text-xs text-yellow-600 mt-2">
            Check the browser console for debugging information.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Accuracy Display */}
      <div className="bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-300 rounded-xl p-8 text-center">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-2">
          Classification Accuracy
        </h3>
        <div className="text-6xl font-bold text-purple-600 mb-4">
          {(accuracy * 100).toFixed(2)}%
        </div>
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-lg border border-purple-200">
          <span className="text-sm font-medium text-gray-700">
            {totalSamples.toLocaleString()} test samples
          </span>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">
          Confusion Matrix
        </h4>
        <div className="overflow-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                <th className="border border-gray-300 bg-gray-50 px-4 py-3 text-sm font-semibold text-gray-600"></th>
                <th
                  className="border border-gray-300 bg-gray-100 px-4 py-3 text-xs font-semibold text-gray-700 uppercase tracking-wide"
                  colSpan={classLabels.length}
                >
                  Predicted
                </th>
              </tr>
              <tr>
                <th className="border border-gray-300 bg-gray-100 px-4 py-3 text-xs font-semibold text-gray-700 uppercase tracking-wide">
                  Actual
                </th>
                {classLabels.map((label: string) => (
                  <th
                    key={label}
                    className="border border-gray-300 bg-purple-50 px-4 py-3 text-sm font-semibold text-purple-900"
                  >
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {classLabels.map((actualLabel: string, i: number) => (
                <tr key={actualLabel}>
                  <th className="border border-gray-300 bg-purple-50 px-4 py-3 text-sm font-semibold text-purple-900">
                    {actualLabel}
                  </th>
                  {classLabels.map((predictedLabel: string, j: number) => {
                    const count = confusionMatrix[i]?.[j] ?? 0;
                    const isCorrect = i === j;
                    return (
                      <td
                        key={predictedLabel}
                        className={`border border-gray-300 px-4 py-3 text-center font-semibold ${
                          isCorrect
                            ? "bg-green-100 text-green-900 text-lg"
                            : count > 0
                              ? "bg-red-50 text-red-700"
                              : "bg-white text-gray-500"
                        }`}
                      >
                        {count}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 flex items-center gap-6 text-xs text-gray-600">
          <div className="flex items-center gap-2">
            <span className="inline-block w-4 h-4 bg-green-100 border border-green-300 rounded"></span>
            <span>Correct Predictions (Diagonal)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-4 h-4 bg-red-50 border border-red-200 rounded"></span>
            <span>Misclassifications</span>
          </div>
        </div>
      </div>

      {/* Per-Class Metrics */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">
          Per-Class Metrics
        </h4>
        <div className="space-y-4">
          {classLabels.map((classLabel: string) => {
            const metrics = getMetrics(classLabel);
            return (
              <div
                key={classLabel}
                className="bg-gray-50 border border-gray-200 rounded-lg p-4"
              >
                <h5 className="text-md font-semibold text-gray-800 mb-3">
                  Class: {classLabel}
                </h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <span className="text-xs text-gray-600">True Positive</span>
                    <p className="text-lg font-semibold text-green-600">
                      {metrics.tp}
                    </p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-600">
                      False Positive
                    </span>
                    <p className="text-lg font-semibold text-red-600">
                      {metrics.fp}
                    </p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-600">True Negative</span>
                    <p className="text-lg font-semibold text-green-600">
                      {metrics.tn}
                    </p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-600">
                      False Negative
                    </span>
                    <p className="text-lg font-semibold text-red-600">
                      {metrics.fn}
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-gray-200">
                  <div>
                    <span className="text-xs text-gray-600">Precision</span>
                    <p className="text-lg font-semibold text-blue-600">
                      {metrics.precision.toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-600">Recall</span>
                    <p className="text-lg font-semibold text-blue-600">
                      {metrics.recall.toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-600">F1-Score</span>
                    <p className="text-lg font-semibold text-purple-600">
                      {metrics.f1Score.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-6">
        <h4 className="text-md font-semibold text-purple-900 mb-3">
          📚 Understanding the Confusion Matrix
        </h4>
        <div className="space-y-2 text-sm text-purple-800">
          <p>
            <strong>True Positive (TP):</strong> Correctly predicted as positive
          </p>
          <p>
            <strong>False Positive (FP):</strong> Incorrectly predicted as
            positive (Type I error)
          </p>
          <p>
            <strong>True Negative (TN):</strong> Correctly predicted as negative
          </p>
          <p>
            <strong>False Negative (FN):</strong> Incorrectly predicted as
            negative (Type II error)
          </p>
          <p className="mt-4 pt-4 border-t border-purple-200">
            <strong>Precision</strong> = TP / (TP + FP) - How many positive
            predictions were correct?
            <br />
            <strong>Recall</strong> = TP / (TP + FN) - How many actual positives
            were found?
            <br />
            <strong>F1-Score</strong> = Harmonic mean of Precision and Recall
          </p>
        </div>
      </div>
    </div>
  );
}
