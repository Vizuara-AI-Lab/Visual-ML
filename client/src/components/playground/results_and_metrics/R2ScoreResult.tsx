/**
 * R² Score Result Visualization Component
 */

interface R2ScoreProps {
  result: any;
}

export function R2ScoreResult({ result }: R2ScoreProps) {
  const r2Score = result.r2_score ?? 0;
  const displayValue = result.display_value ?? "N/A";
  const interpretation = result.interpretation ?? "No interpretation available";
  const modelInfo = result.model_info ?? {};

  return (
    <div className="space-y-6">
      {/* R² Score Display */}
      <div className="bg-gradient-to-br from-green-50 to-blue-50 border-2 border-green-300 rounded-xl p-8 text-center">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-2">
          R² Score (Coefficient of Determination)
        </h3>
        <div className="text-6xl font-bold text-green-600 mb-4">
          {displayValue}
        </div>
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-lg border border-green-200">
          <span className="text-sm font-medium text-gray-700">
            {interpretation}
          </span>
        </div>
      </div>

      {/* Score Interpretation Guide */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-md font-semibold text-gray-800 mb-4">
          Understanding R² Score
        </h4>
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-gradient-to-r from-green-500 to-green-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              0.90 - 1.0
            </div>
            <span className="text-sm text-gray-700">
              Excellent fit - Model explains 90%+ of variance
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              0.70 - 0.90
            </div>
            <span className="text-sm text-gray-700">
              Good fit - Model explains 70-90% of variance
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-gradient-to-r from-yellow-500 to-yellow-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              0.50 - 0.70
            </div>
            <span className="text-sm text-gray-700">
              Moderate fit - Model explains 50-70% of variance
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-gradient-to-r from-orange-500 to-orange-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              0.30 - 0.50
            </div>
            <span className="text-sm text-gray-700">
              Weak fit - Model explains 30-50% of variance
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-gradient-to-r from-red-500 to-red-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              &lt; 0.30
            </div>
            <span className="text-sm text-gray-700">
              Poor fit - Model explains &lt;30% of variance
            </span>
          </div>
        </div>
      </div>

      {/* Model Information */}
      {modelInfo && Object.keys(modelInfo).length > 0 && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-800 mb-4">
            Model Information
          </h4>
          <div className="grid grid-cols-2 gap-4">
            {modelInfo.model_id && (
              <div className="col-span-2">
                <span className="text-sm font-medium text-gray-600">
                  Model ID:
                </span>
                <p className="text-sm text-gray-800 font-mono mt-1 break-all">
                  {modelInfo.model_id}
                </p>
              </div>
            )}
            {modelInfo.training_samples && (
              <div>
                <span className="text-sm font-medium text-gray-600">
                  Training Samples:
                </span>
                <p className="text-sm text-gray-800 mt-1">
                  {modelInfo.training_samples.toLocaleString()}
                </p>
              </div>
            )}
            {modelInfo.n_features && (
              <div>
                <span className="text-sm font-medium text-gray-600">
                  Features:
                </span>
                <p className="text-sm text-gray-800 mt-1">
                  {modelInfo.n_features}
                </p>
              </div>
            )}
            {modelInfo.model_path && (
              <div className="col-span-2">
                <span className="text-sm font-medium text-gray-600">
                  Model Path:
                </span>
                <p className="text-sm text-gray-800 font-mono mt-1 break-all">
                  {modelInfo.model_path}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* What is R²? */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h4 className="text-md font-semibold text-blue-900 mb-3">
          📚 What is R² Score?
        </h4>
        <p className="text-sm text-blue-800 leading-relaxed">
          The R² (R-squared) score, also known as the coefficient of
          determination, measures how well your model's predictions match the
          actual values. It ranges from 0 to 1, where 1 means perfect
          predictions and 0 means the model is no better than simply predicting
          the average value for all observations.
        </p>
      </div>
    </div>
  );
}
